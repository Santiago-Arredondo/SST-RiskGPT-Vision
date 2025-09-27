from __future__ import annotations
import os, cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

from rules_engine import RiskEngine

FLOOR_HAZARD = {"pallet", "cable", "spill", "toolbox"}
MACHINES     = {"machine", "press", "saw", "conveyor"}
VEHICLES     = {"forklift", "truck", "car", "bus", "excavator"}

def _center(b): x1,y1,x2,y2=b; return (0.5*(x1+x2), 0.5*(y1+y2))
def _iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    xi1, yi1 = max(ax1,bx1), max(ay1,by1)
    xi2, yi2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    aA = max(0, ax2-ax1)*max(0, ay2-ay1)
    aB = max(0, bx2-bx1)*max(0, by2-by1)
    return inter/max(1e-6, aA+aB-inter)
def _ndist(a,b,w,h):
    (cx1,cy1),(cx2,cy2)=_center(a),_center(b)
    dx,dy=(cx1-cx2)/max(1e-6,w),(cy1-cy2)/max(1e-6,h)
    return (dx*dx+dy*dy)**0.5

def _parse_yolo_results(results, names):
    dets=[]
    if not results: return dets
    r=results[0]
    if not hasattr(r,"boxes") or r.boxes is None: return dets
    for i in range(len(r.boxes)):
        cls_id=int(r.boxes.cls[i].item())
        conf=float(r.boxes.conf[i].item())
        x1,y1,x2,y2=r.boxes.xyxy[i].tolist()
        dets.append({"cls": names.get(cls_id,str(cls_id)), "conf": conf, "box":[x1,y1,x2,y2]})
    return dets

def proximity_tokens(dets, w, h) -> Set[str]:
    toks=set()
    persons=[d for d in dets if d["cls"]=="person"]
    ladders=[d for d in dets if d["cls"]=="ladder"]
    floor  =[d for d in dets if d["cls"] in FLOOR_HAZARD]
    machs  =[d for d in dets if d["cls"] in MACHINES]

    # Obstáculos piso
    for p in persons:
        px1,py1,px2,py2=p["box"]; feet_y=py2; cx_p=0.5*(px1+px2)
        for hz in floor:
            hx1,hy1,hx2,hy2=hz["box"]; cx_h=0.5*(hx1+hx2)
            if abs(cx_p-cx_h)<0.12*w and (hy1-0.02*h)<=feet_y<=(hy2+0.06*h):
                toks.add("foot_near_floor_hazard")

    # Altura (ladder o base alta)
    if ladders and persons:
        toks.add("at_height")
    else:
        for p in persons:
            _,py1,_,py2=p["box"]; base_rel=py2/float(h); top_rel=py1/float(h)
            if base_rel<0.68 and top_rel<0.42:
                toks.add("at_height")

    # Proximidad a máquina
    for p in persons:
        for m in machs:
            if _ndist(p["box"],m["box"],w,h)<0.22 or _iou(p["box"],m["box"])>0.02:
                toks.add("near_person_machine")

    return toks

def analyze_video(
    video_path: str,
    out_path: str,
    model_path: str|None=None,
    stride: int = 5,
    risk_sustain: int = 6,
    conf: float = 0.35,
    iou: float  = 0.60,
    imgsz: int  = 960
) -> Dict[str,Any]:
    if not YOLO_OK: raise RuntimeError("Ultralytics no disponible. pip install ultralytics")
    default_best=os.path.join("models","best.pt")
    use_model=model_path or (default_best if os.path.exists(default_best) else "yolov8n.pt")
    model=YOLO(use_model); names=model.names if hasattr(model,"names") else {i:str(i) for i in range(1000)}

    eng = RiskEngine("risk_ontology.yaml")
    cap=cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise FileNotFoundError(video_path)

    fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer=cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    counters: Dict[str,int]={}
    opened: Dict[str,int]={}
    risk_names: Dict[str,str]={}
    timeline: List[Dict[str,Any]]=[]
    classes_seen=set()

    idx=0
    last_dets=[]
    while True:
        ok,frame=cap.read()
        if not ok: break
        run_now=(idx%max(1,stride)==0)
        if run_now:
            res=model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu")
            dets=_parse_yolo_results(res,names); last_dets=dets
        else:
            dets=last_dets

        present=sorted({d["cls"] for d in dets})
        classes_seen.update(present)
        toks=proximity_tokens(dets,w,h)
        present_ctx=sorted(set(present).union(toks))

        risks=eng.infer(present_ctx)
        current=[]
        for r in risks:
            rid=r["id"]; current.append(rid)
            risk_names.setdefault(rid, r.get("nombre",rid))
            counters[rid]=risk_sustain

        for rid in list(counters.keys()):
            if rid not in current:
                counters[rid]-=1
                if counters[rid]<=0: counters[rid]=0

        active=[rid for rid,c in counters.items() if c>0]
        # open/close segments
        for rid in set(active)-set(opened.keys()):
            opened[rid]=idx
        for rid in set(opened.keys())-set(active):
            s=opened.pop(rid); e=max(idx-1,s)
            timeline.append({"risk": rid, "nombre": risk_names.get(rid,rid),
                             "start_frame": int(s), "end_frame": int(e),
                             "start_sec": round(s/fps,3), "end_sec": round(e/fps,3)})

        # render overlay sencillo
        txt=f"Frame {idx+1}/{total}  t={idx/fps:.2f}s  Risks: {', '.join(risk_names.get(r,r) for r in active) if active else 'None'}"
        overlay=frame.copy(); cv2.rectangle(overlay,(0,0),(w,28),(30,30,30),-1)
        frame=cv2.addWeighted(overlay,0.5,frame,0.5,0)
        cv2.putText(frame, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),1, cv2.LINE_AA)
        writer.write(frame)
        idx+=1

    for rid,s in opened.items():
        e=idx-1
        timeline.append({"risk": rid, "nombre": risk_names.get(rid,rid),
                         "start_frame": int(s), "end_frame": int(e),
                         "start_sec": round(s/fps,3), "end_sec": round(e/fps,3)})

    cap.release(); writer.release()

    # stats + recomendaciones
    stats={}
    for seg in timeline:
        rid=seg["risk"]
        dur=seg["end_frame"]-seg["start_frame"]+1
        acc=stats.setdefault(rid, {"events":0,"duration_frames":0})
        acc["events"]+=1; acc["duration_frames"]+=max(0,dur)
    for rid,acc in stats.items(): acc["duration_sec"]=round(acc["duration_frames"]/fps,3)
    uniq=sorted(stats.keys())
    recs={rid: eng.recommendations(rid) for rid in uniq}

    return {
        "timeline": sorted(timeline, key=lambda s: (s["start_frame"], s["risk"])),
        "classes_present": sorted(classes_seen),
        "risk_stats": stats,
        "recommendations": recs,
        "risk_names": {rid: risk_names.get(rid,rid) for rid in uniq},
    }
