# video_analyzer.py
# Analiza video con YOLO, aplica fallback COCO (clases limitadas) cuando falte contexto,
# genera tokens de proximidad y construye timeline+recomendaciones vía RiskEngine.

from __future__ import annotations
import os
from typing import List, Dict, Any, Tuple, Set, Optional
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

from rules_engine import RiskEngine


# ---------- utilidades de imagen ----------
def _ensure_color(frame: np.ndarray) -> np.ndarray:
    if frame is None:
        raise ValueError("Frame vacío (None).")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def _center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _norm_dist_boxes(b1, b2, w, h):
    (cx1, cy1), (cx2, cy2) = _center(b1), _center(b2)
    dx, dy = (cx1 - cx2) / max(1e-6, w), (cy1 - cy2) / max(1e-6, h)
    return float((dx*dx + dy*dy) ** 0.5)

def _norm_dist_point_box(px, py, box, w, h):
    cx, cy = _center(box)
    dx, dy = (px - cx) / max(1e-6, w), (py - cy) / max(1e-6, h)
    return float((dx*dx + dy*dy) ** 0.5)

def _iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    union = a1 + a2 - inter + 1e-6
    return float(inter / union)


# ---------- parseos ----------
def _parse_yolo_results(results, names: Dict[int, str]) -> List[Dict[str, Any]]:
    dets: List[Dict[str, Any]] = []
    if not results:
        return dets
    r = results[0]
    if not hasattr(r, "boxes") or r.boxes is None:
        return dets
    boxes = r.boxes
    for i in range(len(boxes)):
        cls_raw = boxes.cls[i]
        conf_raw = boxes.conf[i]
        cls_id = int(getattr(cls_raw, "item", lambda: cls_raw)())
        conf = float(getattr(conf_raw, "item", lambda: conf_raw)())
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        dets.append({"cls": names.get(cls_id, str(cls_id)), "conf": conf, "box": [x1, y1, x2, y2]})
    return dets

def _parse_pose_wrists(results_pose) -> List[Tuple[float, float]]:
    """Devuelve lista de (x,y) de muñecas (COCO: 9 izquierda, 10 derecha) usando YOLOv8-Pose."""
    wrists: List[Tuple[float, float]] = []
    if not results_pose:
        return wrists
    r = results_pose[0]
    if not hasattr(r, "keypoints") or r.keypoints is None:
        return wrists
    try:
        kps = r.keypoints.xy.cpu().numpy()  # (N, K, 2)
    except Exception:
        return wrists
    for person_kp in kps:
        if person_kp.shape[0] >= 11:
            lw = person_kp[9]   # left wrist
            rw = person_kp[10]  # right wrist
            for (x, y) in (lw, rw):
                if float(x) > 0 and float(y) > 0:
                    wrists.append((float(x), float(y)))
    return wrists


# ---------- mapeos / fusión ----------
def _unify_class(name: str) -> str:
    """Unifica clases de distintos modelos (p.ej. COCO) a vocabulario del motor de reglas."""
    n = name.lower().strip()
    # pantallas
    if n in {"tv", "laptop", "monitor"}:
        return "screen"
    # asientos
    if n in {"chair", "bench", "sofa"}:
        return "chair"
    # teléfono
    if n in {"cell phone", "mobile", "phone"}:
        return "phone"
    # vehículos
    if n in {"car", "truck", "bus"}:
        return n  # mismas
    # resto tal cual
    return name

def _merge_by_iou(primary: List[Dict[str, Any]], fallback: List[Dict[str, Any]], iou_thr=0.35) -> List[Dict[str, Any]]:
    """Inserta detecciones fallback que no colisionen (IoU bajo) con las primarias."""
    merged = list(primary)
    for fb in fallback:
        keep = True
        for pr in primary:
            if _unify_class(fb["cls"]) == _unify_class(pr["cls"]) and _iou(fb["box"], pr["box"]) > iou_thr:
                keep = False
                break
        if keep:
            merged.append(fb)
    return merged


# ---------- render ----------
def _draw_detections(frame: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cls_name = _unify_class(d["cls"])
        conf = d.get("conf", 0.0)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(img, f"{cls_name} {conf:.2f}", (x1 + 2, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
    return img

def _draw_header(img: np.ndarray, text_lines: List[str]) -> np.ndarray:
    out = img.copy()
    h = 24 * (len(text_lines) + 1)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], h), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.5, out, 0.5, 0)
    y = 20
    for line in text_lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return out


# ---------- tokens de proximidad ----------
VEHICLES = {"forklift", "truck", "car", "excavator", "bus"}
MACHINES = {"conveyor", "machine", "saw", "press"}
FLOOR_HAZ = {"pallet", "cable", "spill", "toolbox"}

def proximity_tokens(
    dets: List[Dict[str, Any]],
    w: int, h: int,
    wrists: Optional[List[Tuple[float, float]]] = None,
    near_thr: float = 0.20,
    hand_thr: float = 0.10,
    iou_touch: float = 0.01,
) -> Set[str]:
    wrists = wrists or []
    toks: Set[str] = set()

    persons = [d for d in dets if _unify_class(d["cls"]) == "person"]
    vehicles = [d for d in dets if _unify_class(d["cls"]) in VEHICLES]
    machines = [d for d in dets if _unify_class(d["cls"]) in MACHINES]
    floor_haz = [d for d in dets if _unify_class(d["cls"]) in FLOOR_HAZ]

    # Persona cerca de vehículo (atropellamiento)
    for p in persons:
        for v in vehicles:
            if _norm_dist_boxes(p["box"], v["box"], w, h) < near_thr or _iou(p["box"], v["box"]) > iou_touch:
                toks.add("near_person_vehicle")

    # Persona cerca de máquina (contacto con partes móviles)
    for p in persons:
        for m in machines:
            if _norm_dist_boxes(p["box"], m["box"], w, h) < near_thr or _iou(p["box"], m["box"]) > iou_touch:
                toks.add("near_person_machine")

    # Muñecas cerca de máquina (si hay pose)
    for (wx, wy) in wrists:
        for m in machines:
            x1, y1, x2, y2 = m["box"]
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                toks.add("near_hand_machine")
            else:
                d = _norm_dist_point_box(wx, wy, m["box"], w, h)
                if d < hand_thr:
                    toks.add("near_hand_machine")

    # Pies/persona cerca de obstáculo en piso (caídas mismo nivel)
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        feet_y = py2
        cx_p = 0.5 * (px1 + px2)
        for hz in floor_haz:
            hx1, hy1, hx2, hy2 = hz["box"]
            cx_h = 0.5 * (hx1 + hx2)
            if abs(cx_p - cx_h) < 0.10 * w and hy1 <= feet_y <= hy2 + 0.05 * h:
                toks.add("foot_near_floor_hazard")

    return toks


# ---------- analizador principal ----------
FALLBACK_ALLOWED = {
    # oficina / locativos ligeros
    "laptop", "monitor", "tv", "keyboard", "chair", "cell phone",
    # vía / yardas
    "car", "truck", "bus",
}

def analyze_video(
    video_path: str,
    out_path: str,
    model_path: Optional[str] = None,
    stride: int = 5,
    risk_sustain: int = 6,
    conf: float = 0.35,
    iou: float = 0.6,
    imgsz: int = 640,
    use_pose: bool = False,
    pose_model_path: str = "yolov8n-pose.pt",
    near_thr: float = 0.20,
    hand_thr: float = 0.10,
    iou_touch: float = 0.01,
    fallback_model: str = "yolov8n.pt",
) -> Dict[str, Any]:
    """
    Procesa un video, escribe un MP4 anotado y devuelve:
      - timeline: lista de segmentos con riesgo activo
      - classes_present: clases únicas vistas (unificadas)
      - risk_stats: eventos y duración por riesgo
      - recommendations: recomendaciones por riesgo (desde RiskEngine)
      - risk_names: etiquetas legibles por riesgo
    """
    if not YOLO_OK:
        raise RuntimeError("Ultralytics/YOLO no está disponible. Instala 'ultralytics'.")

    # Modelo de detección principal
    if not model_path or not os.path.exists(model_path):
        default_best = os.path.join("models", "best.pt")
        model_path = default_best if os.path.exists(default_best) else "yolov8n.pt"
    model = YOLO(model_path)
    names_main = getattr(model, "names", {i: str(i) for i in range(1000)})

    # Modelo de pose (opcional)
    pose_model = YOLO(pose_model_path) if use_pose else None

    # Fallback COCO (carga on-demand la primera vez que lo necesitemos)
    fallback_yolo = None

    engine = RiskEngine("risk_ontology.yaml")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("No se pudo abrir VideoWriter. Instala FFmpeg o cambia el FOURCC.")

    frame_idx = 0
    last_dets: List[Dict[str, Any]] = []
    last_wrists: List[Tuple[float, float]] = []

    counters: Dict[str, int] = {}
    opened: Dict[str, int] = {}
    risk_names: Dict[str, str] = {}
    timeline: List[Dict[str, Any]] = []
    classes_seen = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = _ensure_color(frame)

        # ¿procesar este frame?
        run_now = (frame_idx % max(1, stride) == 0)

        if run_now:
            # detección principal
            res = model.predict(source=frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu")
            dets = _parse_yolo_results(res, names_main)

            # ¿Hace falta contexto? (pocas clases útiles)
            unified = {_unify_class(d["cls"]) for d in dets}
            needs_fb = len(unified.intersection({"person", "cable", "pallet", "spill", "machine", "truck", "forklift", "chair", "screen"})) < 2

            if needs_fb:
                if fallback_yolo is None:
                    fallback_yolo = YOLO(fallback_model)
                    fb_names = getattr(fallback_yolo, "names", {})
                    # ids de clases permitidas
                    allowed_ids = {i for i, n in fb_names.items() if n in FALLBACK_ALLOWED}
                # inferencia fallback (rápida)
                fb_res = fallback_yolo.predict(source=frame, conf=max(0.25, conf*0.9), iou=0.6, imgsz=min(640, imgsz), verbose=False, device="cpu")
                fb_dets = _parse_yolo_results(fb_res, fallback_yolo.names)
                # filtra por allowlist
                fb_dets = [d for d in fb_dets if fallback_yolo.names and fallback_yolo.names.get(
                    list(fallback_yolo.names.keys())[list(fallback_yolo.names.values()).index(d["cls"])]
                ) if _unify_class(d["cls"]) in {"screen", "chair", "phone", "car", "truck", "bus"}] if hasattr(fallback_yolo, "names") else fb_dets
                # fusión por IoU
                dets = _merge_by_iou(dets, fb_dets, iou_thr=0.35)

            # Pose (muñecas)
            wrists: List[Tuple[float, float]] = []
            if pose_model is not None:
                try:
                    res_pose = pose_model.predict(source=frame, imgsz=imgsz, verbose=False, device="cpu")
                    wrists = _parse_pose_wrists(res_pose)
                except Exception:
                    wrists = []

            last_dets, last_wrists = dets, wrists
        else:
            dets, wrists = last_dets, last_wrists

        present = sorted({_unify_class(d["cls"]) for d in dets})
        classes_seen.update(present)

        toks = proximity_tokens(dets, w, h, wrists=wrists, near_thr=near_thr, hand_thr=hand_thr, iou_touch=iou_touch)
        present_with_ctx = sorted(set(present).union(toks))

        # Inferencia de riesgos
        risks = engine.infer(present_with_ctx)
        current_rids = []
        for r in risks:
            rid = r.get("id", "")
            if not rid:
                continue
            current_rids.append(rid)
            if rid not in risk_names:
                risk_names[rid] = r.get("nombre", rid)
            counters[rid] = risk_sustain

        # decaimiento de riesgos no presentes
        for rid in list(counters.keys()):
            if rid not in current_rids:
                counters[rid] -= 1
                if counters[rid] <= 0:
                    counters[rid] = 0

        active = sorted([rid for rid, c in counters.items() if c > 0])

        # timeline open/close
        active_set = set(active)
        opened_set = set(opened.keys())

        # abrir nuevos
        for rid in active_set - opened_set:
            opened[rid] = frame_idx

        # cerrar los que dejaron de estar activos
        for rid in opened_set - active_set:
            start_f = opened.pop(rid)
            end_f = max(frame_idx - 1, start_f)
            timeline.append({
                "risk": rid,
                "nombre": risk_names.get(rid, rid),
                "start_frame": int(start_f),
                "end_frame": int(end_f),
                "start_sec": round(start_f / fps, 3),
                "end_sec": round(end_f / fps, 3),
            })

        # render
        img_det = _draw_detections(frame, dets)
        info = f"Frame {frame_idx+1}/{total} | t={frame_idx/fps:.2f}s"
        risks_line = "Riesgos activos: " + (", ".join([risk_names.get(r, r) for r in active]) if active else "Ninguno")
        tokens_line = "Tokens: " + (", ".join(sorted(toks)) if toks else "(none)")
        writer.write(_draw_header(img_det, [info, risks_line, tokens_line]))

        frame_idx += 1

    # cerrar segmentos abiertos al final
    for rid, start_f in opened.items():
        timeline.append({
            "risk": rid,
            "nombre": risk_names.get(rid, rid),
            "start_frame": int(start_f),
            "end_frame": int(frame_idx - 1),
            "start_sec": round(start_f / fps, 3),
            "end_sec": round((frame_idx - 1) / fps, 3),
        })

    cap.release()
    writer.release()

    # Estadísticos y recomendaciones
    risk_stats: Dict[str, Dict[str, Any]] = {}
    for seg in timeline:
        rid = seg["risk"]
        dur_frames = seg["end_frame"] - seg["start_frame"] + 1
        acc = risk_stats.setdefault(rid, {"events": 0, "duration_frames": 0})
        acc["events"] += 1
        acc["duration_frames"] += max(0, dur_frames)

    for rid, acc in risk_stats.items():
        acc["duration_sec"] = round(acc["duration_frames"] / fps, 3)

    unique_risks = sorted(risk_stats.keys())
    recommendations = {rid: engine.recommendations(rid) for rid in unique_risks}

    return {
        "timeline": sorted(timeline, key=lambda s: (s["start_frame"], s["risk"])),
        "classes_present": sorted(classes_seen),
        "risk_stats": risk_stats,
        "recommendations": recommendations,
        "risk_names": {rid: risk_names.get(rid, rid) for rid in unique_risks},
    }


if __name__ == "__main__":
    # Prueba CLI
    import argparse, tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "annotated.mp4"))
    ap.add_argument("--model", default=None)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--sustain", type=int, default=6)
    ap.add_argument("--use_pose", action="store_true")
    ap.add_argument("--near_thr", type=float, default=0.20)
    ap.add_argument("--hand_thr", type=float, default=0.10)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.60)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    res = analyze_video(
        args.video, args.out, model_path=args.model, stride=args.stride,
        risk_sustain=args.sustain, use_pose=args.use_pose,
        near_thr=args.near_thr, hand_thr=args.hand_thr,
        conf=args.conf, iou=args.iou, imgsz=args.imgsz
    )
    print("Timeline (primeros 3):", res["timeline"][:3])
    print("Escrito:", args.out)
