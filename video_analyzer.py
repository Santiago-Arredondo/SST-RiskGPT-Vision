import argparse, os, cv2, json
from collections import defaultdict
from typing import Dict, List, Set
from PIL import Image
from detector import YoloDetector
from rules_engine import RiskEngine

def put_label(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def analyze_video(input_path: str, output_path: str, model_path: str = "yolov8n.pt",
                  stride: int = 5, risk_sustain: int = 3, conf: float = 0.25, iou: float = 0.45):
    det = YoloDetector(model_path=model_path, conf=conf, iou=iou)
    eng = RiskEngine("risk_ontology.yaml")

    cap = cv2.VideoCapture(input_path)
    assert cap.isOpened(), f"No se puede abrir el video: {input_path}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    timeline: List[Dict] = []
    counters = defaultdict(int)
    active: Set[str] = set()
    last_dets = []
    last_risks = set()

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        if f % stride == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = det.to_dicts(det.predict(pil))
            present = det.classes_from_dicts(dets)
            risks = eng.infer(present)
            risk_ids = {r["id"] for r in risks}

            for rid in risk_ids:
                counters[rid] += 1
                if counters[rid] >= risk_sustain:
                    active.add(rid)
            for rid in list(active):
                if rid not in risk_ids:
                    counters[rid] = 0
                    active.remove(rid)

            timeline.append({"frame": f, "time_s": round(f / fps, 2), "risks": sorted(list(risk_ids))})
            last_dets = dets
            last_risks = risk_ids

        for d in last_dets:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            put_label(frame, f"{d['cls']} {d['conf']:.2f}", x1 + 3, y1 + 18)

        y0 = 30
        if active:
            put_label(frame, "Riesgos activos:", 10, y0)
            y = y0 + 24
            for rid in sorted(active):
                put_label(frame, f"- {rid}", 10, y)
                y += 22
        elif last_risks:
            put_label(frame, "Riesgos recientes:", 10, 30)
            y = 54
            for rid in sorted(last_risks):
                put_label(frame, f"- {rid}", 10, y)
                y += 22

        out.write(frame)
        f += 1

    cap.release(); out.release()
    return timeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video de entrada (mp4/avi)")
    ap.add_argument("--output", default="annotated.mp4", help="video anotado de salida")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--stride", type=int, default=5, help="procesar 1 de cada N frames")
    ap.add_argument("--risk_sustain", type=int, default=3, help="frames consecutivos para marcar riesgo activo")
    args = ap.parse_args()

    tl = analyze_video(args.input, args.output, model_path=args.model,
                       stride=args.stride, risk_sustain=args.risk_sustain)
    print(json.dumps({"timeline": tl, "output_video": args.output}, ensure_ascii=False, indent=2))
