from __future__ import annotations
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any
from PIL import Image

from ui_gradio import (
    yolo_detect,
    image_tokens,
    load_ontology,
    make_context_tokens,
    infer_risks,
    _ensure_pil,
)
from pose_utils import ergonomic_tokens

ONTO = load_ontology("risk_ontology.yaml")

def _iou(a, b):
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    return inter / (a1 + a2 - inter + 1e-6)

class IoUTracker:
    def __init__(self, iou_thr=0.3, max_age=10):
        self.tracks = {}   # id -> (box, cls, age)
        self.next_id = 1
        self.iou_thr = iou_thr
        self.max_age = max_age

    def update(self, dets: List[Dict]) -> Dict[int, Dict]:
        assigned = set()

        # Asociar con tracks existentes
        for tid, (tbox, tcls, age) in list(self.tracks.items()):
            best = -1
            best_iou = 0.0
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                if d["cls"] != tcls:
                    continue
                iou = _iou(tbox, d["box"])
                if iou > best_iou:
                    best_iou = iou
                    best = j
            if best_iou >= self.iou_thr:
                self.tracks[tid] = (dets[best]["box"], dets[best]["cls"], 0)
                dets[best]["track_id"] = tid
                assigned.add(best)
            else:
                age += 1
                if age > self.max_age:
                    del self.tracks[tid]
                else:
                    self.tracks[tid] = (tbox, tcls, age)

        # Crear nuevos tracks
        for j, d in enumerate(dets):
            if j in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = (d["box"], d["cls"], 0)
            d["track_id"] = tid

        return {tid: {"box": b, "cls": c, "age": a} for tid, (b, c, a) in self.tracks.items()}

def analyze_video(path_in: str,
                  path_out: str = "out.mp4",
                  model_path: str | None = None,
                  conf: float = 0.35,
                  iou_nms: float = 0.60,
                  near_thr: float = 0.20,
                  confirm_frames: int = 3) -> str:
    cap = cv2.VideoCapture(path_in)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {path_in}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_out, fourcc, fps, (w, h))

    tracker = IoUTracker()
    risk_buffer: Dict[str, int] = {}  # conteo de frames consecutivos por riesgo

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets, W, H = yolo_detect(pil, model_path=model_path, conf=conf, iou=iou_nms, imgsz=960)

        tracker.update(dets)

        present = {d["cls"] for d in dets}
        toks = image_tokens(dets, W, H, near_thr=near_thr, iou_touch=0.01, strict=False)

        # Pose -> tokens ergonómicos
        try:
            toks |= ergonomic_tokens(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dets)
        except Exception:
            pass

        ctx = make_context_tokens(ONTO, present)
        present_ctx = present | toks | ctx
        risks = infer_risks(present_ctx, ONTO)  # lista de dicts con id/nombre/tipo

        # Confirmación temporal de riesgos
        active = []
        seen_ids = set()
        for r in risks:
            rid = r["id"]
            seen_ids.add(rid)
            risk_buffer[rid] = risk_buffer.get(rid, 0) + 1
            if risk_buffer[rid] >= confirm_frames:
                active.append(r)
        # Decaer riesgos no vistos en este frame
        for rid in list(risk_buffer.keys()):
            if rid not in seen_ids:
                risk_buffer[rid] = max(0, risk_buffer[rid] - 1)

        # Dibujo
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            lbl = f'{d["cls"]} {d["conf"]:.2f}'
            if "track_id" in d:
                lbl += f' #{d["track_id"]}'
            cv2.putText(frame, lbl, (x1 + 3, max(10, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)

        y = 22
        cv2.putText(frame, "RIESGOS (confirmados)", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        y += 20
        for r in active[:10]:
            cv2.putText(frame, f"- {r.get('nombre', r['id'])}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            y += 18

        out.write(frame)

    cap.release()
    out.release()
    return path_out
