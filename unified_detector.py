# unified_detector.py
from __future__ import annotations
import os, pathlib, yaml
from typing import Any, Dict, List, Optional

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

DEFAULT_OPEN_VOCAB = [
    "ladder","ladder like","scaffold","rebar","roof","edge","hole","trench",
    "guardrail","platform","cable","wire","cord","extension cord","power cord","hose","rope",
    "debris","uneven floor","wet floor sign","rack","shelf","low beam",
    "grinder","saw","cutting","forklift","car","vehicle","hoist","car lift",
    "person","screen","laptop","keyboard","mouse"
]

def _load_open_vocab(paths: List[str]) -> List[str]:
    toks: List[str] = []
    for p in paths:
        pp = pathlib.Path(p)
        if pp.exists():
            data = yaml.safe_load(pp.read_text(encoding="utf-8")) or {}
            ov = (data.get("meta", {}) or {}).get("open_vocab", []) or []
            toks.extend(ov)
    seen, out = set(), []
    for t in (DEFAULT_OPEN_VOCAB + toks):
        if t not in seen:
            seen.add(t); out.append(t)
    return out

class UnifiedDetector:
    """Un solo wrapper para detección de objetos y pose (si hay modelo de pose)."""

    def __init__(self, detect_model: Optional[str] = None, pose_model: Optional[str] = None, ontology_paths: Optional[List[str]] = None):
        if not YOLO_OK:
            raise RuntimeError("Ultralytics no instalado. pip install ultralytics")

        self.detect_model_path = detect_model or self._auto_detect_model()
        if not os.path.exists(self.detect_model_path):
            raise FileNotFoundError(f"No existe el modelo de detección: {self.detect_model_path}")
        self.detect_model = YOLO(self.detect_model_path)
        self.is_world = "world" in os.path.basename(self.detect_model_path).lower()
        self.detect_names = getattr(self.detect_model, "names", None)

        self.pose_model = YOLO(pose_model) if pose_model and os.path.exists(pose_model) else None

        self.ontology_paths = ontology_paths or ["risk_ontology.yaml", "risk_ontology_ext.yaml"]
        self.open_vocab = _load_open_vocab(self.ontology_paths)

    @staticmethod
    def _auto_detect_model() -> str:
        for p in ["models/best.pt", "models/yolov8s.pt", "yolov8n.pt"]:
            if os.path.exists(p):
                return p
        return "yolov8n.pt"

    @staticmethod
    def _normalize_name(name: str) -> str:
        n = (name or "").lower().strip()
        mapping = {
            "cell phone": "phone", "mobile": "phone", "smartphone": "phone",
            "monitor": "screen", "tv": "screen",
            "wet floor": "wet floor sign",
            "ladder like": "ladder like",
        }
        return mapping.get(n, n)

    def _predict_detect(self, image: Any, conf: float, iou: float, imgsz: int, device: str, classes_prompt):
        if self.is_world:
            prompts = classes_prompt or self.open_vocab
            try:
                if hasattr(self.detect_model, "set_classes"):
                    self.detect_model.set_classes(prompts)
                else:
                    self.detect_model.set_classes = prompts
            except Exception:
                pass
        res = self.detect_model.predict(source=image, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
        out: List[Dict] = []
        if not res: return out
        r = res[0]
        if not getattr(r, "boxes", None): return out
        names = self.detect_names or getattr(r, "names", None) or {}
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i].item())
            conf_i = float(r.boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in r.boxes.xyxy[i].tolist()]
            raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            out.append({"cls": self._normalize_name(raw), "conf": conf_i, "box": [x1, y1, x2, y2], "raw": raw})
        return out

    def _predict_pose(self, image: Any, imgsz: int, device: str):
        if self.pose_model is None:
            return []
        res = self.pose_model.predict(source=image, imgsz=imgsz, device=device, verbose=False)
        out: List[Dict] = []
        if not res: return out
        r = res[0]
        if not getattr(r, "keypoints", None): return out
        boxes = r.boxes
        kps = r.keypoints  # (n,17,3)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            conf_i = float(boxes.conf[i].item())
            pts = []
            try:
                arr = kps[i].data.cpu().numpy().tolist()
            except Exception:
                arr = getattr(kps[i], "xy", [])
                arr = arr.tolist() if hasattr(arr, "tolist") else []
            for kp in arr:
                if isinstance(kp, list) and len(kp) >= 3:
                    pts.append((float(kp[0]), float(kp[1]), float(kp[2])))
            out.append({"box": [x1, y1, x2, y2], "conf": conf_i, "keypoints": pts})
        return out

    def infer(self, image_or_path: Any, conf: float = 0.25, iou: float = 0.60, imgsz: int = 640, device: str = "cpu", classes_prompt=None):
        dets = self._predict_detect(image_or_path, conf, iou, imgsz, device, classes_prompt)
        poses = self._predict_pose(image_or_path, imgsz, device)
        classes_present = sorted({d["cls"] for d in dets if isinstance(d, dict) and d.get("cls")})
        return {"detections": dets, "poses": poses, "classes_present": classes_present}
