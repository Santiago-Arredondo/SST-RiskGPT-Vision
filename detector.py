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
    "guardrail","platform","cable","spill","debris","uneven floor","wet floor sign",
    "rack","shelf","low beam","grinder","saw","cutting","forklift","machine","press","gear",
    "car","vehicle","hoist","car lift","person"
]

def _load_open_vocab(paths: List[str]) -> List[str]:
    tokens: List[str] = []
    for p in paths:
        pp = pathlib.Path(p)
        if not pp.exists():
            continue
        data = yaml.safe_load(pp.read_text(encoding="utf-8")) or {}
        ov = (data.get("meta", {}) or {}).get("open_vocab", []) or []
        tokens.extend(list(ov))
    return list(dict.fromkeys(DEFAULT_OPEN_VOCAB + tokens))

class YoloDetector:
    def __init__(self, model_path: Optional[str] = None, ontology_paths: Optional[List[str]] = None):
        if not YOLO_OK:
            raise RuntimeError("Ultralytics no instalado. pip install ultralytics")
        self.model_path = model_path or "runs/detect/train4/weights/best.pt"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.is_world = "world" in os.path.basename(self.model_path).lower()

        self.ontology_paths = ontology_paths or ["risk_ontology.yaml", "risk_ontology_ext.yaml"]
        self.open_vocab = _load_open_vocab(self.ontology_paths)

        self.names = getattr(self.model, "names", None)

    @staticmethod
    def _auto_model() -> str:
        for p in ["models/best.pt", "models/yolov8m.pt", "yolov8m.pt", "yolov8n.pt"]:
            if os.path.exists(p):
                return p
        return "yolov8m.pt"

    def predict(self, image_or_path: Any, conf: float = 0.14, iou: float = 0.6,
                imgsz: int = 640, device: str = "cpu",
                classes_prompt: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if self.is_world:
            prompts = classes_prompt or self.open_vocab
            try:
                if hasattr(self.model, "set_classes"):
                    self.model.set_classes(prompts)
                else:
                    self.model.set_classes = prompts
            except Exception:
                pass

        results = self.model.predict(source=image_or_path, imgsz=imgsz, conf=conf, iou=iou,
                                     device=device, verbose=False)
        dets: List[Dict[str, Any]] = []
        if not results:
            return dets
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None:
            return dets
        boxes = r.boxes
        names = self.names or getattr(r, "names", None) or {}
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf_i = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
            raw = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            norm = self._normalize_name(raw)
            dets.append({"cls": norm, "conf": conf_i, "box": [x1, y1, x2, y2], "raw": raw})
        return dets

    @staticmethod
    def _normalize_name(name: str) -> str:
        n = name.lower().strip()
        mapping = {
            "cell phone": "phone", "mobile": "phone", "smartphone": "phone",
            "monitor": "screen", "tv": "screen",
            "ladder like": "ladder like",
            "wet floor": "wet floor sign",
        }
        return mapping.get(n, n)

    def to_dicts(self, preds) -> List[Dict[str, Any]]:
        if not preds:
            return []
        if isinstance(preds, list) and preds and isinstance(preds[0], dict):
            return preds
        out: List[Dict[str, Any]] = []
        try:
            for p in preds:
                out.append({
                    "cls": p.get("cls") or p.get("class"),
                    "conf": float(p.get("conf", 0.0)),
                    "box": list(p.get("box") or p.get("bbox") or [])
                })
        except Exception:
            return []
        return out

    def classes_from_dicts(self, dets: List[Dict[str, Any]]) -> List[str]:
        return sorted({d.get("cls") for d in dets if isinstance(d, dict) and d.get("cls")})
