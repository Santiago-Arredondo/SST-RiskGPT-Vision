# detector.py
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False


OPEN_VOCAB_PROMPTS = [
    # Clases útiles en obra/altura (open-vocabulary)
    "ladder", "scaffold", "rebar", "roof", "edge", "harness", "safety harness",
    "guardrail", "platform", "crane hook", "toolbox", "spill", "cable"
]

class YoloDetector:
    """
    Pequeño wrapper para Ultralytics YOLO / YOLO-World.
    - Soporta pesos 'clásicos' (COCO) y 'world' (open-vocabulary).
    - Normaliza nombres de clase.
    """
    def __init__(self, model_path: Optional[str] = None):
        if not YOLO_OK:
            raise RuntimeError("Ultralytics no instalado. pip install ultralytics")
        # Selección automática si no se entrega ruta
        self.model_path = model_path or self._auto_model()
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No existe el modelo: {self.model_path}")
        self.model = YOLO(self.model_path)

        # ¿Es un modelo YOLO-World?
        self.is_world = ("world" in os.path.basename(self.model_path).lower())

        # Clases de nombres para mapping posteriormente
        self.names = None
        try:
            self.names = self.model.names
        except Exception:
            self.names = None

    @staticmethod
    def _auto_model() -> str:
        # Preferencias: tu best entrenado, luego v8m, luego v8n
        for p in ["models/best.pt", "models/yolov8m.pt", "yolov8m.pt", "yolov8n.pt"]:
            if os.path.exists(p):
                return p
        # Última carta: que Ultralytics lo resuelva (puede descargar)
        return "yolov8m.pt"

    def predict(
        self,
        image_or_path: Any,
        conf: float = 0.25,
        iou: float = 0.6,
        imgsz: int = 640,
        device: str = "cpu",
        classes_prompt: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Devuelve lista de dicts con: {'cls': str, 'conf': float, 'box': [x1,y1,x2,y2]}
        """
        if self.is_world:
            # YOLO-World: define clases “abiertas”
            prompts = classes_prompt or OPEN_VOCAB_PROMPTS
            try:
                # Algunos builds usan set_classes, otros model.set_classes
                if hasattr(self.model, "set_classes"):
                    self.model.set_classes(prompts)
                else:
                    self.model.set_classes = prompts  # fallback silencioso
            except Exception:
                pass

        results = self.model.predict(
            source=image_or_path, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False
        )
        dets: List[Dict[str, Any]] = []
        if not results:
            return dets
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None:
            return dets
        boxes = r.boxes

        # names puede venir del modelo o del resultado
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
        # Normaliza variantes comunes
        mapping = {
            "tv": "screen", "monitor": "screen", "laptop": "laptop", "keyboard": "keyboard",
            "cell phone": "phone", "mobile": "phone", "smartphone": "phone",
            "chair": "chair", "person": "person", "backpack": "backpack",
            "truck": "truck", "car": "car", "bus": "bus", "motorcycle": "motorcycle",
            # open-vocab típicos:
            "ladder": "ladder", "scaffold": "scaffold", "rebar": "rebar", "roof": "roof",
            "edge": "edge", "harness": "harness", "safety harness": "harness",
            "guardrail": "guardrail", "platform": "platform", "toolbox": "toolbox",
            "spill": "spill", "cable": "cable"
        }
        return mapping.get(n, n)
