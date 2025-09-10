from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class YoloDetector:
    def __init__(self, model_path: str | None = None, conf: float = 0.25, iou: float = 0.6):
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics/YOLO no disponible. Revisa tus dependencias.")
        weights = str(model_path) if model_path else "yolov8n.pt"
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        try:
            self.names = self.model.names
        except Exception:
            nc = getattr(getattr(self.model, "model", None), "nc", 0) or 0
            self.names = {i: str(i) for i in range(nc)}

    @staticmethod
    def _to_numpy_rgb(src: Any) -> np.ndarray:
        if isinstance(src, (str, Path)):
            im = Image.open(src).convert("RGB")
            return np.array(im)
        if isinstance(src, Image.Image):
            return np.array(src.convert("RGB"))
        if isinstance(src, np.ndarray):
            arr = src
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                pass
            else:
                raise ValueError(f"Array con forma no soportada: {arr.shape}")
            return arr.astype(np.uint8)
        raise TypeError(f"Tipo de entrada no soportado: {type(src)}")

    def predict(self, source: Any):
        """Workaround: pasar SIEMPRE lista de imágenes para evitar bug en letterbox()."""
        img = self._to_numpy_rgb(source)
        assert img.ndim == 3 and img.shape[2] == 3, f"Entrada inválida: {img.shape}"
        result_list = self.model.predict(
            source=[img],           # <<<<< CLAVE: lista de un solo frame
            conf=self.conf,
            iou=self.iou,
            imgsz=640,
            verbose=False,
        )
        return result_list

    def to_dicts(self, results) -> list[dict]:
        out: list[dict] = []
        if not results:
            return out
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None or len(r.boxes) == 0:
            return out
        boxes = r.boxes.xyxy.cpu().numpy().astype(int).tolist()
        confs = r.boxes.conf.cpu().numpy().astype(float).tolist()
        clss  = r.boxes.cls.cpu().numpy().astype(int).tolist()
        for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, clss):
            out.append(
                {"cls": self.names.get(cid, str(cid)),
                 "id": int(cid),
                 "conf": float(conf),
                 "box": [int(x1), int(y1), int(x2), int(y2)]}
            )
        return out

    @staticmethod
    def classes_from_dicts(dets: list[dict]) -> list[str]:
        return sorted(set(d["cls"] for d in dets if "cls" in d))
