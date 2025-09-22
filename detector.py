# Envoltura de YOLO para inferencia en imágenes con parámetros configurables.

from __future__ import annotations
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class YoloDetector:
    """
    Carga un modelo YOLO y permite predecir sobre rutas de imagen, PIL.Image o numpy arrays.
    Acepta conf, iou e imgsz desde __init__ para que la UI pueda ajustarlos.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.60,
        imgsz: int = 960,
        device: str = "cpu",
    ):
        if not YOLO_AVAILABLE:
            raise RuntimeError("Ultralytics/YOLO no está disponible. Instala 'ultralytics'.")

        # Resolución del modelo por defecto: models/best.pt -> yolov8n.pt
        resolved = None
        if model_path and os.path.exists(model_path):
            resolved = model_path
        else:
            best_local = os.path.join("models", "best.pt")
            resolved = best_local if os.path.exists(best_local) else "yolov8n.pt"

        self.model_path = resolved
        self.model = YOLO(self.model_path)
        self.names = self.model.names if hasattr(self.model, "names") else {}
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device

    # -------------------- helpers --------------------

    @staticmethod
    def _pil_to_rgb_array(img: Image.Image) -> np.ndarray:
        """Convierte PIL -> numpy RGB uint8 HxWxC."""
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.asarray(img)

    @staticmethod
    def _ensure_hwc_uint8(arr: np.ndarray) -> np.ndarray:
        """Asegura HxWxC y dtype uint8 para YOLO."""
        if arr is None:
            raise ValueError("Imagen vacía.")
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError(f"Array con forma no válida para imagen: {arr.shape}")
        if arr.shape[2] == 4:
            # RGBA -> RGB
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr

    # -------------------- API pública --------------------

    def predict(self, source: Union[str, Image.Image, np.ndarray]):
        """
        Devuelve la lista de Results de Ultralytics para 1 imagen.
        Acepta ruta, PIL.Image o numpy array HxWxC.
        Para mayor compatibilidad, si es PIL o array, escribimos un temporal.
        """
        if isinstance(source, str):
            src = source
            write_temp = False
            tmp_path = None
        elif isinstance(source, Image.Image):
            arr = self._ensure_hwc_uint8(self._pil_to_rgb_array(source))
            # Ultralytics acepta arrays, pero para evitar rarezas, guardamos temporal.
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            Image.fromarray(arr).save(tmp.name, format="JPEG", quality=95)
            src = tmp.name
            write_temp = True
            tmp_path = tmp.name
            tmp.close()
        elif isinstance(source, np.ndarray):
            arr = self._ensure_hwc_uint8(source)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            Image.fromarray(arr).save(tmp.name, format="JPEG", quality=95)
            src = tmp.name
            write_temp = True
            tmp_path = tmp.name
            tmp.close()
        else:
            raise TypeError(f"Tipo de fuente no soportado: {type(source)}")

        try:
            results = self.model.predict(
                source=src,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            return results
        finally:
            if 'write_temp' in locals() and write_temp and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def to_dicts(self, results) -> List[Dict[str, Any]]:
        """
        Convierte Results de Ultralytics (para 1 imagen) en una lista de dicts:
        [{"cls": "person", "conf": 0.87, "box": [x1,y1,x2,y2]}, ...]
        """
        dets: List[Dict[str, Any]] = []
        if not results:
            return dets
        r = results[0]  # una sola imagen
        if not hasattr(r, "boxes") or r.boxes is None:
            return dets
        boxes = r.boxes
        for i in range(len(boxes)):
            cls_raw = boxes.cls[i]
            conf_raw = boxes.conf[i]
            cls_id = int(getattr(cls_raw, "item", lambda: cls_raw)())
            conf = float(getattr(conf_raw, "item", lambda: conf_raw)())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cname = self.names.get(cls_id, str(cls_id))
            dets.append({"cls": cname, "conf": conf, "box": [float(x1), float(y1), float(x2), float(y2)]})
        return dets

    @staticmethod
    def classes_from_dicts(dets: List[Dict[str, Any]]) -> List[str]:
        """Clases únicas presentes en los detections."""
        return sorted({d["cls"] for d in dets})
