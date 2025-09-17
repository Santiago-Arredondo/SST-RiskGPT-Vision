from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import numpy as np
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    YOLO = None  # type: ignore


class YoloDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        model_path:
          - Ruta a .pt entrenado (recomendado: models/best.pt)
          - Si no existe, cae a yolov8n.pt
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "Ultralytics/YOLO no está instalado. Ejecuta: pip install ultralytics"
            )

        if model_path and os.path.exists(model_path):
            self.model_path = model_path
        else:
            default_best = os.path.join("models", "best.pt")
            self.model_path = default_best if os.path.exists(default_best) else "yolov8n.pt"

        self.model = YOLO(self.model_path)
        # nombres de clases
        self.names = self.model.names if hasattr(self.model, "names") else {i: str(i) for i in range(1000)}

    # --------------------
    # Inferencia YOLO
    # --------------------
    def _to_rgb_ndarray(self, image: Any) -> np.ndarray:
        """
        Acepta PIL.Image, ruta str o np.ndarray. Devuelve np.ndarray RGB.
        """
        if isinstance(image, str):
            # Ultralytics acepta rutas; dejamos que lea él
            # pero para homogeneidad devolvemos None aquí
            # y predict usará la ruta directamente.
            return None  # type: ignore

        if isinstance(image, np.ndarray):
            arr = image
            # si es BGR (OpenCV), convertir a RGB opcionalmente
            if arr.ndim == 3 and arr.shape[2] == 3:
                # Intentamos detectar si parece BGR por estadísticos; en la práctica
                # nos sirve igual pasarlo como está, pero preferimos RGB.
                # Convertimos siempre de BGR->RGB si proviene de cv2.
                # Si ya era RGB, la inversión solo cambia canales; no afecta YOLO.
                return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            elif arr.ndim == 2:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError("ndarray con forma no soportada.")
        else:
            # Probablemente PIL.Image
            try:
                from PIL import Image
                if isinstance(image, Image.Image):
                    return np.array(image.convert("RGB"))
            except Exception:
                pass
            raise ValueError("Tipo de imagen no soportado. Usa PIL, ndarray o ruta a archivo.")

    def predict(self, image: Any, conf: float = 0.35, iou: float = 0.6, imgsz: int = 960):
        """
        Ejecuta predicción YOLO.
          - image puede ser ruta (str), PIL.Image o np.ndarray
        """
        if isinstance(image, str):
            src = image
        else:
            # convertir a RGB ndarray
            src = self._to_rgb_ndarray(image)

        result_list = self.model.predict(
            source=src,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device="cpu",
        )
        return result_list

    # --------------------
    # Conversión de resultados
    # --------------------
    def to_dicts(self, result_list) -> List[Dict[str, Any]]:
        dets: List[Dict[str, Any]] = []
        if not result_list:
            return dets

        r = result_list[0]
        if not hasattr(r, "boxes") or r.boxes is None:
            return dets

        boxes = r.boxes
        for i in range(len(boxes)):
            cls_raw = boxes.cls[i]
            conf_raw = boxes.conf[i]
            cls_id = int(getattr(cls_raw, "item", lambda: cls_raw)())
            conf = float(getattr(conf_raw, "item", lambda: conf_raw)())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            dets.append(
                {
                    "cls": self.names.get(cls_id, str(cls_id)),
                    "conf": conf,
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                }
            )
        return dets

    @staticmethod
    def classes_from_dicts(det_list: List[Dict[str, Any]]) -> List[str]:
        return sorted({d["cls"] for d in det_list})

    # --------------------
    # Heurísticas de piso: cable / spill
    # --------------------
    def heuristics_floor_hazards(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Heurísticas simples en la zona baja de la imagen para detectar obstáculos de piso:
        - cable: contornos largos y delgados
        - spill: manchas oscuras/compactas
        Devuelve lista de pseudo-detecciones con cls, conf y box.
        """
        out: List[Dict[str, Any]] = []

        if img_bgr is None or img_bgr.ndim != 3:
            return out

        H, W = img_bgr.shape[:2]
        # Zona inferior (piso)
        y0 = int(0.55 * H)
        roi = img_bgr[y0:H, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # ---- CABLES (bordes alargados/finos) ----
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)), 1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect = (w + 1e-3) / (h + 1e-3)
            # umbrales conservadores para evitar falsos positivos
            if aspect > 6.0 and 60 < area < 40000 and h < 40:
                out.append(
                    {
                        "cls": "cable",
                        "conf": 0.30,
                        "box": [float(x), float(y + y0), float(x + w), float(y + y0 + h)],
                    }
                )

        # ---- SPILL (manchas oscuras) ----
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thr_val = float(np.percentile(blur, 30))  # 30% más oscuro
        _, th = cv2.threshold(blur, thr_val, 255, cv2.THRESH_BINARY_INV)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

        cnts2, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts2:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area > 1500 and w > 40 and h > 20:
                out.append(
                    {
                        "cls": "spill",
                        "conf": 0.30,
                        "box": [float(x), float(y + y0), float(x + w), float(y + y0 + h)],
                    }
                )

        return out
