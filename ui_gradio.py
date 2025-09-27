# ui_gradio.py
from __future__ import annotations
import os, cv2, gradio as gr
import numpy as np
from typing import List, Dict, Any, Tuple, Set, Optional

try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

from rules_engine import RiskEngine

# -------- utilidades -----------
def _ensure_color(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2: return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4: return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def _center(box):
    x1,y1,x2,y2 = box; return ((x1+x2)/2.0,(y1+y2)/2.0)
def _norm_dist_boxes(b1,b2,w,h):
    (cx1,cy1),(cx2,cy2)=_center(b1),_center(b2)
    dx,dy=(cx1-cx2)/max(1e-6,w),(cy1-cy2)/max(1e-6,h)
    return float((dx*dx+dy*dy)**0.5)

# -------- detección YOLO ------
class YoloDetector:
    def __init__(self, model_path: Optional[str] = None):
        if not YOLO_OK: raise RuntimeError("ultralytics no disponible")
        # Fallbacks
        if not model_path or not os.path.exists(model_path):
            model_path = os.path.exists("models/best.pt") and "models/best.pt" or "yolov8n.pt"
        self.m = YOLO(model_path)
        self.names = self.m.names if hasattr(self.m, "names") else {i:str(i) for i in range(1000)}

    def predict(self, image: np.ndarray, conf: float = 0.35, iou: float = 0.6, imgsz: int = 640):
        r = self.m.predict(source=image, conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu")
        dets: List[Dict[str, Any]] = []
        if r and hasattr(r[0], "boxes") and r[0].boxes is not None:
            boxes = r[0].boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf_v = float(boxes.conf[i].item())
                x1,y1,x2,y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                dets.append({"cls": self.names.get(cls_id, str(cls_id)), "conf": conf_v, "box":[x1,y1,x2,y2]})
        return dets

# -------- tokens de imagen ----
OFFICE_HINTS = {"laptop","keyboard","screen","mouse","chair","chair_office"}
VEHICLES = {"forklift","truck","car","bus"}
MACHINES = {"machine","press","saw","conveyor","lathe","grinder"}

def _w_h(image: np.ndarray) -> Tuple[int,int]:
    h,w = image.shape[:2]; return w,h

def image_tokens(dets: List[Dict[str,Any]], w:int, h:int, near_thr: float = 0.15) -> Set[str]:
    toks: Set[str] = set()
    classes = {d["cls"] for d in dets}

    # Contexto oficina si hay suficientes pistas
    if len(OFFICE_HINTS & classes) >= 2:
        toks.add("context_office")

    # Persona cerca de máquina o vehículo (proximidad simple)
    persons = [d for d in dets if d["cls"] == "person"]
    machines = [d for d in dets if d["cls"] in MACHINES]
    vehicles = [d for d in dets if d["cls"] in VEHICLES]

    for p in persons:
        for m in machines:
            if _norm_dist_boxes(p["box"], m["box"], w, h) < near_thr:
                toks.add("near_person_machine")
        for v in vehicles:
            if _norm_dist_boxes(p["box"], v["box"], w, h) < near_thr:
                toks.add("near_person_vehicle")

    # Heurísticas adicionales de riesgos locativos
    if "cable" in classes or "spill" in classes or "pallet" in classes:
        toks.add("foot_near_floor_hazard")

    # Escalera
    if "ladder" in classes:
        toks.add("person_on_ladder")  # si hay 'person' también

    # Laptop por debajo de ojos (aprox.) — si hay laptop+person
    if "laptop" in classes and "person" in classes and "context_office" in toks:
        toks.add("screen_below_eyes")

    return toks

# -------- presentación ----------
def format_risks(risks: List[Dict[str,Any]]) -> str:
    if not risks:
        return "Sin inferencias de riesgo."
    lines = ["Riesgos detectados:"]
    for r in risks:
        lines.append(f"- {r['nombre']}")
    return "\n".join(lines)

def format_detail(risks: List[Dict[str,Any]]) -> str:
    if not risks:
        return "No se identificaron riesgos."
    out = []
    for r in risks:
        out.append(f"**{r['nombre']}**")
        if r.get("jerarquia"):
            out.append("  - *Jerarquía de control:*")
            for j in r["jerarquia"]:
                out.append(f"    - {j}")
        if r.get("normas"):
            out.append("  - *Normas:*")
            for n in r["normas"]:
                out.append(f"    - {n}")
        out.append("")  # separador
    return "\n".join(out)

# -------- Gradio ---------------
engine = RiskEngine("risk_ontology.yaml")

def analyze(image: np.ndarray, model_path: str, conf: float, iou: float, imgsz: int):
    img = _ensure_color(image)
    w,h = _w_h(img)

    det = YoloDetector(model_path=model_path if model_path else None)
    dets = det.predict(img, conf=conf, iou=iou, imgsz=imgsz)

    present = sorted({d["cls"] for d in dets})
    toks = image_tokens(dets, w, h)
    present_ctx = sorted(set(present).union(toks))

    risks = engine.infer(present_ctx)

    summary = format_risks(risks)
    detail_md = format_detail(risks)

    result = {
        "present_classes": present_ctx,
        "detections": dets,
        "risks": risks,
        "params": {"model": model_path or "(auto)", "conf": conf, "iou": iou, "imgsz": imgsz}
    }
    return img, result, summary, detail_md

with gr.Blocks(title="Analizador de Riesgos SST (Imágenes)") as demo:
    gr.Markdown("### Analizador de Riesgos SST — Imágenes")
    with gr.Row():
        inp = gr.Image(type="numpy", label="Imagen")
        out = gr.Image(type="numpy", label="Detecciones (solo para vista)", interactive=False)

    with gr.Row():
        model_path = gr.Textbox(label="Ruta a modelo YOLO entrenado (.pt) — opcional (si vacío usa models/best.pt o yolov8n.pt)")
    with gr.Row():
        conf = gr.Slider(0.1, 0.9, value=0.35, step=0.01, label="Confianza (conf)")
        iou = gr.Slider(0.3, 0.95, value=0.6, step=0.01, label="IoU (NMS)")
        imgsz = gr.Dropdown(choices=[480, 640, 800, 960], value=640, label="Tamaño de inferencia (imgsz)")

    with gr.Row():
        json_out = gr.JSON(label="Salida JSON")
        summary = gr.Textbox(lines=8, label="Resumen", interactive=False)
        detail = gr.Markdown(label="Riesgos identificados y controles")

    btn = gr.Button("Analizar imagen", variant="primary")
    btn.click(analyze, [inp, model_path, conf, iou, imgsz], [out, json_out, summary, detail])

if __name__ == "__main__":
    print(">> Iniciando UI Imágenes (puerto automático)")
    demo.launch(server_name="127.0.0.1", server_port=None, show_api=False)
