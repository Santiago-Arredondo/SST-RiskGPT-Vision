# ui_gradio.py
# UI de análisis por imagen: detecta objetos con YOLO, genera tokens de contexto
# y aplica la ontología de riesgos para mostrar riesgos + recomendaciones.

from __future__ import annotations
import os
import io
import tempfile
from typing import List, Dict, Any, Tuple, Set

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from detector import YoloDetector
from rules_engine import RiskEngine

# ----------------------- utilidades geométricas -----------------------

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

# ----------------------- reglas de tokens por imagen -----------------------

VEHICLES = {"forklift", "truck", "car", "excavator", "bus"}
MACHINES = {"conveyor", "machine", "saw", "press"}
FLOOR_HAZ = {"pallet", "cable", "spill", "toolbox"}  # obstáculos de piso
ERGONOMY = {"chair", "screen", "phone"}  # señales mínimas para ergonomía

def image_tokens(dets: List[Dict[str, Any]], w: int, h: int) -> Set[str]:
    """
    Genera tokens de contexto a partir de las detecciones:
      - near_person_vehicle
      - near_person_machine
      - foot_near_floor_hazard
      - ergo_posture_flag (heurística simple si hay silla/pantalla/persona)
    """
    toks: Set[str] = set()

    persons  = [d for d in dets if d["cls"] == "person"]
    vehicles = [d for d in dets if d["cls"] in VEHICLES]
    machines = [d for d in dets if d["cls"] in MACHINES]
    floorhz  = [d for d in dets if d["cls"] in FLOOR_HAZ]
    chairs   = [d for d in dets if d["cls"] == "chair"]
    screens  = [d for d in dets if d["cls"] == "screen"]

    # A) Persona cerca de vehículo
    for p in persons:
        for v in vehicles:
            if _norm_dist_boxes(p["box"], v["box"], w, h) < 0.20 or _iou(p["box"], v["box"]) > 0.01:
                toks.add("near_person_vehicle")

    # B) Persona cerca de máquina
    for p in persons:
        for m in machines:
            if _norm_dist_boxes(p["box"], m["box"], w, h) < 0.20 or _iou(p["box"], m["box"]) > 0.01:
                toks.add("near_person_machine")

    # C) Pies cerca de obstáculo (heurística por proyección X y borde inferior del bbox)
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        feet_y = py2
        cx_p = 0.5 * (px1 + px2)
        for hz in floorhz:
            hx1, hy1, hx2, hy2 = hz["box"]
            cx_h = 0.5 * (hx1 + hx2)
            if abs(cx_p - cx_h) < 0.10 * w and hy1 <= feet_y <= hy2 + 0.05 * h:
                toks.add("foot_near_floor_hazard")

    # D) Ergonomía (si vemos persona + silla/pantalla, marcamos un flag para que la ontología recomiende)
    if persons and (chairs or screens):
        toks.add("ergo_posture_flag")

    return toks

# ----------------------- dibujo y helpers -----------------------

def draw_boxes(img: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    pil = img.copy().convert("RGB")
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=(0, 140, 255), width=2)
        label = f"{det['cls']} {det.get('conf', 0.0):.2f}"
        d.text((x1 + 3, max(y1 + 3, 2)), label, fill=(0, 140, 255), font=font)
    return pil

def save_temp_image(pil_img: Image.Image) -> str:
    """Guarda la imagen a un archivo temporal (Ultralytics maneja mejor ruta/np.ndarray)."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    pil_img.convert("RGB").save(tmp, format="JPEG", quality=95)
    tmp.flush()
    tmp.close()
    return tmp.name

def format_recommendations(recs: Dict[str, Dict[str, list]]) -> str:
    """Convierte el dict de recomendaciones a Markdown compacto."""
    lines = []
    order = ["eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"]
    titles = {
        "eliminacion": "Eliminación",
        "sustitucion": "Sustitución",
        "ingenieria": "Controles de Ingeniería",
        "administrativos": "Administrativos",
        "epp": "EPP",
    }
    for rid, buckets in recs.items():
        lines.append(f"### Recomendaciones — {rid}")
        for k in order:
            if k in buckets and buckets[k]:
                lines.append(f"- **{titles[k]}:** " + "; ".join(buckets[k]))
    return "\n".join(lines) if lines else "_Sin recomendaciones disponibles._"

# ----------------------- inicialización de modelos -----------------------

# Carga detector: por defecto buscará models/best.pt y si no, usa yolov8n.pt
detector = YoloDetector()
engine = RiskEngine("risk_ontology.yaml")

# ----------------------- función principal -----------------------

def analyze(image: Image.Image, model_path: str):
    """
    1) Guarda imagen a archivo temporal
    2) Detecta objetos con YOLO
    3) Genera tokens de contexto
    4) Infiera riesgos y arma recomendaciones
    5) Devuelve: imagen anotada, JSON, resumen en texto, recomendaciones en Markdown
    """
    if image is None:
        raise gr.Error("No se recibió imagen. Sube una imagen por favor.")

    # Cambiar de pesos en caliente si el usuario pasa una ruta
    global detector
    if model_path and os.path.exists(model_path):
        detector = YoloDetector(model_path=model_path)

    # Guardar a ruta temporal y predecir
    img_path = save_temp_image(image)
    preds = detector.predict(img_path)  # NO pasamos conf/iou para ser compatible con tu detector.py
    det_objs = detector.to_dicts(preds)

    # Dimensiones (desde PIL)
    w, h = image.size

    # Tokens de contexto + clases presentes
    present = sorted({d["cls"] for d in det_objs})
    toks = image_tokens(det_objs, w, h)
    present_with_ctx = sorted(set(present).union(toks))

    # Riesgos y recomendaciones
    risks = engine.infer(present_with_ctx)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}

    # Resumen de texto
    resumen = "Riesgos detectados:\n" + "\n".join(
        f"- {r['nombre']} ({r['tipo']})" for r in risks
    ) if risks else "Sin inferencias de riesgo."

    # Imagen anotada
    img_out = draw_boxes(image, det_objs)

    # JSON resumido
    json_out = {
        "classes_present": present,
        "context_tokens": sorted(toks),
        "detections": det_objs,
        "risks": risks,
        "recommendations": recs
    }

    # Recomendaciones legibles
    md_recs = format_recommendations(recs)

    return img_out, json_out, resumen, md_recs

# ----------------------- UI Gradio -----------------------

with gr.Blocks(title="Analizador SST — Imágenes") as demo:
    gr.Markdown("## Analizador de Riesgos SST (Imágenes)")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Imagen")
            model_path = gr.Textbox("", label="Ruta a modelo YOLO (.pt) — opcional")
            btn = gr.Button("Analizar")
        with gr.Column():
            img_out = gr.Image(type="pil", label="Detecciones")
            json_out = gr.JSON(label="Salida JSON")
            txt_resumen = gr.Textbox(label="Resumen", lines=8)
            md_recos = gr.Markdown()

    btn.click(analyze, inputs=[img_in, model_path], outputs=[img_out, json_out, txt_resumen, md_recos])

if __name__ == "__main__":
    print(">> Iniciando UI Imágenes en http://127.0.0.1:7860")
    demo.queue().launch(server_name="127.0.0.1", server_port=7860, show_api=False)
