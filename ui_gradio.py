import json
import os
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detector import YoloDetector
from rules_engine import RiskEngine
from chat_layer import build_chat_response
from pathlib import Path
DEFAULT_WEIGHTS = Path("models/best.pt")

detector = None
_cached_model_key = None
_cached_conf = None
_cached_iou = None

engine = RiskEngine("risk_ontology.yaml")


def _ensure_detector(model_path: str | None, conf: float, iou: float):
    """
    Carga (o reutiliza) el detector. Orden de prioridad:
    1) Ruta digitada por el usuario (si existe)
    2) models/best.pt (si existe)
    3) pesos por defecto de Ultralytics (yolov8n.pt)
    """
    global detector, _cached_model_key, _cached_conf, _cached_iou

    mp = (model_path or "").strip()
    weights = None
    warn = ""

    # 1) si el usuario escribió algo y existe, úsalo
    if mp and Path(mp).exists():
        weights = mp
    # 2) si no escribió o no existe, intenta models/best.pt
    elif DEFAULT_WEIGHTS.exists():
        weights = str(DEFAULT_WEIGHTS)
    # 3) si tampoco, fallback a yolov8n.pt
    else:
        warn = "⚠️ No se encontró un .pt en la ruta indicada ni en models/best.pt; se usarán los pesos por defecto (yolov8n.pt)."

    key = weights or "DEFAULT"
    need_reload = (
        detector is None
        or _cached_model_key != key
        or _cached_conf != conf
        or _cached_iou != iou
    )
    if need_reload:
        detector = YoloDetector(model_path=weights) if weights else YoloDetector()
        detector.conf = conf
        detector.iou = iou
        _cached_model_key = key
        _cached_conf = conf
        _cached_iou = iou

    return detector, warn


def _save_temp_image(im: Image.Image) -> str:
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    im.save(tmp, format="PNG")
    return tmp


def _save_temp_ndarray(arr: np.ndarray) -> str:
    if arr.ndim == 2:  # grayscale → RGB
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 4:  # RGBA → RGB
        arr = arr[:, :, :3]
    im = Image.fromarray(arr.astype(np.uint8))
    return _save_temp_image(im)


def _coerce_to_filepath(img_input: Any) -> str:
    """
    Acepta: str/Path, PIL.Image, np.ndarray, dict con 'path'/'name'/'image',
    o lista de cualquiera de los anteriores. Devuelve siempre una RUTA.
    """
    if isinstance(img_input, (list, tuple)) and img_input:
        for it in img_input:
            try:
                return _coerce_to_filepath(it)
            except Exception:
                continue
        raise gr.Error("No se pudo leer ninguna imagen de la lista proporcionada.")

    if isinstance(img_input, (str, Path)):
        p = str(img_input)
        if os.path.exists(p):
            return p

    if isinstance(img_input, dict):
        for k in ("path", "name"):
            v = img_input.get(k)
            if isinstance(v, str) and os.path.exists(v):
                return v
        for k in ("image", "composite", "background"):
            v = img_input.get(k)
            if isinstance(v, Image.Image):
                return _save_temp_image(v.convert("RGB"))
            if isinstance(v, np.ndarray):
                return _save_temp_ndarray(v)

    if isinstance(img_input, Image.Image):
        return _save_temp_image(img_input.convert("RGB"))

    if isinstance(img_input, np.ndarray):
        return _save_temp_ndarray(img_input)

    raise gr.Error("No se recibió imagen. Sube una imagen por favor.")


def draw_boxes(image_path: str, dets):
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        label = f"{det['cls']} {det.get('conf', 0):.2f}"
        d.text((x1 + 3, y1 + 3), label, fill=(255, 0, 0), font=font)
    return img


def analyze(image_input, model_path, conf, iou):
    image_path = _coerce_to_filepath(image_input)
    det, warn = _ensure_detector(model_path, conf, iou)

    preds = det.predict(image_path)  # YOLO lee el array/ruta internamente
    det_objs = det.to_dicts(preds)
    present = det.classes_from_dicts(det_objs)

    risks = engine.infer(present)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}
    chat = build_chat_response(present, risks, recs)

    img_out = draw_boxes(image_path, det_objs)
    json_out = {
        "classes_present": present,
        "detections": det_objs,
        "risks": risks,
        "recommendations": recs,
    }

    resumen = "Riesgos detectados:\n" + "\n".join(
        [f"- {r['nombre']} ({r['tipo']})" for r in risks]
    ) if risks else "Sin inferencias de riesgo."
    if warn:
        resumen = warn + "\n\n" + resumen

    return img_out, json_out, resumen, chat


with gr.Blocks(title="Analizador SST — Imágenes") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Imágenes)")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="filepath", label="Imagen")
            model_path = gr.Textbox(
            value=str(DEFAULT_WEIGHTS),   # ← se mostrará models/best.pt por defecto
            label="Ruta a modelo YOLO entrenado (.pt) — opcional",
            placeholder="Pega aquí otra ruta si prefieres usar otro .pt",
            )

            conf = gr.Slider(0.05, 0.90, value=0.25, step=0.05, label="Confianza (conf)")
            iou = gr.Slider(0.20, 0.95, value=0.60, step=0.05, label="IoU (NMS)")
            btn = gr.Button("Analizar", variant="primary")
        with gr.Column():
            img_out = gr.Image(type="pil", label="Detecciones")
            json_out = gr.JSON(label="Salida JSON")
            txt_resumen = gr.Textbox(label="Resumen", lines=8)
            txt_chat = gr.Markdown()

    btn.click(
        analyze,
        inputs=[img_in, model_path, conf, iou],
        outputs=[img_out, json_out, txt_resumen, txt_chat],
    )


if __name__ == "__main__":
    print(">> Iniciando UI Gradio...")
    port = 7860
    try:
        try:  
            demo.queue(default_concurrency_limit=2, max_size=16)
        except TypeError:
            demo.queue()

        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            show_api=False,
            inbrowser=False,
            share=False,
            debug=True,
        )
    except OSError:
        # si el puerto está ocupado, intenta el siguiente
        print(f"[WARN] Puerto {port} ocupado. Intentando 7861...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=7861,
            show_api=False,
            inbrowser=False,
            share=False,
            debug=True,
        )
