# ui_gradio.py
# Analizador de Riesgos SST (Imágenes) con soporte de postura (YOLOv8-Pose)
# - Detecta objetos con YOLOv8 (vía detector.YoloDetector)
# - Genera tokens de postura (espalda/cuello/muñeca) con YOLOv8-Pose
# - Infiera riesgos con rules_engine.RiskEngine a partir de clases + tokens
# - Muestra recomendaciones por jerarquía de control y normas

import os
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from detector import YoloDetector
from rules_engine import RiskEngine
from chat_layer import build_chat_response

# --------- Pose helpers (para ergonomía sin anotar chair/screen) ---------
try:
    from ultralytics import YOLO as _UltralyticsYOLO
    _ULTRA_OK = True
except Exception:
    _ULTRA_OK = False

_POSE_MODEL = None  # cache global

def _get_pose_model():
    """Carga perezosa de YOLOv8-Pose (cpu). Devuelve False si no disponible."""
    global _POSE_MODEL
    if _POSE_MODEL is None:
        if not _ULTRA_OK:
            _POSE_MODEL = False
        else:
            try:
                _POSE_MODEL = _UltralyticsYOLO("yolov8n-pose.pt")
            except Exception:
                _POSE_MODEL = False
    return _POSE_MODEL

def _posture_tokens_pil(pil_img: Image.Image):
    """
    Devuelve un set de tokens de postura:
      - pose_back_flexion
      - pose_neck_flexion
      - pose_wrist_above_elbow
    """
    m = _get_pose_model()
    toks = set()
    if not m:
        return toks

    arr = np.array(pil_img.convert("RGB"))
    try:
        r = m.predict(source=arr, imgsz=640, verbose=False, device="cpu")[0]
        kps = r.keypoints
        if kps is None:
            return toks
        xy = kps.xy.cpu().numpy()  # (N, K, 2) en formato COCO
        for kp in xy:
            # COCO idx: 0 nose, 5 L-shoulder, 6 R-shoulder, 11 L-hip, 12 R-hip,
            # 7/8 elbows, 9/10 wrists
            try:
                nose = kp[0]
                l_sh, r_sh = kp[5], kp[6]
                l_hip, r_hip = kp[11], kp[12]
                l_elb, r_elb = kp[7], kp[8]
                l_wri, r_wri = kp[9], kp[10]
            except Exception:
                continue

            spine_mid = ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
            hip_mid = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)

            # Inclinación de espalda: vector (hip->spine) vs vertical
            v = (spine_mid[0] - hip_mid[0], spine_mid[1] - hip_mid[1])
            n = max((v[0] ** 2 + v[1] ** 2) ** 0.5, 1e-6)
            cosang = (v[1] / n)  # producto con (0,1)
            import math
            back_tilt = abs(90 - abs(math.degrees(math.acos(max(-1.0, min(1.0, cosang))))))
            if back_tilt > 20:  # umbral conservador
                toks.add("pose_back_flexion")

            # Inclinación de cuello: vector (spine_mid->nose) vs vertical
            v2 = (nose[0] - spine_mid[0], nose[1] - spine_mid[1])
            n2 = max((v2[0] ** 2 + v2[1] ** 2) ** 0.5, 1e-6)
            cosang2 = (v2[1] / n2)
            neck_tilt = abs(90 - abs(math.degrees(math.acos(max(-1.0, min(1.0, cosang2))))))
            if neck_tilt > 20:
                toks.add("pose_neck_flexion")

            # Muñecas por encima de codos (postura no neutra)
            if l_wri[1] < l_elb[1] or r_wri[1] < r_elb[1]:
                toks.add("pose_wrist_above_elbow")
    except Exception:
        # Fallo silencioso de pose (no bloquea UX)
        pass
    return toks

# --------- Dibujo de cajas ---------
def draw_boxes(img: Image.Image, dets):
    pil = img.copy().convert("RGB")
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 128, 0), width=2)
        label = f"{det['cls']} {det.get('conf', 0.0):.2f}"
        d.text((x1 + 3, max(3, y1 + 3)), label, fill=(255, 128, 0), font=font)
    return pil

def recs_markdown(risks, recs_by_id):
    """Devuelve un Markdown compacto con jerarquía de control + normas por riesgo."""
    if not risks:
        return ""
    lines = []
    for r in risks:
        rid = r.get("id") or r.get("risk") or ""
        nombre = r.get("nombre", rid)
        tipo = r.get("tipo", "")
        lines.append(f"**{nombre}** ({tipo})")
        recs = recs_by_id.get(rid, {})
        for nivel in ["eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"]:
            items = recs.get(nivel, [])
            if items:
                pretty = nivel.capitalize()
                lines.append(f"- *{pretty}:* " + "; ".join(items))
        normas = r.get("normas", []) or recs.get("normas", [])
        if normas:
            lines.append("- *Normas:* " + ", ".join(normas))
        lines.append("")  # línea en blanco
    return "\n".join(lines).strip()

# --------- Inicialización de motores ---------
detector = YoloDetector()                  # usa yolov8n.pt por defecto si no pasas .pt
engine = RiskEngine("risk_ontology.yaml")  # usa tu ontología con ergonomía por postura

# --------- Lógica principal ---------
def analyze(image: Image.Image, model_path: str):
    if image is None:
        raise gr.Error("No se recibió imagen. Sube una imagen por favor.")

    global detector
    model_path = (model_path or "").strip()
    if model_path:
        if not os.path.exists(model_path):
            raise gr.Error(f"No existe el archivo de modelo: {model_path}")
        # recargar detector con pesos del usuario
        detector = YoloDetector(model_path=model_path)

    # 1) Detección
    np_img = np.array(image.convert("RGB"))
    det_objs = detector.to_dicts(detector.predict(np_img))
    present = set(detector.classes_from_dicts(det_objs))

    # 2) Postura (tokens de ergonomía)
    pose_toks = _posture_tokens_pil(image)
    present |= pose_toks  # unión

    # 3) Reglas de riesgo + recomendaciones
    risks = engine.infer(sorted(present))
    recs = {r.get("id", r.get("risk", "")): engine.recommendations(r.get("id", r.get("risk", ""))) for r in risks}

    # 4) Chat en lenguaje natural (opcional)
    chat = build_chat_response(sorted(present), risks, recs)

    # 5) Imagen con cajas y salidas
    img_out = draw_boxes(image, det_objs)
    json_out = {
        "classes_present": sorted(present),
        "detections": det_objs,
        "risks": risks,
        "recommendations": recs,
        "pose_tokens": sorted(pose_toks) if pose_toks else [],
    }
    resumen = "Riesgos detectados:\n" + "\n".join(
        f"- {r.get('nombre', r.get('id',''))} ({r.get('tipo','')})" for r in risks
    ) if risks else "Sin inferencias de riesgo."

    recs_md = recs_markdown(risks, recs)

    return img_out, json_out, resumen, chat, recs_md

# --------- UI Gradio ---------
with gr.Blocks(title="Analizador de Riesgos SST (Imágenes)") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Imágenes)")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Imagen")
            model_path = gr.Textbox("", label="Ruta a modelo YOLO entrenado (.pt) — opcional")
            btn = gr.Button("Analizar")
        with gr.Column():
            img_out = gr.Image(type="pil", label="Detecciones")
            json_out = gr.JSON(label="Salida JSON")
            txt_resumen = gr.Textbox(label="Resumen", lines=8)
            txt_chat = gr.Markdown()
            md_recs = gr.Markdown(label="Recomendaciones")
    btn.click(analyze, inputs=[img_in, model_path], outputs=[img_out, json_out, txt_resumen, txt_chat, md_recs])

if __name__ == "__main__":
    print(">> UI de video (puerto automático)")
    demo.launch(server_name="127.0.0.1", server_port=None, show_api=False)  # None => elige un puerto libre


