import os
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

from detector import YoloDetector
from rules_engine import RiskEngine

# --- Conjuntos de clases por contexto (ajusta a tu modelo) ---
FLOOR_HAZARD = {"pallet", "cable", "spill", "toolbox"}
MACHINES     = {"machine", "press", "saw", "conveyor"}
VEHICLES     = {"forklift", "truck", "car", "bus", "excavator"}

engine = RiskEngine("risk_ontology.yaml")
detector = None  # se crea al primer uso

def _draw_boxes(img: Image.Image, dets):
    pil = img.copy().convert("RGB")
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 128, 0), width=2)
        label = f"{det['cls']} {det['conf']:.2f}"
        d.text((x1 + 3, max(3, y1 + 3)), label, fill=(255, 128, 0), font=font)
    return pil

# --- NMS por clase (para usar el slider de IoU sin tocar detector.py) ---
def _iou(b1, b2):
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    return inter / max(1e-6, a1 + a2 - inter)

def _nms_by_class(dets, iou_thr):
    out = []
    # agrupar por clase
    by_cls = {}
    for d in dets:
        by_cls.setdefault(d["cls"], []).append(d)
    for cls, lst in by_cls.items():
        # ordenar por confianza
        lst = sorted(lst, key=lambda x: x["conf"], reverse=True)
        keep = []
        while lst:
            best = lst.pop(0)
            keep.append(best)
            lst = [x for x in lst if _iou(best["box"], x["box"]) < iou_thr]
        out.extend(keep)
    return out

def image_tokens(dets, w, h):
    toks = set()
    persons = [d for d in dets if d["cls"] == "person"]
    ladders = [d for d in dets if d["cls"] == "ladder"]
    floor   = [d for d in dets if d["cls"] in FLOOR_HAZARD]
    machs   = [d for d in dets if d["cls"] in MACHINES]

    # A) Obstáculos de piso frente a pies de persona
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        feet_y = py2
        cx_p = 0.5 * (px1 + px2)
        for hz in floor:
            hx1, hy1, hx2, hy2 = hz["box"]
            cx_h = 0.5 * (hx1 + hx2)
            if abs(cx_p - cx_h) < 0.12 * w and (hy1 - 0.02 * h) <= feet_y <= (hy2 + 0.06 * h):
                toks.add("foot_near_floor_hazard")

    # B) Altura (ladder o base alta)
    if ladders and persons:
        toks.add("at_height")
    else:
        for p in persons:
            _, py1, _, py2 = p["box"]
            base_rel = py2 / float(h)
            top_rel  = py1 / float(h)
            if base_rel < 0.68 and top_rel < 0.42:
                toks.add("at_height")

    # C) Proximidad persona-máquina
    def _center(b): x1,y1,x2,y2 = b; return (0.5*(x1+x2), 0.5*(y1+y2))
    def _ndist(a,b):
        (cx1,cy1),(cx2,cy2)=_center(a),_center(b)
        dx,dy=(cx1-cx2)/max(1e-6,w),(cy1-cy2)/max(1e-6,h)
        return (dx*dx+dy*dy)**0.5

    for p in persons:
        for m in machs:
            if _ndist(p["box"], m["box"]) < 0.22 or _iou(p["box"], m["box"]) > 0.02:
                toks.add("near_person_machine")

    return sorted(toks)

def analyze(image: Image.Image, model_path: str, conf: float, iou: float):
    # 1) Instanciar detector sólo con model_path (sin conf/iou)
    global detector
    default_best = os.path.join("models", "best.pt")
    use_model = model_path or (default_best if os.path.exists(default_best) else "yolov8n.pt")
    if detector is None or (model_path and model_path != getattr(detector, "model_path", None)):
        detector = YoloDetector(model_path=use_model)

    # 2) Detectar
    preds = detector.predict(image)
    dets  = detector.to_dicts(preds)

    # 3) Aplicar filtros desde la UI (sin tocar el detector):
    #    - confianza
    dets = [d for d in dets if d.get("conf", 0.0) >= float(conf)]
    #    - NMS por clase con IoU del slider
    dets = _nms_by_class(dets, float(iou))

    # 4) Tokens de contexto + inferencia de riesgos
    w, h = image.size
    present = sorted({d["cls"] for d in dets})
    toks    = image_tokens(dets, w, h)
    present_ctx = sorted(set(present).union(toks))

    engine = engine_cache()  # motor compartido
    risks = engine.infer(present_ctx)
    recs  = {r["id"]: engine.recommendations(r["id"]) for r in risks}

    # 5) Salidas
    img_out = _draw_boxes(image, dets)
    resumen = "Sin inferencias de riesgo." if not risks else \
        "Riesgos detectados:\n" + "\n".join(f"- {r['nombre']}" for r in risks)
    json_out = {
        "present_classes": present_ctx,
        "detections": dets,
        "risks": risks,
        "recommendations": recs,
        "params": {"model": use_model, "conf": conf, "iou": iou}
    }

    # Detalle de recomendaciones
    md = []
    if present:
        md.append(f"**Elementos detectados:** {', '.join(present)}")
    if risks:
        md.append("**Riesgos identificados y controles sugeridos:**")
        for r in risks:
            rid = r["id"]; title = r.get("nombre", rid)
            md.append(f"### {title}")
            reco = recs.get(rid, {})
            jer = reco.get("jerarquia", {})
            if jer:
                md.append("**Jerarquía:**")
                for nivel, items in jer.items():
                    for it in items:
                        md.append(f"- *{nivel}*: {it}")
            norms = reco.get("normativas", [])
            if norms:
                md.append("\n**Normativas:**")
                for n in norms: md.append(f"- {n}")
    md_out = "\n".join(md) if md else "_No se identificaron riesgos._"

    return img_out, json_out, resumen, md_out

# cache simple del motor para no re-cargar el YAML cada click
_engine = None
def engine_cache():
    global _engine
    if _engine is None:
        _engine = RiskEngine("risk_ontology.yaml")
    return _engine

with gr.Blocks(title="Analizador SST — Imágenes") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Imágenes)")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Imagen")
            model  = gr.Textbox("", label="Ruta a modelo YOLO entrenado (.pt) — opcional (si vacío, usa models/best.pt o yolov8n.pt)")
            conf   = gr.Slider(0.10, 0.90, value=0.35, step=0.05, label="Confianza (conf)")
            iou    = gr.Slider(0.30, 0.95, value=0.60, step=0.05, label="IoU (iou)")
            btn    = gr.Button("Analizar imagen", variant="primary")
        with gr.Column():
            img_out   = gr.Image(type="pil", label="Detecciones")
            json_out  = gr.JSON(label="Salida JSON")
            resumen   = gr.Textbox(label="Resumen", lines=6)
            md_detail = gr.Markdown()
    btn.click(analyze, inputs=[img_in, model, conf, iou], outputs=[img_out, json_out, resumen, md_detail])

if __name__ == "__main__":
    print(">> Iniciando UI Imágenes en http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, show_api=False)
