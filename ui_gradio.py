# ui_gradio.py
# UI de imágenes para SST-RiskGPT Vision con ontología expandida + contexto
# - Carga y normaliza risk_ontology.yaml (when_any/when_all/context o if_any/if_all)
# - Tokens de contexto (office/industrial/obra/warehouse)
# - Heurísticas de proximidad (incluye person_on_ladder)
# - Detección YOLO directa (sin depender de detector.py)
# Requisitos: ultralytics, gradio, pillow, pyyaml, numpy, opencv-python

from __future__ import annotations
import os, io, json, math
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# --- YOLO (Ultralytics) ---
try:
    from ultralytics import YOLO
    YOLO_OK = True
except Exception:
    YOLO_OK = False

# --- YAML ---
try:
    import yaml
except Exception as e:
    raise RuntimeError("Falta 'pyyaml'. Instala: pip install pyyaml") from e


# ==========================
# Utilidades geométricas
# ==========================
def _ensure_pil(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            return Image.fromarray(img.astype(np.uint8), mode="L").convert("RGB")
        if img.ndim == 3 and img.shape[2] == 4:
            return Image.fromarray(img.astype(np.uint8), mode="RGBA").convert("RGB")
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    else:
        raise ValueError("Tipo de imagen no soportado")

def _center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def _norm_dist_boxes(b1: List[float], b2: List[float], w: float, h: float) -> float:
    (cx1, cy1), (cx2, cy2) = _center(b1), _center(b2)
    dx, dy = (cx1 - cx2) / max(1e-6, w), (cy1 - cy2) / max(1e-6, h)
    return float((dx*dx + dy*dy) ** 0.5)

def _iou(b1: List[float], b2: List[float]) -> float:
    x1, y1, x2, y2 = b1
    X1, Y1, X2, Y2 = b2
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    union = a1 + a2 - inter + 1e-6
    return float(inter / union)


# ==========================
# Ontología: carga / normalización
# ==========================
def load_ontology(path: str = "risk_ontology.yaml") -> Dict[str, Any]:
    """
    Devuelve un dict normalizado:
    {
      "meta": {...},
      "context_tokens": {ctx: {"any": [...]}, ...},
      "risks": {
         rid: {
           "id","nombre","tipo","if_any":set,"if_all":set,"context":str|None,
           "jerarquia":{...}, "normativas":[...]
         }, ...
      }
    }
    """
    if not os.path.exists(path):
        return {"meta": {}, "context_tokens": {}, "risks": {}}

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    meta = raw.get("meta", {}) if isinstance(raw, dict) else {}
    # context_tokens puede venir como dict o lista
    ctx_tokens: Dict[str, Dict[str, List[str]]] = {}
    if isinstance(raw, dict):
        ct = raw.get("context_tokens")
        if isinstance(ct, dict):
            for ctx, spec in ct.items():
                if isinstance(spec, dict):
                    ctx_tokens[ctx] = {"any": list(spec.get("any", []) or [])}
                elif isinstance(spec, list):
                    ctx_tokens[ctx] = {"any": list(spec)}
    # riesgos puede venir como lista o diccionario disperso
    risk_items: List[Dict[str, Any]] = []
    if isinstance(raw, dict):
        if isinstance(raw.get("riesgos"), list):
            risk_items.extend(raw["riesgos"])
        # admitir riesgos bajo otras claves (menos común)
        for k, v in raw.items():
            if k in ("meta", "context_tokens", "riesgos"):
                continue
            if isinstance(v, dict) and any(x in v for x in ("when_any", "when_all", "if_any", "if_all")):
                vi = v.copy()
                vi.setdefault("id", k)
                risk_items.append(vi)
    elif isinstance(raw, list):
        risk_items.extend(raw)

    risks: Dict[str, Dict[str, Any]] = {}
    for it in risk_items:
        if not isinstance(it, dict):
            continue
        rid = it.get("id") or it.get("clave") or it.get("key")
        if not rid:
            continue
        if_any = set(it.get("if_any", []) or it.get("when_any", []) or [])
        if_all = set(it.get("if_all", []) or it.get("when_all", []) or [])
        risks[rid] = {
            "id": rid,
            "nombre": it.get("nombre", rid),
            "tipo": it.get("tipo"),
            "if_any": set(if_any),
            "if_all": set(if_all),
            "context": it.get("context") or None,
            "jerarquia": it.get("jerarquia", {}),
            "normativas": it.get("normativas", []),
        }
    return {"meta": meta, "context_tokens": ctx_tokens, "risks": risks}


def make_context_tokens(ontology: Dict[str, Any], present: Set[str]) -> Set[str]:
    """Devuelve tokens tipo 'context_office', etc., si se cumple cualquier 'any' de ese contexto."""
    out: Set[str] = set()
    for ctx, spec in ontology.get("context_tokens", {}).items():
        any_list = set(spec.get("any", []))
        if present & any_list:
            out.add(f"context_{ctx}")
    return out


def infer_risks(present_tokens: Set[str], ontology: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dispara riesgos con reglas if_any/if_all + contexto opcional."""
    out: List[Dict[str, Any]] = []
    for rid, rdef in ontology.get("risks", {}).items():
        if not isinstance(rdef, dict):
            continue
        any_ok = True
        all_ok = True
        ctx_ok = True

        if_any = set(rdef.get("if_any", []))
        if_all = set(rdef.get("if_all", []))
        ctx = rdef.get("context")

        if if_any:
            any_ok = bool(present_tokens & if_any)
        if if_all:
            all_ok = if_all.issubset(present_tokens)
        if ctx:
            ctx_ok = (f"context_{ctx}" in present_tokens)

        if any_ok and all_ok and ctx_ok:
            out.append({"id": rid, "nombre": rdef.get("nombre", rid)})
    return out


# ==========================
# Heurísticas (tokens simbólicos)
# ==========================
VEHICLES = {"forklift", "truck", "car", "excavator", "bus"}
MACHINES = {"conveyor", "machine", "saw", "press", "lathe", "grinder"}
FLOOR_HAZ = {"pallet", "cable", "spill", "tool_on_floor", "floor_irregular", "wet_floor", "debris"}

def image_tokens(
    dets: List[Dict[str, Any]],
    w: int, h: int,
    near_thr: float = 0.20,
    iou_touch: float = 0.01,
    strict: bool = False,
) -> Set[str]:
    """
    Genera tokens: near_person_vehicle, near_person_machine, foot_near_floor_hazard,
    person_on_ladder (PARCHE solicitado), y otros básicos.
    """
    toks: Set[str] = set()
    persons = [d for d in dets if d["cls"] == "person"]
    vehicles = [d for d in dets if d["cls"] in VEHICLES]
    machines = [d for d in dets if d["cls"] in MACHINES]
    floor_haz = [d for d in dets if d["cls"] in FLOOR_HAZ]
    ladders = [d for d in dets if d["cls"] in {"ladder", "stepladder"}]

    # A) persona ~ vehículo
    for p in persons:
        for v in vehicles:
            if _norm_dist_boxes(p["box"], v["box"], w, h) < near_thr or _iou(p["box"], v["box"]) > iou_touch:
                toks.add("near_person_vehicle")

    # B) persona ~ máquina
    for p in persons:
        for m in machines:
            if _norm_dist_boxes(p["box"], m["box"], w, h) < near_thr or _iou(p["box"], m["box"]) > iou_touch:
                toks.add("near_person_machine")

    # C) pie cercano a obstáculo en piso (aprox: borde inferior de la persona cerca del bbox del obstáculo)
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        feet_y = py2
        cx_p = 0.5 * (px1 + px2)
        for hz in floor_haz:
            hx1, hy1, hx2, hy2 = hz["box"]
            cx_h = 0.5 * (hx1 + hx2)
            if abs(cx_p - cx_h) < 0.10 * w and hy1 - 0.03 * h <= feet_y <= hy2 + 0.05 * h:
                toks.add("foot_near_floor_hazard")

    # D) PARCHE solicitado: person_on_ladder
    # Lógica: si el bbox de persona y el de escalera se superponen > iou_touch
    # o centros a < near_thr, y la cabeza de la persona (y1) está por encima del 40% superior de la escalera.
    for p in persons:
        px1, py1, px2, py2 = p["box"]
        for ld in ladders:
            lx1, ly1, lx2, ly2 = ld["box"]
            cond_near = (_norm_dist_boxes(p["box"], ld["box"], w, h) < near_thr) or (_iou(p["box"], ld["box"]) > iou_touch)
            if cond_near:
                ladder_top = ly1 + 0.4 * (ly2 - ly1)
                # "y" crece hacia abajo: estar por encima <=> py1 <= ladder_top
                if py1 <= ladder_top:
                    toks.add("person_on_ladder")

    # tokens genéricos por presencia
    if strict:
        # en modo estricto no añadimos extra
        pass
    else:
        if any(d["cls"] in {"screen", "laptop", "keyboard", "mouse", "chair", "chair_office", "desk", "monitor"} for d in dets):
            toks.add("context_office")
        if any(d["cls"] in {"scaffold", "ladder", "helmet", "harness", "edge", "edge_exposure"} for d in dets):
            toks.add("context_obra")

    return toks


# ==========================
# Detección YOLO
# ==========================
def _choose_model_path(user_path: str | None) -> str:
    """
    Prioridad:
      1) user_path si existe
      2) models/best.pt
      3) models/yolov8m.pt
      4) yolov8n.pt (pretrained)
    """
    cands = []
    if user_path and os.path.exists(user_path):
        cands.append(user_path)
    cands.extend([
        os.path.join("models", "best.pt"),
        os.path.join("models", "yolov8m.pt"),
        "yolov8n.pt",
    ])
    for p in cands:
        if os.path.exists(p) or p == "yolov8n.pt":
            return p
    return "yolov8n.pt"

def yolo_detect(
    image: Image.Image | np.ndarray,
    model_path: Optional[str],
    conf: float = 0.25,
    iou: float = 0.60,
    imgsz: int = 960
) -> Tuple[List[Dict[str, Any]], int, int]:
    if not YOLO_OK:
        raise RuntimeError("Ultralytics YOLO no está disponible. Instala 'ultralytics'.")
    pil = _ensure_pil(image)
    w, h = pil.size
    mp = _choose_model_path(model_path)
    model = YOLO(mp)
    results = model.predict(source=np.array(pil), conf=conf, iou=iou, imgsz=imgsz, verbose=False, device="cpu")
    dets: List[Dict[str, Any]] = []
    names = model.names if hasattr(model, "names") else {i: str(i) for i in range(1000)}
    if results:
        r = results[0]
        if hasattr(r, "boxes") and r.boxes is not None:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_raw = boxes.cls[i]
                conf_raw = boxes.conf[i]
                cls_id = int(getattr(cls_raw, "item", lambda: cls_raw)())
                cf = float(getattr(conf_raw, "item", lambda: conf_raw)())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                dets.append({"cls": names.get(cls_id, str(cls_id)), "conf": cf, "box": [x1, y1, x2, y2]})
    return dets, w, h


# ==========================
# Render y armado de salida
# ==========================
def draw_boxes(img: Image.Image, dets: List[Dict[str, Any]]) -> Image.Image:
    pil = img.copy().convert("RGB")
    d = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = None
    for det in dets:
        x1, y1, x2, y2 = det["box"]
        d.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
        label = f"{det['cls']} {det['conf']:.2f}"
        d.text((x1 + 3, max(y1 - 18, 0)), label, fill=(255, 140, 0), font=font)
    return pil

def build_summary_and_md(risks: List[Dict[str, Any]], ontology: Dict[str, Any]) -> Tuple[str, str]:
    """Devuelve (resumen_texto, markdown_detallado)."""
    if not risks:
        return "Sin inferencias de riesgo.", "### Riesgos\n_No se detectaron riesgos con las reglas actuales._"
    lines = []
    md = ["### Riesgos e instrucciones (Jerarquía + Normatividad)"]
    for r in risks:
        rid = r["id"]
        rdef = ontology["risks"].get(rid, {})
        nombre = r.get("nombre", rid)
        lines.append(f"- {nombre}")
        md.append(f"**{nombre}**")
        jer = rdef.get("jerarquia", {})
        if jer:
            for k in ["eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"]:
                if k in jer and jer[k]:
                    md.append(f"- **{k.capitalize()}**: " + "; ".join(jer[k]))
        norms = rdef.get("normativas", [])
        if norms:
            md.append(f"- **Normativas:** " + "; ".join(norms))
        md.append("")  # espacio
    resumen = "Riesgos detectados:\n" + "\n".join(lines)
    return resumen, "\n".join(md)


# ==========================
# Carga ontología (una vez)
# ==========================
ONTOLOGY = load_ontology("risk_ontology.yaml")


# ==========================
# Lógica principal
# ==========================
def analyze(
    image: Image.Image | np.ndarray,
    model_path: str,
    conf: float,
    iou: float,
    near_thr: float,
    strict_mode: bool
):
    if image is None:
        raise gr.Error("Sube una imagen para analizar.")
    # 1) Detección
    dets, w, h = yolo_detect(image, model_path=model_path or None, conf=conf, iou=iou, imgsz=960)
    present = set([d["cls"] for d in dets])

    # 2) Tokens heurísticos (incluye PARCHE person_on_ladder)
    toks = image_tokens(dets, w, h, near_thr=near_thr, iou_touch=0.01, strict=bool(strict_mode))

    # 3) Contexto desde ontología
    ctx = make_context_tokens(ONTOLOGY, present)

    # 4) Unión y reglas
    present_ctx = present | toks | ctx
    risks = infer_risks(present_ctx, ONTOLOGY)

    # 5) Visual y salidas
    img_vis = draw_boxes(_ensure_pil(image), dets)
    resumen, md = build_summary_and_md(risks, ONTOLOGY)
    payload = {
        "classes_present": sorted(list(present)),
        "tokens": sorted(list(toks | ctx)),
        "risks": risks,
        "detections": dets
    }
    return img_vis, payload, resumen, md


# ==========================
# UI Gradio
# ==========================
with gr.Blocks(title="Analizador SST — Imágenes") as demo:
    gr.Markdown("## Analizador de Riesgos SST (Imágenes) — Ontología + Contexto + Heurísticas")
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Imagen")
            model_path = gr.Textbox(value="", label="Ruta a modelo YOLO (.pt) — opcional")
            conf = gr.Slider(0.10, 0.80, value=0.30, step=0.05, label="Confianza YOLO")
            iou = gr.Slider(0.30, 0.90, value=0.60, step=0.05, label="IoU YOLO (NMS)")
            near_thr = gr.Slider(0.05, 0.40, value=0.20, step=0.01, label="Distancia normalizada para proximidad")
            strict_mode = gr.Checkbox(False, label="Modo estricto (menos tokens automáticos)")
            btn = gr.Button("Analizar")
        with gr.Column():
            img_out = gr.Image(type="pil", label="Detecciones")
            json_out = gr.JSON(label="Salida JSON")
            resumen_out = gr.Textbox(label="Resumen", lines=8)
            md_out = gr.Markdown()

    btn.click(
        analyze,
        inputs=[img_in, model_path, conf, iou, near_thr, strict_mode],
        outputs=[img_out, json_out, resumen_out, md_out]
    )

if __name__ == "__main__":
    # Puerto automático (evita error si 7860 está ocupado).
    print(">> Iniciando UI Imágenes (puerto automático)")
    demo.launch(server_name="127.0.0.1", show_api=False)
