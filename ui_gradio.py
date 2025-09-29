from __future__ import annotations
import os
from typing import Any, Dict, List, Set, Tuple, Optional

import yaml
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

import cv2
from ultralytics import YOLO

# Si añadiste pose_utils.py previamente, importamos los tokens ergonómicos
try:
    from pose_utils import ergonomic_tokens  # opcional, pero recomendado
    _HAVE_POSE = True
except Exception:
    _HAVE_POSE = False


# =========================================================
# Utilidades de E/S
# =========================================================
def _ensure_pil(img: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


def load_ontology(path: str = "risk_ontology.yaml") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================================================
# Modelo de detección (YOLOv8)
# =========================================================
_DET_MODEL_CACHE: Dict[str, YOLO] = {}


def _load_model(model_path: Optional[str]) -> YOLO:
    key = model_path or "yolov8n.pt"
    if key not in _DET_MODEL_CACHE:
        _DET_MODEL_CACHE[key] = YOLO(key)
    return _DET_MODEL_CACHE[key]


def yolo_detect(image: Image.Image,
                model_path: Optional[str] = None,
                conf: float = 0.35,
                iou: float = 0.60,
                imgsz: int = 1280) -> Tuple[List[Dict], int, int]:
    """
    Ejecuta YOLO y retorna:
      dets: [{cls, conf, box=[x1,y1,x2,y2]}], w, h
    """
    model = _load_model(model_path)
    res = model.predict(image, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    dets: List[Dict] = []
    if not res:
        w, h = image.size
        return dets, w, h

    r0 = res[0]
    w, h = r0.orig_shape[1], r0.orig_shape[0]
    names = r0.names
    for b in r0.boxes:
        c = int(b.cls)
        cls_name = names.get(c, str(c))
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        confv = float(b.conf[0].item())
        dets.append({
            "cls": cls_name,
            "conf": confv,
            "box": [float(x1), float(y1), float(x2), float(y2)]
        })
    return dets, w, h


# =========================================================
# Heurísticas / tokens de escena
# =========================================================
def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_iou(a, b):
    x1, y1, x2, y2 = a
    X1, Y1, X2, Y2 = b
    xi1, yi1 = max(x1, X1), max(y1, Y1)
    xi2, yi2 = min(x2, X2), min(y2, Y2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, X2 - X1) * max(0, Y2 - Y1)
    return inter / (a1 + a2 - inter + 1e-6)


def _near(p1, p2, thr: float, W: int, H: int) -> bool:
    dx = (p1[0] - p2[0]) / max(1.0, W)
    dy = (p1[1] - p2[1]) / max(1.0, H)
    return (dx*dx + dy*dy) ** 0.5 <= thr


def image_tokens(dets: List[Dict], W: int, H: int,
                 near_thr: float = 0.20,
                 iou_touch: float = 0.01,
                 strict: bool = False) -> Set[str]:
    """
    Tokens de relaciones generales (vehículo-persona, obstáculos, máquina...).
    """
    toks: Set[str] = set()
    persons = [d for d in dets if d["cls"] in {"person"}]
    vehicles = [d for d in dets if d["cls"] in {"forklift", "truck", "vehicle"}]
    machines = [d for d in dets if d["cls"] in {"machine", "press", "gear", "grinder", "saw"}]

    # Persona cerca de vehículo
    for p in persons:
        for v in vehicles:
            if _near(_center(p["box"]), _center(v["box"]), near_thr, W, H):
                toks.add("person_near_vehicle")
                toks.add("vehicle_moving")
                if v["cls"] == "forklift":
                    toks.add("forklift_moving")

    # Pie cerca de obstáculo (cables/charcos/grietas)
    obstacles = [d for d in dets if d["cls"] in {"cable", "debris", "wet_floor", "hole", "floor_irregular"}]
    for p in persons:
        for ob in obstacles:
            if _bbox_iou(p["box"], ob["box"]) > iou_touch or _near(_center(p["box"]), _center(ob["box"]), near_thr, W, H):
                toks.add("foot_near_obstacle")

    # Máquina con partes móviles expuestas
    for m in machines:
        if m["cls"] in {"gear"}:
            toks.add("exposed_gears")
        if m["cls"] in {"grinder", "saw"}:
            toks.add("moving_part")
        if any(d["cls"] in {"guard_missing"} for d in dets):
            toks.add("guard_missing")

    # Persona en escalera (si YOLO entregó ladder)
    if any(d["cls"] in {"ladder"} for d in dets) and persons:
        toks.add("person_on_ladder")

    return toks


# ========= Fallbacks/Heurísticas específicas (obra y escalera) =========
def person_token_from_pose(npimg: np.ndarray) -> bool:
    """
    Devuelve True si MediaPipe detecta pose humana válida (persona presente).
    """
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0)
        res = pose.process(npimg)
        return bool(res and res.pose_landmarks)
    except Exception:
        return False


def edge_tokens(npimg: np.ndarray, dets: List[Dict], W: int, H: int) -> Set[str]:
    """
    Detecta líneas horizontales largas (posible borde de losa) y marca persona cerca de ese borde.
    Devuelve { 'edge_open', 'person_on_edge' } según aplique.
    """
    toks: Set[str] = set()
    try:
        gray = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 50, 140)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90,
                                minLineLength=int(W * 0.28), maxLineGap=18)
        if lines is None:
            return toks

        yhs = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 6:  # casi horizontales
                yhs.append((x1, y1, x2, y2))
        if not yhs:
            return toks

        toks.add("edge_open")
        y_edge = int(np.median([y for _, y, __, ___ in yhs]))

        persons = [d for d in dets if d["cls"] == "person"]
        for p in persons:
            _, _, _, y2 = p["box"]
            y_feet = int(y2)
            if abs(y_feet - y_edge) < int(0.08 * H):  # ±8% alto imagen
                toks.add("person_on_edge")
                break
    except Exception:
        pass
    return toks


def ladder_tokens(npimg: np.ndarray, dets: List[Dict], W: int, H: int) -> Set[str]:
    """
    Busca patrón de peldaños (líneas horizontales cortas apiladas).
    Si hay persona y su bbox intersecta la banda superior, marcamos person_on_ladder.
    """
    toks: Set[str] = set()
    try:
        gray = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80,
                                minLineLength=int(W * 0.06), maxLineGap=8)
        if lines is None:
            return toks

        hs = []
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= 3 and abs(x2 - x1) <= int(W * 0.18):
                hs.append((x1, y1, x2, y2))
        if len(hs) < 4:
            return toks

        xs = np.array([(x1 + x2) / 2 for x1, y1, x2, y2 in hs])
        x_med = np.median(xs)
        band_w = int(W * 0.12)
        band_x1, band_x2 = int(x_med - band_w/2), int(x_med + band_w/2)
        hs_band = [(x1, y1, x2, y2) for x1, y1, x2, y2 in hs if band_x1 <= (x1+x2)/2 <= band_x2]
        if len(hs_band) < 4:
            return toks

        toks.add("ladder")

        y_top_band = min(y1 for _, y1, __, ___ in hs_band)
        persons = [d for d in dets if d["cls"] == "person"]
        for p in persons:
            x1, y1, x2, y2 = p["box"]
            if (x2 >= band_x1 and x1 <= band_x2) and (y2 <= y_top_band + int(0.25 * H)):
                toks.add("person_on_ladder")
                break
    except Exception:
        pass
    return toks


# Coloca este set arriba, cerca de los imports/utilidades:
OFFICE_TOKENS = {
    "desk", "chair", "monitor", "screen", "laptop", "keyboard", "mouse",
    "computer", "pc", "notebook", "printer", "phone"
}

OFFICE_TOKENS = {
    "desk", "chair", "monitor", "screen", "laptop", "keyboard", "mouse",
    "computer", "pc", "notebook", "printer", "phone"
}

def ladder_like_tokens(npimg: np.ndarray, dets: List[Dict], present: Set[str], W: int, H: int) -> Set[str]:
    """
    Detecta 'ladder_like' aunque YOLO etiquete 'chair'. Robusto a fondo blanco.
    - Mejora contraste (CLAHE)
    - Canny sensible + Hough para peldaños
    - Fallback geométrico si no hay peldaños pero hay objeto alto/estrecho solapado con persona
      (sólo si la escena NO huele a oficina)
    """
    toks: Set[str] = set()
    try:
        persons = [d for d in dets if d.get("cls") == "person"]
        others  = [d for d in dets if d.get("cls") != "person"]
        if not persons or not others:
            return toks

        is_office_scene = len(OFFICE_TOKENS & set(present)) >= 2

        gray = cv2.cvtColor(npimg, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        edges = cv2.Canny(gray, 30, 90)

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            xi1, yi1 = max(ax1, bx1), max(ay1, by1)
            xi2, yi2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
            inter = iw * ih
            A = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
            B = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
            return inter / (A + B - inter + 1e-6)

        def overlaps(a, b):
            ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
            xi1, yi1 = max(ax1, bx1), max(ay1, by1)
            xi2, yi2 = min(ax2, bx2), min(ay2, by2)
            return (xi2 > xi1) and (yi2 > yi1)

        MIN_AR     = 1.55   # alto/estrecho
        MIN_H_FRAC = 0.45   # altura relativa mínima
        MAX_W_FRAC = 0.40   # muy ancho => descartar

        for d in others:
            x1, y1, x2, y2 = map(int, d["box"])
            w = max(1, x2 - x1); h = max(1, y2 - y1)
            ar = h / float(w)
            h_frac = h / float(H)
            w_frac = w / float(W)
            if ar < MIN_AR or h_frac < MIN_H_FRAC or w_frac > MAX_W_FRAC:
                continue

            roi = edges[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if roi.size == 0:
                continue

            # ---- Hough de peldaños ----
            lines = cv2.HoughLinesP(
                roi, 1, np.pi/180, threshold=45,
                minLineLength=max(5, int(w * 0.18)),
                maxLineGap=8
            )
            y_vals = []
            if lines is not None:
                for lx1, ly1, lx2, ly2 in lines[:, 0]:
                    if abs(ly1 - ly2) <= 2 and abs(lx2 - lx1) <= int(w * 0.85):
                        y_vals.append(int((ly1 + ly2) / 2))
            y_vals.sort()
            clusters = []
            for yv in y_vals:
                if not clusters or abs(yv - clusters[-1][-1]) > 5:
                    clusters.append([yv])
                else:
                    clusters[-1].append(yv)
            rung_count_hough = len(clusters)

            # ---- Fallback: densidad de bordes horizontales (Sobel Y) ----
            sob = cv2.Sobel(roi, cv2.CV_16S, dx=0, dy=1, ksize=3)
            sob = cv2.convertScaleAbs(sob)
            col_sum = sob.mean(axis=1)
            m, s = float(col_sum.mean()), float(col_sum.std())
            thr = m + 1.0 * s
            rung_count_proj = int((col_sum > thr).sum() / 2.0)

            has_rungs = (rung_count_hough >= 2) or (rung_count_proj >= 3)

            if has_rungs:
                toks.add("ladder_like")
                if any(overlaps((x1, y1, x2, y2), tuple(map(int, p["box"]))) for p in persons) \
                   or any(iou((x1, y1, x2, y2), tuple(map(int, p["box"]))) > 0.05 for p in persons):
                    toks.add("person_on_ladder")
                continue

            # ---- Fallback geométrico (si NO es oficina) ----
            if not is_office_scene:
                VERY_TALL_AR   = 1.80
                VERY_TALL_HFR  = 0.60
                if (ar >= VERY_TALL_AR and h_frac >= VERY_TALL_HFR) and \
                   any(iou((x1, y1, x2, y2), tuple(map(int, p["box"]))) > 0.05 for p in persons):
                    toks.add("ladder_like")
                    toks.add("person_on_ladder")
    except Exception:
        pass
    return toks

def ensure_person_token(npimg, dets, W, H, toks: set, present: set):
    """
    Asegura que exista 'person' cuando YOLO falla (p. ej. lo marca como chair/elephant).
    1) Intenta MediaPipe Pose.
    2) Si no, geometría: bbox alto/angosto razonable -> 'person'.
    """
    if "person" in present or "person" in toks:
        return
    # 1) Pose
    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0)
        res = pose.process(npimg)
        if res and res.pose_landmarks:
            toks.add("person")
            return
    except Exception:
        pass
    # 2) Geometría
    best_h = 0
    for d in dets:
        if d.get("cls") == "person":
            continue
        x1, y1, x2, y2 = map(int, d["box"])
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        ar = h / float(w)
        h_frac = h / float(H); w_frac = w / float(W)
        if ar >= 1.30 and h_frac >= 0.30 and w_frac <= 0.60:
            if h > best_h: best_h = h
    if best_h > 0:
        toks.add("person")


# =========================================================
# Contexto por objetos presentes
# =========================================================
def make_context_tokens(ONTO: Dict[str, Any], present: Set[str]) -> Set[str]:
    ctx = set()
    for name, rule in (ONTO.get("contexts") or {}).items():
        any_tokens = set(rule.get("any", []))
        if any_tokens & present:
            ctx.add(name)
    return ctx


# =========================================================
# Inferencia de riesgos desde ontología
# =========================================================
def _risk_rank(t: str) -> int:
    order = {"LOCATIVO": 0, "MECÁNICO": 1, "ERGONÓMICO": 2}
    return order.get(t, 9)


def infer_risks(tokens: Set[str], ONTO: Dict[str, Any]) -> List[Dict[str, Any]]:
    risks_out: List[Dict[str, Any]] = []
    R = ONTO.get("risks", {}) or {}
    for rid, r in R.items():
        ctx_ok = True
        if r.get("context"):
            ctx_ok = any(c in tokens for c in r["context"])
        any_ok = True
        if r.get("if_any"):
            any_ok = any(t in tokens for t in r["if_any"])
        all_ok = True
        if r.get("if_all"):
            all_ok = all(t in tokens for t in r["if_all"])
        if ctx_ok and any_ok and all_ok:
            risks_out.append({
                "id": rid,
                "tipo": r.get("tipo", ""),
                "nombre": r.get("nombre", rid),
                "descripcion": r.get("descripcion", ""),
                "normativa": r.get("normativa", []),
                "controles": r.get("controles", {}),
                "severidad": r.get("severidad", "MEDIA"),
            })
    risks_out = sorted(risks_out, key=lambda x: _risk_rank(x.get("tipo", "")))
    return risks_out


# =========================================================
# Render de salida
# =========================================================
def _draw_boxes(image: Image.Image, dets: List[Dict]) -> Image.Image:
    im = image.copy()
    dr = ImageDraw.Draw(im)
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        dr.rectangle([x1, y1, x2, y2], outline=(255, 140, 0), width=2)
        label = f'{d["cls"]} {d["conf"]:.2f}'
        dr.text((x1 + 3, max(0, y1 - 12)), label, fill=(255, 140, 0))
    return im


def build_summary_and_md(risks: List[Dict[str, Any]]) -> Tuple[str, str]:
    if not risks:
        return ("Sin riesgos detectados relevantes.",
                "### ✅ Sin riesgos detectados\nNo se encontraron condiciones de riesgo con las reglas actuales.")
    resumen = ", ".join(f'{r["nombre"]} ({r["tipo"]})' for r in risks[:5])
    lines = []
    for r in risks:
        lines.append(f'## {r["nombre"]}  `[{r["tipo"]} | {r["severidad"]}]`')
        if r.get("descripcion"):
            lines.append(f'> {r["descripcion"]}')
        C = r.get("controles", {})
        order = ["Eliminación", "Sustitución", "Ingeniería", "Administrativos", "EPP"]
        for k in order:
            if k in C and C[k]:
                lines.append(f'**{k}:**')
                for it in C[k]:
                    lines.append(f'- {it}')
        N = r.get("normativa", [])
        if N:
            lines.append("**Normativa aplicable:**")
            for n in N[:4]:
                lines.append(f'- {n}')
        lines.append("---")
    md = "\n".join(lines)
    return resumen, md


# =========================================================
# Pipeline principal (imagen)
# =========================================================
def analyze(image: Image.Image | np.ndarray,
            model_path: Optional[str] = None,
            conf: float = 0.35,
            iou: float = 0.60,
            near_thr: float = 0.20,
            strict_mode: bool = False) -> Tuple[Image.Image, Dict[str, Any], str, str]:

    ONTO = load_ontology("risk_ontology.yaml")
    img_pil = _ensure_pil(image)

    # 1) Detección YOLO
    dets, W, H = yolo_detect(img_pil, model_path=model_path, conf=conf, iou=iou, imgsz=1280)
    present = {d["cls"] for d in dets}

    # 2) Tokens generales
    toks = image_tokens(dets, W, H, near_thr=near_thr, iou_touch=0.01, strict=bool(strict_mode))

    # 3) Ergonomía + asegurar 'person'
    npimg = np.array(img_pil)
    if _HAVE_POSE:
        try: toks |= ergonomic_tokens(npimg, dets)
        except Exception: pass
    try: ensure_person_token(npimg, dets, W, H, toks, present)
    except Exception: pass

    # 4) Heurísticas borde/escalera
    try: toks |= edge_tokens(npimg, dets, W, H)
    except Exception: pass
    try: toks |= ladder_tokens(npimg, dets, W, H)
    except Exception: pass
    try: toks |= ladder_like_tokens(npimg, dets, present, W, H)  # ¡con present!
    except Exception: pass

    # 5) Fallback geométrico mínimo para escalera (por si todo lo anterior falla)
    try:
        persons = [d for d in dets if d.get("cls") == "person"] or ([None] if "person" in toks else [])
        others  = [d for d in dets if d.get("cls") != "person"]
        if persons and others:
            # usamos el bbox más alto/angosto como candidata a escalera
            for o in others:
                ox1, oy1, ox2, oy2 = map(int, o["box"])
                w = max(1, ox2 - ox1); h = max(1, oy2 - oy1)
                ar = h / float(w); h_frac = h / float(H); w_frac = w / float(W)
                if ar >= 1.35 and h_frac >= 0.42 and w_frac <= 0.50:
                    toks.add("ladder_like")
                    # si además hay 'person' (por ensure_person_token), inferimos interacción
                    if "person" in toks or any(d.get("cls") == "person" for d in dets):
                        toks.add("person_on_ladder")
                        break
    except Exception:
        pass

    # 6) Contexto (no fuerces obra si huele a oficina)
    OFFICE_TOKENS = {"desk","chair","monitor","screen","laptop","keyboard","mouse",
                     "computer","pc","notebook","printer","phone"}
    ctx = make_context_tokens(ONTO, present)
    is_office_scene = len(OFFICE_TOKENS & set(present)) >= 2
    if ({"edge_open","ladder","ladder_like","person_on_ladder"} & toks) and not is_office_scene:
        ctx |= {"construction"}

    # --- DEBUG (deja esta línea unas pruebas) ---
    # print("TOKENS DEBUG:", sorted(list(present | toks)))

    # 7) Inferencia
    all_tokens = present | toks | ctx
    risks = infer_risks(all_tokens, ONTO)

    # 8) Salida
    vis = _draw_boxes(img_pil, dets)
    resumen, md = build_summary_and_md(risks)
    payload = {
        "tokens_detectados": sorted(list(all_tokens)),
        "n_detecciones": len(dets),
        "riesgos": risks,
    }
    return vis, payload, f"Resumen: {resumen}", md


# =========================================================
# Gradio UI
# =========================================================
def _ui():
    with gr.Blocks(title="SST-RiskGPT-Vision") as demo:
        gr.Markdown("# SST-RiskGPT-Vision\nDetección de **riesgos LOCATIVOS/MECÁNICOS** + recomendaciones normativas.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(label="Imagen", type="pil")
                model = gr.Textbox(label="Ruta modelo YOLO (opcional)", value="")
                conf = gr.Slider(0.1, 0.85, value=0.35, step=0.05, label="Confianza")
                iou = gr.Slider(0.1, 0.9, value=0.60, step=0.05, label="NMS IoU")
                near = gr.Slider(0.05, 0.5, value=0.20, step=0.01, label="Umbral cercanía")
                strict = gr.Checkbox(value=False, label="Modo estricto (tokens heurísticos conservadores)")
                btn = gr.Button("Analizar")
            with gr.Column():
                out_img = gr.Image(label="Detecciones", type="pil")
                out_json = gr.JSON(label="Debug / Tokens / Riesgos")
        md_summary = gr.Markdown()
        md_detail = gr.Markdown()

        def _wrap(image, model_path, conf, iou, near_thr, strict_mode):
            mp = model_path or None
            vis, payload, resumen, md = analyze(
                image, model_path=mp, conf=conf, iou=iou, near_thr=near_thr, strict_mode=strict_mode
            )
            return vis, payload, f"### {resumen}", md

        btn.click(_wrap, [inp, model, conf, iou, near, strict],
                  [out_img, out_json, md_summary, md_detail])
    return demo


if __name__ == "__main__":
    _ui().launch()
