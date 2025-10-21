# image_tokens.py
from __future__ import annotations
from typing import Dict, List, Tuple, Iterable, Set
import math

def _center(box):
    x1, y1, x2, y2 = box
    return (0.5*(x1+x2), 0.5*(y1+y2))

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def select_conf_threshold(meta: Dict, active_contexts: Iterable[str]) -> float:
    thr = (meta or {}).get("thresholds", {}) or {}
    for c in list(active_contexts or []):
        key = f"{c}_conf"
        if key in thr:
            return float(thr[key])
    return float(thr.get("default_conf", 0.25))

def _class_conf_thr(meta: Dict, active_ctx: Iterable[str], cls_name: str, default_thr: float) -> float:
    per_ctx = (meta.get("class_thresholds", {}) or {})
    for c in list(active_ctx or []):
        thr = (per_ctx.get(c, {}) or {}).get(cls_name)
        if thr is not None:
            return float(thr)
    return default_thr

def compute_tokens(
    dets: List[Dict],
    meta: Dict,
    active_contexts: Iterable[str],
    near_thresh_px: float = 80.0
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Traduce detecciones a tokens activadores de riesgos.
    Incluye sinónimos (wire/cord/rope/hose -> cable) y heurística de pie cercano al obstáculo.
    """
    base_thr = select_conf_threshold(meta, active_contexts)
    tokens: Set[str] = set()
    trace: Dict[str, List[str]] = {}

    def add(tok: str, why: str):
        tokens.add(tok)
        trace.setdefault(tok, []).append(why)

    # Indexar por clase normalizada con umbral por clase/contexto
    by: Dict[str, List[Dict]] = {}
    for d in dets:
        c = str(d.get("cls", "")).lower()
        thr_c = _class_conf_thr(meta, active_contexts, c, base_thr)
        if float(d.get("conf", 0.0)) < thr_c:
            continue
        by.setdefault(c, []).append(d)

    # Mapeo directo + sinónimos
    mapping = {
        # locativos
        "ladder": ["ladder"], "ladder like": ["ladder_like"], "scaffold": ["scaffold"],
        "rebar": ["rebar"], "roof": ["roof"], "edge": ["edge"], "hole": ["hole"], "trench": ["trench"],
        "guardrail": ["guardrail"], "platform": ["platform"], "debris": ["debris"],
        "uneven floor": ["floor_irregular"], "wet floor sign": ["wet_floor"],
        # cable y sinónimos
        "cable": ["cable"], "wire": ["cable"], "cord": ["cable"], "extension cord": ["cable"],
        "power cord": ["cable"], "hose": ["cable"], "rope": ["cable"],
        # almacén/industrial
        "rack": ["rack"], "shelf": ["shelf"], "low beam": ["low_beam"],
        "grinder": ["grinder"], "saw": ["saw"], "cutting": ["cutting"], "forklift": ["forklift"],
        # humanos y vehículos
        "person": ["person"], "car": ["vehicle"], "vehicle": ["vehicle"], "car lift": ["lift"], "hoist": ["lift"],
        # oficina/ergo básico
        "screen": ["screen"], "laptop": ["laptop"],
        # derrames
        "spill": ["spill", "wet_floor"]
    }
    for cls_name, toks in mapping.items():
        for d in by.get(cls_name, []):
            for t in toks:
                add(t, f"{cls_name} conf={d['conf']:.2f}")

    # Borde abierto: solo si no hay guardrail
    if "edge" in by and "guardrail" not in by:
        for e in by["edge"]:
            add("edge_open", f"edge sin guardrail conf={e['conf']:.2f}")

    # Persona en borde (proximidad)
    if "person" in by and "edge" in by:
        for e in by["edge"]:
            ce = _center(e["box"])
            for p in by["person"]:
                if _dist(ce, _center(p["box"])) <= near_thresh_px:
                    add("person_on_edge", f"persona cerca de edge ({near_thresh_px}px)")
                    break

    # Persona en escalera
    if "person" in by and ("ladder" in by or "ladder like" in by):
        ladders = by.get("ladder", []) + by.get("ladder like", [])
        for lad in ladders:
            cl = _center(lad["box"])
            for p in by["person"]:
                if _dist(cl, _center(p["box"])) <= near_thresh_px:
                    add("person_on_ladder", f"persona cerca de ladder ({near_thresh_px}px)")
                    break
    if "ladder" in by and "platform" not in by and "guardrail" not in by:
        add("ladder_unstable", "ladder sin plataforma/guardrail visibles")

    # Vehículo elevado y persona debajo
    cars = by.get("car", []) + by.get("vehicle", [])
    lifts = by.get("lift", []) + by.get("hoist", [])
    if cars:
        for car in cars:
            cc = _center(car["box"])
            if lifts:
                add("vehicle_on_lift", "vehículo + elevador detectado")
            for p in by.get("person", []):
                pc = _center(p["box"])
                x1,y1,x2,y2 = car["box"]; px1,py1,px2,py2 = p["box"]
                overlap_x = min(x2,px2) - max(x1,px1) > 0
                if cc[1] < pc[1] and overlap_x:
                    add("vehicle_on_lift", "vehículo alto respecto a persona (heur.)")
                    add("person_under_vehicle", "persona bajo vehículo (heur.)")
                    break

    # Normalizaciones
    if "spill" in tokens and "wet_floor" not in tokens:
        add("wet_floor", "derivado de spill")

    # Heurística: pie cerca de obstáculo (cable/hose/rope) en banda inferior de la persona
    persons = by.get("person", [])
    cables_like = []
    for key in ("cable", "rope", "hose"):
        cables_like += by.get(key, [])
    if persons and cables_like:
        for p in persons:
            px1, py1, px2, py2 = p["box"]
            h = max(1.0, py2 - py1)
            foot_y1 = py2 - 0.12 * h
            foot_y2 = py2
            for c in cables_like:
                cx1, cy1, cx2, cy2 = c["box"]
                overlap_x = min(px2, cx2) - max(px1, cx1)
                overlap_y = min(foot_y2, cy2) - max(foot_y1, cy1)
                if overlap_x > 0 and overlap_y > 0:
                    add("foot_near_obstacle", "cable/obstáculo en banda de pies")
                    break

    # Ergo heurístico mínimo (sin pose)
    screens = by.get("screen", []) + by.get("laptop", [])
    for s in screens:
        sx1, sy1, sx2, sy2 = s["box"]
        scy = (sy1 + sy2) / 2.0
        for p in persons:
            px1, py1, px2, py2 = p["box"]
            eyes_y = py1 + 0.15*(py2 - py1)
            if scy > eyes_y + 8:
                add("screen_below_eyes", "pantalla por debajo de ojos (heurística)")
                break

    return sorted(tokens), trace
