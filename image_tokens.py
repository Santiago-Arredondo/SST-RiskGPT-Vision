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

def compute_tokens(
    dets: List[Dict],
    meta: Dict,
    active_contexts: Iterable[str],
    near_thresh_px: float = 80.0
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Devuelve (tokens, trace). Tokens usan exactamente los nombres de tu YAML:
      edge_open, person_on_edge, person_on_ladder, wet_floor, floor_irregular, ladder_like, ladder_unstable...
    """
    conf_thr = select_conf_threshold(meta, active_contexts)
    tokens: Set[str] = set()
    trace: Dict[str, List[str]] = {}

    def add(tok: str, why: str):
        tokens.add(tok)
        trace.setdefault(tok, []).append(why)

    # Índices por clase
    by_cls: Dict[str, List[Dict]] = {}
    for d in dets:
        c = str(d.get("cls", "")).lower()
        if float(d.get("conf", 0.0)) < conf_thr:
            continue
        by_cls.setdefault(c, []).append(d)

    # Mapeos directos de detección -> token
    direct = {
        "ladder": ["ladder"],
        "ladder like": ["ladder_like"],
        "scaffold": ["scaffold"],
        "rebar": ["rebar"],
        "roof": ["roof"],
        "edge": ["edge"],  # derivado a edge_open más abajo si no hay guardrail
        "hole": ["hole"],
        "trench": ["trench"],
        "guardrail": ["guardrail"],
        "platform": ["platform"],
        "cable": ["cable"],
        "spill": ["spill", "wet_floor"],
        "uneven floor": ["floor_irregular"],
        "wet floor sign": ["wet_floor"],
        "rack": ["rack"],
        "shelf": ["shelf"],
        "low beam": ["low_beam"],
        "grinder": ["grinder"],
        "saw": ["saw"],
        "cutting": ["cutting"],
        "person": ["person"],
        "forklift": ["forklift"],
        "machine": ["machine"],
        "press": ["press"],
        "gear": ["gear"],
        "vehicle": ["vehicle_moving"]  # si tu detector lo saca como 'vehicle'
    }
    for cls_name, toks in direct.items():
        for d in by_cls.get(cls_name, []):
            for t in toks:
                add(t, f"{cls_name} conf={d['conf']:.2f}")

    # Derivaciones locativas
    has_guardrail = "guardrail" in by_cls
    for e in by_cls.get("edge", []):
        if not has_guardrail:
            add("edge_open", f"edge sin guardrail conf={e['conf']:.2f}")

    # Cercanías: persona cerca de edge o ladder
    people = by_cls.get("person", [])
    for lad in by_cls.get("ladder", []) + by_cls.get("ladder like", []):
        c_l = _center(lad["box"])
        for p in people:
            if _dist(c_l, _center(p["box"])) <= near_thresh_px:
                add("person_on_ladder", f"persona cerca de ladder ({near_thresh_px}px)")
                break

    for e in by_cls.get("edge", []):
        c_e = _center(e["box"])
        for p in people:
            if _dist(c_e, _center(p["box"])) <= near_thresh_px:
                add("person_on_edge", f"persona cerca de edge ({near_thresh_px}px)")
                break

    # Heurísticos simples
    if "ladder" in by_cls and "platform" not in by_cls and "guardrail" not in by_cls:
        add("ladder_unstable", "ladder sin plataforma/guardrail visibles")

    # Normalizaciones de compatibilidad
    if "spill" in tokens and "wet_floor" not in tokens:
        add("wet_floor", "derivado de spill")

    return sorted(tokens), trace
