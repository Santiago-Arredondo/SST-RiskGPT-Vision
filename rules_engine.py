from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Set, Union
import copy
import yaml

def _read_yaml(p: Path) -> Dict[str, Any]:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def _merge_lists_unique(a: List[Any], b: List[Any]) -> List[Any]:
    return list(dict.fromkeys(list(a or []) + list(b or [])))

def _merge_contexts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a or {})
    for k, v in (b or {}).items():
        out.setdefault(k, {})
        out[k]["any"] = _merge_lists_unique(out[k].get("any", []), v.get("any", []))
    return out

def _merge_controles(a_ctrl: Dict[str, List[str]] | None, b_ctrl: Dict[str, List[str]] | None) -> Dict[str, List[str]]:
    a_ctrl = a_ctrl or {}
    b_ctrl = b_ctrl or {}
    keys = set(a_ctrl.keys()) | set(b_ctrl.keys())
    out: Dict[str, List[str]] = {}
    for k in keys:
        out[k] = _merge_lists_unique(a_ctrl.get(k, []), b_ctrl.get(k, []))
    return out

def _merge_risk_entry(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k in ["tipo", "nombre", "descripcion", "severidad"]:
        out[k] = out.get(k, b.get(k))
    for k in ["context", "if_any", "if_all", "normativa"]:
        out[k] = _merge_lists_unique(a.get(k, []), b.get(k, []))
    out["controles"] = _merge_controles(a.get("controles"), b.get("controles"))
    return out

def _merge_risks(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a or {})
    for rid, rb in (b or {}).items():
        if rid in out:
            out[rid] = _merge_risk_entry(out[rid], rb)
        else:
            out[rid] = rb
    return out

class RiskEngine:
    def __init__(self, paths: Union[str, Path, List[Union[str, Path]]] = None):
        if paths is None:
            paths = ["risk_ontology.yaml", "risk_ontology_ext.yaml"]
        if not isinstance(paths, list):
            paths = [paths]

        meta: Dict[str, Any] = {}
        contexts: Dict[str, Any] = {}
        risks: Dict[str, Any] = {}

        for p in paths:
            d = _read_yaml(Path(p))
            dm = d.get("meta", {}) or {}
            if "open_vocab" in dm:
                meta.setdefault("open_vocab", [])
                meta["open_vocab"] = _merge_lists_unique(meta["open_vocab"], dm["open_vocab"])
            if "thresholds" in dm:
                meta.setdefault("thresholds", {})
                meta["thresholds"].update(dm["thresholds"] or {})
            contexts = _merge_contexts(contexts, d.get("contexts", {}) or {})
            risks = _merge_risks(risks, d.get("risks", {}) or {})

        self.meta = meta
        self.contexts = contexts
        self.risks = risks

        self._ctrl_key_map = {
            "Eliminación": "eliminacion",
            "Sustitución": "sustitucion",
            "Ingeniería": "ingenieria",
            "Administrativos": "administrativos",
            "EPP": "epp",
        }

    def _normalize_tokens(self, tokens: Union[List[str], Set[str]]) -> Set[str]:
        mapping = {
            "wet floor": "wet floor sign",
            "ladder_like": "ladder like",
            "cell phone": "phone", "mobile": "phone", "smartphone": "phone",
            "monitor": "screen", "tv": "screen",
        }
        normalized = set()
        for t in tokens:
            key = t.lower().strip().replace("_", " ")
            normalized.add(mapping.get(key, key))
        return normalized

    def active_contexts(self, present: Set[str]) -> Set[str]:
        ctx: Set[str] = set()
        for name, rule in self.contexts.items():
            any_tokens = set(rule.get("any", []) or [])
            if any_tokens & present:
                ctx.add(name)
        return ctx

    def infer(self, present_tokens: Union[List[str], Set[str]]) -> List[Dict[str, Any]]:
        P: Set[str] = self._normalize_tokens(present_tokens)
        ctx = self.active_contexts(P)
        TOK = P | ctx
        out: List[Dict[str, Any]] = []
        for rid, r in self.risks.items():
            ctx_list = set(r.get("context", []) or [])
            ctx_ok = True if not ctx_list else bool(ctx_list & ctx)
            any_set = set(t for t in (r.get("if_any") or []) if t)
            all_set = set(t for t in (r.get("if_all") or []) if t)
            any_ok = True if not any_set else bool(any_set & TOK)
            all_ok = True if not all_set else all_set.issubset(TOK)
            if ctx_ok and any_ok and all_ok:
                out.append({
                    "id": rid,
                    "tipo": r.get("tipo", ""),
                    "nombre": r.get("nombre", rid),
                    "descripcion": r.get("descripcion", ""),
                    "severidad": r.get("severidad", "MEDIA"),
                })
        order = {"LOCATIVO": 0, "MECÁNICO": 1, "ERGONÓMICO": 2}
        out.sort(key=lambda x: order.get(x.get("tipo", ""), 9))
        return out

    def recommendations(self, rid: str) -> Dict[str, Any]:
        r = self.risks.get(rid)
        if not r:
            return {"jerarquia": {}, "normativas": []}
        ctrls = r.get("controles", {}) or {}
        jerarquia: Dict[str, List[str]] = {}
        for title, key in self._ctrl_key_map.items():
            vals = ctrls.get(title) or []
            if vals:
                jerarquia[key] = list(vals)
        normativas = list(r.get("normativa", []) or [])
        return {"jerarquia": jerarquia, "normativas": normativas}
