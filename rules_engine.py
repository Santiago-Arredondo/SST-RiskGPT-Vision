# rules_engine.py
from __future__ import annotations
import yaml
from typing import Dict, Any, List, Set, Optional
from pathlib import Path

class RiskEngine:
    """
    Carga risk_ontology.yaml y permite inferir riesgos a partir de:
      - present: set[str] con clases/tokens presentes
      - ctx_tokens: se derivan para office/industrial/obra
    Ontología esperada:
      meta: ...
      context_tokens: {office:{any:[...]}, industrial:{any:[...]}, obra:{any:[...]}}
      riesgos: [ {id, nombre, when_any, when_all?, context?, jerarquia[], normas[]} ]
    """

    def __init__(self, path: str | Path = "risk_ontology.yaml"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Ontología no encontrada: {self.path}")
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.meta: Dict[str, Any] = data.get("meta", {})
        self.ctx_defs: Dict[str, Dict[str, List[str]]] = data.get("context_tokens", {})
        self.risks: List[Dict[str, Any]] = data.get("riesgos", [])
        if not isinstance(self.risks, list):
            raise ValueError("El nodo 'riesgos' debe ser una lista de dicts.")

    # ---------- utilidades de contexto ----------
    def infer_contexts(self, present: Set[str]) -> Set[str]:
        ctx: Set[str] = set()
        for cname, rule in self.ctx_defs.items():
            any_list = set(rule.get("any", []))
            if any_list & present:
                ctx.add(cname)
        return ctx

    def _ctx_match(self, allowed: Optional[str], active_ctx: Set[str]) -> bool:
        if not allowed:
            return True
        # allowed: "office|industrial|obra"
        allow = set([s.strip() for s in allowed.split("|") if s.strip()])
        return bool(allow & active_ctx)

    # ---------- inferencia principal ------------
    def infer(self, present: List[str] | Set[str]) -> List[Dict[str, Any]]:
        P: Set[str] = set(present)
        active_ctx = self.infer_contexts(P)

        inferred: List[Dict[str, Any]] = []
        for r in self.risks:
            rid = r.get("id")
            name = r.get("nombre", rid)
            when_any = set(r.get("when_any", []))
            when_all = set(r.get("when_all", []))
            ctx_rule = r.get("context")  # "office|industrial|obra" or None

            # Debe cumplir contexto si se especifica
            if not self._ctx_match(ctx_rule, active_ctx):
                continue

            ok_any = (not when_any) or bool(when_any & P)
            ok_all = (not when_all) or when_all.issubset(P)

            if ok_any and ok_all:
                inferred.append({
                    "id": rid,
                    "nombre": name,
                    "jerarquia": list(r.get("jerarquia", [])),
                    "normas": list(r.get("normas", [])),
                })
        return inferred

    # Para UI: recomendaciones por ID
    def recommendations(self, rid: str) -> Dict[str, Any]:
        for r in self.risks:
            if r.get("id") == rid:
                return {"jerarquia": r.get("jerarquia", []), "normas": r.get("normas", [])}
        return {"jerarquia": [], "normas": []}
