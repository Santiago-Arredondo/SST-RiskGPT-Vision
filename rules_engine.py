# rules_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Set, Optional
import yaml

class RiskEngine:
    """
    Motor de reglas alineado con risk_ontology.yaml (v0.4):
      - contexts: {nombre: {any: [tokens...]}}
      - risks: {risk_id: {tipo, nombre, descripcion, context[], if_any[], if_all[], severidad, normativa[], controles{...}}}
    """

    def __init__(self, path: str | Path = "risk_ontology.yaml"):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Ontología no encontrada: {self.path}")
        data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        self.meta: Dict[str, Any] = data.get("meta", {}) or {}
        self.contexts: Dict[str, Dict[str, List[str]]] = data.get("contexts", {}) or {}
        self.risks: Dict[str, Dict[str, Any]] = data.get("risks", {}) or {}

        if not isinstance(self.risks, dict):
            raise ValueError("El nodo 'risks' debe ser un dict {id: {...}}")

        # Mapa de títulos -> claves minúsculas que espera la capa de chat
        self._ctrl_key_map = {
            "Eliminación": "eliminacion",
            "Sustitución": "sustitucion",
            "Ingeniería": "ingenieria",
            "Administrativos": "administrativos",
            "EPP": "epp",
        }

    # ---------- Contextos ----------
    def _active_contexts(self, present: Set[str]) -> Set[str]:
        ctx: Set[str] = set()
        for name, rule in self.contexts.items():
            any_tokens = set(rule.get("any", []) or [])
            if any_tokens & present:
                ctx.add(name)
        return ctx

    # ---------- Inferencia ----------
    def infer(self, present: List[str] | Set[str]) -> List[Dict[str, Any]]:
        P: Set[str] = set(present)
        ctx = self._active_contexts(P)
        # Permitimos que las reglas miren también los contextos como “tokens”
        TOK = P | ctx

        out: List[Dict[str, Any]] = []
        for rid, r in self.risks.items():
            # Regla de contexto: basta con que uno coincida
            ctx_list = set(r.get("context", []) or [])
            ctx_ok = True if not ctx_list else bool(ctx_list & ctx)

            # Condiciones “any / all”
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

        # Ordenar por tipo para dar algo estable
        order = {"LOCATIVO": 0, "MECÁNICO": 1, "ERGONÓMICO": 2}
        out.sort(key=lambda x: order.get(x.get("tipo", ""), 9))
        return out

    # ---------- Recomendaciones ----------
    def recommendations(self, rid: str) -> Dict[str, Any]:
        r = self.risks.get(rid)
        if not r:
            return {"jerarquia": {}, "normativas": []}

        ctrls = r.get("controles", {}) or {}
        jerarquia: Dict[str, List[str]] = {}
        for title, key in self._ctrl_key_map.items():
            vals = ctrls.get(title) or []
            if vals:
                # garantizar lista de strings
                jerarquia[key] = list(vals)

        normativas = list(r.get("normativa", []) or [])
        return {"jerarquia": jerarquia, "normativas": normativas}
