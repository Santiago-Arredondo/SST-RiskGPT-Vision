# rules_engine.py
from __future__ import annotations
from typing import Any, Dict, List, Set, Optional
import os, yaml

class RiskEngine:
    def __init__(self, ontology_path: str, default_context: str = "industrial"):
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"No existe ontología: {ontology_path}")
        with open(ontology_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.normas: Dict[str, str] = data.get("normas", {})
        self.risks: Dict[str, Any] = data.get("riesgos", {})
        self.default_context = default_context

    @staticmethod
    def _as_set(x) -> Set[str]:
        if x is None: return set()
        if isinstance(x, str): return {x}
        return set(x)

    def _trigger_matches(
        self,
        trigger: Dict[str, Any],
        present: Set[str],
        tokens: Set[str],
    ) -> bool:
        all_of = self._as_set(trigger.get("all_of"))
        any_of = self._as_set(trigger.get("any_of"))
        none_of = self._as_set(trigger.get("none_of"))
        tokens_any = self._as_set(trigger.get("tokens_any"))

        if all_of and not all_of.issubset(present):
            return False
        if any_of and present.isdisjoint(any_of):
            return False
        if none_of and not none_of.isdisjoint(present):
            return False
        if tokens_any and tokens.isdisjoint(tokens_any):
            return False
        return True

    def infer(
        self,
        present_with_ctx: List[str] | Set[str],
        context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        present_with_ctx = conjunto de clases detectadas + tokens (p.ej. near_person_vehicle).
        context = industrial|bodega|obra|via_publica|oficina  (default: self.default_context)
        """
        context = (context or self.default_context).strip().lower()
        present_set = set(present_with_ctx)
        # separa tokens (por convención: tienen guiones o son "near_*" / "poor_*" / "foot_*" etc.)
        tokens = {t for t in present_set if any(t.startswith(p) for p in ("near_", "foot_", "poor_", "overhead_"))}
        # clases "puras" (sin tokens)
        classes = present_set - tokens

        matches: List[Dict[str, Any]] = []
        for rid, rdef in self.risks.items():
            allowed = set(map(str.lower, rdef.get("contextos_permitidos", [])))
            if allowed and context not in allowed:
                continue

            triggers = rdef.get("disparadores", [])
            if not triggers:
                continue
            # Un riesgo se activa si CUALQUIER trigger OR se cumple
            if any(self._trigger_matches(tr, classes, tokens) for tr in triggers):
                matches.append({
                    "id": rid,
                    "nombre": rdef.get("nombre", rid),
                    "tipo": rdef.get("tipo", "general"),
                })
        return matches

    def recommendations(self, risk_id: str, context: Optional[str] = None) -> Dict[str, List[str]]:
        context = (context or self.default_context).strip().lower()
        rdef = self.risks.get(risk_id, {})
        rec = rdef.get("recomendaciones", {}) or {}
        # Si en el futuro quieres overrides por contexto, aquí puedes fusionar.
        # Por ahora devolvemos tal cual, ordenando niveles.
        ordered = {}
        for k in ["eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"]:
            vals = rec.get(k, []) or []
            ordered[k] = list(dict.fromkeys(vals))  # sin duplicados, conserva orden
        return ordered

    def normas_resueltas(self, risk_id: str) -> List[str]:
        rdef = self.risks.get(risk_id, {})
        keys = rdef.get("normas", []) or []
        out = []
        for k in keys:
            label = self.normas.get(k, k)
            out.append(label)
        return out

    @staticmethod
    def to_markdown_recommendations(
        risk_name: str,
        recs: Dict[str, List[str]],
        normas: List[str] | None = None
    ) -> str:
        lines = [f"### {risk_name} — Recomendaciones"]
        mapping = {
            "eliminacion": "Eliminación",
            "sustitucion": "Sustitución",
            "ingenieria": "Controles de Ingeniería",
            "administrativos": "Controles Administrativos",
            "epp": "EPP"
        }
        for key in ["eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"]:
            items = recs.get(key, [])
            if items:
                lines.append(f"- **{mapping[key]}**: " + "; ".join(items))
        if normas:
            lines.append("- **Normas base**: " + "; ".join(normas))
        return "\n".join(lines)
