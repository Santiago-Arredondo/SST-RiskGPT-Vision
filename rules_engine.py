from typing import List, Dict, Any
import yaml

class RiskEngine:
    """
    Motor declarativo de riesgos. A partir de clases presentes (detecciones),
    aplica reglas definidas en 'risk_ontology.yaml' y devuelve recomendaciones.
    """
    def __init__(self, ontology_path: str = "risk_ontology.yaml"):
        with open(ontology_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.normas: Dict[str, str] = data.get("normas", {})
        self.riesgos: Dict[str, Any] = data.get("riesgos", {})

    @staticmethod
    def _match_trigger(trigger: Dict[str, List[str]], present: List[str]) -> bool:
        all_of = set(map(str.lower, trigger.get("all_of", [])))
        any_of = set(map(str.lower, trigger.get("any_of", [])))
        present_set = set(map(str.lower, present))
        return (not all_of or all_of.issubset(present_set)) and (not any_of or (present_set & any_of))

    def infer(self, present_classes: List[str]) -> List[Dict[str, Any]]:
        out = []
        for rid, spec in self.riesgos.items():
            if any(self._match_trigger(t, present_classes) for t in spec.get("disparadores", [])):
                out.append({
                    "id": rid,
                    "nombre": spec.get("nombre", rid),
                    "tipo": spec.get("tipo", "na"),
                    "evidencia": {"clases_detectadas": present_classes}
                })
        return out

    def recommendations(self, rid: str) -> Dict[str, Any]:
        spec = self.riesgos[rid]
        normas_ext = [self.normas[n] for n in spec.get("normas", []) if n in self.normas]
        return {"jerarquia": spec.get("recomendaciones", {}), "normativas": normas_ext}
