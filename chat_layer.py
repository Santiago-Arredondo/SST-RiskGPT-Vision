from typing import Dict, List, Any

PROMPT_TEMPLATE_ES = """Eres un asesor técnico en SST.
Elementos detectados: {clases}.
Riesgos inferidos: {riesgos}.
Redacta recomendaciones por Jerarquía de Control (Eliminación, Sustitución, Ingeniería, Administrativos, EPP)
y cita de forma breve las normas aplicables. Sé claro y directo.
"""

def _render_from_template(present: List[str], risks: List[Dict[str, Any]], recs: Dict[str, Any]) -> str:
    if not risks:
        return ("No se inferieron riesgos con el modelo actual. "
                "Sugerencia: entrenar un modelo específico del dominio y ajustar las reglas.")
    lines = []
    lines.append(f"**Elementos detectados:** {', '.join(present) if present else 'N/D'}")
    lines.append("**Riesgos identificados y controles:**")
    for r in risks:
        rid = r["id"]
        nombre = r.get("nombre", rid)
        payload = recs.get(rid, {})
        jer = payload.get("jerarquia", {})
        normas = payload.get("normativas", [])
        lines.append(f"- **{nombre}** ({r.get('tipo')})")
        for nivel in ["eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"]:
            if jer.get(nivel):
                etiqueta = nivel.capitalize()
                lines.append(f"  - {etiqueta}: " + "; ".join(jer[nivel]))
        if normas:
            lines.append(f"  - Normas: {', '.join(normas)}")
    return "\n".join(lines)

def build_chat_response(present: List[str], risks: List[Dict[str, Any]], recs: Dict[str, Any]) -> str:
    
    _ = PROMPT_TEMPLATE_ES.format(
        clases=", ".join(present) if present else "N/D",
        riesgos=", ".join([r.get("nombre", r["id"]) for r in risks]) if risks else "N/D"
    )
    return _render_from_template(present, risks, recs)
