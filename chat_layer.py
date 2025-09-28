"""
Capa de chat mejorada para SST-RiskGPT Vision
Genera respuestas conversacionales y educativas sobre riesgos detectados
"""

from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class SST_ChatAssistant:
    """Asistente conversacional especializado en SST"""
    
    def __init__(self, language: str = "es"):
        self.language = language
        self.severity_levels = {
            "CRÍTICO": {"color": "🔴", "priority": 1},
            "ALTO": {"color": "🟠", "priority": 2},
            "MEDIO": {"color": "🟡", "priority": 3},
            "BAJO": {"color": "🟢", "priority": 4}
        }
        
    def _assess_severity(self, risk: Dict[str, Any]) -> str:
        """Evalúa severidad basada en el tipo de riesgo"""
        risk_id = risk.get("id", "")
        
        # Riesgos críticos
        critical = ["caida_distinto_nivel", "atrapamiento_partes_moviles", 
                   "aplastamiento_cargas", "contacto_electrico"]
        high = ["proyeccion_particulas", "cortes_herramientas", 
               "atrapamiento_vehicular"]
        medium = ["caidas_mismo_nivel", "golpes_objetos", "iluminacion_deficiente"]
        
        if any(c in risk_id for c in critical):
            return "CRÍTICO"
        elif any(h in risk_id for h in high):
            return "ALTO"
        elif any(m in risk_id for m in medium):
            return "MEDIO"
        return "BAJO"
    
    def _format_controls(self, jerarquia: Dict[str, List[str]]) -> str:
        """Formatea controles por jerarquía de manera conversacional"""
        if not jerarquia:
            return ""
            
        lines = []
        icons = {
            "eliminacion": "🚫",
            "sustitucion": "🔄", 
            "ingenieria": "⚙️",
            "administrativos": "📋",
            "epp": "🦺"
        }
        
        priority_order = ["eliminacion", "sustitucion", "ingenieria", 
                         "administrativos", "epp"]
        
        for control_type in priority_order:
            if control_type in jerarquia and jerarquia[control_type]:
                icon = icons.get(control_type, "•")
                label = control_type.capitalize()
                controls = jerarquia[control_type]
                
                if isinstance(controls, list) and controls:
                    lines.append(f"\n{icon} **{label}:**")
                    for control in controls[:2]:  # Limitar a 2 más importantes
                        lines.append(f"   • {control}")
                        
        return "\n".join(lines)
    
    def _format_regulations(self, normas: List[str]) -> str:
        """Formatea normativas de manera concisa"""
        if not normas:
            return ""
            
        key_norms = normas[:3]  # Top 3 normas más relevantes
        return "📜 **Normativa aplicable:** " + ", ".join(key_norms)
    
    def build_response(
        self, 
        present: List[str], 
        risks: List[Dict[str, Any]], 
        recommendations: Dict[str, Any],
        image_metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Construye respuesta estructurada con múltiples formatos
        
        Returns:
            Dict con 'summary', 'detailed', 'actionable', 'report'
        """
        
        if not risks:
            return {
                "summary": "✅ No se identificaron riesgos significativos en esta imagen. El área parece segura.",
                "detailed": "No se detectaron condiciones o comportamientos de riesgo que requieran intervención inmediata.",
                "actionable": "Continúe con las buenas prácticas de seguridad.",
                "report": self._generate_safety_report([], present)
            }
        
        # Ordenar riesgos por severidad
        risks_with_severity = [
            {**r, "severity": self._assess_severity(r)} 
            for r in risks
        ]
        risks_sorted = sorted(
            risks_with_severity, 
            key=lambda x: self.severity_levels[x["severity"]]["priority"]
        )
        
        # Generar diferentes formatos
        return {
            "summary": self._build_summary(risks_sorted, present),
            "detailed": self._build_detailed(risks_sorted, recommendations),
            "actionable": self._build_actionable(risks_sorted, recommendations),
            "report": self._generate_safety_report(risks_sorted, present, image_metadata)
        }
    
    def _build_summary(self, risks: List[Dict], present: List[str]) -> str:
        """Resumen ejecutivo de riesgos"""
        if not risks:
            return "✅ Área segura - Sin riesgos detectados"
        
        severities = [r["severity"] for r in risks]
        most_severe = min(severities, key=lambda x: self.severity_levels[x]["priority"])
        
        lines = [f"⚠️ **ALERTA DE SEGURIDAD - Nivel {most_severe}**\n"]
        
        # Elementos detectados relevantes
        relevant_items = [p for p in present if not p.startswith("context_") 
                         and not p.startswith("near_")]
        if relevant_items:
            lines.append(f"**Elementos identificados:** {', '.join(relevant_items[:5])}")
        
        # Resumen de riesgos
        lines.append(f"\n**Se detectaron {len(risks)} riesgo(s):**")
        
        for risk in risks[:3]:  # Top 3 riesgos
            severity_icon = self.severity_levels[risk["severity"]]["color"]
            lines.append(f"{severity_icon} {risk.get('nombre', risk['id'])}")
        
        if len(risks) > 3:
            lines.append(f"   ...y {len(risks)-3} riesgo(s) adicional(es)")
            
        return "\n".join(lines)
    
    def _build_detailed(self, risks: List[Dict], recommendations: Dict) -> str:
        """Análisis detallado con controles"""
        lines = ["## 📊 Análisis Detallado de Riesgos\n"]
        
        for i, risk in enumerate(risks, 1):
            severity_icon = self.severity_levels[risk["severity"]]["color"]
            risk_id = risk.get("id", "")
            
            lines.append(f"### {i}. {severity_icon} {risk.get('nombre', risk_id)}")
            lines.append(f"**Tipo:** {risk.get('tipo', 'GENERAL')}")
            lines.append(f"**Severidad:** {risk['severity']}")
            
            # Controles recomendados
            recs = recommendations.get(risk_id, {})
            if "jerarquia" in recs:
                controls = self._format_controls(recs["jerarquia"])
                if controls:
                    lines.append(f"\n**Controles recomendados:**{controls}")
            
            # Normativa
            if "normativas" in recs:
                norms = self._format_regulations(recs["normativas"])
                if norms:
                    lines.append(f"\n{norms}")
            
            lines.append("\n---")
        
        return "\n".join(lines)
    
    def _build_actionable(self, risks: List[Dict], recommendations: Dict) -> str:
        """Acciones inmediatas requeridas"""
        lines = ["## 🚨 Acciones Inmediatas Requeridas\n"]
        
        immediate_actions = []
        short_term_actions = []
        long_term_actions = []
        
        for risk in risks:
            risk_id = risk.get("id", "")
            severity = risk["severity"]
            recs = recommendations.get(risk_id, {})
            jerarquia = recs.get("jerarquia", {})
            
            # Clasificar acciones por urgencia
            if severity == "CRÍTICO":
                # Acciones inmediatas (< 1 hora)
                if "eliminacion" in jerarquia:
                    for action in jerarquia["eliminacion"][:1]:
                        immediate_actions.append(f"🔴 {action}")
                if "epp" in jerarquia:
                    for action in jerarquia["epp"][:1]:
                        immediate_actions.append(f"🔴 {action}")
                        
            elif severity == "ALTO":
                # Acciones corto plazo (< 24 horas)
                if "ingenieria" in jerarquia:
                    for action in jerarquia["ingenieria"][:1]:
                        short_term_actions.append(f"🟠 {action}")
                        
            else:
                # Acciones largo plazo (< 1 semana)
                if "administrativos" in jerarquia:
                    for action in jerarquia["administrativos"][:1]:
                        long_term_actions.append(f"🟡 {action}")
        
        if immediate_actions:
            lines.append("### ⚡ INMEDIATO (Detener trabajo hasta completar):")
            lines.extend(immediate_actions)
            
        if short_term_actions:
            lines.append("\n### ⏱️ HOY (Completar en las próximas horas):")
            lines.extend(short_term_actions)
            
        if long_term_actions:
            lines.append("\n### 📅 ESTA SEMANA (Programar implementación):")
            lines.extend(long_term_actions)
            
        # Responsables sugeridos
        lines.append("\n### 👥 Responsables sugeridos:")
        lines.append("• **Supervisor SST:** Verificación inmediata")
        lines.append("• **Jefe de área:** Autorizar controles")
        lines.append("• **Mantenimiento:** Implementar barreras físicas")
        lines.append("• **RRHH:** Programar capacitaciones")
        
        return "\n".join(lines)
    
    def _generate_safety_report(
        self, 
        risks: List[Dict], 
        present: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """Genera reporte formal en formato markdown"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# REPORTE DE ANÁLISIS DE SEGURIDAD
**Sistema:** SST-RiskGPT Vision v2.0
**Fecha/Hora:** {timestamp}
**Normatividad:** Decreto 1072/2015, Res. 2400/1979, Res. 0312/2019

## 1. RESUMEN EJECUTIVO
- **Total de riesgos identificados:** {len(risks)}
- **Nivel de riesgo general:** {self._calculate_overall_risk(risks)}
- **Elementos detectados:** {len(present)}

## 2. CLASIFICACIÓN DE RIESGOS

| Severidad | Cantidad | Porcentaje |
|-----------|----------|------------|
"""
        
        severity_count = {"CRÍTICO": 0, "ALTO": 0, "MEDIO": 0, "BAJO": 0}
        for risk in risks:
            severity_count[risk.get("severity", "BAJO")] += 1
            
        total = max(len(risks), 1)
        for sev, count in severity_count.items():
            icon = self.severity_levels[sev]["color"]
            percentage = (count / total) * 100
            report += f"| {icon} {sev} | {count} | {percentage:.1f}% |\n"
        
        report += """
## 3. DETALLE DE RIESGOS IDENTIFICADOS
"""
        
        for i, risk in enumerate(risks, 1):
            report += f"""
### {i}. {risk.get('nombre', risk.get('id', 'Desconocido'))}
- **Tipo:** {risk.get('tipo', 'N/A')}
- **Severidad:** {risk.get('severity', 'N/A')}
- **ID Sistema:** {risk.get('id', 'N/A')}
"""
        
        report += """
## 4. RECOMENDACIONES GENERALES
1. Implementar controles según jerarquía (Eliminación → EPP)
2. Documentar acciones tomadas en el SG-SST
3. Realizar seguimiento en próxima inspección
4. Capacitar al personal sobre riesgos identificados

## 5. FIRMA DIGITAL
*Este reporte fue generado automáticamente por el sistema de visión computacional SST-RiskGPT*
*Para validación manual, contacte al departamento de SST*

---
**Nota:** Este análisis es una herramienta de apoyo. La evaluación final debe ser realizada por personal calificado en SST.
"""
        
        return report
    
    def _calculate_overall_risk(self, risks: List[Dict]) -> str:
        """Calcula el nivel de riesgo general del área"""
        if not risks:
            return "BAJO"
            
        severities = [r.get("severity", "BAJO") for r in risks]
        
        if "CRÍTICO" in severities:
            return "CRÍTICO"
        elif "ALTO" in severities:
            return "ALTO"
        elif "MEDIO" in severities:
            return "MEDIO"
        return "BAJO"


# Función de integración con el sistema existente
def build_enhanced_chat_response(
    present: List[str],
    risks: List[Dict[str, Any]],
    recommendations: Dict[str, Any],
    output_format: str = "all",
    language: str = "es"
) -> Dict[str, str]:
    """
    Wrapper para mantener compatibilidad con sistema existente
    
    Args:
        present: Lista de clases/tokens detectados
        risks: Lista de riesgos inferidos
        recommendations: Dict con recomendaciones por riesgo
        output_format: 'summary'|'detailed'|'actionable'|'report'|'all'
        language: Idioma de respuesta ('es', 'en')
    
    Returns:
        Dict con las respuestas formateadas
    """
    assistant = SST_ChatAssistant(language=language)
    
    # Convertir formato de recommendations si necesario
    formatted_recs = {}
    for risk in risks:
        risk_id = risk.get("id", "")
        if risk_id in recommendations:
            rec = recommendations[risk_id]
            if isinstance(rec, dict):
                formatted_recs[risk_id] = rec
            else:
                # Asumiendo formato legacy
                formatted_recs[risk_id] = {
                    "jerarquia": rec if isinstance(rec, dict) else {},
                    "normativas": []
                }
    
    response = assistant.build_response(present, risks, formatted_recs)
    
    if output_format == "all":
        return response
    elif output_format in response:
        return {output_format: response[output_format]}
    else:
        return {"summary": response.get("summary", "Error en formato de salida")}


# Para mantener retrocompatibilidad
def build_chat_response(present, risks, recommendations):
    """Función legacy para compatibilidad"""
    result = build_enhanced_chat_response(present, risks, recommendations, "summary")
    return result.get("summary", "")