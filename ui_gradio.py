# -*- coding: utf-8 -*-
import yaml
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import hashlib

from detector import YoloDetector
from image_risk_mapping import get_extra_risks


# =============================
# 1) Ontologías
# =============================
ontology_files = ["risk_ontology.yaml", "risk_ontology_ext.yaml"]
category_map = {}
recommendation_map = {}

for file in ontology_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            for _, val in (data.get("risks", {}) or {}).items():
                nombre = (val.get("nombre", "") or "").lower()
                if not nombre:
                    continue
                categoria = (val.get("tipo", "") or "").upper() or "SIN TIPO"
                controles = val.get("controles", {}) or {}

                category_map[nombre] = categoria

                orden_bloques = [
                    "Eliminación",
                    "Ingeniería",
                    "Administrativos",
                    "EPP",
                    "Normas",
                    "Señalización",
                    "Conclusión",
                ]
                recomendaciones = []
                ya = set()
                for bloque in orden_bloques:
                    for item in (controles.get(bloque, []) or []):
                        recomendaciones.append(f"**{bloque.upper()}**: {item}")
                    ya.add(bloque)
                for bloque, lista in controles.items():
                    if bloque not in ya:
                        for item in (lista or []):
                            recomendaciones.append(f"**{bloque.upper()}**: {item}")

                recommendation_map[nombre] = (
                    "\n\n".join(recomendaciones) if recomendaciones else "Sin controles registrados."
                )
    except FileNotFoundError:
        pass


# =============================
# 2) Utilidades
# =============================
def hash_image(image: Image.Image) -> str:
    array = np.array(image.convert("RGB"))
    return hashlib.md5(array).hexdigest()


def _format_extra_risk(r: dict) -> str:
    """
    Formatea riesgos adicionales con títulos separados y saltos de línea claros.
    """
    tipo = r.get("tipo", "RIESGO")
    nombre = r.get("nombre", "Riesgo sin nombre")
    descripcion = r.get("descripcion", "")

    lines = [f"**{nombre}** ({tipo}): {descripcion}".strip()]

    # Consecuencias
    consecuencias = r.get("consecuencias", []) or []
    if consecuencias:
        lines.append("\n**Consecuencias potenciales:**")
        for c in consecuencias:
            lines.append(f"- {c}")

    # Controles con títulos separados
    orden = [
        "Eliminación",
        "Ingeniería",
        "Administrativos",
        "EPP",
        "Normas",
        "Señalización",
        "Conclusión",
    ]
    controles = r.get("controles", {}) or {}

    ya = set()
    for bloque in orden:
        items = controles.get(bloque, [])
        if items:
            lines.append(f"\n**{bloque.upper()}**:")
            for item in items:
                lines.append(f"- {item}")
        ya.add(bloque)

    for bloque, items in controles.items():
        if bloque not in ya and items:
            lines.append(f"\n**{bloque.upper()}**:")
            for item in items:
                lines.append(f"- {item}")

    # Separar bien los bloques
    return "\n".join(lines)


# =============================
# 3) Modelo YOLO y análisis
# =============================
detector_global = None

def cargar_modelo(peso_modelo):
    global detector_global
    if peso_modelo is None:
        return "❌ No se cargó ningún archivo de modelo."
    try:
        detector_global = YoloDetector(model_path=peso_modelo.name, ontology_paths=ontology_files)
        return f"✅ Modelo cargado: {peso_modelo.name}"
    except Exception as e:
        return f"⚠️ Error cargando modelo: {str(e)}"


def analizar_imagen(img: Image.Image, confianza: float, tamano: int):
    if detector_global is None:
        return None, "⚠️ Primero debes cargar un modelo (.pt)."

    np_img = np.array(img.convert("RGB"))
    detections = detector_global.predict(np_img, conf=confianza, imgsz=tamano)
    annotated = np_img.copy()
    informes = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cls = det["cls"]
        conf = det["conf"]
        label = f"{cls} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        clase_key = (cls or "").lower()
        categoria = category_map.get(clase_key, "RIESGO DESCONOCIDO")
        recomendacion_md = recommendation_map.get(
            clase_key, "No se encontró una recomendación registrada para este riesgo."
        )

        informes.append(f"**{cls.title()}** ({categoria}):\n\n{recomendacion_md}")

    # Riesgos adicionales
    img_hash = hash_image(img)
    try:
        extra_risks = get_extra_risks(img_hash) or []
    except Exception as e:
        extra_risks = []
        informes.append(f"⚠️ Error obteniendo riesgos extra: {e}")

    for r in extra_risks:
        informes.append(_format_extra_risk(r))

    imagen_final = Image.fromarray(annotated)
    texto_final = "\n\n---\n\n".join(informes) if informes else "No se detectaron riesgos relevantes."
    return imagen_final, texto_final


# =============================
# 4) Interfaz Gradio
# =============================
modelo_input = gr.File(label="Carga tu modelo YOLOv8 (.pt)", file_types=[".pt"])
imagen_input = gr.Image(type="pil", label="Sube una imagen para análisis")
imagen_output = gr.Image(type="pil", label="Imagen anotada")
texto_output = gr.Markdown()
modelo_status = gr.Textbox(label="Estado del modelo", interactive=False)
conf_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.01, label="Confianza")
imgsz_slider = gr.Slider(minimum=320, maximum=1280, value=640, step=32, label="Tamaño de imagen")

with gr.Blocks(title="SST-RiskGPT Visión") as demo:
    gr.Markdown("### 🔍 Análisis de Riesgos Locativos, Mecánicos y Ergonómicos con Visión por Computador")
    with gr.Row():
        modelo_input.render()
        modelo_status.render()
    modelo_input.change(fn=cargar_modelo, inputs=[modelo_input], outputs=[modelo_status])

    with gr.Row():
        with gr.Column():
            imagen_input.render()
            conf_slider.render()
            imgsz_slider.render()
        with gr.Column():
            imagen_output.render()
            texto_output.render()

    btn = gr.Button("Analizar Imagen")
    btn.click(
        fn=analizar_imagen,
        inputs=[imagen_input, conf_slider, imgsz_slider],
        outputs=[imagen_output, texto_output],
    )

if __name__ == "__main__":
    demo.launch()
