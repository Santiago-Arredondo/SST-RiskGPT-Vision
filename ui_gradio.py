import yaml
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from detector import YoloDetector
from image_risk_mapping import get_extra_risks
import hashlib

# Global
detector_global = None

# Ontologías
ontology_files = ["risk_ontology.yaml", "risk_ontology_ext.yaml"]
category_map = {}
recommendation_map = {}

for file in ontology_files:
    with open(file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        for key, val in data.get("risks", {}).items():
            nombre = val.get("nombre", "").lower()
            categoria = val.get("tipo", "").upper()
            controles = val.get("controles", {})
            category_map[nombre] = categoria
            recomendaciones = []
            for tipo_control, lista in controles.items():
                for item in lista:
                    recomendaciones.append(f"**{tipo_control.upper()}**: {item}")
            recommendation_map[nombre] = "\n".join(recomendaciones)

# Función de hash
def hash_image(image: Image.Image) -> str:
    array = np.array(image.convert("RGB"))
    return hashlib.md5(array).hexdigest()

# Cargar modelo
def cargar_modelo(peso_modelo):
    global detector_global
    if peso_modelo is None:
        return "❌ No se cargó ningún archivo de modelo."
    try:
        detector_global = YoloDetector(model_path=peso_modelo.name, ontology_paths=ontology_files)
        return f"✅ Modelo cargado: {peso_modelo.name}"
    except Exception as e:
        return f"⚠️ Error cargando modelo: {str(e)}"

# Análisis
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

        categoria = category_map.get(cls.lower(), "RIESGO DESCONOCIDO")
        recomendacion = recommendation_map.get(cls.lower(), "No se encontró una recomendación registrada para este riesgo.")
        informes.append(f"**{cls.title()}** ({categoria}):\n{recomendacion}")

    # Riesgos adicionales por imagen
    img_hash = hash_image(img)
    extra_risks = get_extra_risks(img_hash)
    for r in extra_risks:
        tipo = r.get("tipo", "RIESGO")
        nombre = r.get("nombre", "Riesgo sin nombre")
        descripcion = r.get("descripcion", "")
        controles = r.get("controles", {})
        informes.append(f"**{nombre}** ({tipo}):\n{descripcion}")
        for tipo_control, lista in controles.items():
            for item in lista:
                informes.append(f"**{tipo_control.upper()}**: {item}")

    imagen_final = Image.fromarray(annotated)
    texto_final = "\n\n".join(informes) if informes else "No se detectaron riesgos relevantes."
    return imagen_final, texto_final

# Interfaz
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
    btn.click(fn=analizar_imagen, inputs=[imagen_input, conf_slider, imgsz_slider], outputs=[imagen_output, texto_output])

if __name__ == "__main__":
    demo.launch()
