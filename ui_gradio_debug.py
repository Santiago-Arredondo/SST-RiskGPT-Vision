import yaml
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from detector import YoloDetector

# Cargar ontología y mapa de categorías/recomendaciones
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
                    recomendaciones.append(f"{tipo_control.upper()}: {item}")
            recommendation_map[nombre] = "\n".join(recomendaciones)

# Instanciar detector
model = YoloDetector(model_path="runs/detect/train4/weights/best.pt")

def analizar_imagen_debug(img: Image.Image):
    np_img = np.array(img.convert("RGB"))
    detections = model.predict(np_img, conf=0.2)

    annotated = np_img.copy()
    informes = []
    debug_log = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cls = det["cls"]
        conf = det["conf"]
        raw_label = det.get("raw", cls)

        label = f"{raw_label} ({conf:.2f})"
        debug_log.append(f"- {raw_label}: {conf:.2f}")

        # Dibujar caja y etiqueta
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Buscar coincidencia en mapas
        categoria = category_map.get(cls.lower(), "RIESGO DESCONOCIDO")
        recomendacion = recommendation_map.get(cls.lower(), "No se encontró una recomendación para este riesgo.")

        informes.append(f"**{cls.title()}** ({categoria}):\n{recomendacion}")

    imagen_final = Image.fromarray(annotated)

    info_final = "\n\n".join(informes) if informes else "**Sin riesgos inferidos con ontología.**"
    debug_output = "\n".join(debug_log) if debug_log else "❌ No se detectaron objetos con confianza suficiente."

    return imagen_final, info_final, debug_output

# Interfaz Gradio con salida extra
demo = gr.Interface(
    fn=analizar_imagen_debug,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Image(type="pil", label="Imagen Anotada"),
        gr.Markdown(label="Informe de Riesgos"),
        gr.Textbox(label="DEBUG: Clases detectadas (crudas)")
    ],
    title="SST-RiskGPT Visión (Modo Debug)",
    description="Modo DEBUG: Verás los objetos detectados, confianza, y riesgos inferidos según ontología."
)

if __name__ == "__main__":
    demo.launch()
