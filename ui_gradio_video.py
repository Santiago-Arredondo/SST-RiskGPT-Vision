import gradio as gr, tempfile, os
from video_analyzer import analyze_video

def run(video, model_path, stride, sustain):
    out_path = os.path.join(tempfile.gettempdir(), "annotated.mp4")
    timeline = analyze_video(video, out_path, model_path=model_path or "yolov8n.pt",
                             stride=stride, risk_sustain=sustain)
    return out_path, {"timeline": timeline}

with gr.Blocks(title="Analizador SST — Video") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Video)")
    with gr.Row():
        with gr.Column():
            v_in = gr.Video(label="Video (mp4)")
            model = gr.Textbox("", label="Ruta a modelo YOLO (.pt) — opcional")
            stride = gr.Slider(1, 10, value=5, step=1, label="Stride (1 procesa más, 10 menos)")
            sustain = gr.Slider(1, 10, value=3, step=1, label="Persistencia (frames) para riesgo activo")
            btn = gr.Button("Analizar video")
        with gr.Column():
            v_out = gr.Video(label="Video anotado")
            j_out = gr.JSON(label="Timeline de riesgos")
    btn.click(run, inputs=[v_in, model, stride, sustain], outputs=[v_out, j_out])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, show_api=False)

