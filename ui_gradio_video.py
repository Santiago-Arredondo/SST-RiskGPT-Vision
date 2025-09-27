# ui_gradio_video.py — UI para video (corrigido)
import gradio as gr, tempfile, os
from video_analyzer import analyze_video

def run(video, model_path, stride, sustain, conf, iou, imgsz, use_pose, near_thr, hand_thr):
    if video is None:
        raise gr.Error("Sube un video .mp4 para analizar.")
    out_path = os.path.join(tempfile.gettempdir(), "annotated.mp4")
    # Llamada con firma compatible (sin conf_thres)
    result = analyze_video(
        video_path=video,
        out_path=out_path,
        model_path=model_path or None,
        stride=int(stride),
        risk_sustain=int(sustain),
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        use_pose=bool(use_pose),
        near_thr=float(near_thr),
        hand_thr=float(hand_thr),
    )

    # Resumen + detalle
    risk_stats = result.get("risk_stats", {})
    risk_names = result.get("risk_names", {})
    recs       = result.get("recommendations", {})

    if not risk_stats:
        md = "_No se identificaron riesgos._"
    else:
        lines = []
        lines.append("**Riesgos identificados y controles sugeridos:**")
        for rid, info in risk_stats.items():
            nm = risk_names.get(rid, rid)
            ev = info.get("events", 0)
            dur = info.get("duration_sec", 0.0)
            lines.append(f"\n### {nm}")
            lines.append(f"- **Eventos:** {ev} · **Duración total:** ~{dur:.1f}s")
            for r in recs.get(rid, []):
                lines.append(f"- {r}")
        md = "\n".join(lines)

    # Para tener algo estructurado al lado
    j = {
        "timeline": result.get("timeline", []),
        "classes_present": result.get("classes_present", []),
        "risk_stats": risk_stats,
        "risk_names": risk_names,
    }
    return out_path, j, md

with gr.Blocks(title="Analizador SST — Video") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Video)")
    with gr.Row():
        with gr.Column():
            v_in = gr.Video(label="Video (mp4)")
            model = gr.Textbox("", label="Ruta a modelo YOLO (.pt) — opcional (si vacío, usa models/best.pt o yolov8n.pt)")
            conf = gr.Slider(0.1, 0.9, value=0.35, step=0.01, label="Confianza detección (conf)")
            iou  = gr.Slider(0.3, 0.9, value=0.6, step=0.01, label="IoU NMS (iou)")
            imgsz = gr.Dropdown(choices=[640, 800, 960], value=640, label="Tamaño de inferencia (imgsz)")
            stride = gr.Slider(1, 10, value=5, step=1, label="Stride (1 procesa más frames; 10 menos)")
            sustain = gr.Slider(1, 12, value=6, step=1, label="Persistencia (frames) para riesgo activo")
            use_pose = gr.Checkbox(value=False, label="Usar YOLOv8-Pose (muñecas)")
            near_thr = gr.Slider(0.08, 0.35, value=0.20, step=0.01, label="Umbral proximidad persona–máquina/vehículo")
            hand_thr = gr.Slider(0.04, 0.20, value=0.10, step=0.01, label="Umbral proximidad mano–máquina")
            btn = gr.Button("Analizar video")
        with gr.Column():
            v_out = gr.Video(label="Video anotado")
            j_out = gr.JSON(label="Timeline / Stats")
            md_out = gr.Markdown()
    btn.click(run, inputs=[v_in, model, stride, sustain, conf, iou, imgsz, use_pose, near_thr, hand_thr],
              outputs=[v_out, j_out, md_out])

if __name__ == "__main__":
    print(">> UI de video (puerto automático)")
    demo.launch(server_name="127.0.0.1", server_port=None, show_api=False)
