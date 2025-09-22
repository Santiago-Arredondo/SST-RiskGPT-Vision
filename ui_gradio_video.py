# ui_gradio_video.py
# Interfaz Gradio para análisis de VIDEO:
# - Llama a video_analyzer.analyze_video(...)
# - Muestra video anotado, JSON con timeline
# - Resume riesgos con eventos, duración y recomendaciones por jerarquía

import os
import tempfile
from typing import Dict, Any
import gradio as gr

# Tu analizador de video (debe existir en el proyecto)
from video_analyzer import analyze_video


def _format_recs_md(recs: Dict[str, Any]) -> str:
    """Formatea recomendaciones por jerarquía en Markdown compacto."""
    if not recs:
        return "_Sin recomendaciones._"
    lines = []
    for nivel in ("eliminacion", "sustitucion", "ingenieria", "administrativos", "epp"):
        items = recs.get(nivel, [])
        if items:
            lines.append(f"**{nivel.capitalize()}:**")
            lines += [f"- {x}" for x in items]
    normas = recs.get("normas", [])
    if normas:
        lines.append("**Normas:** " + "; ".join(normas))
    return "\n".join(lines) if lines else "_Sin recomendaciones._"


def _result_md(result: Dict[str, Any]) -> str:
    """Construye un informe Markdown con elementos detectados + riesgos y controles."""
    if not result:
        return "_Sin resultados._"
    classes = result.get("classes_present", [])
    risk_stats = result.get("risk_stats", {})
    recs_all = result.get("recommendations", {})
    risk_names = result.get("risk_names", {})

    md = []
    md.append(f"**Elementos detectados:** " + (", ".join(classes) if classes else "—"))

    if not risk_stats:
        md.append("\n_No se identificaron riesgos._")
        return "\n".join(md)

    md.append("\n**Riesgos identificados y controles sugeridos:**")

    # Ordena por duración descendente
    sorted_risks = sorted(
        risk_stats.items(),
        key=lambda kv: kv[1].get("duration_sec", 0.0),
        reverse=True,
    )

    for rid, stat in sorted_risks:
        nombre = risk_names.get(rid, rid)
        events = stat.get("events", 0)
        dur_s = stat.get("duration_sec", 0.0)
        md.append(f"\n### {nombre}")
        md.append(f"- **Eventos:** {events} · **Duración total:** ~{dur_s:.1f}s")
        md.append(_format_recs_md(recs_all.get(rid, {})))

    return "\n".join(md)


def run(
    video_path,
    model_path,
    stride,
    sustain,
    conf,
    iou,
    imgsz,
    use_pose,
    near_thr,
    hand_thr,
):
    if not video_path:
        raise gr.Error("Sube un video (.mp4).")

    out_path = os.path.join(tempfile.gettempdir(), "annotated_sst.mp4")

    # Modelo por defecto si está vacío
    model_path = model_path.strip()
    if not model_path:
        default_best = os.path.join("models", "best.pt")
        model_path = default_best if os.path.exists(default_best) else "yolov8n.pt"

    try:
        result = analyze_video(
            video_path,
            out_path,
            model_path=model_path,
            stride=int(stride),
            risk_sustain=int(sustain),
            conf=float(conf),
            iou=float(iou),
            imgsz=int(imgsz),
            use_pose=bool(use_pose),
            near_thr=float(near_thr),
            hand_thr=float(hand_thr),
        )
    except Exception as e:
        raise gr.Error(f"Error procesando el video: {e}")

    # Construye salidas
    timeline_json = {"timeline": result.get("timeline", [])}
    md = _result_md(result)

    return out_path, timeline_json, md


with gr.Blocks(title="Analizador de Riesgos SST (Video)") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Video)")

    with gr.Row():
        with gr.Column():
            v_in = gr.Video(label="Video (mp4)")
            model = gr.Textbox(
                "",
                label="Ruta a modelo YOLO (.pt) — opcional (si vacío, usa models/best.pt o yolov8n.pt)",
            )
            conf = gr.Slider(0.1, 0.9, value=0.35, step=0.05, label="Confianza detección (conf)")
            iou = gr.Slider(0.3, 0.95, value=0.6, step=0.05, label="IoU NMS (iou)")
            imgsz = gr.Dropdown([640, 960], value=640, label="Tamaño de inferencia (imgsz)")

            stride = gr.Slider(1, 10, value=5, step=1, label="Stride (1 procesa más frames; 10 menos)")
            sustain = gr.Slider(1, 10, value=6, step=1, label="Persistencia (frames) para riesgo activo")

            use_pose = gr.Checkbox(False, label="Usar YOLOv8-Pose (muñecas)")
            near_thr = gr.Slider(0.05, 0.35, value=0.20, step=0.01, label="Umbral proximidad persona–máquina/vehículo")
            hand_thr = gr.Slider(0.03, 0.20, value=0.10, step=0.01, label="Umbral proximidad mano–máquina")

            btn = gr.Button("Analizar video", variant="primary")

        with gr.Column():
            v_out = gr.Video(label="Video anotado")
            j_out = gr.JSON(label="Timeline de riesgos")
            md_out = gr.Markdown(label="Informe")

    btn.click(
        run,
        inputs=[v_in, model, stride, sustain, conf, iou, imgsz, use_pose, near_thr, hand_thr],
        outputs=[v_out, j_out, md_out],
    )

if __name__ == "__main__":
    print(">> UI de video en http://127.0.0.1:7861")
    demo.launch(server_name="127.0.0.1", server_port=7861, show_api=False)
