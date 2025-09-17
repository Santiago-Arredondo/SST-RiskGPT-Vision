# ui_gradio_video.py
# UI Gradio para análisis de VIDEO con recomendaciones por riesgo (jerarquía + normas)

from __future__ import annotations
import os
import tempfile
import gradio as gr

from video_analyzer import analyze_video


# ------------------------- helpers de presentación ------------------------- #

def _fmt_seconds(sec: float) -> str:
    try:
        return f"{sec:.1f}s"
    except Exception:
        return f"{sec}"

def build_summary(result: dict) -> str:
    """Texto corto tipo 'Riesgos detectados' con eventos y duración."""
    stats = result.get("risk_stats", {}) or {}
    names = result.get("risk_names", {}) or {}
    if not stats:
        return "Sin inferencias de riesgo."
    lines = ["Riesgos detectados:"]
    for rid, data in stats.items():
        ev = int(data.get("events", 0))
        dur = _fmt_seconds(float(data.get("duration_sec", 0.0)))
        lines.append(f"- {names.get(rid, rid)} — Eventos: {ev}, Duración total: ~{dur}")
    return "\n".join(lines)

def recs_section_html(risk_title: str, stats: dict, recs: dict) -> str:
    """Bloque HTML por riesgo con Jerarquía y Normas."""
    ev = int(stats.get("events", 0))
    dur = _fmt_seconds(float(stats.get("duration_sec", 0.0)))
    html = [f"### {risk_title}", f"*Eventos: {ev} · Duración total: ~{dur}*"]

    # Jerarquía de control
    html.append("**Jerarquía de control:**")
    order = [
        ("eliminacion", "eliminación"),
        ("sustitucion", "sustitución"),
        ("ingenieria", "ingeniería"),
        ("administrativos", "administrativos"),
        ("epp", "epp"),
    ]
    any_item = False
    for k, label in order:
        items = recs.get(k) or []
        if items:
            any_item = True
            html.append(f"- **{label.capitalize()}:**")
            for it in items:
                html.append(f"  - {it}")
    if not any_item:
        html.append("- (sin recomendaciones registradas)")

    # Normas
    normas = recs.get("normas") or recs.get("normativas") or []
    if normas:
        html.append("**Normativas:**")
        for n in normas:
            html.append(f"- {n}")

    return "\n".join(html)

def build_recs_panel(result: dict) -> str:
    """Panel Markdown con recomendaciones por cada riesgo activo en el video."""
    stats = result.get("risk_stats", {}) or {}
    names = result.get("risk_names", {}) or {}
    recs_all = result.get("recommendations", {}) or {}

    if not stats:
        return "_No se detectaron riesgos en el período analizado._"

    # Ordenar por duración descendente para priorizar lo más importante
    ordered = sorted(stats.items(), key=lambda kv: float(kv[1].get("duration_sec", 0.0)), reverse=True)

    sections = []
    sections.append(f"**Elementos detectados:** {', '.join(result.get('classes_present', [])) or '—'}")
    sections.append("")
    for rid, st in ordered:
        title = names.get(rid, rid)
        recs = recs_all.get(rid, {}) or {}
        sections.append(recs_section_html(title, st, recs))
        sections.append("")  # espacio entre riesgos

    return "\n".join(sections).strip()


# ------------------------------- lógica UI -------------------------------- #

def run(
    video,
    model_path,
    conf,
    iou,
    stride,
    sustain,
    imgsz,
    use_pose,
    near_thr,
    hand_thr,
):
    if video is None:
        raise gr.Error("Sube un video (.mp4) para analizar.")

    # salida del video anotado en temp
    out_path = os.path.join(tempfile.gettempdir(), "annotated.mp4")

    # llamar al analizador
    result = analyze_video(
        video_path=video,
        out_path=out_path,
        model_path=(model_path or None),
        stride=int(stride),
        risk_sustain=int(sustain),
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        use_pose=bool(use_pose),
        near_thr=float(near_thr),
        hand_thr=float(hand_thr),
    )

    # salidas a UI
    resumen = build_summary(result)
    recs_md = build_recs_panel(result)

    # Nota: Gradio Video acepta la ruta al mp4 generado
    return out_path, result, resumen, recs_md


with gr.Blocks(title="Analizador SST — Video") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Video)")

    with gr.Row():
        # --------------------- Entrada y parámetros --------------------- #
        with gr.Column(scale=1):
            v_in = gr.Video(label="Video (mp4)")
            model = gr.Textbox("", label="Ruta a modelo YOLO (.pt) — opcional")
            conf = gr.Slider(0.05, 0.95, value=0.35, step=0.01, label="Confianza detección (conf)")
            iou = gr.Slider(0.3, 0.9, value=0.6, step=0.01, label="IoU NMS (iou)")
            stride = gr.Slider(1, 10, value=5, step=1, label="Stride (1 procesa más frames; 10 menos)")
            sustain = gr.Slider(1, 10, value=6, step=1, label="Persistencia (frames) de riesgo activo")
            imgsz = gr.Dropdown(choices=["640", "960", "1280"], value="640", label="Tamaño de inferencia (imgsz)")
            use_pose = gr.Checkbox(value=True, label="Usar YOLOv8-Pose (muñecas)")
            near_thr = gr.Slider(0.05, 0.5, value=0.16, step=0.01, label="Umbral proximidad persona–máquina (dist. normalizada)")
            hand_thr = gr.Slider(0.03, 0.2, value=0.06, step=0.01, label="Umbral proximidad mano–máquina (dist. normalizada)")

            btn = gr.Button("Analizar video", variant="primary")

        # --------------------- Salidas --------------------- #
        with gr.Column(scale=1):
            v_out = gr.Video(label="Video anotado")
            j_out = gr.JSON(label="Resultados (JSON)")  # timeline, stats, recomendaciones…
            resumen = gr.Textbox(label="Resumen", lines=6)
            recs_panel = gr.Markdown()  # recomendaciones por riesgo

    btn.click(
        run,
        inputs=[v_in, model, conf, iou, stride, sustain, imgsz, use_pose, near_thr, hand_thr],
        outputs=[v_out, j_out, resumen, recs_panel],
    )

if __name__ == "__main__":
    print(">> UI de video (puerto automático)")
    # Puerto automático para evitar el error "Cannot find empty port"
    demo.launch(server_name="127.0.0.1", server_port=None, show_api=False)
