# ui_gradio_video.py
# UI para analizar VIDEO con YOLO + motor de riesgos.
# - Muestra video anotado
# - Devuelve timeline JSON de riesgos
# - Renderiza recomendaciones por riesgo (jerarquía + normas) de risk_ontology.yaml

from __future__ import annotations
import os
import tempfile
import gradio as gr

from video_analyzer import analyze_video
from rules_engine import RiskEngine

# Cargamos la ontología para poder mapear los IDs de normas a texto legible
ENGINE_UI = RiskEngine("risk_ontology.yaml")


def render_risk_recos(risk_id: str, risk_name: str, recos: dict, stats: dict | None) -> str:
    """
    Construye un bloque Markdown con:
      - Título del riesgo
      - Estadísticos (eventos y duración)
      - Acciones por jerarquía de control
      - Normativas aplicables (texto desde risk_ontology.yaml)
    """
    md = [f"### {risk_name}"]

    if stats:
        ev = stats.get("events", 0)
        dur = stats.get("duration_sec", 0.0)
        md.append(f"- **Eventos**: {ev} · **Duración total**: ~{dur:.1f}s")

    md.append("**Jerarquía de control:**")
    for capa, titulo in [
        ("eliminacion", "Eliminación"),
        ("sustitucion", "Sustitución"),
        ("ingenieria", "Ingeniería"),
        ("administrativos", "Administrativos"),
        ("epp", "EPP"),
    ]:
        items = recos.get(capa, [])
        if items:
            md.append(f"- **{titulo}:**")
            for it in items:
                md.append(f"  - {it}")

    normas_ids = recos.get("normas", [])
    normas_txt = [ENGINE_UI.normas[nid] for nid in normas_ids if nid in ENGINE_UI.normas]
    if normas_txt:
        md.append("**Normativas:**")
        for n in normas_txt:
            md.append(f"- {n}")

    return "\n".join(md)


def run(
    video_path: str | None,
    model_path: str,
    stride: int,
    sustain: int,
    conf: float,
    iou: float,
    imgsz: int,
    use_pose: bool,
    near_thr: float,
    hand_thr: float,
):
    if not video_path:
        raise gr.Error("Sube un video .mp4 para analizar.")

    # Salida temporal del mp4 anotado
    out_path = os.path.join(tempfile.gettempdir(), "annotated.mp4")

    try:
        res = analyze_video(
            video_path,
            out_path,
            model_path=model_path.strip() or None,
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

    # Markdown de salida con clases presentes y recos por riesgo
    clases = ", ".join(res.get("classes_present", [])) or "—"
    md_blocks = [f"**Elementos detectados:** {clases}"]

    recos = res.get("recommendations", {})
    rnames = res.get("risk_names", {})
    rstats = res.get("risk_stats", {})

    if recos:
        for rid in sorted(recos.keys(), key=lambda k: rnames.get(k, k).lower()):
            md_blocks.append(
                render_risk_recos(
                    rid,
                    rnames.get(rid, rid),
                    recos[rid],
                    rstats.get(rid, {}),
                )
            )
    else:
        md_blocks.append("_Sin riesgos inferidos._")

    md_final = "\n\n".join(md_blocks)

    # JSON compacto útil para depurar / auditoría
    json_out = {
        "timeline": res.get("timeline", []),
        "risk_stats": rstats,
    }

    return out_path, json_out, md_final


with gr.Blocks(title="Analizador SST — Video") as demo:
    gr.Markdown("# Analizador de Riesgos SST (Video)")

    with gr.Row():
        with gr.Column():
            v_in = gr.Video(label="Video (mp4)")
            model = gr.Textbox("", label="Ruta a modelo YOLO (.pt) — opcional")
            conf = gr.Slider(0.05, 0.95, value=0.35, step=0.01, label="Confianza detección (conf)")
            iou = gr.Slider(0.3, 0.9, value=0.6, step=0.01, label="IoU NMS (iou)")
            stride = gr.Slider(1, 10, value=5, step=1, label="Stride (1 procesa más frames; 10 menos)")
            sustain = gr.Slider(1, 10, value=6, step=1, label="Persistencia (frames) de riesgo activo")
            imgsz = gr.Dropdown(choices=[640, 960, 1280], value=640, label="Tamaño de inferencia (imgsz)")

            gr.Markdown("### Proximidad avanzada")
            use_pose = gr.Checkbox(value=True, label="Usar YOLOv8-Pose (muñecas)")
            near_thr = gr.Slider(0.05, 0.30, value=0.16, step=0.01, label="Umbral proximidad persona–máquina")
            hand_thr = gr.Slider(0.03, 0.20, value=0.06, step=0.01, label="Umbral proximidad mano–máquina")

            btn = gr.Button("Analizar video", variant="primary")

        with gr.Column():
            v_out = gr.Video(label="Video anotado")
            j_out = gr.JSON(label="Timeline de riesgos")
            md_recos = gr.Markdown(label="Recomendaciones y normativas")

    btn.click(
        run,
        inputs=[v_in, model, stride, sustain, conf, iou, imgsz, use_pose, near_thr, hand_thr],
        outputs=[v_out, j_out, md_recos],
    )

if __name__ == "__main__":
    print(">> UI de video en http://127.0.0.1:7861")
    demo.launch(server_name="127.0.0.1", server_port=7861, show_api=False)
