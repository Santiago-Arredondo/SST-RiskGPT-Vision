# app_fastapi.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from unified_detector import UnifiedDetector
from rules_engine import RiskEngine
from image_tokens import compute_tokens
# si integras pose:
# from pose_tokens import compute_pose_tokens
from chat_layer import build_enhanced_chat_response

engine = RiskEngine(["risk_ontology.yaml", "risk_ontology_ext.yaml"])
detector = UnifiedDetector(
    detect_model=None,                 # autodetecta en models/
    pose_model="models/pose.pt",       # o None si no tienes YOLO pose
    ontology_paths=["risk_ontology.yaml", "risk_ontology_ext.yaml"]
)

app = FastAPI(title="API Analizador SST (Unificado)")

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["GET /docs", "POST /analyze", "POST /analyze-chat"]}

@app.get("/healthz")
def healthz():
    return {"ok": True}

def _pipeline(pil, output_format="all"):
    out = detector.infer(pil, conf=0.25, iou=0.60, imgsz=640, device="cpu", classes_prompt=None)
    dets = out["detections"]
    classes_present = out["classes_present"]
    active_ctx = engine.active_contexts(set(classes_present))
    tokens, trace = compute_tokens(dets, engine.meta, active_ctx)

    # pose opcional:
    # if out.get("poses"):
    #     pose_toks, pose_trace = compute_pose_tokens(out["poses"], screen_boxes=[d["box"] for d in dets if d["cls"] in ("screen","laptop")])
    #     tokens = sorted(set(tokens) | set(pose_toks))
    #     for k,v in pose_trace.items(): trace.setdefault(k,[]).extend(v)

    risks = engine.infer(tokens)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}
    chat = build_enhanced_chat_response(tokens, risks, recs, output_format=output_format, language="es")
    return out, active_ctx, tokens, trace, risks, recs, chat

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    out, active_ctx, tokens, trace, risks, recs, _ = _pipeline(pil, output_format="all")
    return JSONResponse({
        "classes_present": out["classes_present"],
        "active_contexts": sorted(list(active_ctx)),
        "tokens": tokens,
        "token_trace": trace,
        "detections": out["detections"],
        "poses": out.get("poses", []),
        "risks": risks,
        "recommendations": recs
    })

@app.post("/analyze-chat")
async def analyze_chat(image: UploadFile = File(...), output_format: str = "all"):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")
    out, active_ctx, tokens, trace, risks, recs, chat = _pipeline(pil, output_format=output_format)
    return JSONResponse({
        "classes_present": out["classes_present"],
        "active_contexts": sorted(list(active_ctx)),
        "tokens": tokens,
        "token_trace": trace,
        "detections": out["detections"],
        "poses": out.get("poses", []),
        "risks": risks,
        "recommendations": recs,
        "chat": chat
    })
