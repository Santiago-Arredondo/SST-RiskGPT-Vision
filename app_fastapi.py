from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from detector import YoloDetector
from rules_engine import RiskEngine
from image_tokens import compute_tokens
from chat_layer import build_enhanced_chat_response

engine = RiskEngine(["risk_ontology.yaml", "risk_ontology_ext.yaml"])
detector = YoloDetector(ontology_paths=["risk_ontology.yaml", "risk_ontology_ext.yaml"])

app = FastAPI(title="API Analizador SST")

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")

    dets = detector.to_dicts(detector.predict(pil))
    classes_present = detector.classes_from_dicts(dets)
    active_ctx = engine.active_contexts(set(classes_present))
    tokens, trace = compute_tokens(dets, engine.meta, active_ctx)

    risks = engine.infer(tokens)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}

    return JSONResponse({
        "classes_present": classes_present,
        "active_contexts": sorted(list(active_ctx)),
        "tokens": tokens,
        "token_trace": trace,
        "detections": dets,
        "risks": risks,
        "recommendations": recs
    })

@app.post("/analyze-chat")
async def analyze_chat(image: UploadFile = File(...), output_format: str = "all"):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")

    dets = detector.to_dicts(detector.predict(pil))
    classes_present = detector.classes_from_dicts(dets)
    active_ctx = engine.active_contexts(set(classes_present))
    tokens, trace = compute_tokens(dets, engine.meta, active_ctx)

    risks = engine.infer(tokens)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}
    chat = build_enhanced_chat_response(tokens, risks, recs, output_format=output_format, language="es")

    return JSONResponse({
        "classes_present": classes_present,
        "active_contexts": sorted(list(active_ctx)),
        "tokens": tokens,
        "token_trace": trace,
        "detections": dets,
        "risks": risks,
        "recommendations": recs,
        "chat": chat
    })
