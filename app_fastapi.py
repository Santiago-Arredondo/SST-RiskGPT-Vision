from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from detector import YoloDetector
from rules_engine import RiskEngine
from chat_layer import build_enhanced_chat_response  # usar la versión completa

detector = YoloDetector()
engine = RiskEngine("risk_ontology.yaml")

app = FastAPI(title="API Analizador SST")

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")

    dets = detector.to_dicts(detector.predict(pil))
    present = detector.classes_from_dicts(dets)
    risks = engine.infer(present)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}

    return JSONResponse({
        "classes_present": present,
        "detections": dets,
        "risks": risks,
        "recommendations": recs
    })

@app.post("/analyze-chat")
async def analyze_chat(image: UploadFile = File(...), output_format: str = "all"):
    content = await image.read()
    pil = Image.open(io.BytesIO(content)).convert("RGB")

    dets = detector.to_dicts(detector.predict(pil))
    present = detector.classes_from_dicts(dets)
    risks = engine.infer(present)
    recs = {r["id"]: engine.recommendations(r["id"]) for r in risks}
    chat = build_enhanced_chat_response(present, risks, recs, output_format=output_format, language="es")

    return JSONResponse({
        "classes_present": present,
        "detections": dets,
        "risks": risks,
        "recommendations": recs,
        "chat": chat
    })
    