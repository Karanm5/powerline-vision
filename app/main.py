"""
main.py
-------
PowerLine Vision API - Aerial cable detection from uploaded imagery.
"""

import io
import os
import time
import base64
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
from ultralytics import YOLO

from app.schemas import DetectionResponse, Detection, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("powerline-vision")

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
MAX_IMAGE_SIZE_MB = 10

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model from: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        logger.error(f"Model weights not found at {MODEL_PATH}")
    else:
        model = YOLO(MODEL_PATH)
        logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="PowerLine Vision API",
    description="Automated detection of overhead power line cables from aerial imagery using YOLOv8.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

detections_total = Counter("powerline_detections_total", "Total cables detected")
images_processed = Counter("powerline_images_processed_total", "Total images processed")
inference_latency = Histogram(
    "powerline_inference_latency_seconds",
    "Inference time in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)
confidence_scores = Histogram(
    "powerline_detection_confidence",
    "Detection confidence distribution",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def draw_detections(image, detections):
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = int(det.bbox_x1), int(det.bbox_y1), int(det.bbox_x2), int(det.bbox_y2)
        colour = (0, 200, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(y1 - 4, th + 4)
        cv2.rectangle(annotated, (x1, label_y - th - 4), (x1 + tw, label_y), colour, -1)
        cv2.putText(annotated, label, (x1, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated

def image_to_base64(image):
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")

def parse_detections(results, model):
    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for xyxy, conf, cls_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        ):
            det = Detection(
                class_name=model.names.get(cls_id, str(cls_id)),
                confidence=round(float(conf), 4),
                bbox_x1=round(float(xyxy[0]), 1),
                bbox_y1=round(float(xyxy[1]), 1),
                bbox_x2=round(float(xyxy[2]), 1),
                bbox_y2=round(float(xyxy[3]), 1),
            )
            detections.append(det)
            confidence_scores.observe(det.confidence)
    detections_total.inc(len(detections))
    return detections

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    return HTMLResponse(content=BROWSER_UI, status_code=200)

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    import torch
    return HealthResponse(
        status="ok" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG and PNG supported.")
    content = await file.read()
    if len(content) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_SIZE_MB}MB.")
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    h, w = image.shape[:2]
    image_processed = apply_clahe(image)
    t0 = time.perf_counter()
    results = model.predict(source=image_processed, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    elapsed_s = time.perf_counter() - t0
    inference_latency.observe(elapsed_s)
    images_processed.inc()
    detections = parse_detections(results, model)
    logger.info(f"Detected {len(detections)} objects in {elapsed_s*1000:.1f}ms")
    return DetectionResponse(
        filename=file.filename or "upload.jpg",
        image_width=w, image_height=h,
        detections=detections,
        total_detections=len(detections),
        inference_time_ms=round(elapsed_s * 1000, 2),
    )

@app.post("/detect/visualise", tags=["Detection"])
async def detect_visualise(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    content = await file.read()
    arr = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")
    image_processed = apply_clahe(image)
    t0 = time.perf_counter()
    results = model.predict(source=image_processed, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
    elapsed_s = time.perf_counter() - t0
    inference_latency.observe(elapsed_s)
    images_processed.inc()
    detections = parse_detections(results, model)
    annotated = draw_detections(image, detections)
    encoded = image_to_base64(annotated)
    logger.info(f"Visualise: {len(detections)} detections in {elapsed_s*1000:.1f}ms")
    return JSONResponse({
        "filename": file.filename,
        "total_detections": len(detections),
        "inference_time_ms": round(elapsed_s * 1000, 2),
        "detections": [d.model_dump() for d in detections],
        "annotated_image_base64": encoded,
    })

BROWSER_UI = open("app/main.py").read() if False else """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PowerLine Vision</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
    .header { background: #1a1d27; border-bottom: 1px solid #2d3748; padding: 20px 32px; display: flex; align-items: center; }
    .header h1 { font-size: 1.4rem; font-weight: 600; color: #00d4ff; }
    .header p { font-size: 0.85rem; color: #718096; margin-top: 2px; }
    .badge { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff44; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; margin-left: auto; }
    .container { max-width: 1000px; margin: 40px auto; padding: 0 24px; }
    .card { background: #1a1d27; border: 1px solid #2d3748; border-radius: 12px; padding: 28px; margin-bottom: 24px; }
    .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; color: #a0aec0; }
    .drop-zone { border: 2px dashed #2d3748; border-radius: 8px; padding: 40px 20px; text-align: center; cursor: pointer; transition: all 0.2s; background: #0f1117; }
    .drop-zone:hover { border-color: #00d4ff; background: #00d4ff08; }
    .drop-zone p { color: #718096; font-size: 0.9rem; margin-top: 8px; }
    .drop-icon { font-size: 2.5rem; }
    input[type=file] { display: none; }
    .btn { background: #00d4ff; color: #0f1117; border: none; padding: 10px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 0.9rem; display: inline-flex; align-items: center; gap: 8px; }
    .btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-secondary { background: #2d3748; color: #e2e8f0; }
    .btn-row { margin-top: 16px; display: flex; gap: 12px; }
    .slider-row { display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap; }
    .slider-group label { font-size: 0.8rem; color: #a0aec0; display: block; margin-bottom: 6px; }
    .slider-group input { width: 180px; accent-color: #00d4ff; }
    .slider-group span { font-size: 0.8rem; color: #00d4ff; margin-left: 8px; }
    .results { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    @media (max-width: 640px) { .results { grid-template-columns: 1fr; } }
    .result-img { width: 100%; border-radius: 8px; border: 1px solid #2d3748; }
    .detections-list { max-height: 340px; overflow-y: auto; }
    .det-item { background: #0f1117; border: 1px solid #2d3748; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
    .det-item .cls { font-weight: 600; color: #00d4ff; font-size: 0.85rem; }
    .det-item .conf { font-size: 0.8rem; color: #68d391; margin-left: 8px; }
    .det-item .bbox { font-size: 0.75rem; color: #718096; margin-top: 3px; }
    .stat { display: inline-flex; flex-direction: column; background: #0f1117; border: 1px solid #2d3748; border-radius: 8px; padding: 12px 18px; margin-right: 12px; margin-bottom: 12px; }
    .stat .val { font-size: 1.4rem; font-weight: 700; color: #00d4ff; }
    .stat .lbl { font-size: 0.75rem; color: #718096; margin-top: 2px; }
    .spinner { display: none; width: 20px; height: 20px; border: 2px solid #2d3748; border-top-color: #00d4ff; border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .empty { color: #718096; font-size: 0.9rem; text-align: center; padding: 24px; }
    .error { background: #ff4d4d18; border: 1px solid #ff4d4d44; border-radius: 8px; padding: 12px 16px; color: #fc8181; font-size: 0.9rem; margin-top: 12px; display: none; }
    .model-info { font-size: 0.78rem; color: #718096; line-height: 1.6; }
    .model-info strong { color: #a0aec0; }
    .model-info a { color: #00d4ff; }
  </style>
</head>
<body>
<div class="header">
  <div><h1>PowerLine Vision</h1><p>Automated overhead cable detection from aerial imagery</p></div>
  <span class="badge">YOLOv8s</span>
</div>
<div class="container">
  <div class="card">
    <h2>Upload Image</h2>
    <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
      <div class="drop-icon">&#128247;</div>
      <p id="dropLabel">Drop an aerial image here or click to browse</p>
      <p style="font-size:0.78rem;margin-top:6px;">JPEG or PNG, max 10MB</p>
    </div>
    <input type="file" id="fileInput" accept="image/jpeg,image/png">
    <div class="slider-row">
      <div class="slider-group">
        <label>Confidence threshold <span id="confVal">0.25</span></label>
        <input type="range" id="confSlider" min="0.05" max="0.95" step="0.05" value="0.25"
               oninput="document.getElementById('confVal').textContent=parseFloat(this.value).toFixed(2)">
      </div>
      <div class="slider-group">
        <label>IoU threshold <span id="iouVal">0.45</span></label>
        <input type="range" id="iouSlider" min="0.1" max="0.9" step="0.05" value="0.45"
               oninput="document.getElementById('iouVal').textContent=parseFloat(this.value).toFixed(2)">
      </div>
    </div>
    <div class="btn-row">
      <button class="btn" id="detectBtn" disabled onclick="runDetection()">
        <span id="btnText">Detect Cables</span>
        <div class="spinner" id="spinner"></div>
      </button>
      <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
    </div>
    <div class="error" id="errorBox"></div>
  </div>
  <div id="statsRow" style="display:none;margin-bottom:24px;">
    <span class="stat"><span class="val" id="statCount">0</span><span class="lbl">Detections</span></span>
    <span class="stat"><span class="val" id="statTime">0ms</span><span class="lbl">Inference time</span></span>
    <span class="stat"><span class="val" id="statFile">--</span><span class="lbl">File</span></span>
  </div>
  <div id="resultsSection" style="display:none;">
    <div class="results">
      <div class="card" style="padding:16px;">
        <h2>Annotated Output</h2>
        <img id="resultImg" class="result-img" src="" alt="Detection result">
      </div>
      <div class="card">
        <h2>Detections (<span id="detCount">0</span>)</h2>
        <div class="detections-list" id="detectionsList"></div>
      </div>
    </div>
  </div>
  <div class="card">
    <div class="model-info">
      <strong>Model:</strong> YOLOv8s &nbsp;|&nbsp;
      <strong>mAP@0.5:</strong> 0.688 &nbsp;|&nbsp;
      <strong>Precision:</strong> 0.758 &nbsp;|&nbsp;
      <strong>Recall:</strong> 0.647 &nbsp;|&nbsp;
      <a href="/docs">API Docs</a> &nbsp;|&nbsp;
      <a href="/metrics">Metrics</a> &nbsp;|&nbsp;
      <a href="https://github.com/Karanm5/powerline-vision" target="_blank">GitHub</a>
    </div>
  </div>
</div>
<script>
  let selectedFile = null;
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.style.borderColor='#00d4ff'; });
  dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor='#2d3748'; });
  dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.style.borderColor='#2d3748'; if(e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
  fileInput.addEventListener('change', () => { if(fileInput.files[0]) setFile(fileInput.files[0]); });
  function setFile(f) {
    selectedFile = f;
    document.getElementById('dropLabel').textContent = f.name;
    document.getElementById('detectBtn').disabled = false;
    document.getElementById('errorBox').style.display = 'none';
  }
  async function runDetection() {
    if (!selectedFile) return;
    const btn = document.getElementById('detectBtn');
    const spinner = document.getElementById('spinner');
    const btnText = document.getElementById('btnText');
    btn.disabled = true; spinner.style.display = 'block'; btnText.textContent = 'Detecting...';
    const form = new FormData();
    form.append('file', selectedFile);
    try {
      const res = await fetch('/detect/visualise', { method: 'POST', body: form });
      if (!res.ok) {
        let msg = 'Detection failed';
        try { const err = await res.json(); msg = err.detail || msg; } catch { msg = await res.text(); }
        throw new Error(msg);
      }
      const data = await res.json();
      document.getElementById('statCount').textContent = data.total_detections;
      document.getElementById('statTime').textContent = data.inference_time_ms.toFixed(1) + 'ms';
      document.getElementById('statFile').textContent = selectedFile.name;
      document.getElementById('statsRow').style.display = 'block';
      document.getElementById('resultImg').src = 'data:image/jpeg;base64,' + data.annotated_image_base64;
      document.getElementById('detCount').textContent = data.total_detections;
      const list = document.getElementById('detectionsList');
      if (data.detections.length === 0) {
        list.innerHTML = '<div class="empty">No cables detected. Try lowering the confidence slider.</div>';
      } else {
        list.innerHTML = data.detections.map(d => `
          <div class="det-item">
            <span class="cls">${d.class_name}</span>
            <span class="conf">${(d.confidence*100).toFixed(1)}% confidence</span>
            <div class="bbox">Box: (${d.bbox_x1.toFixed(0)}, ${d.bbox_y1.toFixed(0)}) to (${d.bbox_x2.toFixed(0)}, ${d.bbox_y2.toFixed(0)})</div>
          </div>`).join('');
      }
      document.getElementById('resultsSection').style.display = 'block';
    } catch(err) {
      const box = document.getElementById('errorBox');
      box.textContent = err.message; box.style.display = 'block';
    } finally {
      btn.disabled = false; spinner.style.display = 'none'; btnText.textContent = 'Detect Cables';
    }
  }
  function clearAll() {
    selectedFile = null; fileInput.value = '';
    document.getElementById('dropLabel').textContent = 'Drop an aerial image here or click to browse';
    document.getElementById('detectBtn').disabled = true;
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('statsRow').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
  }
</script>
</body>
</html>"""
