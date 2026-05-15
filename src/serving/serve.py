"""
PCB Defect Detection – FastAPI Inference Service
=================================================
Project : PCB Defect Detection System (Spring 2026)
Team    : PCB Defect Detection MLOps Team

Endpoints
---------
GET  /health              → service status + model info
POST /predict             → single image inference
POST /predict/batch       → batch inference (up to 16 images)
GET  /classes             → list of defect class names
GET  /docs                → Swagger UI (auto-generated)

Prerequisites
-------------
    pip install fastapi uvicorn ultralytics opencv-python-headless numpy pillow python-multipart

Usage
-----
    # Start the server (PyTorch weights)
    python serve.py --weights runs/train/yolov8n_pcb/weights/best.pt

    # Start with ONNX weights (faster CPU inference)
    python serve.py --weights runs/train/yolov8n_pcb/weights/best.onnx

    # Custom host/port
    python serve.py --weights best.pt --host 0.0.0.0 --port 8080

    # Test with curl
    curl -X POST http://localhost:8000/predict \\
         -F "file=@test_image.jpg" | python -m json.tool

    # In Colab — expose via ngrok:
    # pip install pyngrok
    # from pyngrok import ngrok; ngrok.connect(8000)
"""

import argparse
import io
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# FastAPI imports (fail gracefully with install hint)
try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    raise ImportError(
        "FastAPI stack not installed.\n"
        "Run: pip install fastapi uvicorn python-multipart pillow"
    )

try:
    from PIL import Image as PILImage
except ImportError:
    raise ImportError("Pillow not installed. Run: pip install pillow")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
]

MAX_BATCH_SIZE  = 16
DEFAULT_CONF    = 0.10
DEFAULT_IOU     = 0.45
DEFAULT_IMG_SIZE= 640

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------

class Detection(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  List[float]      # [x1, y1, x2, y2] in pixels
    bbox_xywhn: List[float]      # [cx, cy, w, h] normalised to [0,1]


class PredictionResponse(BaseModel):
    filename:          str
    image_width:       int
    image_height:      int
    num_detections:    int
    detections:        List[Detection]
    inference_time_ms: float
    pass_fail:         str          # "PASS" if no defects above threshold


class BatchPredictionResponse(BaseModel):
    results:           List[PredictionResponse]
    total_images:      int
    total_detections:  int
    batch_time_ms:     float


class HealthResponse(BaseModel):
    status:      str
    model_path:  str
    classes:     List[str]
    device:      str
    img_size:    int


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

class ModelManager:
    """Thread-safe singleton that holds the loaded YOLO model."""
    _model      = None
    _weights    = None
    _img_size   = DEFAULT_IMG_SIZE
    _device     = "cpu"

    @classmethod
    def load(cls, weights: str, img_size: int):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        cls._weights  = weights
        cls._img_size = img_size
        cls._model    = YOLO(weights)

        # detect device (CUDA for NVIDIA, MPS for Mac, CPU as fallback)
        try:
            import torch
            if torch.cuda.is_available():
                cls._device = "cuda"
            elif torch.backends.mps.is_available():
                cls._device = "mps"
            else:
                cls._device = "cpu"
        except (ImportError, AttributeError):
            cls._device = "cpu"

        print(f"Model loaded: {weights}  |  device: {cls._device}  |  img_size: {img_size}")

    @classmethod
    def get(cls):
        if cls._model is None:
            raise RuntimeError("Model not loaded. Call ModelManager.load() first.")
        return cls._model


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes → BGR numpy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Ensure it is a valid JPEG/PNG.")
    return img


def run_inference(
    img_bgr:    np.ndarray,
    filename:   str,
    conf:       float,
    iou:        float,
) -> PredictionResponse:
    model    = ModelManager.get()
    h, w     = img_bgr.shape[:2]

    t0       = time.perf_counter()
    results  = model.predict(
        source  = img_bgr,
        imgsz   = ModelManager._img_size,
        conf    = conf,
        iou     = iou,
        verbose = False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections: List[Detection] = []
    r = results[0]
    if r.boxes is not None and len(r.boxes):
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        scores     = r.boxes.conf.cpu().numpy()
        cls_ids    = r.boxes.cls.cpu().numpy().astype(int)

        for box, score, cls_id in zip(boxes_xyxy, scores, cls_ids):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            detections.append(Detection(
                class_id   = int(cls_id),
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
                confidence = round(float(score), 4),
                bbox_xyxy  = [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                bbox_xywhn = [round(cx, 4), round(cy, 4), round(bw, 4), round(bh, 4)],
            ))

    return PredictionResponse(
        filename         = filename,
        image_width      = w,
        image_height     = h,
        num_detections   = len(detections),
        detections       = detections,
        inference_time_ms= round(elapsed_ms, 2),
        pass_fail        = "FAIL" if detections else "PASS",
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "PCB Defect Detection API",
    description = (
        "YOLOv8-powered PCB defect detection service.\n\n"
        "Detects 6 defect types: open, short, mousebite, spur, "
        "spurious_copper, pin_hole."
    ),
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve raw data for Label Studio (Zero-Config fix)
app.mount("/data", StaticFiles(directory="data"), name="data")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns service status and loaded model information."""
    return HealthResponse(
        status     = "ok",
        model_path = ModelManager._weights or "not loaded",
        classes    = CLASS_NAMES,
        device     = ModelManager._device,
        img_size   = ModelManager._img_size,
    )


@app.get("/classes", tags=["Info"])
def get_classes():
    """Returns the list of detectable defect class names."""
    return {"classes": {i: name for i, name in enumerate(CLASS_NAMES)}}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_single(
    file: UploadFile = File(..., description="PCB image (JPEG or PNG)"),
    conf: float      = Query(DEFAULT_CONF, ge=0.01, le=1.0,
                             description="Confidence threshold"),
    iou:  float      = Query(DEFAULT_IOU,  ge=0.01, le=1.0,
                             description="IoU threshold for NMS"),
):
    """
    Run defect detection on a single PCB image.

    Returns detected defects with class names, confidence scores,
    bounding boxes, and an overall PASS/FAIL decision.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400,
                            detail=f"Expected image file, got: {file.content_type}")

    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        img_bgr = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        response = run_inference(img_bgr, file.filename or "image.jpg", conf, iou)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return response


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Up to 16 PCB images"),
    conf: float             = Query(DEFAULT_CONF, ge=0.01, le=1.0),
    iou:  float             = Query(DEFAULT_IOU,  ge=0.01, le=1.0),
):
    """
    Run defect detection on a batch of PCB images (max 16 per request).
    """
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum batch size is {MAX_BATCH_SIZE}. Got {len(files)}."
        )

    t0      = time.perf_counter()
    results = []
    for f in files:
        raw = await f.read()
        try:
            img_bgr = decode_image(raw)
            result  = run_inference(img_bgr, f.filename or "image.jpg", conf, iou)
        except Exception as e:
            # include a placeholder with error info rather than aborting the batch
            result = PredictionResponse(
                filename          = f.filename or "unknown",
                image_width       = 0,
                image_height      = 0,
                num_detections    = 0,
                detections        = [],
                inference_time_ms = 0.0,
                pass_fail         = f"ERROR: {e}",
            )
        results.append(result)

    batch_ms = (time.perf_counter() - t0) * 1000
    return BatchPredictionResponse(
        results          = results,
        total_images     = len(results),
        total_detections = sum(r.num_detections for r in results),
        batch_time_ms    = round(batch_ms, 2),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(weights: str = None, host: str = "0.0.0.0", port: int = 8000,
         img_size: int = DEFAULT_IMG_SIZE, workers: int = 1):
    """
    Colab usage:
        main(weights="/content/.../best.pt")
        # then in a separate cell: start_server()

    CLI usage:
        python serve.py --weights best.pt
    """
    import sys, types

    # ── Programmatic call (Colab) ─────────────────────────────────────────
    if weights is not None:
        weights_path = Path(weights)
        if not weights_path.exists():
            print(f"[error] Weights not found: {weights_path}")
            return
        ModelManager.load(str(weights_path), img_size)
        print(f"\nStarting PCB Defect Detection API")
        print(f"  Host  : {host}:{port}")
        print(f"  Docs  : http://localhost:{port}/docs\n")
        print("  Tip   : In Colab use a thread so the cell stays interactive:")
        print("          import threading")
        print(f"          t = threading.Thread(target=uvicorn.run, kwargs=dict(")
        print(f"                app=app, host='{host}', port={port}, log_level='info'))")
        print("          t.daemon = True; t.start()")
        return  # caller starts uvicorn themselves (see start_server below)

    # ── CLI path ──────────────────────────────────────────────────────────
    raw_argv = [a for a in sys.argv[1:] if not a.startswith("-f")]
    
    # Smart weights discovery
    # Smart weights discovery
    if "--weights" not in raw_argv:
        # 1. Try MLflow first (The Modern Human-Gatekeeper Way)
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5555")
            print("Checking MLflow Model Registry for '@champion' alias (Waiting for your Approval)...")
            # We look for the model with the '@champion' alias
            weights = mlflow.artifacts.download_artifacts(artifact_uri="models:/pcb-defect-model@champion/weights/best.pt")
            print(f"✅ Approved! Using your @champion model: {weights}")
        except Exception as e:
            print(f"⏳ Waiting for you to add the '@champion' alias in MLflow UI. Falling back to local files...")
            
            # 2. Local Fallbacks
            potential_paths = [
                "models/pcb-defect-detector/best.pt",
                "models/best.pt",
            ]
            # Also check all YOLO runs
            potential_paths.extend([str(p) for p in Path("runs/detect").glob("**/weights/best.pt")])
            
            # Take the first one that exists
            for p in potential_paths:
                if Path(p).exists():
                    weights = p
                    print(f"Auto-discovered local weights: {weights}")
                    break
        
        if not weights:
            print("Error: No weights found in MLflow or locally.")
            return
    else:
        # Get weights from argv if provided
        parser_temp = argparse.ArgumentParser(add_help=False)
        parser_temp.add_argument("--weights")
        args_temp, _ = parser_temp.parse_known_args()
        weights = args_temp.weights

    parser = argparse.ArgumentParser(description="PCB Defect Detection Inference API")
    parser.add_argument("--weights",  default=weights)
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     type=int, default=8000)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--workers",  type=int, default=1)
    args, _ = parser.parse_known_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[error] Weights not found: {weights_path}")
        raise SystemExit(1)

    ModelManager.load(str(weights_path), args.img_size)
    print(f"\nStarting PCB Defect Detection API")
    print(f"  Host     : {args.host}:{args.port}")
    print(f"  Docs     : http://{args.host}:{args.port}/docs\n")
    uvicorn.run(app, host=args.host, port=args.port,
                workers=args.workers, log_level="info")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start uvicorn in a background daemon thread — safe to call from Colab.
    Call main(weights=...) first to load the model.

    Example
    -------
        main(weights="/content/.../best.pt")
        start_server()
        # optionally expose via ngrok:
        # from pyngrok import ngrok; print(ngrok.connect(8000))
    """
    import threading
    if ModelManager._model is None:
        print("[error] Load the model first:  main(weights='path/to/best.pt')")
        return
    t = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=app, host=host, port=port, log_level="info"),
        daemon=True,
    )
    t.start()
    print(f"Server running at http://localhost:{port}")
    print(f"Swagger UI      : http://localhost:{port}/docs")


if __name__ == "__main__":
    main()
