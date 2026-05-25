import os
import sys
import yaml
import torch
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
import shutil

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5556")
MODEL_NAME = "pcb-defect-model"
DATASET_YAML = "data/yolo/dataset.yaml"
# Strict threshold for governance gate
MIN_MAP50_THRESHOLD = 0.70

def main():
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI}...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print(f"Looking for @staging version of {MODEL_NAME}...")
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, "staging")
    except Exception as e:
        print(f"Error: Could not find @staging alias for {MODEL_NAME}.")
        sys.exit(1)

    print(f"Found @staging model: Version {mv.version} (Run ID: {mv.run_id})")

    # Download best.onnx — exported on CPU so it is hardware-agnostic and
    # works identically on Mac, Linux, and inside Docker/Airflow.
    print("Downloading model weights (best.onnx) from MLflow Artifact Store...")
    try:
        model_path = client.download_artifacts(mv.run_id, "pcb-yolo-model/best.onnx")
        print(f"Model downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading artifacts: {e}")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset YAML not found at {DATASET_YAML}. Ensure preprocessing has run.")
        sys.exit(1)

    # Read img_size from params.yaml to match training configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    img_size = params["train"]["img_size"]

    # Use CPU for evaluation so results are deterministic across Mac/Linux/Docker
    device = 'cpu'

    print(f"Initializing YOLO model from {model_path}...")
    model = YOLO(model_path)

    print(f"Starting evaluation on Golden Dataset using device={device}, imgsz={img_size}...")
    results = model.val(data=DATASET_YAML, split='val', device=device, imgsz=img_size)

    # Extract metrics
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)

    print("-" * 40)
    print("EVALUATION RESULTS")
    print(f"mAP50:     {map50:.4f}")
    print(f"Precision: {metrics.get('metrics/precision(B)', 0.0):.4f}")
    print(f"Recall:    {metrics.get('metrics/recall(B)', 0.0):.4f}")
    print(f"Required Threshold: {MIN_MAP50_THRESHOLD}")
    print("-" * 40)

    if map50 < MIN_MAP50_THRESHOLD:
        print(f"EVALUATION FAILED: Model mAP50 ({map50:.4f}) is below the {MIN_MAP50_THRESHOLD} threshold!")
        sys.exit(1)

    print("EVALUATION PASSED: Model meets quality standards.")
    sys.exit(0)

if __name__ == "__main__":
    main()
