import os
import sys
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO
import shutil

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5556")
MODEL_NAME = "pcb-defect-model"
DATASET_YAML = "data/yolo/dataset.yaml"
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

    # Download the artifact
    print("Downloading model weights from MLflow Artifact Store...")
    try:
        # In train.py, we used mlflow.log_artifacts to upload the raw YOLO run directory.
        # This means the native YOLO .pt file is located exactly at "weights/best.pt" in the artifact root!
        model_path = client.download_artifacts(mv.run_id, "weights/best.pt")
            
        print(f"Model downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading artifacts: {e}")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset YAML not found at {DATASET_YAML}. Ensure preprocessing has run.")
        sys.exit(1)

    print("Initializing YOLO model...")
    model = YOLO(model_path)
    
    import torch
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Starting evaluation on Golden Dataset (Validation Set) using device: {device}...")
    results = model.val(data=DATASET_YAML, split='val', device=device)

    # Extract metrics
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)

    print("-" * 40)
    print("EVALUATION RESULTS")
    print(f"mAP50: {map50:.4f}")
    print(f"Required Threshold: {MIN_MAP50_THRESHOLD}")
    print("-" * 40)

    if map50 < MIN_MAP50_THRESHOLD:
        print(f"EVALUATION FAILED: Model mAP50 ({map50:.4f}) is below the {MIN_MAP50_THRESHOLD} threshold!")
        sys.exit(1)

    print("EVALUATION PASSED: Model meets quality standards.")
    sys.exit(0)

if __name__ == "__main__":
    main()
