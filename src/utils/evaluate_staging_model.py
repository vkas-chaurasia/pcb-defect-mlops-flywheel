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
        # We download the entire pcb-yolo-model artifact dir which contains the .pt file
        artifact_path = client.download_artifacts(mv.run_id, "pcb-yolo-model")
        # MLflow saves the PyTorch YOLO model in a specific sub-path depending on how it was logged,
        # usually under data/model.pt or we can just pull best.pt from the run itself.
        # Wait, in train.py we used mlflow.register_model(f"runs:/{run_id}/pcb-yolo-model", model_name)
        # Actually, ultralytics saves the model inside the mlflow artifact directory.
        # Let's dynamically find the .pt file
        pt_files = list(Path(artifact_path).rglob("*.pt"))
        if not pt_files:
            # Maybe it's a pyfunc model? If so, we need to extract the underlying weights.
            # Ultralytics mlflow logging stores the weights in 'model.pt' inside 'artifacts' or 'data'.
            model_path = os.path.join(artifact_path, "data", "model.pt")
            if not os.path.exists(model_path):
                print(f"Error: Could not locate .pt file inside downloaded artifact {artifact_path}")
                sys.exit(1)
        else:
            model_path = str(pt_files[0])
            
        print(f"Model downloaded successfully to {model_path}")
    except Exception as e:
        print(f"Error downloading artifacts: {e}")
        sys.exit(1)

    if not os.path.exists(DATASET_YAML):
        print(f"Error: Dataset YAML not found at {DATASET_YAML}. Ensure preprocessing has run.")
        sys.exit(1)

    print("Initializing YOLO model...")
    model = YOLO(model_path)

    print("Starting evaluation on Golden Dataset (Validation Set)...")
    results = model.val(data=DATASET_YAML, split='val', device='cpu')  # Using CPU for reliable CI eval

    # Extract metrics
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)

    print("-" * 40)
    print("EVALUATION RESULTS")
    print(f"mAP50: {map50:.4f}")
    print(f"Required Threshold: {MIN_MAP50_THRESHOLD}")
    print("-" * 40)

    if map50 < MIN_MAP50_THRESHOLD:
        print(f"❌ EVALUATION FAILED: Model mAP50 ({map50:.4f}) is below the {MIN_MAP50_THRESHOLD} threshold!")
        sys.exit(1)

    print("✅ EVALUATION PASSED: Model meets quality standards.")
    sys.exit(0)

if __name__ == "__main__":
    main()
