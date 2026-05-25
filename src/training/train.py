# CI/CD Trigger: Kickstarting the pipeline
import argparse
import json
import os
import shutil
import sys
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
import mlflow
import mlflow.pytorch
import mlflow.data
from ultralytics import settings
from tqdm import tqdm
from dotenv import load_dotenv

# Load local .env if present
load_dotenv()

# --- Configuration ---
PROJECT_ROOT  = Path(os.getcwd()).absolute()
CLASS_NAMES   = ["open", "short", "mousebite", "spur", "spurious_copper", "pin_hole"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
YOLO_DIR      = PROJECT_ROOT / "data" / "yolo"
RUNS_DIR      = PROJECT_ROOT / "runs" / "detect"
# Detect environment: Always use port 5555 for the Smart MLflow service
DEFAULT_MLFLOW_URI = "http://localhost:5555"

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)

# S3 Configuration is now handled by the MLflow Artifact Proxy (Server-side)
# Direct client-side S3 access is no longer required.

# ---------------------------------------------------------------------------
# 1. Data Conversion (NPZ -> YOLO)
# ---------------------------------------------------------------------------

def prepare_yolo_data(processed_dir: Path, yolo_dir: Path, img_size: int):
    """Convert .npz files into YOLOv8 format."""
    yaml_path = yolo_dir / "dataset.yaml"
    if yaml_path.exists():
        print("[skip] YOLO dataset already prepared.")
        return yaml_path

    print("Converting processed data to YOLO format...")
    for split in ("train", "val", "test"):
        npz_path = processed_dir / f"{split}.npz"
        if not npz_path.exists(): continue

        img_out = yolo_dir / "images" / split
        lbl_out = yolo_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        data = np.load(npz_path, allow_pickle=True)
        images, boxes, labels = data["images"], data["boxes"], data["labels"]

        # Re-load stats for un-normalisation (to save as readable JPEG)
        stats_path = processed_dir / "dataset_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)
        mean, std = np.array(stats["mean"]), np.array(stats["std"])

        for i in tqdm(range(len(images)), desc=f"  {split}", leave=False):
            # Save Image
            img_uint8 = np.clip((images[i] * (std + 1e-7) + mean) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(img_out / f"{split}_{i:06d}.jpg"), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            # Save Labels (YOLO format: cls cx cy w h)
            lines = []
            for (x1, y1, x2, y2), cls in zip(boxes[i], labels[i]):
                cx, cy = ((x1 + x2) / 2) / img_size, ((y1 + y2) / 2) / img_size
                bw, bh = (x2 - x1) / img_size, (y2 - y1) / img_size
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_out / f"{split}_{i:06d}.txt").write_text("\n".join(lines))

    print("YOLO dataset images and labels prepared.")
    dataset_cfg = {"path": str(yolo_dir), "train": "images/train", "val": "images/val", "test": "images/test", "nc": len(CLASS_NAMES), "names": CLASS_NAMES}
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f)
    return yaml_path

# ---------------------------------------------------------------------------
# 2. Training with MLflow Tracking
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCB Training with MLflow")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    # Prep Data
    yaml_path = prepare_yolo_data(PROCESSED_DIR, YOLO_DIR, args.img_size)

    # MLflow Setup
    print(f"Connecting to MLflow at {MLFLOW_URI}...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    # Unified experiment for both local and production runs
    experiment_name = "pcb-defect-detection"
    mlflow.set_experiment(experiment_name)
    print(f"Connected to MLflow Experiment: {experiment_name}")
    exp = mlflow.get_experiment_by_name(experiment_name)

    # Disable YOLO's internal MLflow callback to prevent duplicate runs
    from ultralytics import YOLO, settings
    settings.update({"mlflow": False})
    
    # Hide tracking URI from YOLO's internal environment
    _uri = os.environ.pop("MLFLOW_TRACKING_URI", None)
    
    model = YOLO(f"{args.model}.pt")

    # Detect device (priority: Nvidia -> Mac -> CPU)
    import torch
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = run.info.run_name
        print(f"Training on device: {device} | MLflow Run: {run_name}")

        # Log hyperparameters
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "img_size": args.img_size,
        })
        
        # Log Git and DVC Metadata
        try:
            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT).decode("utf-8").strip()
            mlflow.set_tag("git_commit", git_commit)
        except Exception:
            pass
            
        try:
            import yaml
            from mlflow.data.dataset import Dataset
            from mlflow.data.http_dataset_source import HTTPDatasetSource
            
            with open("dvc.lock", "r") as f:
                dvc_lock = yaml.safe_load(f)
                for out in dvc_lock.get("stages", {}).get("preprocess", {}).get("outs", []):
                    if out.get("path") == "data/processed":
                        dvc_hash = out.get("md5")
                        
                        # Log formal Dataset to MLflow UI
                        source = HTTPDatasetSource("s3://dvc-storage/data/processed")
                        dataset = Dataset(source=source, name="pcb-defect-images", digest=dvc_hash)
                        mlflow.log_input(dataset, context="training")
                        
                        # Keep the tag for easy filtering in the table view
                        mlflow.set_tag("dataset_hash", dvc_hash)
                        break
        except Exception as e:
            print(f"Failed to log DVC Dataset: {e}")
            
        # Log Environment Artifacts
        if Path("pyproject.toml").exists():
            mlflow.log_artifact("pyproject.toml")
        if Path("uv.lock").exists():
            mlflow.log_artifact("uv.lock")

        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            project=str(RUNS_DIR / "pcb-defect-detection"),
            name=run_name,
            exist_ok=True,
            device=device
        )

        # Restore URI
        if _uri:
            os.environ["MLFLOW_TRACKING_URI"] = _uri

        # Save meaningful metrics for DVC and MLflow
        metrics = results.results_dict
        meaningful_keys = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]
        clean_metrics = {k.replace("(", "_").replace(")", ""): v for k, v in metrics.items() if k in meaningful_keys}
        mlflow.log_metrics(clean_metrics)
        
        with open("metrics.json", "w") as f:
            json.dump(clean_metrics, f, indent=4)

        # Log the formal PyTorch model (Master branch pattern)
        yolo_run_dir = RUNS_DIR / "pcb-defect-detection" / run_name
        
        # --- Time-Series Metrics Logging ---
        results_csv = yolo_run_dir / "results.csv"
        if results_csv.exists():
            print("Logging time-series metrics from results.csv...")
            with open(results_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned_row = {k.strip().replace("(", "_").replace(")", ""): float(v.strip()) for k, v in row.items() if v.strip() and k.strip() != "epoch"}
                    step = int(float(row.get("epoch", row.get("                  epoch", 0))))
                    if step > 0 or "epoch" in "".join(row.keys()): # Ensure it's valid
                        mlflow.log_metrics(cleaned_row, step=step)
        
        best_pt = yolo_run_dir / "weights" / "best.pt"
        
        if best_pt.exists():
            print("Exporting model to ONNX format for universal compatibility...")
            from ultralytics import YOLO
            
            # Load the PyTorch model
            model = YOLO(best_pt)
            
            # Export to ONNX
            # dynamic=False is often safer for YOLO inference unless specifically needed
            onnx_path = model.export(format="onnx", device="cpu")
            
            print(f"Logging ONNX model to MLflow from {onnx_path}...")
            # We log the raw ONNX file as an artifact. 
            # Ultralytics natively loads .onnx files via YOLO('model.onnx')
            mlflow.log_artifact(onnx_path, artifact_path="pcb-yolo-model")

        # Log all YOLO artifacts (Unspoiled)
        if yolo_run_dir.exists():
            mlflow.log_artifacts(str(yolo_run_dir))
            
            # Export metadata for CI/CD
            with open("last_run_path.txt", "w") as f:
                f.write(str(yolo_run_dir))
            
            # THE KEY: Save the Run ID for the CI/CD to promote
            with open("mlflow_run.txt", "w") as f:
                f.write(f"RUN_ID={run_id}\n")
                f.write(f"EXP_ID={exp.experiment_id}\n")
                f.write(f"RUN_URL={MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run_id}\n")
                f.write(f"EXP_URL={MLFLOW_URI}/#/experiments/{exp.experiment_id}\n")
                
            # Save persistent metadata to be committed to Git
            os.makedirs("models", exist_ok=True)
            with open("models/model_meta.json", "w") as f:
                json.dump({"RUN_ID": run_id, "EXP_ID": exp.experiment_id}, f, indent=4)

        # Ensure DVC sees the history folder exists (to avoid errors)
        os.makedirs(PROJECT_ROOT / "mlflow-history", exist_ok=True)

    print(f"\nTraining and Logging Complete. Run ID: {run_id}")

if __name__ == "__main__":
    main()
