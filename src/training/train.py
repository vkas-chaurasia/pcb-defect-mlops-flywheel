# CI/CD Trigger: Kickstarting the pipeline
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
import mlflow
import mlflow.pytorch
import mlflow.data
from ultralytics import settings
from tqdm import tqdm

# --- Configuration ---
PROJECT_ROOT  = Path(os.getcwd()).absolute()
CLASS_NAMES   = ["open", "short", "mousebite", "spur", "spurious_copper", "pin_hole"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
YOLO_DIR      = PROJECT_ROOT / "data" / "yolo"
RUNS_DIR      = PROJECT_ROOT / "runs" / "detect"
# Detect environment: Use Prod (5555) in CI/CD, Local (5556) for exploration
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
    mlflow.set_experiment("pcb-defect-detection")
    print("Connected to MLflow Experiment: pcb-defect-detection")
    exp = mlflow.get_experiment_by_name("pcb-defect-detection")

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

        # Save metrics for DVC
        metrics = results.results_dict
        clean_metrics = {k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics)
        
        with open("metrics.json", "w") as f:
            json.dump(clean_metrics, f, indent=4)

        # Log the formal PyTorch model (Master branch pattern)
        yolo_run_dir = RUNS_DIR / "pcb-defect-detection" / run_name
        best_pt = yolo_run_dir / "weights" / "best.pt"
        
        if best_pt.exists():
            print("Logging formal PyTorch model flavor...")
            import torch
            ckpt = torch.load(best_pt, weights_only=False)
            brain = ckpt['model']
            
            
            
            mlflow.pytorch.log_model(
                pytorch_model=brain,
                artifact_path="pcb-yolo-model"
            )

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

        # Ensure DVC sees the history folder exists (to avoid errors)
        os.makedirs(PROJECT_ROOT / "mlflow-history", exist_ok=True)

    print(f"\nTraining and Logging Complete. Run ID: {run_id}")

if __name__ == "__main__":
    main()
