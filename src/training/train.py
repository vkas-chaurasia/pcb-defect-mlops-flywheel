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
MLFLOW_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")

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
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.set_experiment("pcb-defect-detection")

    # Disable YOLO's internal MLflow callback to prevent duplicate runs
    settings.update({"mlflow": False})

    from ultralytics import YOLO
    model = YOLO(f"{args.model}.pt")

    with mlflow.start_run() as run:
        run_name = run.info.run_name
        print(f"MLflow Run Name: {run_name}")

        # Log hyperparameters
        mlflow.log_params({
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "img_size": args.img_size,
        })

        # Train — temporarily hide the tracking URI so YOLO's internal
        # MLflow callback doesn't create a second run
        _uri = os.environ.pop("MLFLOW_TRACKING_URI", None)
        # Detect device (priority: Nvidia -> Mac -> CPU)
        import torch
        if torch.cuda.is_available():
            device = 0 # Use first NVIDIA GPU
        elif torch.backends.mps.is_available():
            device = 'mps' # Use Apple Silicon GPU
        else:
            device = 'cpu' # Fallback to CPU
            
        # Smart Cache: Check if training results already exist
        yolo_run_dir = RUNS_DIR / "pcb-defect-detection" / run_name
        results_csv = yolo_run_dir / "results.csv"
        
        if results_csv.exists():
            print(f"✨ Smart Cache Hit: Found existing results in {yolo_run_dir}. Skipping training and proceeding to logging.")
            # Create a mock results object to satisfy the logging logic
            from types import SimpleNamespace
            results = SimpleNamespace(results_dict={})
        else:
            print(f"🚀 Training on device: {device}")
            # Train
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
        
        if _uri:
            os.environ["MLFLOW_TRACKING_URI"] = _uri

        # Log training metrics
        metrics = results.results_dict
        clean_metrics = {k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics)
        
        # Save metrics for DVC CI/CD
        with open("metrics.json", "w") as f:
            json.dump(clean_metrics, f, indent=4)
        
        print("Final training metrics logged to MLflow and metrics.json.")

        # Log the formal PyTorch model (enables "Register Model" button)
        best_pt = yolo_run_dir / "weights" / "best.pt"

        if best_pt.exists():
            import torch
            ckpt = torch.load(best_pt, weights_only=False)
            brain = ckpt['model']

            print("Logging formal PyTorch model flavor...")
            mlflow.pytorch.log_model(
                pytorch_model=brain,
                artifact_path="pcb-yolo-model"
            )

            print("Formal PyTorch model flavour logged.")

        # Log dataset profile (mean/std)
        stats_path = PROCESSED_DIR / "dataset_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)

        mlflow.log_params({
            "mean_rgb": stats["mean"],
            "std_rgb": stats["std"],
        })

        # Log all YOLO artifacts (plots, charts, labels, etc.)
        if yolo_run_dir.exists():
            print(f"Uploading visual plots and charts from {yolo_run_dir} to MLflow...")
            mlflow.log_artifacts(str(yolo_run_dir))
            
            # Export the path for CI/CD reporting
            with open("last_run_path.txt", "w") as f:
                f.write(str(yolo_run_dir))

        # Export URLs for CI/CD reporting
        with open("mlflow_urls.txt", "w") as f:
            f.write(f"RUN_URL={MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}\n")
            f.write(f"EXP_URL={MLFLOW_URI}/#/experiments/{exp.experiment_id}\n")

        print(f"🏃 View run {run_name} at: {MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}")
        print(f"🧪 View experiment at: {MLFLOW_URI}/#/experiments/{exp.experiment_id}")
        print("Data profile and artifacts logged.")

    print(f"\nTraining Complete. View logs at: {MLFLOW_URI}")

if __name__ == "__main__":
    main()
