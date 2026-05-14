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
CLASS_NAMES = ["open", "short", "mousebite", "spur", "spurious_copper", "pin_hole"]
PROCESSED_DIR = Path("data/processed")
YOLO_DIR      = Path("data/yolo")
RUNS_DIR      = Path("runs/train")
MLFLOW_URI    = "http://localhost:5555"

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
    mlflow.set_experiment("pcb-defect-detection")

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
            
        print(f"🚀 Training on device: {device}")

        # Train
        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            project="pcb-defect-detection",
            name=run_name,
            exist_ok=True,
            device=device
        )
        if _uri:
            os.environ["MLFLOW_TRACKING_URI"] = _uri

        # Log training metrics
        metrics = results.results_dict
        mlflow.log_metrics({k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()})

        # Log the formal PyTorch model (enables "Register Model" button)
        best_pt = Path(f"runs/detect/pcb-defect-detection/{run_name}/weights/best.pt")

        if best_pt.exists():
            import torch
            ckpt = torch.load(best_pt, weights_only=False)
            brain = ckpt['model']

            print("Logging formal PyTorch model flavor...")
            mlflow.pytorch.log_model(
                pytorch_model=brain,
                artifact_path="pcb-yolo-model"
            )

            # Also log the raw .pt file directly to the root for easy manual download
            print("Logging raw best.pt weights...")
            mlflow.log_artifact(str(best_pt))

        # Log dataset formally for lineage
        try:
            import subprocess
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            
            # Log the YOLO yaml as the dataset source with Git info
            dataset_source = mlflow.data.dataset_source.DatasetSource()
            mlflow.log_input(
                mlflow.data.dataset.Dataset(
                    source=dataset_source, 
                    name="pcb-yolo-dataset", 
                    targets=str(yaml_path),
                    digest=commit_hash[:8] # Use short commit as digest
                ),
                context="training"
            )
            mlflow.set_tag("git_commit", commit_hash)
            print(f"Formal dataset lineage logged (Commit: {commit_hash[:8]})")
        except Exception as e:
            print(f"Note: Could not log formal dataset lineage: {e}")

        # Log dataset profile (mean/std)
        stats_path = PROCESSED_DIR / "dataset_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)

        mlflow.log_params({
            "mean_rgb": stats["mean"],
            "std_rgb": stats["std"],
        })

        # Log all YOLO artifacts (plots, charts, labels, etc.)
        yolo_run_dir = Path(f"runs/detect/pcb-defect-detection/{run_name}")
        if yolo_run_dir.exists():
            print("Uploading visual plots and charts to MLflow...")
            mlflow.log_artifacts(str(yolo_run_dir))

        print("Data profile and artifacts logged.")

    print(f"\nTraining Complete. View logs at: {MLFLOW_URI}")

if __name__ == "__main__":
    main()
