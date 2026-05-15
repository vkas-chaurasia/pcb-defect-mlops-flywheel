import argparse
import json
import os
from pathlib import Path
import yaml
import mlflow
import mlflow.pytorch
from ultralytics import YOLO, settings
import torch
from tqdm import tqdm
import cv2
import numpy as np

# --- Configuration ---
PROJECT_ROOT  = Path(os.getcwd()).absolute()
CLASS_NAMES   = ["open", "short", "mousebite", "spur", "spurious_copper", "pin_hole"]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
YOLO_DIR      = PROJECT_ROOT / "data" / "yolo"
RUNS_DIR      = PROJECT_ROOT / "runs" / "detect"

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

        stats_path = processed_dir / "dataset_stats.json"
        with open(stats_path) as f:
            stats = json.load(f)
        mean, std = np.array(stats["mean"]), np.array(stats["std"])

        for i in tqdm(range(len(images)), desc=f"  {split}", leave=False):
            img_uint8 = np.clip((images[i] * (std + 1e-7) + mean) * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(str(img_out / f"{split}_{i:06d}.jpg"), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

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

def main():
    parser = argparse.ArgumentParser(description="PCB Training with MLflow")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n")
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    # 1. Prep Data
    yaml_path = prepare_yolo_data(PROCESSED_DIR, YOLO_DIR, args.img_size)

    # 2. MLflow Infrastructure Setup (Environment-Aware)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
    exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pcb-defect-exploration")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)
    
    # Hide direct S3 access from the client to force Proxy Mode (High-Fidelity)
    # This ensures YOLO uses the MLflow server's proxy for uploads.
    os.environ.pop("MLFLOW_S3_ENDPOINT_URL", None)
    os.environ.pop("AWS_ACCESS_KEY_ID", None)
    os.environ.pop("AWS_SECRET_ACCESS_KEY", None)
    
    # 3. ENABLE YOLO's native MLflow integration (This restores artifacts/models)
    settings.update({"mlflow": True})

    # 4. Initialize Model and Hardware
    model = YOLO(f"{args.model}.pt")
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    # 5. Execute Training within a Manual Run for DNA Tagging
    # YOLOv8 will automatically join this active run and log everything.
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        run_name = run.info.run_name
        print(f"Industry Mode Active | Experiment: {exp_name} | Run: {run_name}")
        
        # Tag the run with the unique Job ID for surgical traceability
        github_run_id = os.getenv("GITHUB_RUN_ID", "local")
        mlflow.set_tag("github_run_id", github_run_id)

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

        # 6. Export Metrics for DVC
        metrics = results.results_dict
        clean_metrics = {k.replace("(", "_").replace(")", ""): v for k, v in metrics.items()}
        with open("metrics.json", "w") as f:
            json.dump(clean_metrics, f, indent=4)

    print(f"\nTraining Complete. View in MLflow.")

if __name__ == "__main__":
    main()
