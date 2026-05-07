"""
PCB Defect Detection – YOLOv8 Training Script
==============================================
Project : PCB Defect Detection System (Spring 2026)
Team    : Bhatia Isha, Chaurasia Vikas, Duss Karin, Müller Jonathan

What this script does
---------------------
1. Converts the preprocessed .npz + COCO manifest into YOLOv8 folder layout
2. Trains YOLOv8 (nano/small/medium) with Weights & Biases logging
3. Saves best & last checkpoints, exports to ONNX for serving
4. Logs final metrics, confusion matrix, and sample predictions to W&B

Prerequisites
-------------
    pip install ultralytics wandb opencv-python-headless numpy tqdm pyyaml

Usage
-----
    # First login to W&B (once per environment)
    wandb login

    # Train with defaults (YOLOv8n, 50 epochs, img-size 640)
    python train.py

    # Larger model, more epochs
    python train.py --model yolov8s --epochs 100 --img-size 640

    # Dry-run without W&B (offline mode)
    python train.py --wandb-mode offline

    # Skip W&B entirely
    python train.py --no-wandb
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
]

PROCESSED_DIR = Path("data/processed")
YOLO_DIR      = Path("data/yolo")          # YOLOv8 expects this layout
RUNS_DIR      = Path("runs/train")

WANDB_PROJECT = "pcb-defect-detection"

# ---------------------------------------------------------------------------
# Step 1 – Convert .npz + manifest.json → YOLOv8 folder layout
# ---------------------------------------------------------------------------
# YOLOv8 expects:
#   data/yolo/
#     images/train/*.jpg
#     images/val/*.jpg
#     images/test/*.jpg
#     labels/train/*.txt   (YOLO format: cls cx cy w h, normalised 0-1)
#     labels/val/*.txt
#     labels/test/*.txt
#     dataset.yaml

def npz_to_yolo(processed_dir: Path, yolo_dir: Path, img_size: int) -> Path:
    """
    Convert train/val/test .npz files into YOLOv8 directory structure.
    Skips conversion if yolo_dir already exists and is populated.
    Returns path to dataset.yaml.
    """
    yaml_path = yolo_dir / "dataset.yaml"
    if yaml_path.exists() and (yolo_dir / "images" / "train").exists():
        n_existing = len(list((yolo_dir / "images" / "train").glob("*.jpg")))
        if n_existing > 0:
            print(f"[skip] YOLO dataset already exists ({n_existing} train images)")
            return yaml_path

    print("Converting .npz → YOLO format …")
    for split in ("train", "val", "test"):
        npz_path = processed_dir / f"{split}.npz"
        if not npz_path.exists():
            print(f"  [warn] {npz_path} not found, skipping {split} split")
            continue

        img_out = yolo_dir / "images" / split
        lbl_out = yolo_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        data   = np.load(npz_path, allow_pickle=True)
        images = data["images"]   # (N, H, W, 3) float32, normalised
        boxes  = data["boxes"]    # object array of (K,4) int32 arrays
        labels = data["labels"]   # object array of (K,) int64 arrays

        # recover uint8 for saving as JPEG
        # images were normalised with per-channel mean/std; we just save
        # the raw pixel representation directly from the original dataset.
        # Re-load stats to un-normalise properly.
        stats_path = processed_dir / "dataset_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
            mean = np.array(stats["mean_rgb"], dtype=np.float32)
            std  = np.array(stats["std_rgb"],  dtype=np.float32)
            images_uint8 = np.clip(
                (images * (std + 1e-7) + mean) * 255, 0, 255
            ).astype(np.uint8)
        else:
            # fallback: stretch to 0-255
            images_uint8 = np.clip(images * 255, 0, 255).astype(np.uint8)

        h, w = img_size, img_size   # images were already resized to this

        for i in tqdm(range(len(images_uint8)), desc=f"  {split}", leave=False):
            # ── Save image ────────────────────────────────────────────
            img_bgr  = cv2.cvtColor(images_uint8[i], cv2.COLOR_RGB2BGR)
            img_file = img_out / f"{split}_{i:06d}.jpg"
            cv2.imwrite(str(img_file), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # ── Save label ────────────────────────────────────────────
            bboxes = boxes[i]    # (K, 4) or empty
            lbls   = labels[i]   # (K,)   or empty
            lbl_file = lbl_out / f"{split}_{i:06d}.txt"

            lines = []
            if len(bboxes) > 0:
                for (x1, y1, x2, y2), cls in zip(bboxes, lbls):
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    # clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    bw = max(0.0, min(1.0, bw))
                    bh = max(0.0, min(1.0, bh))
                    lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lbl_file.write_text("\n".join(lines))

        print(f"  {split}: {len(images_uint8)} images written → {img_out}")

    # ── Write dataset.yaml ────────────────────────────────────────────────
    dataset_cfg = {
        "path":  str(yolo_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote dataset.yaml → {yaml_path}\n")
    return yaml_path


# ---------------------------------------------------------------------------
# Step 2 – W&B setup
# ---------------------------------------------------------------------------

def setup_wandb(args):
    """Initialise a W&B run and return the run object (or None)."""
    if args.no_wandb:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed. Run: pip install wandb")
        return None

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"{args.model}-ep{args.epochs}-img{args.img_size}",
        config={
            "model":         args.model,
            "epochs":        args.epochs,
            "img_size":      args.img_size,
            "batch_size":    args.batch,
            "lr0":           args.lr,
            "optimizer":     args.optimizer,
            "classes":       CLASS_NAMES,
            "augment":       not args.no_augment,
            "pretrained":    not args.no_pretrained,
            "dataset":       "DeepPCB",
        },
        tags=["pcb", "yolov8", "defect-detection"],
    )
    print(f"W&B run initialised → {run.url}\n")
    return run


# ---------------------------------------------------------------------------
# Step 3 – Train YOLOv8
# ---------------------------------------------------------------------------

def train(args, yaml_path: Path, wandb_run):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # ── Build training kwargs ─────────────────────────────────────────────
    train_kwargs = dict(
        data       = str(yaml_path),
        epochs     = args.epochs,
        imgsz      = args.img_size,
        batch      = args.batch,
        lr0        = args.lr,
        optimizer  = args.optimizer,
        pretrained = not args.no_pretrained,
        augment    = not args.no_augment,
        project    = str(RUNS_DIR),
        name       = f"{args.model}_pcb",
        exist_ok   = True,
        verbose    = True,
        # built-in W&B integration — ultralytics logs automatically if W&B is active
        # additional augmentation hyperparams
        hsv_h      = 0.0,    # PCB images are mostly greyscale — disable colour jitter
        hsv_s      = 0.0,
        hsv_v      = 0.3,    # slight brightness jitter is fine
        fliplr     = 0.5,
        flipud     = 0.5,
        degrees    = 10.0,
        translate  = 0.1,
        scale      = 0.2,
        mosaic     = 0.5,
    )

    # ── Load model ────────────────────────────────────────────────────────
    weights = f"{args.model}.pt" if not args.no_pretrained else f"{args.model}.yaml"
    print(f"Loading model: {weights}")
    model = YOLO(weights)

    print(f"\n{'='*60}")
    print(f"  Training {args.model} for {args.epochs} epochs")
    print(f"  Image size : {args.img_size}  |  Batch : {args.batch}")
    print(f"  Classes    : {CLASS_NAMES}")
    print(f"{'='*60}\n")

    # ── Train ─────────────────────────────────────────────────────────────
    results = model.train(**train_kwargs)

    run_dir  = RUNS_DIR / f"{args.model}_pcb"
    best_pt  = run_dir / "weights" / "best.pt"
    last_pt  = run_dir / "weights" / "last.pt"

    # ── Export to ONNX ────────────────────────────────────────────────────
    onnx_path = None
    if best_pt.exists():
        print("\nExporting best.pt → ONNX …")
        best_model = YOLO(str(best_pt))
        export_result = best_model.export(
            format  = "onnx",
            imgsz   = args.img_size,
            dynamic = True,       # dynamic batch axis for serving
            simplify= True,
        )
        onnx_path = Path(str(export_result))
        print(f"  ONNX saved → {onnx_path}")

    # ── Log final artefacts to W&B ────────────────────────────────────────
    if wandb_run is not None:
        import wandb
        _log_to_wandb(wandb_run, model, results, run_dir,
                      best_pt, onnx_path, yaml_path, args)

    print(f"\n{'='*60}")
    print("  Training complete!")
    print(f"  Best weights : {best_pt}")
    print(f"  Last weights : {last_pt}")
    if onnx_path:
        print(f"  ONNX model   : {onnx_path}")
    print(f"{'='*60}\n")

    return results, best_pt, onnx_path


# ---------------------------------------------------------------------------
# W&B logging helpers
# ---------------------------------------------------------------------------

def _log_to_wandb(run, model, results, run_dir, best_pt, onnx_path,
                  yaml_path, args):
    import wandb

    # ── Final scalar metrics ──────────────────────────────────────────────
    if hasattr(results, "results_dict"):
        metrics = results.results_dict
        run.summary.update({
            "mAP50":     metrics.get("metrics/mAP50(B)",    0),
            "mAP50-95":  metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall":    metrics.get("metrics/recall(B)",   0),
            "box_loss":  metrics.get("train/box_loss",      0),
            "cls_loss":  metrics.get("train/cls_loss",      0),
        })

    # ── Upload model checkpoints ──────────────────────────────────────────
    artifact = wandb.Artifact(
        name=f"pcb-yolov8-{args.model}",
        type="model",
        description=f"YOLOv8 PCB defect detector – {args.model}",
        metadata={"classes": CLASS_NAMES, "img_size": args.img_size},
    )
    if best_pt and best_pt.exists():
        artifact.add_file(str(best_pt), name="best.pt")
    if onnx_path and onnx_path.exists():
        artifact.add_file(str(onnx_path), name="model.onnx")
    artifact.add_file(str(yaml_path), name="dataset.yaml")
    run.log_artifact(artifact)

    # ── Upload training plots (confusion matrix, PR curve, etc.) ─────────
    for plot_name in [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "results.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
    ]:
        plot_path = run_dir / plot_name
        if plot_path.exists():
            run.log({plot_name.replace(".png", "").replace(".jpg", ""):
                     wandb.Image(str(plot_path))})

    # ── Per-class metrics table ───────────────────────────────────────────
    try:
        metrics_csv = run_dir / "results.csv"
        if metrics_csv.exists():
            import csv
            with open(metrics_csv) as f:
                reader = csv.DictReader(f)
                rows   = list(reader)
            if rows:
                table = wandb.Table(
                    columns=list(rows[0].keys()),
                    data=[[r[k] for k in rows[0].keys()] for r in rows],
                )
                run.log({"training_history": table})
    except Exception as e:
        print(f"[warn] Could not log metrics table: {e}")

    run.finish()
    print("W&B run finished and synced.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on DeepPCB with W&B logging"
    )
    # Model
    parser.add_argument("--model",    default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                        help="YOLOv8 variant (default: yolov8n = nano, fastest)")
    # Training
    parser.add_argument("--epochs",   type=int,   default=50)
    parser.add_argument("--img-size", type=int,   default=640)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=0.01)
    parser.add_argument("--optimizer",default="SGD",
                        choices=["SGD", "Adam", "AdamW"])
    # Data
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--yolo-dir",      default=str(YOLO_DIR))
    parser.add_argument("--img-size-npz",  type=int, default=224,
                        help="Image size used during preprocessing (for un-normalisation)")
    # Flags
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch (no COCO pretrained weights)")
    parser.add_argument("--no-augment",    action="store_true")
    # W&B
    parser.add_argument("--no-wandb",    action="store_true")
    parser.add_argument("--wandb-mode",  default="online",
                        choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb-project", default=WANDB_PROJECT)

    args, _ = parser.parse_known_args()   # Jupyter/Colab safe

    processed_dir = Path(args.processed_dir)
    yolo_dir      = Path(args.yolo_dir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Convert data ───────────────────────────────────────────────────
    yaml_path = npz_to_yolo(processed_dir, yolo_dir, args.img_size_npz)

    # ── 2. Init W&B ───────────────────────────────────────────────────────
    wandb_run = setup_wandb(args)

    # ── 3. Train ──────────────────────────────────────────────────────────
    results, best_pt, onnx_path = train(args, yaml_path, wandb_run)

    # ── 4. Print next steps ───────────────────────────────────────────────
    print("Next steps:")
    print(f"  Evaluate  : python evaluate.py --weights {best_pt}")
    print(f"  Serve     : python serve.py --weights {best_pt}")
    if onnx_path:
        print(f"  ONNX serve: python serve.py --weights {onnx_path}")


if __name__ == "__main__":
    main()
