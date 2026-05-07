"""
PCB Defect Detection – Evaluation Script
=========================================
Project : PCB Defect Detection System (Spring 2026)
Team    : Bhatia Isha, Chaurasia Vikas, Duss Karin, Müller Jonathan

What this script does
---------------------
1. Runs YOLOv8 validation on the test split
2. Computes per-class precision, recall, F1, mAP50, mAP50-95
3. Plots confusion matrix and PR curves
4. Saves a visual sample of predictions (first N images)
5. Logs everything to W&B (optional)
6. Writes a summary report to evaluation_report.json

Usage
-----
    python evaluate.py --weights runs/train/yolov8n_pcb/weights/best.pt
    python evaluate.py --weights runs/train/yolov8n_pcb/weights/best.pt --no-wandb
    python evaluate.py --weights model.onnx   # ONNX inference
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASS_NAMES = [
    "open",
    "short",
    "mousebite",
    "spur",
    "spurious_copper",
    "pin_hole",
]

YOLO_DIR = Path("data/yolo")
EVAL_DIR = Path("runs/evaluate")
WANDB_PROJECT = "pcb-defect-detection"

# Colour palette for bounding box visualisation (BGR)
COLOURS = [
    (255,  80,  80),   # open          – red
    ( 80, 200, 255),   # short         – sky blue
    ( 80, 255, 130),   # mousebite     – green
    (255, 200,  50),   # spur          – amber
    (180,  80, 255),   # spurious_copper – purple
    (255, 140,  30),   # pin_hole      – orange
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yolo_model(weights: Path):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    print(f"Loading weights: {weights}")
    return YOLO(str(weights))


def draw_predictions(img_bgr: np.ndarray, boxes, scores, cls_ids,
                     conf_threshold: float = 0.25) -> np.ndarray:
    """Draw bounding boxes and labels on image. Returns annotated copy."""
    out = img_bgr.copy()
    for box, score, cls_id in zip(boxes, scores, cls_ids):
        if score < conf_threshold:
            continue
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        colour = COLOURS[int(cls_id) % len(COLOURS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label  = f"{CLASS_NAMES[int(cls_id)]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), colour, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out


def make_grid(images: list, cols: int = 4, cell_size: int = 224) -> np.ndarray:
    """Arrange a list of BGR images into a grid."""
    rows = (len(images) + cols - 1) // cols
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (cell_size, cell_size))
        grid[r*cell_size:(r+1)*cell_size, c*cell_size:(c+1)*cell_size] = resized
    return grid


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_evaluation(args):
    weights   = Path(args.weights)
    yolo_dir  = Path(args.yolo_dir)
    eval_dir  = Path(args.eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = yolo_dir / "dataset.yaml"
    if not yaml_path.exists():
        print(f"[error] dataset.yaml not found at {yaml_path}")
        print("  Run train.py first to generate the YOLO dataset.")
        sys.exit(1)

    model = load_yolo_model(weights)

    # ── 1. YOLOv8 built-in validation on test split ───────────────────────
    print("\nRunning validation on test split …")

    # patch yaml to set test as the val split temporarily
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    cfg_test          = dict(cfg)
    cfg_test["val"]   = cfg.get("test", "images/test")
    test_yaml         = eval_dir / "test_dataset.yaml"
    with open(test_yaml, "w") as f:
        yaml.dump(cfg_test, f, default_flow_style=False, sort_keys=False)

    val_results = model.val(
        data    = str(test_yaml),
        imgsz   = args.img_size,
        batch   = args.batch,
        conf    = args.conf,
        iou     = args.iou,
        project = str(eval_dir),
        name    = "val_run",
        exist_ok= True,
        verbose = True,
        plots   = True,
        save_json= True,
    )

    # ── 2. Extract per-class metrics ──────────────────────────────────────
    metrics = _extract_metrics(val_results)
    print("\nPer-class metrics:")
    print(f"  {'Class':<20} {'P':>6} {'R':>6} {'F1':>6} {'mAP50':>7} {'mAP50-95':>9}")
    print("  " + "-" * 60)
    for cls, m in metrics["per_class"].items():
        print(f"  {cls:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['map50']:>7.3f} {m['map50_95']:>9.3f}")
    print("  " + "-" * 60)
    om = metrics["overall"]
    print(f"  {'OVERALL':<20} {om['precision']:>6.3f} {om['recall']:>6.3f} "
          f"{om['f1']:>6.3f} {om['map50']:>7.3f} {om['map50_95']:>9.3f}")

    # ── 3. Visual prediction samples ──────────────────────────────────────
    sample_grid_path = _make_prediction_samples(
        model, yolo_dir, eval_dir, args.img_size, args.conf, args.num_samples
    )

    # ── 4. Save JSON report ───────────────────────────────────────────────
    report = {
        "weights":     str(weights),
        "img_size":    args.img_size,
        "conf":        args.conf,
        "iou":         args.iou,
        "metrics":     metrics,
    }
    report_path = eval_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report → {report_path}")

    # ── 5. W&B logging ────────────────────────────────────────────────────
    if not args.no_wandb:
        _log_eval_to_wandb(report, eval_dir, sample_grid_path, weights, args)

    return report


def _extract_metrics(val_results) -> dict:
    """Pull per-class and overall metrics from ultralytics results object."""
    try:
        # ultralytics stores per-class AP in val_results.box
        box = val_results.box
        per_class = {}
        nc = len(CLASS_NAMES)

        # ap_class_index tells us which classes had detections
        class_indices = (
            box.ap_class_index.tolist()
            if hasattr(box, "ap_class_index") and box.ap_class_index is not None
            else list(range(nc))
        )

        for i, cls_idx in enumerate(class_indices):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else str(cls_idx)
            p  = float(box.p[i])  if hasattr(box, "p")  and box.p  is not None else 0.0
            r  = float(box.r[i])  if hasattr(box, "r")  and box.r  is not None else 0.0
            ap50    = float(box.ap50[i])   if hasattr(box, "ap50")   and box.ap50   is not None else 0.0
            ap50_95 = float(box.ap[i])     if hasattr(box, "ap")     and box.ap     is not None else 0.0
            f1 = 2 * p * r / (p + r + 1e-9)
            per_class[cls_name] = {
                "precision": round(p,  4),
                "recall":    round(r,  4),
                "f1":        round(f1, 4),
                "map50":     round(ap50,    4),
                "map50_95":  round(ap50_95, 4),
            }

        mp   = float(box.mp)   if hasattr(box, "mp")   else 0.0
        mr   = float(box.mr)   if hasattr(box, "mr")   else 0.0
        map50    = float(box.map50)   if hasattr(box, "map50")   else 0.0
        map50_95 = float(box.map)     if hasattr(box, "map")     else 0.0

        overall = {
            "precision": round(mp, 4),
            "recall":    round(mr, 4),
            "f1":        round(2 * mp * mr / (mp + mr + 1e-9), 4),
            "map50":     round(map50, 4),
            "map50_95":  round(map50_95, 4),
        }
    except Exception as e:
        print(f"[warn] Could not parse detailed metrics: {e}")
        overall    = {"precision": 0, "recall": 0, "f1": 0, "map50": 0, "map50_95": 0}
        per_class  = {c: overall.copy() for c in CLASS_NAMES}

    return {"overall": overall, "per_class": per_class}


def _make_prediction_samples(model, yolo_dir, eval_dir, img_size,
                              conf, num_samples) -> Path:
    """Run inference on first num_samples test images, draw boxes, save grid."""
    test_img_dir = yolo_dir / "images" / "test"
    if not test_img_dir.exists():
        return None

    img_paths = sorted(test_img_dir.glob("*.jpg"))[:num_samples]
    if not img_paths:
        return None

    print(f"\nGenerating prediction visualisations for {len(img_paths)} images …")
    annotated = []
    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        results = model.predict(
            source=str(img_path),
            imgsz=img_size,
            conf=conf,
            verbose=False,
        )
        r = results[0]
        if r.boxes is not None and len(r.boxes):
            boxes   = r.boxes.xyxy.cpu().numpy()
            scores  = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy()
        else:
            boxes, scores, cls_ids = [], [], []
        annotated.append(draw_predictions(img_bgr, boxes, scores, cls_ids, conf))

    grid      = make_grid(annotated, cols=4, cell_size=img_size)
    grid_path = eval_dir / "prediction_samples.jpg"
    cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  Saved prediction grid → {grid_path}")
    return grid_path


def _log_eval_to_wandb(report, eval_dir, sample_grid_path, weights, args):
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed, skipping W&B logging")
        return

    run = wandb.init(
        project  = WANDB_PROJECT,
        name     = f"eval-{Path(weights).parent.parent.name}",
        config   = {"weights": str(weights), "conf": args.conf, "iou": args.iou},
        tags     = ["evaluation", "pcb", "yolov8"],
        job_type = "evaluation",
    )

    # overall metrics as summary
    run.summary.update(report["metrics"]["overall"])

    # per-class metrics as a W&B table
    cols = ["class", "precision", "recall", "f1", "mAP50", "mAP50-95"]
    rows = []
    for cls, m in report["metrics"]["per_class"].items():
        rows.append([cls, m["precision"], m["recall"],
                     m["f1"], m["map50"], m["map50_95"]])
    run.log({"per_class_metrics": wandb.Table(columns=cols, data=rows)})

    # prediction samples grid
    if sample_grid_path and sample_grid_path.exists():
        run.log({"prediction_samples": wandb.Image(str(sample_grid_path))})

    # confusion matrix and curves from val run
    val_run_dir = eval_dir / "val_run"
    for fname in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                  "PR_curve.png", "F1_curve.png", "R_curve.png", "P_curve.png"]:
        fpath = val_run_dir / fname
        if fpath.exists():
            run.log({fname.replace(".png", ""): wandb.Image(str(fpath))})

    # upload report JSON as artifact
    artifact = wandb.Artifact("pcb-eval-report", type="evaluation")
    artifact.add_file(str(eval_dir / "evaluation_report.json"))
    if sample_grid_path and sample_grid_path.exists():
        artifact.add_file(str(sample_grid_path))
    run.log_artifact(artifact)

    run.finish()
    print(f"W&B evaluation run finished → {run.url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(weights: str = None, **kwargs):
    """
    Can be called directly from Colab:
        main(weights="/content/.../best.pt")
    or via CLI:
        python evaluate.py --weights best.pt
    """
    import types

    # ── If called programmatically (e.g. from a Colab cell) ──────────────
    if weights is not None:
        args = types.SimpleNamespace(
            weights     = weights,
            yolo_dir    = kwargs.get("yolo_dir",    str(YOLO_DIR)),
            eval_dir    = kwargs.get("eval_dir",    str(EVAL_DIR)),
            img_size    = kwargs.get("img_size",    640),
            batch       = kwargs.get("batch",       16),
            conf        = kwargs.get("conf",        0.25),
            iou         = kwargs.get("iou",         0.5),
            num_samples = kwargs.get("num_samples", 16),
            no_wandb    = kwargs.get("no_wandb",    False),
        )
        return run_evaluation(args)

    # ── CLI path: inject a sentinel --weights so parse_known_args
    #    doesn't fail when Jupyter passes its -f kernel.json flag ─────────
    import sys
    # Check if --weights was actually provided on the real argv
    raw_argv = [a for a in sys.argv[1:] if not a.startswith("-f")]
    if "--weights" not in raw_argv:
        print("Usage (Colab):  main(weights='path/to/best.pt')")
        print("Usage (CLI):    python evaluate.py --weights path/to/best.pt")
        return

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 PCB defect detector")
    parser.add_argument("--weights",     required=True)
    parser.add_argument("--yolo-dir",    default=str(YOLO_DIR))
    parser.add_argument("--eval-dir",    default=str(EVAL_DIR))
    parser.add_argument("--img-size",    type=int,   default=640)
    parser.add_argument("--batch",       type=int,   default=16)
    parser.add_argument("--conf",        type=float, default=0.25)
    parser.add_argument("--iou",         type=float, default=0.5)
    parser.add_argument("--num-samples", type=int,   default=16)
    parser.add_argument("--no-wandb",    action="store_true")
    args, _ = parser.parse_known_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()

# main(weights="path/to/best.pt")  # example usage
