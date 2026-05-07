import argparse
import json
import os
import random
import shutil
import struct
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

DEFAULT_IMG_SIZE   = 224
DEFAULT_SPLIT      = (0.70, 0.15, 0.15)
DEFAULT_SEED       = 42
DEEPPCB_ZIP_URL    = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"
RAW_DIR            = Path("data/raw")
PROCESSED_DIR      = Path("data/processed")

CLASS_NAMES = ["open", "short", "mousebite", "spur", "spurious_copper", "pin_hole"]
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# 1. Download & extract
# ---------------------------------------------------------------------------

def download_deeppcb(raw_dir: Path) -> Path:
    """Download DeepPCB zip from GitHub and unpack it. Skips if already done."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path   = raw_dir / "DeepPCB-master.zip"
    unzip_path = raw_dir / "DeepPCB-master"

    if unzip_path.exists():
        print(f"[skip] DeepPCB already extracted at {unzip_path}")
        return unzip_path

    print(f"Downloading DeepPCB from GitHub …")
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        print(f"\r  {pct:5.1f}%  ({downloaded // 1024:,} KB)", end="", flush=True)

    urllib.request.urlretrieve(DEEPPCB_ZIP_URL, zip_path, _progress)
    print()

    print("Extracting …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()           # remove zip to save space
    print(f"Extracted to {unzip_path}")
    return unzip_path


# ---------------------------------------------------------------------------
# 2. Parse annotations
# ---------------------------------------------------------------------------

def parse_annotation_file(ann_path: Path):
    """
    DeepPCB annotation format (one line per defect, comma or space separated):
        x1,y1,x2,y2,type   OR   x1 y1 x2 y2 type
    Class IDs are 1-indexed: 0=background(unused), 1=open, 2=short,
    3=mousebite, 4=spur, 5=copper, 6=pin-hole.
    We remap to 0-indexed: subtract 1 so labels are 0–5.

    Returns
    -------
    boxes  : list of [x1, y1, x2, y2]  (int)
    labels : list of int class indices  (0-indexed)
    """
    boxes, labels = [], []
    if not ann_path.exists():
        return boxes, labels
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # support both comma-separated and space-separated formats
            parts = line.replace(",", " ").split()
            if len(parts) < 5:
                continue
            x1, y1, x2, y2, cls = (int(p) for p in parts[:5])
            # remap 1-indexed → 0-indexed; skip class 0 (background)
            cls_0 = cls - 1
            if 0 <= cls_0 < len(CLASS_NAMES):
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_0)
    return boxes, labels


def collect_samples(dataset_root: Path):
    """
    Walk the DeepPCB directory structure and collect
    (image_path, annotation_path) pairs.

    DeepPCB layout:
        PCBData/
            group00000/
                00000000/
                    test/     ← defective images  (*.jpg)
                    temp/     ← template images   (not used here)
                    *.txt     ← annotation files matching test/*.jpg names
    """
    samples = []
    pcb_root = dataset_root / "PCBData"
    if not pcb_root.exists():
        # some versions place images directly under DeepPCB-master/
        pcb_root = dataset_root

    for img_path in sorted(pcb_root.rglob("*.jpg")):
        # skip template images (in 'temp' subfolder)
        if "temp" in img_path.parts:
            continue
        # DeepPCB actual layout:
        #   images : .../group00041/00041/00041000_test.jpg
        #   labels : .../group00041/00041_not/00041000.txt
        # Annotation folder = image_folder + "_not"; stem strips "_test" suffix.
        stem     = img_path.stem
        ann_stem = stem[:-5] if stem.endswith("_test") else stem
        ann_dir  = img_path.parent.parent / (img_path.parent.name + "_not")
        ann_path = ann_dir / (ann_stem + ".txt")
        # fallbacks for alternative layouts
        if not ann_path.exists():
            ann_path = img_path.parent / (ann_stem + ".txt")
        if not ann_path.exists():
            ann_path = img_path.parent.parent / (ann_stem + ".txt")
        samples.append((img_path, ann_path))

    if not samples:
        raise FileNotFoundError(
            f"No .jpg images found under {pcb_root}. "
            "Check the DeepPCB directory structure."
        )
    print(f"Found {len(samples):,} defective PCB images.")
    return samples


# ---------------------------------------------------------------------------
# 3. Resize
# ---------------------------------------------------------------------------

def resize_image_and_boxes(image: np.ndarray, boxes, target_size: int):
    """
    Resize image to (target_size × target_size) and scale bounding boxes
    proportionally.
    """
    h, w = image.shape[:2]
    scale_x = target_size / w
    scale_y = target_size / h
    resized  = cv2.resize(image, (target_size, target_size),
                          interpolation=cv2.INTER_LINEAR)
    scaled_boxes = []
    for x1, y1, x2, y2 in boxes:
        scaled_boxes.append([
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y),
        ])
    return resized, scaled_boxes


# ---------------------------------------------------------------------------
# 4. Compute per-channel statistics on training images
# ---------------------------------------------------------------------------

def compute_mean_std(images: list) -> tuple:
    """
    Compute per-channel mean and std over a list of uint8 images
    (H×W×3). Returns mean, std as (3,) float32 in [0,1] range.
    """
    acc   = np.zeros(3, dtype=np.float64)
    acc2  = np.zeros(3, dtype=np.float64)
    count = 0
    for img in images:
        pix    = img.astype(np.float64) / 255.0
        acc   += pix.reshape(-1, 3).sum(axis=0)
        acc2  += (pix ** 2).reshape(-1, 3).sum(axis=0)
        count += img.shape[0] * img.shape[1]
    mean = (acc  / count).astype(np.float32)
    std  = np.sqrt(np.maximum(acc2 / count - mean ** 2, 0)).astype(np.float32)
    return mean, std


def normalize(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize a uint8 image to float32 using per-channel mean/std."""
    img = image.astype(np.float32) / 255.0
    img = (img - mean) / (std + 1e-7)
    return img


# ---------------------------------------------------------------------------
# 5. Augmentation  (training split only)
# ---------------------------------------------------------------------------

def augment(image: np.ndarray, boxes, labels, img_size: int, rng: random.Random):
    """
    Apply random augmentations. Returns augmented (image, boxes, labels).

    Transforms applied independently with stated probabilities:
      • Horizontal flip          p=0.5
      • Vertical flip            p=0.5
      • 90° rotation             p=0.3
      • Brightness jitter ±20%   p=0.5
      • Contrast  jitter ±20%    p=0.5
      • Cutout (random erase)    p=0.3
    """
    h, w = image.shape[:2]
    aug_img    = image.copy()
    aug_boxes  = [list(b) for b in boxes]
    aug_labels = list(labels)

    # ── Horizontal flip ──────────────────────────────────────────────
    if rng.random() < 0.5:
        aug_img = cv2.flip(aug_img, 1)
        aug_boxes = [[w - x2, y1, w - x1, y2]
                     for x1, y1, x2, y2 in aug_boxes]

    # ── Vertical flip ────────────────────────────────────────────────
    if rng.random() < 0.5:
        aug_img = cv2.flip(aug_img, 0)
        aug_boxes = [[x1, h - y2, x2, h - y1]
                     for x1, y1, x2, y2 in aug_boxes]

    # ── 90° CW rotation ──────────────────────────────────────────────
    if rng.random() < 0.3:
        aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
        aug_boxes = [[h - y2, x1, h - y1, x2]
                     for x1, y1, x2, y2 in aug_boxes]
        h, w = w, h     # update after rotation

    # ── Brightness jitter ────────────────────────────────────────────
    if rng.random() < 0.5:
        factor = rng.uniform(0.8, 1.2)
        aug_img = np.clip(aug_img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # ── Contrast jitter ──────────────────────────────────────────────
    if rng.random() < 0.5:
        mean_val = aug_img.mean()
        factor   = rng.uniform(0.8, 1.2)
        aug_img  = np.clip((aug_img.astype(np.float32) - mean_val) * factor + mean_val,
                           0, 255).astype(np.uint8)

    # ── Cutout (random erase) ─────────────────────────────────────────
    if rng.random() < 0.3:
        cut_h = rng.randint(img_size // 8, img_size // 4)
        cut_w = rng.randint(img_size // 8, img_size // 4)
        cy    = rng.randint(0, h - cut_h)
        cx    = rng.randint(0, w - cut_w)
        aug_img[cy:cy + cut_h, cx:cx + cut_w] = 0   # zero-fill

    return aug_img, aug_boxes, aug_labels


# ---------------------------------------------------------------------------
# 6. Train / val / test split
# ---------------------------------------------------------------------------

def split_samples(samples, split, seed):
    """Reproducible stratified-ish random split."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n      = len(shuffled)
    n_tr   = int(n * split[0])
    n_val  = int(n * split[1])
    train  = shuffled[:n_tr]
    val    = shuffled[n_tr:n_tr + n_val]
    test   = shuffled[n_tr + n_val:]
    return train, val, test


# ---------------------------------------------------------------------------
# 7. Save processed data
# ---------------------------------------------------------------------------

def process_and_save(
    split_name: str,
    samples: list,
    img_size: int,
    mean: np.ndarray,
    std: np.ndarray,
    out_dir: Path,
    augment_flag: bool,
    rng: random.Random,
) -> dict:
    """
    Process a split, apply augmentation (training only), normalize,
    and save to .npz + return per-class counts.
    """
    images_list  = []
    boxes_list   = []
    labels_list  = []
    class_counts = {c: 0 for c in CLASS_NAMES}

    for img_path, ann_path in tqdm(samples, desc=f"Processing {split_name}"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels = parse_annotation_file(ann_path)
        img, boxes    = resize_image_and_boxes(img, boxes, img_size)

        if augment_flag and split_name == "train":
            img, boxes, labels = augment(img, boxes, labels, img_size, rng)

        img_norm = normalize(img, mean, std)

        images_list.append(img_norm)
        boxes_list.append(np.array(boxes,  dtype=np.int32)  if boxes  else np.zeros((0, 4), dtype=np.int32))
        labels_list.append(np.array(labels, dtype=np.int64) if labels else np.zeros((0,),   dtype=np.int64))

        for lbl in labels:
            class_counts[CLASS_NAMES[lbl]] += 1

    images_arr = np.stack(images_list, axis=0).astype(np.float32)   # (N,H,W,3)

    out_path = out_dir / f"{split_name}.npz"
    np.savez_compressed(
        out_path,
        images=images_arr,
        boxes=np.array(boxes_list,  dtype=object),
        labels=np.array(labels_list, dtype=object),
    )
    print(f"  Saved {split_name}.npz  →  shape {images_arr.shape}  |  {out_path}")
    return class_counts


def build_coco_manifest(all_samples, img_size, out_dir):
    """
    Write a minimal COCO-style JSON covering all splits.
    Useful for downstream frameworks that accept COCO format.
    """
    categories = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
    images_json, annotations_json = [], []
    ann_id = 0

    for img_id, (img_path, ann_path) in enumerate(all_samples):
        images_json.append({
            "id":        img_id,
            "file_name": str(img_path),
            "width":     img_size,
            "height":    img_size,
        })
        boxes, labels = parse_annotation_file(ann_path)
        for (x1, y1, x2, y2), lbl in zip(boxes, labels):
            bw = x2 - x1
            bh = y2 - y1
            annotations_json.append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  lbl,
                "bbox":         [x1, y1, bw, bh],   # COCO: [x, y, w, h]
                "area":         bw * bh,
                "iscrowd":      0,
            })
            ann_id += 1

    manifest = {
        "info":        {"description": "DeepPCB preprocessed", "version": "1.0"},
        "categories":  categories,
        "images":      images_json,
        "annotations": annotations_json,
    }
    out_path = out_dir / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved COCO manifest  →  {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCB defect preprocessing pipeline")
    parser.add_argument("--img-size",    type=int,   default=DEFAULT_IMG_SIZE,
                        help="Target image size (square). Default: 224")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT[0])
    parser.add_argument("--val-ratio",   type=float, default=DEFAULT_SPLIT[1])
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--raw-dir",     type=str,   default=str(RAW_DIR))
    parser.add_argument("--out-dir",     type=str,   default=str(PROCESSED_DIR))
    parser.add_argument("--no-augment",  action="store_true",
                        help="Disable training-set augmentation")
    args, _ = parser.parse_known_args()   # parse_known_args ignores Jupyter kernel args

    split = (
        args.train_ratio,
        args.val_ratio,
        round(1.0 - args.train_ratio - args.val_ratio, 6),
    )
    raw_dir  = Path(args.raw_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  PCB Defect Detection – Preprocessing Pipeline")
    print("=" * 60)
    print(f"  Image size  : {args.img_size}×{args.img_size}")
    print(f"  Split       : train={split[0]:.0%}  val={split[1]:.0%}  test={split[2]:.0%}")
    print(f"  Augmentation: {'OFF' if args.no_augment else 'ON'}")
    print()

    # ── Step 1: Download ───────────────────────────────────────────────
    dataset_root = download_deeppcb(raw_dir)

    # ── Step 2: Collect samples ────────────────────────────────────────
    samples = collect_samples(dataset_root)

    # ── Step 3: Split ──────────────────────────────────────────────────
    train_samples, val_samples, test_samples = split_samples(samples, split, args.seed)
    print(f"\nSplit sizes → train: {len(train_samples)},  "
          f"val: {len(val_samples)},  test: {len(test_samples)}\n")

    # ── Step 4: Compute normalisation stats on raw training images ─────
    print("Computing per-channel mean & std on training images …")
    raw_train_imgs = []
    for img_path, _ in tqdm(train_samples, desc="  Reading train images"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _ = resize_image_and_boxes(img, [], args.img_size)
        raw_train_imgs.append(img)

    mean, std = compute_mean_std(raw_train_imgs)
    print(f"  Mean (RGB) : {mean.tolist()}")
    print(f"  Std  (RGB) : {std.tolist()}\n")

    # ── Steps 5-7: Process each split and save ─────────────────────────
    all_counts = {}
    for split_name, split_data in [
        ("train", train_samples),
        ("val",   val_samples),
        ("test",  test_samples),
    ]:
        counts = process_and_save(
            split_name=split_name,
            samples=split_data,
            img_size=args.img_size,
            mean=mean,
            std=std,
            out_dir=out_dir,
            augment_flag=not args.no_augment,
            rng=rng,
        )
        all_counts[split_name] = counts

    # ── Save dataset stats ─────────────────────────────────────────────
    stats = {
        "img_size":       args.img_size,
        "mean_rgb":       mean.tolist(),
        "std_rgb":        std.tolist(),
        "class_names":    CLASS_NAMES,
        "split_sizes":    {
            "train": len(train_samples),
            "val":   len(val_samples),
            "test":  len(test_samples),
        },
        "class_counts":   all_counts,
    }
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved dataset_stats.json  →  {stats_path}")

    # ── COCO manifest ──────────────────────────────────────────────────
    build_coco_manifest(samples, args.img_size, out_dir)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Preprocessing complete!")
    print("=" * 60)
    print("\n  Class distribution (train):")
    for cls, cnt in all_counts["train"].items():
        bar = "█" * min(cnt // 10, 40)
        print(f"    {cls:<20}  {cnt:>4}  {bar}")
    print(f"\n  Output directory: {out_dir.resolve()}")
    print("\n  Files written:")
    for f in sorted(out_dir.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"    {f.name:<25}  {size_kb:>6} KB")

    print("""
  ── Usage in PyTorch ────────────────────────────────────
  data = np.load('data/processed/train.npz', allow_pickle=True)
  images, boxes, labels = data['images'], data['boxes'], data['labels']
  # images: (N, H, W, 3) float32  → permute to (N,3,H,W) for PyTorch
  import torch
  images_t = torch.from_numpy(images).permute(0,3,1,2)

  ── Usage in TensorFlow ─────────────────────────────────
  import tensorflow as tf
  dataset = tf.data.Dataset.from_tensor_slices({
      'image': images,   # (N,H,W,3) float32 – ready to use
      'label': labels,
  })
""")


if __name__ == "__main__":
    main()
