import argparse
import json
import os
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# --- Defaults & Configuration ---
DEFAULT_IMG_SIZE   = 224
DEFAULT_SPLIT      = (0.70, 0.15, 0.15)
DEFAULT_SEED       = 42
DEEPPCB_ZIP_URL    = "https://github.com/tangsanli5201/DeepPCB/archive/refs/heads/master.zip"
RAW_DIR            = Path("data/raw")
PROCESSED_DIR      = Path("data/processed")

CLASS_NAMES = ["open", "short", "mousebite", "spur", "spurious_copper", "pin_hole"]
CLASS2IDX   = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# 1. Data Acquisition
# ---------------------------------------------------------------------------

def download_and_extract(raw_dir: Path) -> Path:
    """Download the base dataset and extract it. Used for initial setup."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path   = raw_dir / "DeepPCB-master.zip"
    unzip_path = raw_dir / "DeepPCB-master"

    if unzip_path.exists():
        print(f"[skip] Dataset already extracted at {unzip_path}")
        return unzip_path

    print(f"Downloading base dataset ...")
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
        print(f"\r  {pct:5.1f}%  ({downloaded // 1024:,} KB)", end="", flush=True)

    urllib.request.urlretrieve(DEEPPCB_ZIP_URL, zip_path, _progress)
    print()

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink()  # Clean up zip
    print(f"Extracted to {unzip_path}")
    return unzip_path

# ---------------------------------------------------------------------------
# 2. Annotation & Sample Parsing
# ---------------------------------------------------------------------------

def parse_annotation(ann_path: Path):
    """
    Parses PCB annotation format (x1,y1,x2,y2,type).
    Remaps 1-indexed classes to 0-indexed.
    """
    boxes, labels = [], []
    if not ann_path.exists():
        return boxes, labels
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.replace(",", " ").split()
            if len(parts) < 5: continue
            x1, y1, x2, y2, cls = (int(p) for p in parts[:5])
            cls_0 = cls - 1
            if 0 <= cls_0 < len(CLASS_NAMES):
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_0)
    return boxes, labels

def collect_samples(raw_dir: Path):
    """
    Recursively finds all .jpg images and their corresponding .txt annotations.
    Designed to work with both the initial DeepPCB structure and Active Learning folders.
    """
    samples = []
    # Search for all images in the raw directory
    for img_path in sorted(raw_dir.rglob("*.jpg")):
        if "temp" in img_path.parts: continue  # Skip template images
        
        # Determine annotation path (DeepPCB pattern: image_folder + "_not" / stem.txt)
        stem = img_path.stem
        ann_stem = stem[:-5] if stem.endswith("_test") else stem
        ann_dir = img_path.parent.parent / (img_path.parent.name + "_not")
        ann_path = ann_dir / (ann_stem + ".txt")

        # Fallback for simplified structures (Active Learning loop)
        if not ann_path.exists():
            ann_path = img_path.with_suffix(".txt")
        
        if ann_path.exists():
            samples.append((img_path, ann_path))

    if not samples:
        print(f"Warning: No valid samples found in {raw_dir}")
    else:
        print(f"Found {len(samples):,} valid samples.")
    return samples

# ---------------------------------------------------------------------------
# 3. Image Processing
# ---------------------------------------------------------------------------

def process_image(image: np.ndarray, boxes, target_size: int):
    """Resize image and scale bounding boxes."""
    h, w = image.shape[:2]
    scale_x = target_size / w
    scale_y = target_size / h
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    scaled_boxes = [[int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)] for x1, y1, x2, y2 in boxes]
    return resized, scaled_boxes

def compute_stats(images: list):
    """Compute per-channel mean and std."""
    acc = np.zeros(3, dtype=np.float64)
    acc2 = np.zeros(3, dtype=np.float64)
    count = 0
    for img in images:
        pix = img.astype(np.float64) / 255.0
        acc += pix.reshape(-1, 3).sum(axis=0)
        acc2 += (pix ** 2).reshape(-1, 3).sum(axis=0)
        count += img.shape[0] * img.shape[1]
    mean = (acc / count).astype(np.float32)
    std = np.sqrt(np.maximum(acc2 / count - mean ** 2, 0)).astype(np.float32)
    return mean, std

def normalize(image: np.ndarray, mean: np.ndarray, std: np.ndarray):
    """Normalize image to float32."""
    img = image.astype(np.float32) / 255.0
    return (img - mean) / (std + 1e-7)

# ---------------------------------------------------------------------------
# 4. Augmentation (Team-ready standard)
# ---------------------------------------------------------------------------

def augment(image: np.ndarray, boxes, labels, rng: random.Random):
    """Applies basic augmentations for training."""
    h, w = image.shape[:2]
    aug_img, aug_boxes = image.copy(), [list(b) for b in boxes]
    
    if rng.random() < 0.5: # Horizontal flip
        aug_img = cv2.flip(aug_img, 1)
        aug_boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in aug_boxes]
    
    if rng.random() < 0.5: # Vertical flip
        aug_img = cv2.flip(aug_img, 0)
        aug_boxes = [[x1, h - y2, x2, h - y1] for x1, y1, x2, y2 in aug_boxes]
    
    return aug_img, aug_boxes, labels

# ---------------------------------------------------------------------------
# 5. Main Execution Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PCB Dataset Preprocessing Pipeline")
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT[1])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--download", action="store_true", help="Download base dataset if missing")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    raw_path, out_path = Path(args.raw_dir), Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    # Step 1: Optional Download
    if args.download:
        download_and_extract(raw_path)

    # Step 2: Collect & Split
    samples = collect_samples(raw_path)
    if not samples: return
    
    rng.shuffle(samples)
    n = len(samples)
    n_tr, n_val = int(n * args.train_ratio), int(n * args.val_ratio)
    splits = {
        "train": samples[:n_tr],
        "val":   samples[n_tr:n_tr+n_val],
        "test":  samples[n_tr+n_val:]
    }

    # Step 3: Compute stats on train set
    print("Computing normalization stats ...")
    train_imgs = []
    for img_path, _ in splits["train"][:500]: # Sample 500 for stats speed
        img = cv2.imread(str(img_path))
        if img is not None:
            img, _ = process_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), [], args.img_size)
            train_imgs.append(img)
    mean, std = compute_stats(train_imgs)

    # Step 4: Process and Save
    all_counts = {}
    for name, data in splits.items():
        imgs, bxs, lbls = [], [], []
        counts = {c: 0 for c in CLASS_NAMES}
        
        for img_p, ann_p in tqdm(data, desc=f"Processing {name}"):
            img = cv2.imread(str(img_p))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            boxes, labels = parse_annotation(ann_p)
            img, boxes = process_image(img, boxes, args.img_size)
            
            if name == "train" and not args.no_augment:
                img, boxes, labels = augment(img, boxes, labels, rng)
            
            imgs.append(normalize(img, mean, std))
            bxs.append(np.array(boxes, dtype=np.int32) if boxes else np.zeros((0,4), dtype=np.int32))
            lbls.append(np.array(labels, dtype=np.int64))
            for l in labels: counts[CLASS_NAMES[l]] += 1
        
        np.savez_compressed(out_path / f"{name}.npz", images=np.stack(imgs), boxes=np.array(bxs, dtype=object), labels=np.array(lbls, dtype=object))
        all_counts[name] = counts

    # Step 5: Save Metadata
    with open(out_path / "dataset_stats.json", "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "counts": all_counts, "classes": CLASS_NAMES}, f, indent=2)
    
    print(f"\nPreprocessing complete. Data saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
