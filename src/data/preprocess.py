import cv2
import os
import yaml
from pathlib import Path

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
TARGET_SIZE = (224, 224)

def preprocess():
    """
    Resizes images from raw directory and saves them to processed directory.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    raw_path = Path(RAW_DIR)
    processed_count = 0
    
    for img_path in raw_path.glob("*"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipping corrupt image: {img_path}")
                continue
            
            # Resize
            img_resized = cv2.resize(img, TARGET_SIZE)
            
            # Save
            save_path = os.path.join(PROCESSED_DIR, img_path.name)
            cv2.imwrite(save_path, img_resized)
            processed_count += 1
            
    print(f"Preprocessing complete. Processed {processed_count} images.")

if __name__ == "__main__":
    preprocess()
