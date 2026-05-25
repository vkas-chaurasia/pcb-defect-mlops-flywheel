import argparse
from pathlib import Path
import cv2
import sys
import os

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.serving.serve import ModelManager, run_inference
from src.monitoring.prediction_logger import PredictionLogger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--img-dir", default="data/yolo/images/test")
    parser.add_argument("--out-csv", default="monitoring/reference_predictions.csv")
    args = parser.parse_args()

    ModelManager.load(args.weights, 640)
    logger = PredictionLogger(log_path=args.out_csv)
    img_dir = Path(args.img_dir)
    
    if not img_dir.exists():
        print(f"Error: {img_dir} does not exist.")
        return

    images = list(img_dir.glob("*.jpg"))
    print(f"Generating reference from {len(images)} images...")
    
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            response = run_inference(img_bgr, img_path.name, 0.10, 0.45)
            logger.log(response)
            
    print(f"Reference baseline saved to {args.out_csv}")

if __name__ == "__main__":
    main()
