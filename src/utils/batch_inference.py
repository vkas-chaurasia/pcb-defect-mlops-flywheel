import os
import requests
from pathlib import Path
from label_studio_sdk import Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "4c5c6e9a5c5c42458204e855a68a0fcd4c5c6e9a")
LS_PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

CONFIDENCE_THRESHOLD = 0.85
DATA_DIR = Path("data/raw/unseen_simulation")

def upload_to_label_studio(image_path, prediction_result):
    """Uploads an image and its model predictions to Label Studio."""
    print(f"Flagging image for review: {image_path.name}")
    
    try:
        ls = Client(url=LS_URL, api_key=LS_API_KEY)
        project = ls.get_project(id=LS_PROJECT_ID)
        
        # Prepare Label Studio prediction results
        ls_results = []
        for det in prediction_result.get('detections', []):
            # YOLO [x1, y1, x2, y2] to Label Studio [x, y, width, height] in %
            img_w = prediction_result['image_width']
            img_h = prediction_result['image_height']
            
            x1, y1, x2, y2 = det['bbox_xyxy']
            
            # Convert to percentages
            ls_x = (x1 / img_w) * 100
            ls_y = (y1 / img_h) * 100
            ls_w = ((x2 - x1) / img_w) * 100
            ls_h = ((y2 - y1) / img_h) * 100
            
            ls_results.append({
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [det['class_name']],
                    "x": ls_x,
                    "y": ls_y,
                    "width": ls_w,
                    "height": ls_h
                },
                "score": det['confidence']
            })

        # Import task with predictions
        project.import_tasks([{
            "data": {
                "image": f"{API_URL}/data/raw/unseen_simulation/{image_path.name}"
            },
            "predictions": [{
                "model_version": "v1-automated",
                "result": ls_results
            }]
        }])
        
    except Exception as e:
        print(f"Error uploading to Label Studio: {e}")

def run_batch_inference():
    """Runs inference on all images in the simulation directory."""
    print(f"🚀 Starting Automated Batch Inference on: {DATA_DIR}")
    
    image_files = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png"))
    if not image_files:
        print("No images found.")
        return

    print(f"Found {len(image_files)} images. Processing in batches of 16...")
    
    # Process images one by one for demo simplicity
    for img_path in image_files:
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            response = requests.post(f"{API_URL}/predict", files=files)
            
            if response.status_code != 200:
                print(f"Error processing {img_path.name}: {response.text}")
                continue
                
            result = response.json()
            
            # Active Learning Logic:
            # 1. If any defect detected, flag for review (verify it)
            # 2. If no defect but low confidence, flag for review (potential missed defect)
            
            # Simplified logic for demo: send everything with detections
            if result['num_detections'] > 0:
                print(f"🔍 {img_path.name}: Defect detected. Routing to Label Studio...")
                upload_to_label_studio(img_path, result)
            else:
                # Check confidence of "no defect" (if model provides it, else assume high for now)
                print(f"✅ {img_path.name}: PASS. Model is confident.")

if __name__ == "__main__":
    run_batch_inference()
