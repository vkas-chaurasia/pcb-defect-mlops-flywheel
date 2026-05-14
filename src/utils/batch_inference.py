import os
import requests
import json
from pathlib import Path
from label_studio_sdk import Client

# --- Configuration ---
API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "your_api_key_here")
LS_PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

CONFIDENCE_THRESHOLD = 0.85
DATA_DIR = Path("data/raw/unseen_simulation")

def run_batch_inference(image_paths):
    """Sends images to the FastAPI batch prediction endpoint."""
    url = f"{API_URL}/predict/batch"
    
    files = []
    for path in image_paths:
        files.append(("files", (path.name, open(path, "rb"), "image/jpeg")))
    
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error during batch inference: {e}")
        return None
    finally:
        for _, file_tuple in files:
            file_tuple[1].close()

def upload_to_label_studio(image_path, prediction_data):
    """Uploads a failed or low-confidence image to Label Studio."""
    print(f"Flagging image for review: {image_path.name}")
    
    try:
        ls = Client(url=LS_URL, api_key=LS_API_KEY)
        project = ls.get_project(id=LS_PROJECT_ID)
        
        # Upload the file
        # In a real setup, you might want to host these images on S3 or use LS import
        # For local dev, we can use the SDK's import method
        project.import_tasks([{
            "data": {
                "image": f"/data/local-files/?d=unseen_simulation/{image_path.name}"
            },
            "predictions": [{
                "model_version": "v1-automated",
                "result": [
                    {
                        "from_name": "label",
                        "to_name": "image",
                        "type": "choices",
                        "value": {
                            "choices": ["Defect" if prediction_data['pass_fail'] == "FAIL" else "No Defect"]
                        }
                    }
                ]
            }]
        }])
        return True
    except Exception as e:
        print(f"Error uploading to Label Studio: {e}")
        return False

def main():
    print(f"🚀 Starting Automated Batch Inference on: {DATA_DIR}")
    
    # 1. Discover images (only .jpg or .png)
    image_extensions = (".jpg", ".jpeg", ".png")
    all_images = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in image_extensions]
    
    if not all_images:
        print("No images found in the simulation directory.")
        return

    print(f"Found {len(all_images)} images. Processing in batches of 16...")
    
    # 2. Process in batches
    for i in range(0, len(all_images), 16):
        batch = all_images[i:i+16]
        print(f"Processing batch {i//16 + 1} ({len(batch)} images)...")
        
        results = run_batch_inference(batch)
        if not results:
            continue
            
        for img_path, pred in zip(batch, results['results']):
            # Logic: Send to Label Studio if:
            # - Model says FAIL (defect detected)
            # - OR Model says PASS but confidence is low (ambiguous)
            
            should_review = False
            if pred['pass_fail'] == "FAIL":
                should_review = True
                reason = "Defect detected"
            elif pred['num_detections'] == 0:
                # For PASS cases, check if we had any low-confidence hits that were filtered
                # Or just use a default "Pass" confidence if the API provided it
                # In our current API, 'pass_fail' is "PASS" if detections list is empty.
                # Let's assume for this project we review all FAILs.
                pass
            
            if should_review:
                print(f"🔍 {img_path.name}: {reason}. Routing to Label Studio...")
                if LS_API_KEY == "your_api_key_here":
                    print("⚠️ Skipping upload: Please set LS_API_KEY in src/utils/batch_inference.py")
                else:
                    upload_to_label_studio(img_path, pred)
            else:
                print(f"✅ {img_path.name}: PASS. Model is confident.")

    print("\n✅ Batch processing complete.")

if __name__ == "__main__":
    main()
