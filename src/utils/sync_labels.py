import os
import json
import shutil
from pathlib import Path
from label_studio_sdk import Client
from dotenv import load_dotenv

load_dotenv()

# Label Studio Credentials
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "4c5c6e9a5c5c42458204e855a68a0fcd4c5c6e9a")
PROJECT_ID = int(os.getenv("LABEL_STUDIO_PROJECT_ID", "1"))

# Paths
# We save to raw/active_learning so they can be tracked by DVC later
OUTPUT_DIR = Path("data/raw/active_learning")
IMAGE_SOURCE_DIR = Path("data/raw/unseen_simulation")

# YOLO Class Mapping (Must match serve.py)
CLASS_MAPPING = {
    "open": 0,
    "short": 1,
    "mousebite": 2,
    "spur": 3,
    "spurious_copper": 4,
    "pin_hole": 5,
}

def convert_ls_to_yolo(ls_rect, img_width, img_height):
    """Converts Label Studio rectangle to YOLO format (normalized cx, cy, w, h)."""
    # LS values are percentages (0-100)
    x = ls_rect['x'] / 100.0
    y = ls_rect['y'] / 100.0
    w = ls_rect['width'] / 100.0
    h = ls_rect['height'] / 100.0
    
    # YOLO format is center-based
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    
    return cx, cy, w, h

def sync_labels():
    """Downloads annotations from Label Studio and saves them in YOLO format."""
    print(f"📡 Connecting to Label Studio at {LS_URL}...")
    
    try:
        ls = Client(url=LS_URL, api_key=LS_API_KEY)
        project = ls.get_project(id=PROJECT_ID)
        
        # Get all tasks with annotations
        tasks = project.get_tasks()
        exported_count = 0
        
        for task in tasks:
            if not task.get('annotations'):
                continue
                
            # Get latest annotation
            annotation = task['annotations'][0]
            results = annotation.get('result', [])
            
            if not results:
                continue
                
            # Get original filename from URL or data
            # URL format: http://localhost:8000/data/raw/unseen_simulation/filename.jpg
            image_url = task['data']['image']
            filename = os.path.basename(image_url).split('?')[0] # Remove query params if any
            base_name = os.path.splitext(filename)[0]
            
            # YOLO Label content
            yolo_lines = []
            
            for res in results:
                if res['type'] != 'rectanglelabels':
                    continue
                
                val = res['value']
                label_name = val['rectanglelabels'][0]
                class_id = CLASS_MAPPING.get(label_name, 0)
                
                # Convert coordinates
                cx, cy, w, h = convert_ls_to_yolo(val, res['original_width'], res['original_height'])
                yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
            if yolo_lines:
                # Save label file
                label_file = OUTPUT_DIR / f"{base_name}.txt"
                with open(label_file, "w") as f:
                    f.write("\n".join(yolo_lines))
                
                # Copy image to active_learning for a complete dataset
                image_src = IMAGE_SOURCE_DIR / filename
                if image_src.exists():
                    shutil.copy(image_src, OUTPUT_DIR / filename)
                
                print(f"✅ Synced: {filename} ({len(yolo_lines)} defects)")
                exported_count += 1
        
        print(f"\n✨ Exported {exported_count} labels to {OUTPUT_DIR}")
        print("💡 These are now ready to be moved to your training folder!")
        
    except Exception as e:
        print(f"❌ Error during sync: {e}")

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sync_labels()
