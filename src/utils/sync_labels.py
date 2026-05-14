import os
import requests
import yaml
from label_studio_sdk import Client
from dotenv import load_dotenv

load_dotenv()

# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Label Studio Credentials (defaults for local setup)
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
LS_API_KEY = os.getenv("LABEL_STUDIO_API_KEY", "your_api_key_here")
PROJECT_ID = os.getenv("LABEL_STUDIO_PROJECT_ID", "1")

RAW_DATA_DIR = "data/raw"

def sync_from_label_studio():
    """
    Connects to Label Studio, finds completed tasks, and downloads
    them into the project's raw data folder to trigger the flywheel.
    """
    print(f"Connecting to Label Studio at {LS_URL}...")
    
    try:
        ls = Client(url=LS_URL, api_key=LS_API_KEY)
        project = ls.get_project(id=PROJECT_ID)
        
        # Get all tasks with annotations
        tasks = project.get_tasks()
        exported_count = 0
        
        for task in tasks:
            if task.get('annotations'):
                # Get the latest annotation
                annotation = task['annotations'][0]
                result = annotation.get('result', [])
                
                if not result:
                    continue
                
                # Check if it was labeled as a 'Defect'
                label = result[0]['value']['choices'][0]
                is_defect = "Defect" in label
                
                # Get image URL and filename
                image_url = task['data']['image']
                filename = os.path.basename(image_url)
                
                # Add 'defect' prefix if applicable to help the dataset class
                prefix = "defect_" if is_defect else "no_defect_"
                save_name = prefix + filename
                save_path = os.path.join(RAW_DATA_DIR, save_name)
                
                # Download the image
                if not os.path.exists(save_path):
                    print(f"Downloading new labeled image: {save_name}")
                    # Note: You might need to pass headers if LS is authenticated
                    r = requests.get(image_url)
                    with open(save_path, 'wb') as f:
                        f.write(r.content)
                    exported_count += 1
        
        print(f"Sync complete. Added {exported_count} new samples to {RAW_DATA_DIR}.")
        
    except Exception as e:
        print(f"Error syncing from Label Studio: {e}")
        print("💡 Simulation Tip: In a demo, you can manually move images into 'data/raw' to simulate this sync.")

if __name__ == "__main__":
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    sync_from_label_studio()
