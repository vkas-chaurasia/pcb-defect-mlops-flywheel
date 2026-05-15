import json
import os
from pathlib import Path
import mlflow

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pcb-defect-production")

def find_latest_run(base_path):
    """Finds the most recently modified subdirectory in the YOLO runs folder."""
    p = Path(base_path)
    if not p.exists():
        return None
    subdirs = [d for d in p.iterdir() if d.is_dir() and d.name != "weights"]
    if not subdirs:
        return None
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0]

def generate_report():
    print(f"Reporting to GitHub for Official Experiment: {EXP_NAME}...")
    
    # 1. Read the fresh metadata produced by train.py
    if not Path("mlflow_run.txt").exists():
        print("Error: mlflow_run.txt not found. Did the forced training run fail?")
        return
        
    with open("mlflow_run.txt", "r") as f:
        lines = {line.split('=')[0]: line.split('=')[1].strip() for line in f if '=' in line}
        
    # 2. Find the local artifact path for PR visualization
    base_run_path = "runs/detect/pcb-defect-detection"
    latest_run = find_latest_run(base_run_path)
    
    if not latest_run:
        print(f"Error: No artifacts found in {base_run_path}")
        return

    # 3. Finalize the official metadata for CML
    # We use the existing RUN_ID and URLs that autolog just created
    with open("mlflow_run_official.txt", "w") as f:
        f.write(f"RUN_ID={lines['RUN_ID']}\n")
        f.write(f"EXP_ID={lines['EXP_ID']}\n")
        f.write(f"RUN_URL={lines['RUN_URL']}\n")
        f.write(f"EXP_URL={lines['EXP_URL']}\n")
        f.write(f"RUN_PATH={latest_run}\n")

    print(f"Official Metadata Ready: {lines['RUN_URL']}")

if __name__ == "__main__":
    generate_report()
