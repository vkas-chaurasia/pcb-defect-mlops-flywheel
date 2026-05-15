import json
import os
from pathlib import Path
import mlflow

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pcb-defect-production")

def find_latest_run_folder(base_path):
    """Finds the most recently modified subdirectory in the YOLO runs folder."""
    p = Path(base_path)
    if not p.exists(): return None
    subdirs = [d for d in p.iterdir() if d.is_dir() and d.name != "weights"]
    if not subdirs: return None
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0]

def generate_report():
    print(f"📡 API Auditor Active. Searching Experiment: {EXP_NAME}...")
    
    # 1. Connect to MLflow API
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name(EXP_NAME)
    if not exp:
        print(f"Error: Experiment {EXP_NAME} not found.")
        return

    # 2. Search for the EXACT run tied to this specific GitHub Job (DNA Match)
    github_run_id = os.getenv("GITHUB_RUN_ID", "local")
    filter_string = f"tags.github_run_id = '{github_run_id}'"
    
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id], 
        filter_string=filter_string, 
        max_results=1
    )
    
    if runs.empty:
        print(f"Error: No run found with tag github_run_id='{github_run_id}' in {EXP_NAME}. Did training fail?")
        return
        
    latest_run_data = runs.iloc[0]
    run_id = latest_run_data.run_id
    print(f"🎯 DNA Match Found! Official Run: {run_id}")

    # 3. Find the local artifact path for PR visualization
    base_run_path = "runs/detect/pcb-defect-detection"
    latest_folder = find_latest_run_folder(base_run_path)
    if not latest_folder:
        print(f"Error: No artifacts found in {base_run_path}")
        return

    # 4. Export the Metadata for GitHub Actions (This file is ONLY for CML, not tracked by DVC)
    run_url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run_id}"
    exp_url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}"
    
    with open("mlflow_run_official.txt", "w") as f:
        f.write(f"RUN_ID={run_id}\n")
        f.write(f"EXP_ID={exp.experiment_id}\n")
        f.write(f"RUN_URL={run_url}\n")
        f.write(f"EXP_URL={exp_url}\n")
        f.write(f"RUN_PATH={latest_folder}\n")

    print(f"Success. Metadata prepared for GitHub PR: {run_url}")

if __name__ == "__main__":
    generate_report()
