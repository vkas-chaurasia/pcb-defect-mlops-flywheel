import json
import os
from pathlib import Path
import mlflow

# Configuration
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5555")
EXP_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pcb-defect-production") # Default to production in the auditor

def find_latest_run(base_path):
    """Finds the most recently modified subdirectory in the YOLO runs folder."""
    p = Path(base_path)
    if not p.exists():
        return None
    
    # Get all subdirectories, excluding weights/ and other internal folders
    subdirs = [d for d in p.iterdir() if d.is_dir() and d.name != "weights"]
    if not subdirs:
        return None
        
    # Sort by modification time (latest first)
    subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return subdirs[0]

def generate_report():
    print(f"Smart Auditor active. Target Experiment: {EXP_NAME}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXP_NAME)
    
    # 1. Discover the results DVC just processed
    # Path follows the YOLOv8 project structure we defined in train.py
    base_run_path = "runs/detect/pcb-defect-detection"
    latest_run = find_latest_run(base_run_path)
    
    if not latest_run:
        print(f"Error: No run artifacts found in {base_run_path}. Did training finish?")
        return
        
    print(f"Found latest artifacts in: {latest_run}")

    # 2. Read metrics (cached/reproduced by DVC)
    if not Path("metrics.json").exists():
        print("Error: metrics.json not found.")
        return
        
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    # 3. Create the Official Validation Run
    with mlflow.start_run(run_name=f"Official-Validation-{latest_run.name}") as run:
        run_id = run.info.run_id
        exp = mlflow.get_experiment_by_name(EXP_NAME)
        
        print(f"Logging Official Results to Run ID: {run_id}")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(latest_run))
        
        # 4. Export the Metadata for GitHub Actions
        run_url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run_id}"
        exp_url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}"
        
        with open("mlflow_run_official.txt", "w") as f:
            f.write(f"RUN_ID={run_id}\n")
            f.write(f"EXP_ID={exp.experiment_id}\n")
            f.write(f"RUN_URL={run_url}\n")
            f.write(f"EXP_URL={exp_url}\n")
            # Also save the path for CI/CD visualization script if needed
            f.write(f"RUN_PATH={latest_run}\n")

    print(f"Success. Official Report generated: {run_url}")

if __name__ == "__main__":
    generate_report()
