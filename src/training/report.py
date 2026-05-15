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
    print(f"API Auditor Active. Searching Experiment: {EXP_NAME}...")
    
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
        print(f"Error: No run found with tag github_run_id='{github_run_id}' in {EXP_NAME}.")
        return
        
    latest_run_data = runs.iloc[0]
    run_id = latest_run_data.run_id
    print(f"DNA Match Found! Official Run: {run_id}")

    # 3. Find the local artifact path for PR visualization
    base_run_path = "runs/detect/pcb-defect-detection"
    latest_folder = find_latest_run_folder(base_run_path)
    if not latest_folder:
        print(f"Error: No artifacts found in {base_run_path}")
        return

    # 4. Generate the Unified Markdown Report (report.md)
    # This file is consumed directly by CML
    run_url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}/runs/{run_id}"
    exp_url = f"{MLFLOW_URI}/#/experiments/{exp.experiment_id}"
    
    report_content = [
        "# MLOps Flywheel: Official Validation Report",
        "\nAn automated, system-verified training run has completed successfully.",
        "\n## Official Record",
        f"* **Experiment**: [View All Runs]({exp_url})",
        f"* **Validation Run**: [View Detailed Metrics & Weights]({run_url})",
        f"* **System Job ID**: `{github_run_id}`",
        "\n## Visual Evidence",
        f"![Results]({latest_folder}/results.png)",
        "\n## Model Performance Summary",
        "| Class | Images | Instances | Box(P) | R | mAP50 | mAP50-95 |",
        "|-------|--------|-----------|--------|---|-------|----------|"
    ]
    
    # Optionally add rows from metrics.json if needed, but YOLO results.png is usually better
    # For now, we'll keep it focused on the High-Fidelity evidence.
    
    with open("report.md", "w") as f:
        f.write("\n".join(report_content))

    print(f"Success. Unified Report generated: report.md")

if __name__ == "__main__":
    generate_report()
