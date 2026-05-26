import argparse
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="monitoring/reference_predictions.csv")
    parser.add_argument("--curr", default="monitoring/prediction_log.csv")
    parser.add_argument("--auto-pr", action="store_true")
    args = parser.parse_args()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    if not Path(args.ref).exists() or not Path(args.curr).exists():
        print(f"Missing reference ({args.ref}) or current logs ({args.curr}).")
        return

    ref_df = pd.read_csv(args.ref)
    curr_df = pd.read_csv(args.curr)
    
    if len(curr_df) < 5:
        print(f"Not enough data to check drift ({len(curr_df)} < 5).")
        return
        
    # Take the last 100 rows for current window
    curr_window_df = curr_df.tail(100)
    
    # ---------------------------------------------------------
    # EVIDENTLY AI DRIFT DETECTION & WORKSPACE REGISTRATION
    # ---------------------------------------------------------
    print("Running Evidently AI Data Drift Report...")
    evidently_drift_detected = False
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        from evidently.ui.workspace import Workspace
        
        # Initialize or load Workspace
        workspace_path = "monitoring/evidently_workspace"
        try:
            workspace = Workspace.create(workspace_path)
        except Exception:
            workspace = Workspace(workspace_path)
            
        # Find or create project
        project = None
        for p in workspace.list_projects():
            if p.name == "PCB Defect Detection":
                project = p
                break
                
        if not project:
            project = workspace.create_project("PCB Defect Detection")
            project.description = "Production data drift monitoring for PCB YOLOv8 model."
            # Set up dashboard panels for UI
            from evidently.legacy.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, PanelValue, PlotType, ReportFilter
            project.dashboard.add_panel(
                DashboardPanelCounter(
                    title="PCB Defect Detection Monitoring",
                    filter=ReportFilter(metadata_values={}, tag_values=[]),
                    value=PanelValue(
                        metric_id="DatasetDriftMetric",
                        field_path="number_of_drifted_features",
                        legend="Drifted Features"
                    ),
                    size=1
                )
            )
            project.save()

        # Run the Data Drift Report
        features = ["avg_confidence", "num_detections", "avg_bbox_area"]
        ref_data = ref_df[features].dropna()
        curr_data = curr_window_df[features].dropna()
        
        evidently_report = Report(metrics=[DataDriftPreset()])
        snapshot = evidently_report.run(reference_data=ref_data, current_data=curr_data)
        
        # Save to Workspace
        workspace.add_run(project.id, snapshot)
        print(f"Evidently report registered in workspace: {workspace_path}")
        
        # Save a local HTML copy as well for reference
        evidently_report_path = reports_dir / "evidently_drift_report.html"
        snapshot.save_html(str(evidently_report_path))
        print(f"Saved Evidently AI HTML report to {evidently_report_path}")
        
        # Extract drift status from DriftedColumnsCount (Evidently 0.7.x API)
        for result in snapshot.dump_dict().get("metric_results", {}).values():
            if "DriftedColumnsCount" in result.get("type", ""):
                share = result.get("share", {}).get("value", 0)
                evidently_drift_detected = share >= 0.5
                break
        print(f"Evidently AI dataset_drift result: {evidently_drift_detected}")
    except Exception as e:
        print(f"⚠️ Evidently AI failed to run or parse: {e}.")

    drift_detected = evidently_drift_detected
    print(f"Drift check complete. Drift detected: {drift_detected}")
    
    # Save JSON results compatible with pipeline consumers
    results_json = {
        "metrics": [
            {
                "result": {
                    "dataset_drift": drift_detected
                }
            }
        ],
        "checked_at": datetime.now().isoformat()
    }
    
    with open(reports_dir / "drift_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
        
    print(f"Saved drift results JSON to {reports_dir}")
    
    if drift_detected and args.auto_pr:
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        if not token or not repo:
            print("GITHUB_TOKEN or GITHUB_REPO missing. Cannot create PR.")
            return

        import requests
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

        # Skip if an open drift PR already exists
        existing = requests.get(
            f"https://api.github.com/repos/{repo}/pulls",
            headers=headers,
            params={"state": "open", "base": "main"},
        )
        if existing.status_code == 200:
            open_drift_prs = [pr for pr in existing.json() if pr["head"]["ref"].startswith("retrain/drift-")]
            if open_drift_prs:
                print(f"Open drift PR already exists ({open_drift_prs[0]['html_url']}). Skipping PR creation.")
                return
        else:
            print(f"Warning: could not check existing PRs ({existing.status_code}). Proceeding with PR creation.")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"retrain/drift-{timestamp}"
        
        # Git and DVC actions
        print("Registering drift data changes via DVC and pushing to Git...")
        os.system("dvc add data/raw && dvc push")
        os.system(f"git checkout -b {branch_name}")
        os.system("git add data/raw.dvc reports/evidently_drift_report.html reports/drift_results.json")
        os.system(f"git commit -m 'chore: drift detected, syncing data'")
        os.system(f"git push origin {branch_name}")
        
        pr_body = (
            "## 🔄 Drift Detected - Retraining Required\n\n"
            "Data drift has been detected in production predictions using Evidently AI.\n"
            "The updated data has been synced and versioned with DVC.\n\n"
            "### Drift Summary\n"
            f"- **Evidently AI Dataset Drift:** {evidently_drift_detected}\n\n"
            "Please review the attached drift report (`reports/evidently_drift_report.html`) and check the Evidently AI UI dashboard before approving retraining.\n\n"
            "**To trigger training on this branch, approve this PR or comment `/train`.**"
        )
        
        url = f"https://api.github.com/repos/{repo}/pulls"
        data = {
            "title": f"🔄 Drift Detected - Retraining Required ({timestamp})",
            "head": branch_name,
            "base": "main",
            "body": pr_body
        }
        res = requests.post(url, headers=headers, json=data)
        if res.status_code == 201:
            print(f"PR created successfully: {res.json()['html_url']}")
        else:
            print(f"Failed to create PR: {res.status_code} - {res.text}")

if __name__ == "__main__":
    main()
