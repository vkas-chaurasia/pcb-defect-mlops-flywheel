import argparse
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="monitoring/reference_predictions.csv")
    parser.add_argument("--curr", default="monitoring/prediction_log.csv")
    args = parser.parse_args()

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    if not Path(args.ref).exists() or not Path(args.curr).exists():
        print(f"Missing reference ({args.ref}) or current logs ({args.curr}).")
        return

    ref_df = pd.read_csv(args.ref)
    curr_df = pd.read_csv(args.curr)

    if len(curr_df) < 50:
        print(f"Not enough data to check drift ({len(curr_df)} < 50).")
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
            project.save()

        # Run the Data Drift Report on confidence and bbox area only.
        # num_detections is excluded — it reflects actual defect rate, not distribution shift.
        features = ["avg_confidence", "avg_bbox_area", "pass_fail"]
        ref_data = ref_df[features].dropna()
        curr_data = curr_window_df[features].dropna()

        evidently_report = Report(metrics=[DataDriftPreset()])
        snapshot = evidently_report.run(reference_data=ref_data, current_data=curr_data)

        # Save to Workspace
        workspace.add_run(project.id, snapshot)
        print(f"Evidently report registered in workspace: {workspace_path}")

        # Save a local HTML copy
        evidently_report_path = reports_dir / "evidently_drift_report.html"
        snapshot.save_html(str(evidently_report_path))
        print(f"Saved Evidently AI HTML report to {evidently_report_path}")

        # Extract drift status from DriftedColumnsCount (Evidently 0.7.x API)
        # Note: top-level "type" is "CountValue"; DriftedColumnsCount is in nested params
        for result in snapshot.dump_dict().get("metric_results", {}).values():
            params_type = result.get("metric_value_location", {}).get("metric", {}).get("params", {}).get("type", "")
            if "DriftedColumnsCount" in params_type:
                share = result.get("share", {}).get("value", 0)
                evidently_drift_detected = share >= 0.5
                break
        print(f"Evidently AI dataset_drift result: {evidently_drift_detected}")
    except Exception as e:
        print(f"Evidently AI failed to run or parse: {e}.")

    drift_detected = evidently_drift_detected
    print(f"Drift check complete. Drift detected: {drift_detected}")

    results_json = {
        "metrics": [{"result": {"dataset_drift": drift_detected}}],
        "checked_at": datetime.now().isoformat()
    }

    with open(reports_dir / "drift_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"Saved drift results JSON to {reports_dir}")
    if drift_detected:
        print("Drift detected. Review the Evidently UI at http://localhost:8005 and open a retraining PR when ready.")

if __name__ == "__main__":
    main()
