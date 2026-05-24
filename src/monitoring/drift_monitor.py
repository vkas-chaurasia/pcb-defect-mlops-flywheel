import argparse
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def calculate_psi(expected, actual, num_bins=10):
    """
    Calculate PSI (Population Stability Index) between expected (reference)
    and actual (current) datasets.
    """
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    unique_vals = np.unique(expected)
    if len(unique_vals) <= num_bins:
        # Use unique values as bins for discrete variables
        bins = np.sort(unique_vals)
        bins = np.append(bins, bins[-1] + 1e-5)
    else:
        # Quantile binning
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(expected, percentiles)
        bins = np.unique(bins)
        if len(bins) < 2:
            # Fallback to equal width
            bins = np.linspace(expected.min(), expected.max() + 1e-5, num_bins + 1)
            
    # Calculate frequencies in bins
    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)
    
    # Convert to proportions (percentages)
    expected_prop = expected_counts / len(expected)
    actual_prop = actual_counts / len(actual)
    
    # Handle zero counts using a small epsilon
    eps = 1e-4
    expected_prop = np.where(expected_prop == 0, eps, expected_prop)
    actual_prop = np.where(actual_prop == 0, eps, actual_prop)
    
    # Calculate PSI
    psi_value = np.sum((actual_prop - expected_prop) * np.log(actual_prop / expected_prop))
    return float(psi_value)

def get_status_label(psi):
    if psi < 0.1:
        return "Stable", "badge-stable"
    elif psi < 0.25:
        return "Moderate Shift", "badge-moderate"
    else:
        return "Drift Detected", "badge-drift"

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
    
    # Calculate PSI for key metrics
    psi_conf = calculate_psi(ref_df["avg_confidence"].values, curr_window_df["avg_confidence"].values)
    psi_detections = calculate_psi(ref_df["num_detections"].values, curr_window_df["num_detections"].values)
    
    # Define drift if any key metric exceeds threshold
    drift_detected = (psi_conf >= 0.25) or (psi_detections >= 0.25)
    
    status_conf, badge_conf = get_status_label(psi_conf)
    status_detections, badge_detections = get_status_label(psi_detections)
    
    # Compute descriptive metrics
    metrics_summary = {
        "avg_confidence": {
            "ref_mean": float(ref_df["avg_confidence"].mean()),
            "curr_mean": float(curr_window_df["avg_confidence"].mean()),
            "psi": psi_conf,
            "status": status_conf,
            "badge_class": badge_conf
        },
        "num_detections": {
            "ref_mean": float(ref_df["num_detections"].mean()),
            "curr_mean": float(curr_window_df["num_detections"].mean()),
            "psi": psi_detections,
            "status": status_detections,
            "badge_class": badge_detections
        }
    }
    
    print(f"Drift check complete. Drift detected: {drift_detected}")
    print(f"PSI (avg_confidence): {psi_conf:.4f} ({status_conf})")
    print(f"PSI (num_detections): {psi_detections:.4f} ({status_detections})")
    
    # Save JSON results compatible with pipeline consumers
    results_json = {
        "metrics": [
            {
                "result": {
                    "dataset_drift": drift_detected
                }
            }
        ],
        "psi_scores": {
            "avg_confidence": round(psi_conf, 4),
            "num_detections": round(psi_detections, 4)
        },
        "summary": metrics_summary,
        "checked_at": datetime.now().isoformat()
    }
    
    with open(reports_dir / "drift_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
        
    # Generate a gorgeous HTML report
    banner_class = "banner-drift" if drift_detected else "banner-stable"
    banner_text = "🔄 Drift Detected - Retraining Recommended" if drift_detected else "✅ System Stable - No retrain required"
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PCB Defect Detection - Drift Report</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border: #334155;
        }}
        body {{
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 2rem;
            line-height: 1.5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1.5rem;
        }}
        h1 {{
            margin: 0;
            font-size: 2.2rem;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{
            color: var(--text-muted);
            margin: 0.5rem 0 0 0;
            font-size: 1rem;
        }}
        .banner {{
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            font-weight: 600;
            font-size: 1.2rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        }}
        .banner-drift {{
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(245, 158, 11, 0.15));
            border: 1px solid var(--danger);
            color: #fca5a5;
        }}
        .banner-stable {{
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(99, 102, 241, 0.15));
            border: 1px solid var(--success);
            color: #a7f3d0;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
            margin-bottom: 2rem;
        }}
        .card h2 {{
            margin-top: 0;
            font-size: 1.2rem;
            color: var(--text-main);
            border-bottom: 1px solid var(--border);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            text-align: left;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            color: var(--text-muted);
            font-weight: 500;
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.6rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-stable {{
            background-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .badge-moderate {{
            background-color: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        .badge-drift {{
            background-color: rgba(239, 68, 68, 0.2);
            color: #fca5a5;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        .footer {{
            margin-top: 3rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PCB Defect Detection - Drift Report</h1>
            <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="banner {banner_class}">
            {banner_text}
        </div>
        
        <div class="card">
            <h2>Data Drift Metrics (PSI Analysis)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Reference Mean</th>
                        <th>Current Window Mean</th>
                        <th>PSI Score</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Average Confidence</strong></td>
                        <td>{metrics_summary['avg_confidence']['ref_mean']:.4f}</td>
                        <td>{metrics_summary['avg_confidence']['curr_mean']:.4f}</td>
                        <td>{metrics_summary['avg_confidence']['psi']:.4f}</td>
                        <td><span class="badge {metrics_summary['avg_confidence']['badge_class']}">{metrics_summary['avg_confidence']['status']}</span></td>
                    </tr>
                    <tr>
                        <td><strong>Average Detections</strong></td>
                        <td>{metrics_summary['num_detections']['ref_mean']:.4f}</td>
                        <td>{metrics_summary['num_detections']['curr_mean']:.4f}</td>
                        <td>{metrics_summary['num_detections']['psi']:.4f}</td>
                        <td><span class="badge {metrics_summary['num_detections']['badge_class']}">{metrics_summary['num_detections']['status']}</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
        
        <div class="card">
            <h2>Drift Threshold Reference</h2>
            <ul>
                <li><strong>PSI &lt; 0.10:</strong> Stable distribution / No drift.</li>
                <li><strong>0.10 &le; PSI &lt; 0.25:</strong> Moderate shift detected / Monitor closely.</li>
                <li><strong>PSI &ge; 0.25:</strong> Significant distribution drift / Action (Retraining) required.</li>
            </ul>
        </div>
        
        <div class="footer">
            PCB Defect Detection System &bull; Spring 2026 &bull; MLOps Monitoring Loop
        </div>
    </div>
</body>
</html>
"""
    
    with open(reports_dir / "drift_report.html", "w") as f:
        f.write(html_content)
        
    print(f"Saved drift reports to {reports_dir}")
    
    if drift_detected and args.auto_pr:
        token = os.getenv("GITHUB_TOKEN")
        repo = os.getenv("GITHUB_REPO")
        if not token or not repo:
            print("GITHUB_TOKEN or GITHUB_REPO missing. Cannot create PR.")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"retrain/drift-{timestamp}"
        
        # Git and DVC actions
        print("Registering drift data changes via DVC and pushing to Git...")
        os.system("dvc add data/raw && dvc push")
        os.system(f"git checkout -b {branch_name}")
        os.system("git add data/raw.dvc reports/drift_report.html reports/drift_results.json")
        os.system(f"git commit -m 'chore: drift detected, syncing data'")
        os.system(f"git push origin {branch_name}")
        
        pr_body = (
            "## 🔄 Drift Detected - Retraining Required\n\n"
            "Data drift has been detected in production predictions using Population Stability Index (PSI).\n"
            "The updated data has been synced and versioned with DVC.\n\n"
            "### Drift Summary\n"
            f"- **Average Confidence PSI:** {psi_conf:.4f} ({status_conf})\n"
            f"- **Average Detections PSI:** {psi_detections:.4f} ({status_detections})\n\n"
            "Please review the attached drift reports before approving retraining.\n\n"
            "**To trigger training on this branch, approve this PR or comment `/train`.**"
        )
        
        import requests
        url = f"https://api.github.com/repos/{repo}/pulls"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
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
