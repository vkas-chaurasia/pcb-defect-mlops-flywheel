# PCB Defect Detection: The MLOps Flywheel

This repository implements a production-grade MLOps ecosystem for automated PCB (Printed Circuit Board) defect detection. By integrating real-time inference, human-in-the-loop annotation, continuous monitoring, and automated CI/CD auditing, this project creates a "flywheel" effect—where every run makes the model smarter and the pipeline more robust.

---

## The Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Detection** | [YOLOv8](https://ultralytics.com/) | State-of-the-art object detection for 6 defect types. |
| **Inference API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance model serving (Port 8000) with integrated prediction logging. |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive sandbox for real-time defect analysis (Port 8501). |
| **Orchestration** | [Docker Compose](https://www.docker.com/) | Unified management of all infrastructure services. |
| **Workflow / Monitoring**| [Airflow](https://airflow.apache.org/) | Workflow orchestration (Port 8081), automating label synchronization and scheduled data drift monitoring. |
| **Tracking** | [MLflow](https://mlflow.org/) | Dual-instance tracking for Sandbox (5555) and Official (5556) runs. |
| **Annotation** | [Label Studio](https://labelstud.io/) | Active learning and dataset refinement (Port 8080). |
| **Versioning** | [DVC](https://dvc.org/) | Large data and model versioning with S3-compatible backends. |
| **Storage** | [RustFS](https://github.com/cloud-native-ml/rustfs) | Local S3-compatible object storage for model vaults. |
| **Reporting** | [CML](https://cml.dev/) | Automated performance reporting in GitHub PRs. |

---

## Important: Infrastructure Prerequisite

This project uses a self-hosted CI/CD architecture. Before running any scripts or triggering CI/CD, you MUST have the Docker services running on the host machine:

```bash
# Launch the Flywheel Infrastructure
docker compose -f docker/docker-compose.yml up -d
```

*Note: The self-hosted CI/CD runner communicates with these containers via localhost. If they are not running, the pipeline will fail its health checks.*

### Starting the Self-Hosted Runner
If you plan to test or run the automated GitHub CI/CD pipelines (e.g., triggering `/train` on a PR), you must also have a local GitHub Actions runner active:
1. Go to your GitHub repository -> **Settings** -> **Actions** -> **Runners**.
2. Click **New self-hosted runner** and follow the instructions to download and configure it.
3. Start the runner in a separate terminal:
   ```bash
   ./run.sh
   ```

---

## Onboarding: Quick Start

### 1. Environment Setup
```bash
# Clone and Enter
git clone <repository_url>
cd pcb-defect-mlops-flywheel

# Initialize Virtual Environment (using UV or Pip)
uv sync  # Recommended
# OR: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Synchronize Data & Models
dvc pull
```

### 2. Access the Ecosystem
Once Docker is up, your ecosystem is live at:
- **FastAPI Docs**: http://localhost:8000/docs
- **Streamlit UI**: http://localhost:8501
- **Airflow UI**: http://localhost:8081 (admin / admin)
- **MLflow (Sandbox)**: http://localhost:5555
- **MLflow (Official)**: http://localhost:5556
- **Label Studio**: http://localhost:8080 (Admin: admin@example.com / mlops123)

---

## The Complete MLOps Flywheel Workflow

### Phase 1: Training and Validation
Execute the pipeline with DVC to establish your baseline model. This ensures every run is reproducible and tracked.
```bash
dvc repro
```
- **Local Dev**: Logs results to Port 5555.
- **CI/CD Retraining**: The retraining pipeline is designed for **Human-in-the-Loop** validation. It triggers when:
  1. A Pull Request is **Approved** via GitHub review.
  2. A repository member comments **`/train`** on a Pull Request.
  The CI pipeline resolves the target branch dynamically, runs `dvc repro`, updates the model and reference baselines, and pushes the new artifacts back to the PR branch. A visual report (Confusion Matrix, F1-Curves) is posted as a PR comment.

### Phase 2: Serving and Monitoring
Deploy your champion model to the FastAPI server for real-time inference. The server continuously logs prediction confidence and metrics for drift analysis.
```bash
# Serves the champion model from local weights or MLflow registry
python src/serving/serve.py --weights models/best.pt
```
- **Data Drift Detection**: Data drift is calculated using Population Stability Index (PSI). 
  - *Note for macOS users*: The drift calculation is implemented using pure Python/NumPy to strictly avoid `scipy.linalg` (or `evidently`) thread-lock/deadlock issues on Apple Silicon.
- **Airflow Automation**: The `drift_monitoring_dag` runs continuously in Airflow to analyze live FastAPI prediction logs against the model baseline. If significant drift is detected, it flags the system for retraining.

### Phase 3: Active Learning and Refinement
Once the model is serving and monitored, you can close the loop:
1. Upload new "unseen" or drifted PCB images to Label Studio.
2. Use the Active Learning Loop in the Streamlit UI to trigger batch inference and identify high-uncertainty cases.
3. Sync refined labels back to the repo (automated via Airflow or manually via `src/utils/sync_labels.py`).
4. Return to Phase 1 (Trigger `/train` on a PR to incorporate new data).

---

## Repository Structure
- `src/app/`: Streamlit dashboard for real-time detection.
- `src/serving/`: FastAPI inference service logic and prediction logger.
- `src/training/`: YOLOv8 training and evaluation scripts.
- `src/monitoring/`: Data drift detection, baseline generation, and pure-Python PSI implementation.
- `src/utils/`: Label Studio sync and batch inference helpers.
- `dags/`: Airflow DAGs for drift monitoring and synchronization tasks.
- `docker/`: Unified Docker Compose configuration (FastAPI, MLflow, Label Studio, RustFS, Airflow).
- `mlflow-official-history/`: Committed history of team-verified runs.
- `mlflow-history/`: Private local sandbox history (ignored by Git).

---

## Best Practices
- **Never Commit data/raw**: Always use `dvc push` to store large images in RustFS.
- **Official Runs**: Only the CI/CD pipeline should log to Port 5556 to keep the "Official Showroom" clean.
- **PR Reports**: Always review the CML report in your Pull Request before merging to main.
- **Model Approvals**: Model retraining in CI only runs on explicit approval or a `/train` trigger comment to avoid runaway compute costs.

---
**Developed for the ZHAW MLOps Course (Spring 2026)**
🏆 *Stabilizing the Flywheel, one PCB at a time.*
