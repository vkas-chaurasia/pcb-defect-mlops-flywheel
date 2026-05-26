# PCB Defect Detection: The MLOps Flywheel

This repository implements a production-grade MLOps ecosystem for automated PCB (Printed Circuit Board) defect detection. By integrating real-time inference, human-in-the-loop annotation, continuous monitoring, and automated CI/CD auditing, this project creates a "flywheel" effect—where every run makes the model smarter and the pipeline more robust.

---

## The Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Detection** | [YOLOv8](https://ultralytics.com/) | State-of-the-art object detection for 6 defect types. |
| **Model Format** | [ONNX](https://onnx.ai/) | Universal, hardware-agnostic computation graph (resolves Mac vs Linux inference bugs). |
| **Inference API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance model serving (Port 8000) with integrated prediction logging. |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive sandbox for real-time defect analysis (Port 8501). |
| **Orchestration** | [Docker Compose](https://www.docker.com/) | Unified management of all infrastructure services. |
| **Workflow / Monitoring**| [Airflow](https://airflow.apache.org/) | Workflow orchestration (Port 8085), automating label synchronization and scheduled data drift monitoring. |
| **Drift Monitoring** | [Evidently AI](https://www.evidentlyai.com/) | Live data drift evaluation & dashboard service (Port 8005). |
| **Tracking** | [MLflow](https://mlflow.org/) | Dual-instance tracking for Sandbox (5555) and Official (5556) runs. |
| **Annotation** | [Label Studio](https://labelstud.io/) | Active learning and dataset refinement (Port 8080). |
| **Versioning** | [DVC](https://dvc.org/) | Large data and model versioning with S3-compatible backends. |
| **Storage** | [RustFS](https://github.com/cloud-native-ml/rustfs) | Local S3-compatible object storage for model vaults. |
| **Reporting** | [CML](https://cml.dev/) | Automated performance reporting in GitHub PRs. |

---

## System Architecture

Two-pipeline design: the **CI/CD training loop** (top) continuously produces better models; the **serving and monitoring loop** (bottom) runs them in production. The flywheel closes when drift triggers retraining.

```mermaid
%%{init: {
  'theme': 'base',
  'themeVariables': {
    'fontSize': '14px',
    'fontFamily': 'Inter'
  }
}}%%
graph LR

classDef infra fill:#1C2B3A,stroke:#0d1821,color:#fff,font-weight:bold
classDef ml fill:#1A4731,stroke:#0d2419,color:#fff,font-weight:bold
classDef registry fill:#1E6B3C,stroke:#0f3a21,color:#fff,font-weight:bold
classDef decision fill:#C47A1E,stroke:#8B5615,color:#fff,font-weight:bold
classDef human fill:#FFD700,stroke:#8B6914,color:#1a1a1a,font-weight:bold

A["1. Training Pipeline\nGit · DVC · YOLOv8\nGitHub Actions · CML"]:::ml
H_PR["HUMAN APPROVAL\nReview CML report\n& merge PR"]:::human
B["2. MLflow Registry\nTracking · Versioning"]:::registry
C["3. Governance Evaluation\nAirflow model_governance_eval"]:::infra
H_GOV["HUMAN APPROVAL\nGovernance PR\n→ promote @champion"]:::human
E["4. Production Serving\nFastAPI · ONNX · Streamlit"]:::infra
F{"5. Drift Detected?\nAirflow + Evidently AI\n① sync_labels  ② drift_monitor"}:::decision
H_RETRAIN["HUMAN APPROVAL\nReview retraining PR\n& merge"]:::human
G["6. Label Studio\nAnnotation"]:::ml
H_ANNOT["HUMAN\nAnnotate uncertain\ndefect images"]:::human

A -->|train + validate| H_PR
H_PR -->|"merge to main\nregister @staging"| B
B -->|trigger governance| C
C -->|"eval passed · open PR"| H_GOV
H_GOV -.->|set @champion alias| B
B -.->|hot reload| E
E -->|prediction_log.csv| F
F -->|No — keep serving| E
F -->|"Yes — open retraining PR"| H_RETRAIN
H_RETRAIN -.->|merge triggers CI/CD| A
E -->|"defect detected\nbut low confidence"| G
G --> H_ANNOT
H_ANNOT -.->|"Airflow sync_labels\n→ data/raw"| A
```

---

## Onboarding: Quick Start

### 1. Clone the Repository
```bash
git clone <repository_url>
cd pcb-defect-mlops-flywheel
```

### 2. Infrastructure Prerequisite (Docker)
This project uses a self-hosted CI/CD architecture. Before running any local scripts or triggering CI/CD, the environment must be configured and Docker Desktop must be running:

1. **Configure `.env`**: Ensure a `.env` file is present in the root directory and contains the `GITHUB_TOKEN` and `GITHUB_REPO` variables (required for Airflow to open Pull Requests).
2. **Launch the Infrastructure**:
```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d
```
*Note: The self-hosted CI/CD runner communicates with these containers via localhost. If they are not running, the pipeline will fail its health checks.*

### 3. Access the Docker Infrastructure
The following core services are now running via Docker:
- **Airflow UI**: http://localhost:8085 (admin / admin)
- **MLflow (Sandbox)**: http://localhost:5555
- **MLflow (Official)**: http://localhost:5556
- **Label Studio**: http://localhost:8080 (Admin: admin@example.com / mlops123)
- **Evidently UI**: http://localhost:8005
- **RustFS S3 Console**: http://localhost:9001 (rustfsadmin / rustfsadmin)

*(Note: The FastAPI Server and Streamlit UI are run locally via Python scripts later in the workflow, not via Docker).*

### 4. Local Environment Setup
Initialize Python environment and download data:
```bash
# Initialize Virtual Environment & Install Dependencies (using UV)
uv sync
# (Optional) To manually activate the shell environment: source .venv/bin/activate

```

### 5. Start Local Python Services
The FastAPI backend and Streamlit frontend must be started in separate terminals:

```bash
# 1. Start FastAPI Inference Server
uv run python -m src.serving.serve
# API Docs available at: http://localhost:8000/docs

# 2. Start Streamlit Frontend (In a new terminal)
uv run python -m streamlit run src/app/main.py
# Sandbox UI available at: http://localhost:8501
```

### 6. Starting the Self-Hosted Runner (For CI/CD)
To use the automated GitHub CI/CD pipelines, a local runner is required:
1. Go to the GitHub repository -> **Settings** -> **Actions** -> **Runners**.
2. Click **New self-hosted runner** and follow the instructions to download and configure it.
3. Start the runner in a separate terminal:
   ```bash
   ./run.sh
   ```

---

## The Complete MLOps Flywheel Workflow

### Phase 1: Training and Validation
Pull the raw data from local S3 storage:
```bash
dvc pull
```

**Execution Strategies:**
- **Fast Iteration (Flexible)**: Developers can manually run `uv run python src/training/train.py` for quick debugging. This bypasses DVC and logs directly to the local Sandbox MLflow (Port 5555).
- **Pipeline Verification (Recommended)**: Before opening a Pull Request, verify your `params.yaml` is set correctly, then run `dvc repro` locally. Since CI/CD relies strictly on `params.yaml`, any command-line flags you used during fast iteration will be ignored by the CI pipeline! 
- **Adding New Data (Strict)**: If you added new images, you **must** update the dataset pointer before pushing:
  ```bash
  dvc commit data/raw
  dvc push
  ```

```bash
# Execute the strict pipeline before committing
dvc repro

# Commit the updated contract!
git add dvc.lock params.yaml
```

- **Local Dev**: Logs results to Port 5555 (Local Sandbox MLflow).
- **CI/CD Retraining**: Logs results to Port 5556 (Official Team MLflow). The retraining pipeline triggers automatically whenever a Pull Request is opened or updated.
  The CI pipeline dynamically resolves the branch, runs `dvc repro`, and posts a visual report (Confusion Matrix, F1-Curves) as a PR comment.
- **Merge to Main**: When the PR is merged to `main`, the CI pipeline automatically exports the model to the **ONNX format**, registers the `best.onnx` artifact to the MLflow Model Registry, and applies the `@staging` alias. This guarantees the model will run flawlessly on any CPU architecture.

### Phase 2: Model Governance & GitOps Promotion
Before a model reaches production, it must pass governance:
- **Automated Evaluation**: The Airflow `model_evaluation_dag` fetches the `@staging` model from MLflow and runs simulated tests.
- **Governance PR**: If the model passes, Airflow automatically creates a Pull Request updating the `deployment-repo/production_version.json` file.
- **Promotion to Champion**: Once a human approves and merges the Governance PR, a GitHub Action automatically updates the MLflow Registry, promoting the model to `@champion`.

### Phase 3: Serving and Monitoring
Deploy the champion model to the FastAPI server for real-time inference. The server continuously logs prediction metrics to `monitoring/prediction_log.csv` for drift analysis.
```bash
# Serves the champion model from the MLflow registry
uv run python -m src.serving.serve
```
- **Data Drift Detection**: Data drift is calculated using Evidently AI.
- **Airflow Automation**: The `data_sync_and_drift_check_dag` runs daily — it first syncs any completed annotations from Label Studio, then analyzes live FastAPI prediction logs against the baseline using Evidently AI. If significant drift is detected, Airflow automatically opens a retraining Pull Request via the GitHub API.

### Phase 4: Active Learning and Refinement
FastAPI automatically routes uncertain predictions to Label Studio for human review:
1. **Automatic routing**: When FastAPI detects defects with low confidence, it saves those images to `data/raw/unseen_simulation/`. Label Studio reads them directly via FastAPI's static file server — no manual upload needed.
2. **Human annotation**: Open Label Studio and annotate the queued images with defect bounding boxes.
3. **Automated sync**: The Airflow `data_sync_and_drift_check_dag` runs daily — it pulls completed annotations from Label Studio via the API, converts them to YOLO format, and writes them to `data/raw/active_learning/`.
4. **Trigger retraining**: Airflow opens a retraining PR automatically when drift is detected. A developer can also open a PR manually after versioning new data:
   ```bash
   dvc add data/raw
   dvc push
   git add data/raw.dvc
   git commit -m "data: add new annotated samples"
   git push origin <branch>
   ```

---

## Repository Structure
- `src/app/`: Streamlit dashboard for real-time detection.
- `src/serving/`: FastAPI inference service logic and prediction logger.
- `src/training/`: YOLOv8 training and evaluation scripts.
- `src/monitoring/`: Data drift detection, monitoring scripts, and baseline generation using Evidently AI.
- `src/utils/`: Label Studio sync and batch inference helpers.
- `dags/`: Airflow DAGs for drift monitoring and synchronization tasks.
- `docker/`: Unified Docker Compose configuration (FastAPI, MLflow, Label Studio, RustFS, Airflow).
- `mlflow-official-history/`: Committed history of team-verified runs.
- `mlflow-history/`: Private local sandbox history (ignored by Git).

---

## Best Practices
- **Never Commit data/raw**: Always use `dvc push` to store large images in RustFS.
- **Official Runs**: Only the CI/CD pipeline should log to Port 5556 to keep the "Official Showroom" clean.
- **PR Reports**: Always review the CML report in the Pull Request before merging to main.

---
**Developed for the ZHAW MLOps Course (Spring 2026)**