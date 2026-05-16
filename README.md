# PCB Defect Detection: MLOps Flywheel

This repository implements an end-to-end MLOps pipeline for automated PCB defect detection. The architecture integrates model training, data versioning, and automated reporting into a high-fidelity feedback loop.

---

## Infrastructure Overview

The pipeline utilizes a containerized infrastructure managed via Docker Compose:

*   **Label Studio**: Human-in-the-loop data annotation and active learning.
*   **MLflow**: Centralized experiment tracking and model registry.
*   **RustFS**: S3-compatible object storage serving as the DVC remote cache.
*   **DVC**: Data and model versioning with automated pipeline reproduction.
*   **CML**: Continuous Machine Learning for visual reporting in GitHub Pull Requests.

---

## Important: Infrastructure Prerequisite

This project uses a self-hosted CI/CD architecture. Before running any training or triggering a GitHub Action, you **MUST** ensure the local infrastructure is running on your Mac:

```bash
docker compose -f docker/docker-compose.yml up -d
```

The CI/CD pipeline performs a health check on ports **5556** (MLflow-Official) and **9000** (RustFS). If these services are not reachable, the pipeline will fail.

---

## Quick Start (Onboarding)

Follow these steps to initialize the environment and synchronize the latest model artifacts.

### 1. Environment Setup
```bash
# Clone the repository
git clone <repository_url>
cd pcb-defect-mlops-flywheel

# Configure environment variables
cp .env.example .env

# Initialize virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Synchronize data and models from remote storage
dvc pull
```

### 2. Launch Infrastructure
```bash
docker compose -f docker/docker-compose.yml up -d
```
Access points:
*   Label Studio: http://localhost:8080 (admin@example.com / mlops123)
*   MLflow: http://localhost:5555
*   RustFS Console: http://localhost:9001

---

## The MLOps Workflow

### 1. Local Development and Training
We use DVC to manage the reproduction of the training pipeline.

```bash
# Execute the training pipeline
dvc repro
```

**What happens behind the scenes:**
1.  **Data Prep**: Raw data is processed into YOLOv8 format.
2.  **Training**: The model is trained. Performance metrics are saved to `metrics.json`.
3.  **Logging**: The script logs all parameters, metrics, and visual artifacts to MLflow.
4.  **Mapping**: A specialized metadata file, `mlflow_run.txt`, is generated to map the current Git/DVC state to the specific MLflow Run ID.

### 2. Versioning and Promotion
To share results with the team and activate CI/CD validation:

```bash
# Snapshot the results and push to remote storage
dvc push
git add dvc.lock mlflow_run.txt metrics.json
git commit -m "feat: optimize hyperparameters for pcb-detection"
git push origin <branch_name>
```

---

## CI/CD Validation Architecture

Our GitHub Actions pipeline follows a "Pure Promotion" model to ensure absolute traceability and computational efficiency.

### Dual-Mode Execution
The CI/CD runner executes `dvc repro` in a clean environment:

1.  **Cached Mode (Success Path)**: If you have already trained and pushed the results locally, DVC identifies that the code and parameters have not changed. It skips the training process and simply downloads your `runs/` folder and `mlflow_run.txt` from RustFS.
2.  **Validation Mode**: The pipeline reads the downloaded `mlflow_run.txt` and uses its metadata to generate a deep-linked report in your Pull Request.

### Visual Reporting
Every Pull Request automatically receives a performance report containing:
*   **Metrics vs Main**: A side-by-side comparison of accuracy against the production baseline.
*   **Deep Links**: Direct URLs to the original MLflow experiment and specific run.
*   **Visual Analysis**: Loss curves, F1-curves, and confusion matrices embedded directly in the PR comment.

---

## Tech Stack
*   **Detection**: YOLOv8 (Ultralytics)
*   **Orchestration**: Docker Compose, GitHub Actions
*   **Storage**: DVC, RustFS (S3)
*   **Tracking**: MLflow
*   **Reporting**: CML (Iterative)
