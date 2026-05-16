# 🔍 PCB Defect Detection: The MLOps Flywheel 🎡

This repository implements a production-grade MLOps ecosystem for automated PCB (Printed Circuit Board) defect detection. By integrating real-time inference, human-in-the-loop annotation, and automated CI/CD auditing, this project creates a "flywheel" effect—where every run makes the model smarter and the pipeline more robust.

---

## 🛠️ The Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Detection** | [YOLOv8](https://ultralytics.com/) | State-of-the-art object detection for 6 defect types. |
| **Inference API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance model serving (Port 8000). |
| **Frontend** | [Streamlit](https://streamlit.io/) | Interactive sandbox for real-time defect analysis (Port 8501). |
| **Orchestration** | [Docker Compose](https://www.docker.com/) | Unified management of all infrastructure services. |
| **Tracking** | [MLflow](https://mlflow.org/) | Dual-instance tracking for Sandbox (5555) and Official (5556) runs. |
| **Annotation** | [Label Studio](https://labelstud.io/) | Active learning and dataset refinement (Port 8080). |
| **Versioning** | [DVC](https://dvc.org/) | Large data and model versioning with S3-compatible backends. |
| **Storage** | [RustFS](https://github.com/cloud-native-ml/rustfs) | Local S3-compatible object storage for model vaults. |
| **Reporting** | [CML](https://cml.dev/) | Automated performance reporting in GitHub PRs. |

---

## ⚠️ Infrastructure Prerequisite

This project is built on a **Shared Infrastructure Model**. Before running any scripts or triggering CI/CD, you **MUST** have the Docker services running on the host machine:

```bash
# Launch the Flywheel Infrastructure
docker compose -f docker/docker-compose.yml up -d
```

*Note: The self-hosted CI/CD runner communicates with these containers via `localhost`. If they are not running, the pipeline will fail its health checks.*

---

## 🚀 Onboarding: Quick Start

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
- **MLflow (Sandbox)**: http://localhost:5555
- **MLflow (Official)**: http://localhost:5556
- **Label Studio**: http://localhost:8080 (Admin: `admin@example.com` / `mlops123`)

---

## 🔄 The Flywheel Workflow

### Phase 1: Annotation & Active Learning
1. Upload new PCB images to Label Studio.
2. Use the **Active Learning Loop** in the Streamlit UI to trigger batch inference on "unseen" images.
3. Sync labels back to the repo using `src/utils/sync_labels.py`.

### Phase 2: Training & Validation
Execute the pipeline with DVC. This ensures every run is reproducible and tracked.
```bash
dvc repro
```
- **Local Dev**: Logs results to Port 5555.
- **CI/CD**: When you push to a PR, the runner executes a "Showcase" run on Port 5556 and posts a visual report (Confusion Matrix, F1-Curves) directly to your GitHub PR comment.

### Phase 3: Serving & Monitoring
Deploy your champion model to the FastAPI server:
```bash
# Serves the champion model from MLflow registry or local weights
python src/serving/serve.py --weights models/best.pt
```

---

## 📂 Repository Structure
- `src/app/`: Streamlit dashboard for real-time detection.
- `src/serving/`: FastAPI inference service logic.
- `src/training/`: YOLOv8 training and evaluation scripts.
- `src/utils/`: Label Studio sync and batch inference helpers.
- `docker/`: Unified Docker Compose configuration.
- `mlflow-official-history/`: **Committed** history of team-verified runs.
- `mlflow-history/`: **Private** local sandbox history (ignored by Git).

---

## 🛡️ Best Practices
- **Never Commit `data/raw`**: Always use `dvc push` to store large images in RustFS.
- **Official Runs**: Only the CI/CD pipeline should log to Port 5556 to keep the "Official Showroom" clean.
- **PR Reports**: Always review the CML report in your Pull Request before merging to `main`.

---
**Developed for the ZHAW MLOps Course (Spring 2026)**
🏆 *Stabilizing the Flywheel, one PCB at a time.*
