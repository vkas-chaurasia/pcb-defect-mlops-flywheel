# PCB Defect Detection MLOps Flywheel

This repository implements an end-to-end MLOps pipeline for detecting defects on PCBs (Printed Circuit Boards). It features an **Active Learning Flywheel** that allows for continuous model improvement via human-in-the-loop labeling.

## 🏗 Architecture

The system is designed to be orchestrated entirely **locally** using Docker and DVC.

1.  **DVC Data Versioning**: Manages datasets in `data/raw` and `data/processed`.
2.  **PyTorch Training**: CNN-based defect detection trained in `src/training/`.
3.  **MLflow Tracking**: All experiments, parameters, and model versions are logged to a local MLflow server.
4.  **Local Orchestration**: Docker Compose manages MLflow, Label Studio, and MLServer.
5.  **MLServer (Backend)**: Serves the PyTorch model via a REST API.
6.  **Streamlit (Frontend)**: User interface for uploading PCB images and visualizing predictions.
7.  **Active Learning Loop**: 
    *   Low-confidence predictions are automatically flagged.
    *   Images are sent to **Label Studio** for human review.
    *   Labeled data is synced back to trigger a **DVC Reproduction** (`dvc repro`).

## 📁 Project Structure

```text
.
├── data/                   # Versioned data (Raw & Processed)
├── models/                 # Trained model artifacts
├── src/
│   ├── data/               # OpenCV preprocessing scripts
│   ├── training/           # PyTorch training logic
│   ├── serving/            # MLServer configuration
│   ├── app/                # Streamlit UI
│   └── utils/              # Sync & MLflow helpers
├── docker/                 # Docker Compose & Service configs
├── dvc.yaml                # MLOps Pipeline definition
└── requirements.txt        # Project dependencies
```

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

2. **Launch Services**:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

3. **Run the Flywheel**:
   ```bash
   dvc repro
   ```
