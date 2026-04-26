# PCB Defect Detection MLOps Flywheel

This repository implements an end-to-end MLOps pipeline for detecting defects on PCBs (Printed Circuit Boards). It features an **Active Learning Flywheel** that allows for continuous model improvement via human-in-the-loop labeling.

## 🏗 Architecture

The system is designed to be orchestrated entirely **locally** using Docker and DVC.

1.  **DVC Data Versioning**: Manages datasets in `data/raw` and `data/processed`.
2.  **PyTorch Training**: CNN-based defect detection trained in `src/training/`.
3.  **MLflow Tracking**: All experiments, parameters, and model versions are logged to a local MLflow server.
4.  **Local Orchestration**: Docker Compose manages MLflow and Label Studio.
5.  **MLServer (Professional Serving)**: Serves the PyTorch model via a V2-compatible REST API on port **8081**.
6.  **Streamlit (Frontend)**: User interface that communicates with MLServer for real-time predictions.
7.  **Model Promotion**: The CI/CD pipeline (act) acts as a "Judge," only deploying models that meet accuracy thresholds.
8.  **Active Learning Loop**: 
    *   **Trigger**: Low-confidence predictions (e.g. < 80%) are automatically flagged.
    *   **Label**: Images are sent to **Label Studio** for human review.
    *   **Sync**: Labeled data is pulled back into the repo using `src/utils/sync_labels.py`.
    *   **Automated Retrain**: Running `act push` triggers the **CI/CD Driver**.
    *   **The Gatekeeper**: The CI/CD script only runs `dvc repro` if the **new sample threshold** (e.g. 5 images) is met.
    *   **Promotion**: If the new model is better, it is automatically deployed to **MLServer**.

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
