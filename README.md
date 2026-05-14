# PCB Defect Detection MLOps Flywheel

This repository implements an end-to-end MLOps pipeline for detecting defects on PCBs (Printed Circuit Boards). It features an **Active Learning Flywheel** that allows for continuous model improvement via human-in-the-loop labeling.

## 🏗 Architecture

The system is designed to be orchestrated entirely **locally** using Docker and DVC.

1.  **DVC Data Versioning**: Manages datasets in `data/raw`.
2.  **PyTorch Training**: CNN-based defect detection trained in `src/training/`.
3.  **MLflow Tracking**: All experiments, parameters, and model versions are logged to a local MLflow server.
4.  **Local Orchestration**: Docker Compose manages MLflow and Label Studio.
5.  **FastAPI (Custom Serving)**: High-performance inference engine with built-in **PASS/FAIL** logic and Swagger UI on port **8000**.
6.  **Human-in-the-Loop (HITL)**: All "FAIL" results (defects detected) can be routed to humans for verification.
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
├── data/                   # Versioned data (Raw)
├── models/                 # Trained model artifacts
├── src/
│   ├── data/               # OpenCV preprocessing scripts
│   ├── training/           # PyTorch training logic
│   ├── serving/            # FastAPI (serve.py) logic
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

4. **Promote the Champion**:
   * Open MLflow UI (`http://localhost:5555`).
   * Select your best model version and add the alias **`champion`** (or change stage to **`Production`**).

5. **Start Inference API**:
   ```bash
   .venv/bin/python src/serving/serve.py
   ```

## ⚖️ License & Attribution

### Dataset
This project uses the **DeepPCB** dataset from the research paper:
> *Tang, S., et al. "On-line PCB Defect Detector On A New PCB Defect Dataset." (2019).*

**Important**: Per the original authors' request, this dataset is for **research purposes only**. Please refer to the [DeepPCB GitHub Repository](https://github.com/tangsanli5201/DeepPCB) for more details.

### Code
The MLOps infrastructure and FastAPI implementation in this repository are licensed under the **MIT License**.
