# PCB Defect Detection System

This repository implements an end-to-end MLOps pipeline for detecting defects on PCBs (Printed Circuit Boards). It features an **Active Learning Flywheel** that allows for continuous model improvement via human-in-the-loop labeling.

## 🏗 Architecture

The system is designed to be orchestrated entirely **locally** using Docker and DVC.

1.  **DVC Data Versioning**: Manages datasets in `data/raw`.
2.  **PyTorch Training**: CNN-based defect detection trained in `src/training/`.
3.  **MLflow Tracking**: All experiments, parameters, and model versions are logged to a local MLflow server.
4.  **Local Orchestration**: Docker Compose manages MLflow and Label Studio.
5.  **FastAPI (Custom Serving)**: High-performance inference engine with built-in **PASS/FAIL** logic and Swagger UI on port **8000**. Supports `/predict/batch`.
6.  **Automated Pipeline**: A batch processor (`src/utils/batch_inference.py`) that automatically routes failures to Label Studio.
7.  **Streamlit Dashboard**: A premium UI for live testing and manual pipeline orchestration.
8.  **Active Learning Loop**: 
    *   **Trigger**: The system controller scans `data/raw/unseen_simulation`.
    *   **Filter**: All "FAIL" results or low-confidence predictions are automatically flagged.
    *   **Label**: Images are sent to **Label Studio** for human review.
    *   **Sync**: Labeled data is pulled back using `src/utils/sync_labels.py`.

## 📁 Project Structure

```text
.
├── data/                   # Versioned data (Raw)
├── models/                 # Trained model artifacts
├── src/
│   ├── data/               # OpenCV preprocessing scripts
│   ├── training/           # PyTorch training logic
│   ├── serving/            # FastAPI (serve.py) logic
│   ├── app/                # Streamlit Dashboard
│   └── utils/              # batch_inference.py & sync_labels.py
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
