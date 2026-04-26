# MLOps Flywheel: Step-by-Step Execution Guide

## 🎙 The Technical Elevator Pitch (For Exams/Interviews)
> *"In the data flywheel, retraining is controlled by a threshold-based trigger implemented inside the pipeline. While DVC detects data changes and GitHub Actions orchestrates execution, the pipeline logic compares the current dataset size with the last trained state and only retrains when a predefined threshold (e.g., 5-10 new samples) is reached."*

---

## Phase 1: Environment Initialization

### 1.1 Project Setup (with `uv`)
We use `uv` for lightning-fast package management.
```bash
# Initialize a new Python project
uv init

# Install core MLOps dependencies
uv pip install dvc mlflow streamlit torch torchvision opencv-python label-studio-sdk
```

### 1.2 Git & DVC Initialization
Git tracks your code; DVC tracks your data and models.
```bash
# Initialize Git (if not already done)
git init

# Initialize DVC
dvc init

# (Optional) Set up local storage for data
mkdir /tmp/dvc_cache
dvc cache dir /tmp/dvc_cache
```

---

## Phase 2: Building the Architecture

### 2.1 Standard MLOps Folder Structure
A modular structure ensures scalability.
```bash
mkdir -p data/raw data/processed  # Data storage
mkdir -p models                   # Model weights
mkdir -p src/data src/training    # Pipeline logic
mkdir -p src/serving src/app      # Deployment & UI
mkdir -p docker configs           # Orchestration
```

### 2.2 Local Service Orchestration (Docker)
We use Docker Compose to run infrastructure without installing things manually on the OS.
*   **MLflow**: Port 5000 (Tracking)
*   **Label Studio**: Port 8080 (Human labeling)
*   **MLServer**: Port 8081 (Inference)

Run with: `docker-compose -f docker/docker-compose.yml up -d`

---

## Phase 3: The Active Learning Flywheel

### 3.1 Data Versioning (DVC)
When you add new images to `data/raw/`:
```bash
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Add new raw data"
```

### 3.2 Defining the Pipeline (`dvc.yaml`)
This file automates the execution. You define "stages" like `preprocess` and `train`.
```yaml
stages:
  preprocess:
    cmd: python src/data/preprocess.py
    deps:
      - data/raw
      - src/data/preprocess.py
    outs:
      - data/processed
  train:
    cmd: python src/training/train.py
    deps:
      - data/processed
      - src/training/train.py
    outs:
      - models/model.pth
```

### 3.3 Executing the Loop
Whenever new labels arrive from Label Studio:
```bash
# Pull new labels (using our sync script)
python src/utils/sync_labels.py

# Reproduce the entire pipeline automatically
dvc repro
```

---

## Phase 4: Verification for Exams
To prove your system works, show these three "Truths":
1.  **Code Truth**: `git log` shows the history of your experiments.
2.  **Data Truth**: `dvc dag` shows the visual graph of your data pipeline.
3.  **Performance Truth**: Open `http://localhost:5000` (MLflow) to show accuracy improvements over time.

---

## Phase 5: Local CI/CD with `act` (Optional/Pro)

If you want to simulate GitHub Actions locally without pushing to GitHub:
1.  **Install act**: `brew install nektos/tap/act`
2.  **Create Workflow**: Define your pipeline in `.github/workflows/train.yml`.
3.  **Run Locally**:
    ```bash
    # Run the "push" event workflows locally
    act push
    ```
This is perfect for testing your automation logic entirely offline.

---

## Phase 6: Team Collaboration (The "Lean" Way)

In a university project, you don't need paid servers. You use **Git** as your shared logbook.

### 6.1 Collaborative Training Flow
1.  **Teammate A** runs `dvc repro` and gets a new model.
2.  They commit:
    *   `data/raw.dvc` (The new data pointer)
    *   `models/model.pth.dvc` (The new model pointer)
    *   `metrics.json` (The accuracy/loss results)
3.  **Teammate B** runs `git pull`.
4.  They immediately see the new `metrics.json` and know the model improved.
5.  They run `dvc pull` to download the actual model files.

### 6.2 Shared Data Storage
*   If you have a shared network drive: `dvc remote add -d shared_drive /path/to/drive`
*   If working independently: Everyone keeps their own `../dvc_storage`. DVC ensures everyone is "in sync" via the `.dvc` receipts.

---

## Phase 7: Advanced Data Lake Management (lakeFS)

For large-scale industry projects, we use **lakeFS** to manage our data like a Git repository.

### 7.1 Why lakeFS?
- **Branching**: Create a "sandbox" of 10TB of data in 1 second.
- **Hooks**: Automatically run data quality tests before merging new labels.
- **Rollback**: Instantly revert to a previous state if a bad dataset is introduced.

### 7.2 Running Locally
1.  **Start Services**: `docker-compose up -d`
2.  **Access UI**: `http://localhost:8000`
3.  **Exam Pro Tip**: Mention that lakeFS provides "Git-like operations" for object storage (S3/GCS/MinIO), allowing for reproducible data environments at the petabyte scale.
