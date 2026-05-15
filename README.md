# PCB Defect Detection – MLOps Pipeline

This repository contains our MLOps pipeline for automated PCB defect detection. We have implemented a complete loop where our YOLOv8 model identifies defects and routes ambiguous cases to Label Studio for verification and retraining data collection.

---

## Quick Start (Team Setup)

We follow these steps to get the entire pipeline running locally.

### 1. Environment Setup
```bash
# Clone the repository and enter the directory
cd pcb-defect-mlops-flywheel

# Create our local environment bridge
cp .env.example .env

# Install dependencies (recommended in a virtual environment)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Pull heavy data and models from DVC remote storage
dvc pull
```

### 2. Launch Infrastructure (Label Studio, MLflow, & RustFS)
We use Docker to orchestrate Label Studio, MLflow, and RustFS (our DVC remote storage server).
```bash
docker compose -f docker/docker-compose.yml up -d
```
*   **Label Studio**: http://localhost:8080
*   **Credentials**: admin@example.com / mlops123
*   **Automated Project**: We have pre-configured a project named "PCB Defect Detection" with the correct defect labels (open, short, mousebite, spur, spurious_copper, pin_hole).

### 3. Start the Pipeline Services
We open two terminal tabs (with our virtual environment active):

**Tab 1: Inference Server (Fetching from Registry)**
```bash
# We pull and run the latest champion model from our MLflow registry
python src/serving/serve.py --weights models:/pcb-defect-model@champion
```

**Tab 2: Streamlit Dashboard**
```bash
streamlit run src/app/main.py
```

---

## The Pipeline Loop (Demo Workflow)

### Step 1: Detection
We open our Streamlit Dashboard at http://localhost:8501. We can upload images or use the simulation directory to see the model predict defects in real-time.

### Step 2: Trigger the Loop
We click the "Trigger Loop" button. This script (batch_inference.py) will:
1.  Run our model on the unseen_simulation dataset.
2.  Route all detections and low-confidence images to Label Studio.
3.  Apply Pre-Annotations: We send the model's boxes automatically so we do not have to label from scratch.

### Step 3: Human-in-the-loop (Label Studio)
1.  We log in to Label Studio at http://localhost:8080.
2.  We open our PCB Defect Detection project.
3.  We review the pre-annotated boxes, correcting any mistakes or adding missing detections.
4.  We click Submit.

### Step 4: Syncing New Data
Once we have submitted our labels, we run the sync script to pull them back into our training set:
```bash
python src/utils/sync_labels.py
```
*   **Output**: Our new YOLO-formatted data (Images + .txt labels) appears in data/raw/active_learning/.
*   **DVC**: We then use `dvc add` to version these new files.

---

## Tech Stack
*   **Model**: YOLOv8 (Ultralytics)
*   **Orchestration**: Docker Compose
*   **Annotation**: Label Studio
*   **Dashboard**: Streamlit
*   **Backend**: FastAPI
*   **Data Versioning**: DVC

---

## Team Collaboration
*   **Shared Credentials**: We standardize all API tokens and login credentials via .env.example.
*   **Infrastructure**: Our docker-compose configuration ensures that all team members are running identical versions of the environment.

---

## Experiment Tracking & CI/CD Validation

We use DVC to separate local model exploration from automated pipeline validation.

### 1. Local Exploration
We use `dvc exp run` to test multiple hyperparameters locally without writing custom bash scripts.
```bash
# Example: Queueing multiple variations
dvc exp run --queue -n exp1 -S train.epochs=2
dvc exp run --queue -n exp2 -S train.epochs=3 -S train.batch=8
dvc exp run --run-all
```
Once we identify the best performing experiment, we apply it to our workspace:
```bash
dvc exp apply <experiment_name>
```
This updates our `params.yaml` and `dvc.lock` files with the winning configuration. 

Finally, we must push the heavy model artifacts to our shared remote storage (RustFS) so the rest of the team and CI/CD can access them:
```bash
dvc push
```
We then commit `params.yaml` and `dvc.lock` to Git.

### 2. CI/CD Validation
Our automated CI/CD pipelines (e.g., GitHub Actions) do *not* run multiple experiments. Instead, they use:
```bash
dvc repro
```
By reading the committed `params.yaml` and `dvc.lock`, the CI/CD pipeline knows exactly which configuration is approved. 

**Crucially, it does not retrain the model.** Because we use a shared DVC remote cache (our RustFS server), when CI/CD runs `dvc repro`, DVC detects that the training step has already been computed for these exact parameters and code. It skips the expensive training process entirely and simply downloads the cached model artifacts, saving massive amounts of compute time and money.
