# PCB Defect Detection – MLOps Pipeline

This repository contains an MLOps pipeline for automated PCB defect detection. It implements a complete loop where a YOLOv8 model identifies defects and routes ambiguous cases to Label Studio for verification and retraining data collection.

---

## Quick Start (Team Setup)

Follow these steps to get the entire pipeline running locally on your machine.

### 1. Environment Setup
```bash
# Clone the repository and enter the directory
git checkout feature/automated-pipeline

# Create your local environment bridge
cp .env.example .env

# Install dependencies (recommended in a virtual environment)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch Infrastructure (Label Studio)
We use Docker to orchestrate Label Studio and MLflow.
```bash
docker compose -f docker/docker-compose.yml up -d
```
*   **Label Studio**: http://localhost:8080
*   **Credentials**: admin@example.com / mlops123
*   **Automated Project**: A pre-configured project named "PCB Defect Detection" is automatically created with the correct defect labels (open, short, mousebite, spur, spurious_copper, pin_hole).

### 3. Start the Pipeline Services
Open two terminal tabs (with your virtual environment active):

**Tab 1: Inference Server**
```bash
python src/serving/serve.py --weights runs/detect/pcb-defect-detection/indecisive-hare-189/weights/best.pt
```

**Tab 2: Streamlit Dashboard**
```bash
streamlit run src/app/main.py
```

---

## The Pipeline Loop (Demo Workflow)

### Step 1: Detection
Open the Streamlit Dashboard at http://localhost:8501. Upload images or use the simulation directory. The model will predict defects in real-time.

### Step 2: Trigger the Loop
Click the "Trigger Loop" button. This script (batch_inference.py) will:
1.  Run the model on the unseen_simulation dataset.
2.  Route all detections and low-confidence images to Label Studio.
3.  Apply Pre-Annotations: The script sends the model's boxes so labels do not need to be created from scratch.

### Step 3: Human-in-the-loop (Label Studio)
1.  Log in to Label Studio at http://localhost:8080.
2.  Open the PCB Defect Detection project.
3.  Review the pre-annotated boxes. Correct any mistakes or add missing detections.
4.  Click Submit.

### Step 4: Syncing New Data
Once labels have been submitted, run the sync script to pull them back into the project:
```bash
python src/utils/sync_labels.py
```
*   **Output**: New YOLO-formatted data (Images + .txt labels) will appear in data/raw/active_learning/.
*   **DVC**: These files are now ready to be versioned via `dvc add`.

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
*   **Shared Credentials**: All API tokens and login credentials are standardized via .env.example.
*   **Infrastructure**: The docker-compose configuration ensures that all team members are running identical versions of the environment.
