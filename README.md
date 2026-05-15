# PCB Defect Detection – Active Learning Pipeline

This repository contains a professional MLOps pipeline for automated PCB defect detection. It implements a complete Active Learning Loop (Human-in-the-loop) where a YOLOv8 model identifies defects and routes ambiguous cases to Label Studio for expert verification and retraining data collection.

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
We use Docker to orchestrate Label Studio and MLflow with a Zero-Config approach.
```bash
docker compose -f docker/docker-compose.yml up -d
```
*   **Label Studio**: http://localhost:8080
*   **Credentials**: admin@example.com / mlops123 (Shared for the whole team)
*   **Automated Project**: A pre-configured project named "PCB Defect Detection" is automatically created with the correct defect labels (open, short, mousebite, spur, spurious_copper, pin_hole).

### 3. Start the Pipeline Services
Open two terminal tabs (with your virtual environment active):

**Tab 1: Inference Server (AI Engine)**
```bash
python src/serving/serve.py --weights runs/detect/pcb-defect-detection/indecisive-hare-189/weights/best.pt
```

**Tab 2: Streamlit Dashboard (UI)**
```bash
streamlit run src/app/main.py
```

---

## The Active Learning Loop (Demo Workflow)

### Step 1: Real-time Detection
Open the Streamlit Dashboard at http://localhost:8501. Upload images or use the simulation directory. The model will predict defects in real-time.

### Step 2: Trigger the Loop
Click the "Trigger Active Learning Loop" button. This script (batch_inference.py) will:
1.  Run the model on the unseen_simulation dataset.
2.  Route all detections and low-confidence images to Label Studio.
3.  Apply Pre-Annotations: The script sends the model's "best guess" boxes so labels do not need to be created from scratch.

### Step 3: Human-in-the-loop (Label Studio)
1.  Log in to Label Studio at http://localhost:8080.
2.  Open the PCB Defect Detection project.
3.  Review the pre-annotated boxes. Correct any mistakes or add missing detections.
4.  Click Submit to finalize the annotation.

### Step 4: Syncing New Training Data
Once labels have been submitted, run the sync script to pull them back into the project for future training:
```bash
python src/utils/sync_labels.py
```
*   **Output**: New YOLO-formatted data (Images + .txt labels) will appear in data/raw/active_learning/.
*   **DVC**: These files are now ready to be versioned via `dvc add`.

---

## Tech Stack
*   **Model**: YOLOv8 (Ultralytics)
*   **Orchestration**: Docker Compose
*   **Annotation**: Label Studio (with automated API integration)
*   **Dashboard**: Streamlit
*   **Backend**: FastAPI
*   **Data Versioning**: DVC

---

## Team Collaboration
*   **Shared Credentials**: All API tokens and login credentials are standardized via .env.example to ensure team-wide consistency.
*   **Infrastructure**: The docker-compose configuration ensures that all team members are running identical versions of the labeling and tracking environments.
