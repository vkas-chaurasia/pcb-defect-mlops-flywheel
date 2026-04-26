# MLOps Flywheel: Step-by-Step Execution Guide (Portable Edition)

## 🎙 The Technical Elevator Pitch (For Exams/Interviews)
> *"In the data flywheel, retraining is controlled by a threshold-based trigger implemented inside the pipeline. While DVC detects data changes and GitHub Actions orchestrates execution, the pipeline logic compares the current dataset size with the last trained state and only retrains when a predefined threshold (e.g., 5-10 new samples) is reached."*

---

## Phase 1: Environment Initialization

### 1.1 Project Setup
```bash
# Initialize project
uv init
uv pip install dvc mlflow streamlit torch torchvision opencv-python label-studio-sdk
```

### 1.2 Portable DVC Setup
This step makes the project 100% portable for your team.
```bash
# Initialize DVC
dvc init

# Create the INTERNAL remote folder
mkdir -p dvc-storage

# Tell DVC to use this folder as the "Remote Server"
dvc remote add -d localremote ./dvc-storage
```

---

## Phase 2: Building the Architecture

### 2.1 Folder Structure
```bash
mkdir -p data/raw data/processed  # Working data
mkdir -p models                   # Model weights
mkdir -p src/data src/training    # Logic
mkdir -p docker                   # Infrastructure
```

### 2.2 Local Services
Start MLflow, Label Studio, and lakeFS:
`docker-compose -f docker/docker-compose.yml up -d`

---

## Phase 3: The Flywheel Workflow

### 3.1 Adding Data (The DVC Cycle)
```bash
# 1. Add images to the working folder
cp my_pcb_image.jpg data/raw/

# 2. Add to DVC (this creates the .dvc receipt)
dvc add data/raw/my_pcb_image.jpg

# 3. Push to the internal "Remote"
dvc push
```
*   If working independently: Everyone keeps their own `../dvc_storage`. DVC ensures everyone is "in sync" via the `.dvc` receipts.

---

## Phase 4: Running the Engine

### 4.1 Initial Training (Local)
Before starting the flywheel, you need a baseline model.
```bash
# Run the pipeline locally on your Mac
dvc repro
```
This creates your first `models/model.pth`.

### 4.2 The Automated Flywheel (via act)
Once the project is live, you use **Automation** to retrain.
1. **Sync**: `python src/utils/sync_labels.py` (Pull new human labels).
2. **Automate**: `act push` (The Driver).
3. **Internal Logic**:
   - `act` counts the images.
   - If >= 5, it runs `dvc repro` inside a Docker container.
   - It promotes the model to `models/` ONLY if accuracy is high.
Defense (If asked about `dvc-storage`)
> *"For this university project, we placed the DVC remote folder inside the repository. This ensures that the project is 100% portable for the grading team. In a real-world industry setup, this `dvc-storage` would be an external S3 bucket, but the commands (`dvc push/pull`) would be identical."*

---

## Phase 5: Team Collaboration (Zero-Config)

1. **Member A (You)**: Pushes code and the `dvc-storage` folder to GitHub.
2. **Member B (Teammate)**: 
   ```bash
   git clone <repo_url>
   dvc pull  # Works instantly because dvc-storage is right there!
   ```

---

## Phase 6: Local Automation with `act`
- **act**: Run `act push` to test your GitHub Actions locally.

---

## Phase 7: Professional Model Serving (MLServer)

Instead of the app loading the model directly, we use **MLServer** for production-grade inference.

### 7.1 The "Hot-Reload" Logic
- **Architecture**: The `mlserver` container watches the `models/` folder.
- **Inference**: The Streamlit app sends REST requests to `http://localhost:8081`.
- **Zero Downtime**: When a new `model.pth` is saved, MLServer reloads it in milliseconds without stopping the app.

---

## Phase 8: Model Promotion & The "Judge"

We never publish a bad model. The **GitHub Action (act)** acts as the Judge.

### 8.1 The Promotion Workflow
1.  **Evaluate**: The CI/CD script checks the new model's accuracy in MLflow.
2.  **Threshold**: If accuracy > 90% AND better than the current model.
3.  **Deploy**: The script copies the new model to `models/model.pth`.
1. **Member A (You)**: Pushes code and the `dvc-storage` folder to GitHub.
2. **Member B (Teammate)**: 
   ```bash
   git clone <repo_url>
   dvc pull  # Works instantly because dvc-storage is right there!
   ```

---

## Phase 6: Local Automation with `act`
- **act**: Run `act push` to test your GitHub Actions locally.
