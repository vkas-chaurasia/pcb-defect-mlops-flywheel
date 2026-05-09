# Data Version Control (DVC) Guide

This guide explains how we handle large datasets (images, model weights) in this project using DVC. Since Git isn't designed for heavy files, we use DVC to track the metadata in Git while storing the actual files in our local RustFS storage.

## Core Concepts

### 1. Workspace vs. Vault
*   **Workspace (`data/raw/`)**: This is where you work. Your scripts read data from here.
*   **Vault (`dvc-storage` / RustFS)**: This is our shared backup. DVC stores every version of our data here. We never touch these files manually.

### 2. How it works (Portable Mode)
Our project is configured for **100% Portability**. This means our "Vault" is stored directly inside the repository in the `dvc-storage/` folder.

1.  **Track**: Put your data in `data/raw/` and run `dvc add`. This creates a tiny `.dvc` receipt.
2.  **Push**: Run `dvc push`. This moves your data into the local `dvc-storage/` vault.
3.  **Commit**: Run `git add .` to stage both your `.dvc` receipts AND the new data in `dvc-storage/`.
4.  **Sync**: Your team runs `git pull` followed by `dvc pull`. DVC will automatically reconstruct the files in their `data/raw/` folder using the vault data they just pulled from Git.

---

## Local Setup & Testing

To make sure your local environment is ready for DVC, follow these steps:

### 1. Set up the Environment
We use a modular virtual environment to keep DVC and its S3 dependencies isolated.
```bash
# Create the environment
python3 -m venv dvc/.venv

# Activate and install dependencies
source dvc/.venv/bin/activate
pip install -r dvc/requirements.txt
```

### 2. Start Infrastructure
We use RustFS (S3-compatible) running in Docker as our local remote storage.
```bash
docker compose -f docker/docker-compose.yml up rustfs -d
```

### 3. Access the Storage Console
Open [http://localhost:9001](http://localhost:9001) to browse the data.
*   **Username**: `rustfsadmin`
*   **Password**: `rustfsadmin`

### 4. Verification Lab
I've included a test script to verify your setup without touching our main training data.
1. Create a bucket named `pcb-test-bucket` in the console.
2. Run the verification script:
   ```bash
   ./dvc/tests/test_dvc.sh
   ```

This script will automatically configure a test remote, push a dummy file, delete it, and pull it back to confirm everything is working.
