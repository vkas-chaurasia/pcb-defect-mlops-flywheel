import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient

def main():
    """
    Registers the model from the latest MLflow run and applies the @staging alias.
    Expected to be run by the CI/CD pipeline when merging to main.
    """
    if not os.path.exists("models/model_meta.json"):
        print("Error: models/model_meta.json not found. Cannot register model.")
        sys.exit(1)

    with open("models/model_meta.json", "r") as f:
        meta = json.load(f)
        run_id = meta.get("RUN_ID")

    if not run_id:
        print("Error: RUN_ID not found in models/model_meta.json.")
        sys.exit(1)

    model_name = "pcb-defect-model"
    artifact_path = "pcb-yolo-model"
    model_uri = f"runs:/{run_id}/{artifact_path}"

    print(f"Checking if run {run_id} is already registered as @staging...")
    client = MlflowClient()
    
    try:
        staging_mv = client.get_model_version_by_alias(model_name, "staging")
        if staging_mv.run_id == run_id:
            print(f"Skipping registration: Run {run_id} is already registered as version {staging_mv.version} and aliased as @staging.")
            sys.exit(0)
    except Exception:
        # Alias doesn't exist yet, which is fine, proceed to register
        pass

    print(f"Registering model from run {run_id} as {model_name}...")
    
    # Register the model (creates a new version if model_name exists)
    mv = mlflow.register_model(model_uri, model_name)
    
    # Set the staging alias
    client = MlflowClient()
    print(f"Setting alias '@staging' for {model_name} version {mv.version}...")
    client.set_registered_model_alias(name=model_name, alias="staging", version=mv.version)
    
    print("Model successfully registered and tagged as @staging.")

if __name__ == "__main__":
    main()
