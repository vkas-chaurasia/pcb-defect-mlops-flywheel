import json
import os
import requests
import base64
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException

import mlflow
from mlflow.tracking import MlflowClient

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5556"  # Official MLflow
MODEL_NAME = "pcb-defect-model"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO", "owner/repo") # Make sure this is set in docker-compose if we really want it to work

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'model_governance_eval',
    default_args=default_args,
    description='Evaluate Staging model and open Governance PR',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['governance', 'mlops'],
) as dag:

    @task
    def evaluate_staging_model():
        """
        Simulates model evaluation. Fetches the Staging model from MLflow.
        If it meets criteria (mocked here), returns the version to promote.
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        
        try:
            mv = client.get_model_version_by_alias(MODEL_NAME, "staging")
        except Exception as e:
            raise AirflowSkipException(f"No staging model found for {MODEL_NAME}.")
            
        print(f"Found staging model version {mv.version}.")
        
        # 1. State Machine: Check if already evaluated
        status = mv.tags.get("validation_status")
        if status in ["passed", "failed"]:
            raise AirflowSkipException(f"Staging model v{mv.version} was already evaluated (Status: {status}). Skipping.")
        
        # REAL EVALUATION: Execute the industry-standard YOLO validation script
        import subprocess
        print(f"Executing PyTorch evaluation script against Golden Dataset...")
        
        # We need to run the script using the uv python environment if possible
        # but standard python is fine if uv environment is active.
        result = subprocess.run(["python", "src/utils/evaluate_staging_model.py"])
        
        passed_evaluation = (result.returncode == 0)
        
        # 2. State Machine: Record the outcome
        if not passed_evaluation:
            client.set_model_version_tag(MODEL_NAME, mv.version, "validation_status", "failed")
            raise AirflowSkipException(f"Staging model v{mv.version} failed evaluation.")
            
        client.set_model_version_tag(MODEL_NAME, mv.version, "validation_status", "passed")
        print(f"Model v{mv.version} passed evaluation. Tagged as 'passed'.")
        return mv.version

    @task
    def create_governance_pr(model_version: str):
        """
        Creates a GitOps Pull Request to promote the model to Production.
        Updates deployment-repo/production_version.json.
        """
        if not GITHUB_TOKEN or GITHUB_REPO == "owner/repo":
            print(f"Would create PR for model v{model_version}, but GITHUB_TOKEN or GITHUB_REPO is not set properly.")
            print("To see this fully simulate, set valid GitHub credentials in docker-compose.")
            return

        api_base = f"https://api.github.com/repos/{GITHUB_REPO}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        branch_name = f"promote-model-v{model_version}"
        
        # 1. Get SHA of main branch
        res = requests.get(f"{api_base}/git/ref/heads/main", headers=headers)
        if res.status_code != 200:
            raise Exception("Could not get main branch SHA")
        main_sha = res.json()['object']['sha']
        
        # 2. Create new branch
        res = requests.post(f"{api_base}/git/refs", headers=headers, json={
            "ref": f"refs/heads/{branch_name}",
            "sha": main_sha
        })
        # Ignore 422 if branch exists
        if res.status_code not in (201, 422):
            raise Exception(f"Failed to create branch: {res.text}")
            
        # 3. Get existing file (to get its SHA for update)
        file_path = "deployment-repo/production_version.json"
        res = requests.get(f"{api_base}/contents/{file_path}?ref=main", headers=headers)
        file_sha = res.json().get('sha') if res.status_code == 200 else None
        
        # 4. Update file on new branch
        new_content = {"model_name": MODEL_NAME, "production_version": model_version}
        content_b64 = base64.b64encode(json.dumps(new_content, indent=2).encode()).decode()
        
        payload = {
            "message": f"gov: Promote {MODEL_NAME} v{model_version} to production",
            "content": content_b64,
            "branch": branch_name
        }
        if file_sha:
            payload["sha"] = file_sha
            
        res = requests.put(f"{api_base}/contents/{file_path}", headers=headers, json=payload)
        if res.status_code not in (200, 201):
            raise Exception(f"Failed to update file: {res.text}")
            
        # 5. Create Pull Request
        pr_payload = {
            "title": f"🚀 Model Governance: Promote {MODEL_NAME} v{model_version}",
            "body": f"Automated Governance PR.\n\nThe Airflow evaluation pipeline has verified that **v{model_version}** of **{MODEL_NAME}** passes all safety and performance checks.\n\nApprove and merge this PR to officially deploy this version as the new Champion.",
            "head": branch_name,
            "base": "main"
        }
        res = requests.post(f"{api_base}/pulls", headers=headers, json=pr_payload)
        if res.status_code == 201:
            print(f"Successfully created Governance PR: {res.json()['html_url']}")
        elif res.status_code == 422:
            print("PR already exists.")
        else:
            raise Exception(f"Failed to create PR: {res.text}")

    # Set dependencies
    version = evaluate_staging_model()
    create_governance_pr(version)
