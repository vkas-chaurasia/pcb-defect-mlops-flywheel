import os
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pcb_data_sync_and_drift_check',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    sync_labels = BashOperator(
        task_id='sync_labels',
        bash_command='cd /opt/airflow/project && python src/utils/sync_labels.py',
    )

    drift_monitor = BashOperator(
        task_id='drift_monitor',
        bash_command='cd /opt/airflow/project && python src/monitoring/drift_monitor.py --auto-pr',
        env={
            'GITHUB_TOKEN': os.getenv('GITHUB_TOKEN') or '{{ var.value.get("github_token", "") }}',
            'GITHUB_REPO': os.getenv('GITHUB_REPO') or '{{ var.value.get("github_repo", "") }}',
        }
    )

    sync_labels >> drift_monitor
