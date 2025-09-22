from datetime import datetime
import os, subprocess
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

PROJECT_DIR = "/opt/project"

RAW = os.path.join(PROJECT_DIR, "data/raw/spam.csv")
PROC_DIR = os.path.join(PROJECT_DIR, "data/processed")
TRAIN = os.path.join(PROJECT_DIR, "data/processed/train.csv")
TEST  = os.path.join(PROJECT_DIR, "data/processed/test.csv")
MODEL = os.path.join(PROJECT_DIR, "models/model.pkl")

def run_prepare():
    cmd = [
        "python", os.path.join(PROJECT_DIR, "code/datasets/prepare_data.py"),
        "--raw", RAW, "--outdir", PROC_DIR, "--test-size", "0.2", "--seed", "42",
    ]
    subprocess.run(cmd, check=True)

def run_train():
    cmd = [
        "python", os.path.join(PROJECT_DIR, "code/models/train.py"),
        "--train", TRAIN, "--test", TEST,
        "--out", MODEL,
        "--metrics-out", os.path.join(PROJECT_DIR, "models/metrics.json"),
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_DIR)

with DAG(
    dag_id="mlops_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="*/5 * * * *",  # каждые 5 минут
    catchup=False,
    default_args={"owner": "you", "retries": 0},
    tags=["sms-spam", "mlops"],
) as dag:

    data_stage = PythonOperator(
        task_id="data_stage",
        python_callable=run_prepare,
    )

    model_stage = PythonOperator(
        task_id="model_stage",
        python_callable=run_train,
    )

    deploy_stage = BashOperator(
        task_id="deploy_stage",
        bash_command=(
            "set -euxo pipefail; "
            "docker --version; "
            "docker compose version; "
            "ls -l /var/run/docker.sock || true; "
            "cd /opt/project/code/deployment; "
            "docker compose up --build -d"
        ),
    )

    data_stage >> model_stage >> deploy_stage
