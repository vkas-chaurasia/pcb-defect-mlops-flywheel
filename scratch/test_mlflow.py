import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5555")
mlflow.set_experiment("connection-test")

with mlflow.start_run(run_name="Ping"):
    mlflow.log_param("status", "working")
    mlflow.log_metric("latency", 0.1)
    print("Logged test run to http://localhost:5555")
