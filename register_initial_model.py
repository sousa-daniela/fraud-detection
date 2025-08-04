# Import libraries and modules
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score
from dotenv import load_dotenv

# 1. Setup
# Load environment variables for local testing
load_dotenv()

# Load pre-trained model and test data
xgb_model = joblib.load("models/xgb_model.joblib")
X_test = joblib.load("data/train-test-splits/X_test.pkl")
y_test = joblib.load("data/train-test-splits/y_test.pkl")

# Set up MLflow tracking
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Define experiment name and S3 artifact location
experiment_name = "fraud_detection_project"
artifact_location = "s3://dss-fraud-detection/"

# Create the experiment if it doesn't exist, setting the artifact location
try:
    mlflow.create_experiment(name=experiment_name, artifact_location=artifact_location)
except mlflow.exceptions.MlflowException:
    pass # Experiment already exists
mlflow.set_experiment(experiment_name)

# 2. Model evaluation and logging
# Evaluate F1 on the test set
y_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.25).astype(int)
f1 = f1_score(y_test, y_pred)

client = MlflowClient()

with mlflow.start_run(run_name="Initial XGBoost Production Model") as run:
    print(f"Logging initial model with F1 score: {f1}")
    
    # Log parameters and metrics
    mlflow.log_params({"model_type": "initial_xgboost", "threshold": 0.25})
    mlflow.log_metric("f1_score", f1)

    # Log the model and capture its information
    input_example = X_test.iloc[[0]]
    signature = infer_signature(X_test, y_proba)
    model_info = mlflow.sklearn.log_model(
        sk_model=xgb_model,
        name="model",
        registered_model_name="xgboost_fraud_detector",
        input_example=input_example,
        signature=signature
    )

    # 3. Promote to production dynamically
    # Get the version number from the correct attribute
    new_version = model_info.registered_model_version
    
    print(f"Model registered as version {new_version}. Promoting to 'prod'.")
    client.set_registered_model_alias(
        name="xgboost_fraud_detector",
        alias="prod",
        version=new_version
        )
    
print("Initial model registration complete.")

# Run mlflow server --backend-store-uri "postgresql://neondb_owner:npg_c0iPKGM4tXhQ@ep-quiet-forest-a29wt3ze-pooler.eu-central-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require" --default-artifact-root "s3://dss-fraud-detection" --host 0.0.0.0 --port 5500
# Run python register_initial_model.py
# MLflow UI: http://127.0.1:5500