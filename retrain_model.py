# Import libraries and modules
import pandas as pd
import os
import sys
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# 1. Setup
# Load environment variables for local testing
load_dotenv()

# Set up MLflow tracking
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500")
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Define experiment and S3 locations
experiment_name = "fraud_detection_project"
s3_bucket = "dss-fraud-detection"
training_path_s3 = f"s3://{s3_bucket}/training_data/df_training_current.csv"

# Create the experiment if it doesn't exist, setting the artifact location
try:
    mlflow.create_experiment(name=experiment_name, artifact_location=f"s3://{s3_bucket}/")
except mlflow.exceptions.MlflowException:
    pass # Experiment already exists
mlflow.set_experiment(experiment_name)

# Check for command-line argument
if len(sys.argv) != 2:
    print("Usage: python retrain_model.py <path_to_month_file.csv>")
    sys.exit(1)
month_file = sys.argv[1]

# 2. Data loading and processing
# Load base dataset from S3. Handle case where it doesn't exist yet (first run).
try:
    print(f"Loading base training data from {training_path_s3}")
    df_base = pd.read_csv(training_path_s3)
except FileNotFoundError:
    print("Base training data not found in S3. Starting with initial training set.")
    # Assuming initial training set is stored locally for the very first run
    df_base = pd.read_csv("data/df_training_current.csv")

# Load new monthly data (from local path provided by GitHub Actions)
print(f"Loading new monthly data from {month_file}")
df_month = pd.read_csv(month_file)

# Rebalance incoming month
desired_ratio = 0.25
df_fraud = df_month[df_month["TARGET"] == 1]
n_fraud = len(df_fraud)
n_nonfraud = int(n_fraud * (1 - desired_ratio) / desired_ratio)
# A small robustness check in case a month has no fraud cases or not enough non-fraud
if n_fraud > 0 and len(df_month[df_month["TARGET"] == 0]) >= n_nonfraud:
    df_nonfraud = df_month[df_month["TARGET"] == 0].sample(n=n_nonfraud, random_state=42)
    df_month_balanced = pd.concat([df_fraud, df_nonfraud], ignore_index=True)
else:
    df_month_balanced = df_month # Use unbalanced month if rebalancing isn't possible

# Combine with existing training data
df_combined = pd.concat([df_base, df_month_balanced], ignore_index=True)

# Create new feature and prepare data for training
df_combined["CREDIT_TO_INCOME"] = df_combined["AMT_CREDIT"] / df_combined["AMT_INCOME_TOTAL"]
X = df_combined.drop(columns=["SK_ID_CURR", "TARGET"])
y = df_combined["TARGET"]

# Stratified split for train/test from the entire combined pool
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Model training and evaluation
print("Training new XGBoost model...")
model = XGBClassifier(
    objective="binary:logistic", learning_rate=0.1, max_depth=7,
    min_child_weight=25, subsample=0.9, colsample_bytree=0.6,
    n_jobs=-1, seed=1237
)
model.fit(X_train, y_train)

# Evaluate model
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.25).astype(int)
f1 = f1_score(y_test, y_pred)
print(f"New Model F1 Score: {f1}")


# 4. MLflow logging and model promotion
client = MlflowClient()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

with mlflow.start_run(run_name=f"retrain_{timestamp}") as run:
    print(f"Logging run {run.info.run_id} to MLflow...")
    mlflow.log_params({ "max_depth": 7, "threshold": 0.25, "month_file": os.path.basename(month_file) })
    mlflow.log_metric("f1_score", f1)

    # Use the efficient method to get the new model version
    signature = infer_signature(X_test, y_proba)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="xgboost_fraud_detector",
        signature=signature
    )
    new_model_version = model_info.registered_model_version
    print(f"New model registered as version: {new_model_version}")


# 5. Champion/Challenger Logic
try:
    prod_model_details = client.get_model_version_by_alias("xgboost_fraud_detector", "prod")
    prod_run_id = prod_model_details.run_id
    prev_f1 = client.get_run(prod_run_id).data.metrics["f1_score"]
    print(f"Current Production Model F1: {prev_f1} (Version: {prod_model_details.version})")
except mlflow.exceptions.RestException:
    prev_f1 = -1
    print("No Production model found to compare against.")

if f1 > prev_f1:
    print(f"New model is better. Promoting version {new_model_version} to Production.")
    client.set_registered_model_alias(name="xgboost_fraud_detector", alias="prod", version=new_model_version)
else:
    print("New model is not better than current Production model. Not promoting.")


# 6. Update Master Training Data in S3
# Overwrite the main training file in S3 with the newly combined data from this run
print(f"Saving updated master training data to {training_path_s3}...")
df_combined.to_csv(training_path_s3, index=False)
print("Process complete.")