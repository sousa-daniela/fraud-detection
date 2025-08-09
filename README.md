# From Model to Production: A Complete MLOps Lifecycle for a Fraud API

**Author:** Daniela de Sousa Silva

## Description

This project implements a complete, end-to-end MLOps pipeline for a fraud detection system. The scenario is a government agency that needs to automate the detection of suspicious funding applications to ensure support reaches those in need.

The system is designed to be robust, adaptable, and automated. It features a cloud-native backend, a CT/CD pipeline for continuous training and deployment of a simple machine learning model, and a RESTful API to serve real-time predictions.

## Key MLOps Features

- **Automated Retraining Pipeline:** A scheduled GitHub Actions workflow automatically retrains the model with "new monthly data" (simulated), simulating a real-world production environment.
- **Cloud-Native Backend:** Decouples the MLOps infrastructure from the application code by using **AWS S3** for data and model storage and a **Neon cloud PostgreSQL database** for experiment tracking.
- **Champion/Challenger Deployment:** Implements automated promotion logic where a newly trained "challenger" model is only deployed to production if its F1 score surpasses the current "champion" model.
- **Centralized Model Management:** Leverages **MLflow** for comprehensive experiment tracking and a Model Registry to manage the lifecycle and versions of all trained models.
- **RESTful API for Predictions:** A **FastAPI** application serves predictions from the current production model, with on-demand loading to ensure it always uses the latest promoted version without downtime.
- **Secure by Design:** Manages all credentials and secrets securely using a local `.env` file (via `python-dotenv`) and encrypted **GitHub Secrets** for automation workflows.

## Tech Stack

- **Modeling & Data:** Python, Pandas, Scikit-learn, XGBoost
- **MLOps & Monitoring:** MLflow
- **Automation (CT/CD):** GitHub Actions
- **API:** FastAPI, Uvicorn
- **Cloud Infrastructure:** AWS S3, Neon (Cloud PostgreSQL)

## Project Structure

The repository is organized as follows:
```
fraud-detection/
│
├── .github/workflows/
│   └── retrain.yml                  # GitHub Actions workflow for automated retraining
├── data/
│   ├── train-test-splits/           # Pickled data splits for the initial model
│   └── df_train_balanced.csv        # Original initial balanced training set (see Notebook Section 2.1)
│   └── final_cleaned_data.csv       # Cleaned dataset for easy reproducibility (see Notebook Section 1)
│   └── merged_application_data.csv  # Raw merged data for easy reproducibility (see Notebook Section 1)
├── models/
│   └── mljar_results/
│   └── xgb_model.joblib             # The pre-trained initial model file
├── fraud_detection_pipeline.ipynb   # Main notebook including dataset cleaning, initial model training, evaluation and testing
├── main.py                          # FastAPI application for serving predictions
├── register_initial_model.py        # Script to register the first model and seed data
├── retrain_model.py                 # Script for the automated retraining process
├── requirements.txt                 # Project dependencies
├── requirements_dev.txt             # Developer dependencies
├── last_retrained_month.txt         # Keeps track of last month automatically retrained using Actions
└── README.md
```

---

## Setup and Installation

To run this project, a local setup and cloud infrastructure are required.

### 1. Local Environment Setup
1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/sousa-daniela/fraud-detection.git](https://github.com/sousa-daniela/fraud-detection.git)
    cd fraud-detection
    ```
2.  **Create and Activate a Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### 2. Cloud Infrastructure Setup
This project requires access to AWS and Neon. You will need to create your own free-tier accounts and resources.

1.  **AWS S3:**
    - Create a private AWS S3 bucket (e.g., `dss-fraud-detection`).
    - Create an IAM user with programmatic access (`Access key ID` and `Secret access key`).
    - Attach a policy to this user that grants it `s3:ListBucket`, `s3:GetObject`, `s3:PutObject`, and `s3:DeleteObject` permissions for the bucket you created.

2.  **Neon Database:**
    - Create a free PostgreSQL project on [Neon.tech](https://neon.tech).
    - From your project dashboard, copy the full database connection URL (URI).

### 3. Environment Configuration
1.  **Local (`.env` file):**
    - Create a file named `.env` in the project's root directory.
    - Paste the content below into the file and replace the placeholders with your own credentials from the previous step.
      ```
      # .env file
      export MLFLOW_TRACKING_URI="<YOUR_NEON_DATABASE_URL>"
      export AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>"
      export AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>"
      ```
2.  **GitHub Actions (Secrets):**
    - If you fork this repository and wish to run the automated workflow, you must add the same three variables (`MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) to your repository's `Settings > Secrets and variables > Actions`.

---

## Usage

Ensure you have sourced your environment variables before running commands: `source .env`

1.  **Run the Initial Model Registration:**
    This script registers the first model and uploads the initial training dataset to S3.
    ```bash
    python register_initial_model.py
    ```
2.  **Start the MLflow UI:**
    ```bash
    mlflow server \
      --backend-store-uri $MLFLOW_TRACKING_URI \
      --default-artifact-root "s3://<YOUR_BUCKET_NAME>" \
      --host 0.0.0.0 \
      --port 5500
    ```
3.  **Start the Prediction API:**
    ```bash
    uvicorn main:app --reload
    ```

---

## Author Notes

This project was developed to meet the requirements of a "Model to Production" course. The scenario is a government agency detecting application fraud. To simulate this, a comparable dataset from Kaggle concerning credit card applications was used and reframed for presentation.

The primary focus of this project is the design and implementation of the MLOps architecture, from cloud infrastructure and CI/CD automation to model monitoring and serving. The model itself is **intentionally kept simple** to emphasize the operational pipeline.

## Dataset Source

Credit Card Fraud Detection – [Kaggle](https://www.kaggle.com/c/home-credit-default-risk)

## License

This project is licensed under the MIT License. See the `LICENSE.md` file for details.
