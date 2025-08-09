# Import libraries and modules
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import mlflow.sklearn
from mlflow.sklearn import load_model
from dotenv import load_dotenv
import os

# Looad the MLFLOW_TRACKING_URI and AWS credentials from .env file
load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Initialize FastAPI app instance
app = FastAPI()

# Response schema for a single prediction
class RiskPrediction(BaseModel):
    probability: float
    risk_level: str

# Input schema for a single application record
class ApplicationRecord(BaseModel):
    SK_ID_CURR: int
    TARGET: Optional[int] = None  # Present in incoming data but not used for prediction
    NAME_CONTRACT_TYPE: int
    CODE_GENDER: int
    FLAG_OWN_CAR: int
    FLAG_OWN_REALTY: int
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    CNT_FAM_MEMBERS: float
    FLAG_EMP_PHONE: int
    FLAG_WORK_PHONE: int
    FLAG_CONT_MOBILE: int
    FLAG_EMAIL: int
    REGION_RATING_CLIENT: int
    REG_REGION_NOT_LIVE_REGION: int
    PREV_AMT_CREDIT_mean: float
    PREV_AMT_CREDIT_max: float
    PREV_CNT_PAYMENT_mean: float
    PREV_NFLAG_LAST_APPL_IN_DAY_sum: int
    PREV_DAYS_DECISION_mean: float
    PREV_NAME_CONTRACT_STATUS_Approved_sum: int
    PREV_NAME_CONTRACT_STATUS_Refused_sum: int
    PREV_NAME_CONTRACT_STATUS_Canceled_sum: int
    PREV_NAME_CONTRACT_STATUS_Unused_offer_sum: int = Field(alias="PREV_NAME_CONTRACT_STATUS_Unused offer_sum")
    PREV_NAME_CLIENT_TYPE_New_sum: int
    PREV_NAME_CLIENT_TYPE_Repeater_sum: int
    PREV_NAME_CLIENT_TYPE_Refreshed_sum: int
    PREV_NAME_CLIENT_TYPE_XNA_sum: int
    NAME_FAMILY_STATUS_Married: int
    NAME_FAMILY_STATUS_Separated: int
    NAME_FAMILY_STATUS_Single_not_married: int = Field(alias="NAME_FAMILY_STATUS_Single / not married")
    NAME_FAMILY_STATUS_Widow: int
    NAME_INCOME_TYPE_Public: int
    NAME_INCOME_TYPE_Retired: int
    NAME_INCOME_TYPE_Working: int
    NAME_EDUCATION_TYPE_Lower: int
    NAME_EDUCATION_TYPE_Secondary: int
    NAME_HOUSING_TYPE_Stable: int
    NAME_HOUSING_TYPE_With_parents: int = Field(alias="NAME_HOUSING_TYPE_With parents")
    OCCUPATION_TYPE_Service: int
    OCCUPATION_TYPE_Unknown: int
    OCCUPATION_TYPE_White_collar: int = Field(alias="OCCUPATION_TYPE_White collar")
    ORGANIZATION_TYPE_Business: int
    ORGANIZATION_TYPE_Government: int
    ORGANIZATION_TYPE_Health_Education: int = Field(alias="ORGANIZATION_TYPE_Health/Education")
    ORGANIZATION_TYPE_Industry: int
    ORGANIZATION_TYPE_Services: int
    ORGANIZATION_TYPE_Trade: int
    ORGANIZATION_TYPE_Transport: int
    
    # Enable alias usage so fields with special characters or spaces can be passed and parsed correctly
    class Config:
        validation_by_name = True
        populate_by_name = True
        allow_population_by_alias = True

# Input schema for a batch of application records
class ApplicationData(BaseModel):
    data: List[ApplicationRecord]

# Root endpoint to verify that the API is running
@app.get("/")
def read_root():
    return {"message": "Risk prediction API is running"}

# Function to predict risk previously defined in the notebook
# Risk prediction using model and defined threshold
def predict_risk(df_month, model):
    df_month = df_month.copy()
    df_month["CREDIT_TO_INCOME"] = df_month["AMT_CREDIT"] / df_month["AMT_INCOME_TOTAL"]

    # Drop columns that shouldn't be used for prediction
    cols_to_drop = [col for col in ["SK_ID_CURR", "TARGET"] if col in df_month.columns]
    X = df_month.drop(columns=cols_to_drop)
    
    # Get probability of fraud (positive class)
    proba = model.predict_proba(X)[:, 1]

    # Assign risk levels based on probability thresholds
    risk = pd.Series("moderate_risk", index=X.index)
    risk[proba < 0.25] = "low_risk"
    risk[proba > 0.45] = "high_risk"

    # Combine probabilities and risk levels into a response list
    return [{"probability": round(p, 4), "risk_level": r} for p, r in zip(proba, risk)]

# Endpoint to receive input data and return predictions
@app.post("/predict", response_model=Dict[str, List[RiskPrediction]])
def predict(data: ApplicationData):
    model = mlflow.sklearn.load_model("models:/xgboost_fraud_detector@prod")
    df_input = pd.DataFrame([record.dict(by_alias=True) for record in data.data])
    results = predict_risk(df_input, model)
    return {"predictions": results}

# To run the server: uvicorn main:app --reload
# API available at http://127.0.0.1:8000/docs
