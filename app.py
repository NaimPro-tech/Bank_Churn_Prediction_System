import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from schemas import CustomerData
from utils import banking_feature_eng, model, threshold, feature_columns, mean_balance

#fastapi app initiate
app = FastAPI(title="Bank Customer Churn Predictor API")

@app.get("/")
def home():
    return {"message":"Bank Customer Churn Prediction API is Running..."}

@app.post("/predict")
def predict_churn(data: CustomerData):
    input_data = pd.DataFrame([data.model_dump()])

    processed_df = banking_feature_eng(input_data)
    ordered_df = processed_df.reindex(columns=feature_columns, fill_value=0)

    ordered_df['is_welthy_active'] = (1 if (ordered_df['balance']>mean_balance) and (ordered_df['active_member']== 1) else 0)
    prob = model.predict_proba(ordered_df)[:,1][0]

    decision = (
        "Customer Likely to Churn."
        if prob>threshold
        else "Customer Likely to Stay."
    )

    return {
        "churn_probability": float(prob),
        "decision": decision,
        "applied_threshold": float(threshold)
    }


