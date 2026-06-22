import joblib
import pandas as pd
import numpy as np
from feature_pipeline import banking_feature_eng


model = joblib.load("bank_churn_predictor.pkl")
threshold = joblib.load("threshold.pkl")
mean_balance = joblib.load("mean_balance.pkl")
feature_columns = joblib.load("feature_columns.pkl")


new_customer = pd.DataFrame({
    "credit_score": [650],
    "country": ["Germany"],
    "gender": ["Male"],
    "age": [45],
    "tenure": [5],
    "balance": [80000],
    "products_number": [2],
    "credit_card": [1],
    "active_member": [1],
    "estimated_salary": [70000]
})



new_customer = banking_feature_eng(new_customer)


new_customer["is_welthy_active"] = (
    (new_customer["balance"] > mean_balance) &
    (new_customer["active_member"] == 1)
).astype(int)

for col in feature_columns:
    if col not in new_customer.columns:
        new_customer[col] = 0

new_customer = new_customer[feature_columns]


prob = model.predict_proba(new_customer.to_numpy())[:, 1][0]


print(f"Churn Probability: {prob:.4f}")

if prob >= threshold:
    print("Customer likely to churn.")
else:
    print("Customer likely to stay.")