import joblib
import numpy as np
import pandas as pd
from feature_pipeline import banking_feature_eng

# ১. মডেল এবং প্রসেসড ফাইলগুলো লোড করা
model = joblib.load("bank_churn_predictor.pkl")
threshold = joblib.load("threshold.pkl")
mean_balance = joblib.load("mean_balance.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# ২. কাস্টমার ডেটা (DataFrame)
new_customer = pd.DataFrame(
    {
        "credit_score": [650],
        "country": ["Germany"],
        "gender": ["Male"],
        "age": [45],
        "tenure": [5],
        "balance": [80000],
        "products_number": [2],
        "credit_card": [1],
        "active_member": [1],
        "estimated_salary": [70000],
    }
)

#feature engineering pipeline
new_customer = banking_feature_eng(new_customer)

new_customer_ordered = new_customer.reindex(columns=feature_columns, fill_value=0)

#new binary feature include
new_customer_ordered["is_welthy_active"] = (
    (new_customer_ordered["balance"] > mean_balance)
    & (new_customer_ordered["active_member"] == 1)
).astype(int)

#direct pass dataframe into model
prob = model.predict_proba(new_customer_ordered)[:, 1][0]

print(f"Churn Probability: {prob:.4f}")

if prob >= threshold:
    print("Customer likely to churn.")
else:
    print("Customer likely to stay.")