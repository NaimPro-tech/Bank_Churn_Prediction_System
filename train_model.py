import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

import joblib


#load dataset
df = pd.read_csv(r"D:\Projects\bank_customer_churn\Bank Customer Churn Prediction.csv")
df_copy = df.copy()


#feature engineering
def banking_feature_eng(dataset):
    df_fe = dataset.copy()

    #interaction feature
    df_fe['balance_salary_ratio'] = df_fe['balance']/(df_fe['estimated_salary']+1)
    df_fe['tenure_age_ratio'] = df_fe['tenure']/(df_fe['age']+1)
    df_fe['credit_per_tenure'] = df_fe['credit_score']/(df_fe['tenure']+1)



    #create bin based on age
    age_bins = [18, 30, 50, 100]
    age_labels = ['Young', 'Middle_Aged', 'Senior']
    df_fe['age_group'] = pd.cut(df_fe['age'], bins=age_bins, labels=age_labels, include_lowest=True)

    #one hot for age_group
    df_fe = pd.get_dummies(df_fe, columns=['age_group', 'country', 'gender'], drop_first=True)

    # capping age outlier with 99 percentile method
    upper_limit = df_fe['age'].quantile(0.99)
    df_fe['age'] = np.where(df_fe['age']>upper_limit, upper_limit, df_fe['age'])

    return df_fe


df = banking_feature_eng(df)


X = df.drop(columns=['customer_id', 'churn'])
y = df['churn']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

mean_balance = X_train['balance'].mean()

X_train['is_welthy_active'] = ((X_train['balance']>mean_balance) & (X_train['active_member'] == 1)).astype(int)
X_val['is_welthy_active'] = ((X_val['balance']>mean_balance) & (X_val['active_member'] == 1)).astype(int)
X_test['is_welthy_active'] = ((X_test['balance']>mean_balance) & (X_test['active_member'] == 1)).astype(int)


# create a estimated ratio of positive & negative class for XGBClassifier parameter
neg = np.sum(y_train == 0)
pos = np.sum(y_train == 1)
scale_weight = neg / pos

model = XGBClassifier(
    max_depth=5,
    n_estimators=200,
    learning_rate=0.1,
    subsample=0.8,
    eval_metric="logloss",
    scale_pos_weight=scale_weight,
    random_state=42
)

model.fit(X_train, y_train)


val_prob = model.predict_proba(X_val)[:, 1]


#threshold tuning on validation set
thresholds = np.arange(0.1, 0.9, 0.01)

best_t = 0
best_f1 = 0

for t in thresholds:
    val_pred = (val_prob >= t).astype(int)
    score = f1_score(y_val, val_pred)

    if score > best_f1:
        best_f1 = score
        best_t = t

# print("Best Threshold:", best_t)
# print("Best Validation F1:", best_f1)


#final test evaluation
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_t).astype(int)

# print("\n===== FINAL TEST RESULTS =====")
# print(classification_report(y_test, test_pred))

thresholds = [0.20,0.25,0.30,0.35,0.40,0.45,0.50]
y_prob = model.predict_proba(X_test)[:,1]
for t in thresholds:
    y_pred = (y_prob>=t).astype(int)
    
    # print(f"\nThreshold:{t}")
    # print(classification_report(y_test, y_pred))


precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

target_recall = 0.80
idx = np.where(recall >= target_recall)[0][-1]
best_threshold = thresholds[idx]
# print(best_threshold)

test_pred = (test_prob>=best_threshold).astype(int)


joblib.dump(model, "bank_churn_predictor.pkl")
joblib.dump(best_threshold, "threshold.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")
joblib.dump(mean_balance, "mean_balance.pkl")



