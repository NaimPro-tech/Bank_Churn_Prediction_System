import joblib

#load model and necessary file
model = joblib.load("bank_churn_predictor.pkl")
threshold = joblib.load("threshold.pkl")
feature_columns = joblib.load("feature_columns.pkl")
mean_balance = joblib.load("mean_balance.pkl")

#featuure engineering to process main features
def banking_feature_eng(df):

    df['balance_salary_ratio'] = df['balance']/(df['estimated_salary']+1)
    df['tenure_age_ratio'] = df['tenure']/(df['age']+1)
    df['credit_per_tenure'] = df['credit_score']/(df['tenure']+1)

    #create age group based on different ages
    age = df['age']
    df['age_group_Middle_Aged'] = 1 if 30 < age <=50 else 0
    df['age_group_Senior_Aged'] = 1 if age>50 else 0

    #create direct country column
    df["country_Germany"] = 1 if df["country"] == "Germany" else 0
    df["country_Spain"] = 1 if df["country"] == "Spain" else 0
    df["gender_Male"] = 1 if df["gender"] == "Male" else 0

    return df


