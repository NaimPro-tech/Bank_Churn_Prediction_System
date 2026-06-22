import numpy as np
import pandas as pd

def banking_feature_eng(dataset):
    df_fe = dataset.copy()

    df_fe['balance_salary_ratio'] = df_fe['balance']/(df_fe['estimated_salary']+1)
    df_fe['tenure_age_ratio'] = df_fe['tenure']/(df_fe['age']+1)
    df_fe['credit_per_tenure'] = df_fe['credit_score']/(df_fe['tenure']+1)

    age_bins = [18, 30, 50, 100]
    age_labels = ['Young', 'Middle_Aged', 'Senior']
    df_fe['age_group'] = pd.cut(df_fe['age'], bins=age_bins,
                                labels=age_labels, include_lowest=True)

    df_fe = pd.get_dummies(df_fe, columns=['age_group'], drop_first=True)

    upper_limit = df_fe['age'].quantile(0.99)
    df_fe['age'] = np.where(df_fe['age'] > upper_limit,
                            upper_limit, df_fe['age'])

    return df_fe