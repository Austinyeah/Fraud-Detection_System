import pandas as pd
import numpy as np
import pandera as pa
from pandera import Column, Check
import streamlit as st

# Schema for data validation
schema = pa.DataFrameSchema({
    "policy_number": Column(int, nullable=False),
    "age": Column(int, Check(lambda s: s > 0), nullable=False),
    "fraud_reported": Column(str, Check.isin(["Y", "N"]), nullable=False),
    # Add more columns as needed
})

# Function to validate data
def validate_data(data):
    try:
        schema.validate(data)
        return True, None
    except pa.errors.SchemaError as e:
        return False, str(e)

# Function to preprocess the data
def preprocess_data(data):
    data.replace('?', np.nan, inplace=True)
    columns_to_fill = ['collision_type', 'property_damage', 'police_report_available']
    for column in columns_to_fill:
        data[column] = data[column].fillna(data[column].mode()[0])
    data.drop(['policy_number', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location', 'incident_date',
               'incident_state', 'incident_city', 'insured_hobbies', 'auto_make', 'auto_model', 'auto_year', '_c39'],
              axis=1, inplace=True)
    data.drop(columns=['age', 'total_claim_amount'], inplace=True, axis=1)
    return data

# Function to perform feature engineering
def feature_engineering(data):
    X = data.drop('fraud_reported', axis=1)
    y = data['fraud_reported'].map({'Y': 1, 'N': 0})  # Map Y/N to 1/0
    categorical_cols = X.select_dtypes(include=['object'])
    categorical_cols = pd.get_dummies(categorical_cols, drop_first=True)
    numerical_col = X.select_dtypes(include=['int64'])
    X = pd.concat([numerical_col, categorical_cols], axis=1)
    return X, y

# Function to display data distribution statistics
def display_data_distribution(data):
    st.write("### Data Distribution")
    st.write(data.describe())