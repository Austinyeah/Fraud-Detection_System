import streamlit as st
import pandas as pd
import numpy as np

from model_training import get_feature_importances


def interpret_standardized_values(record):
    interpretation = {}
    interpretation["Vehicle Claim Amount"] = (
        f"{record['num__vehicle_claim']:.2f} - This value represents the standardized amount claimed for vehicle damage. "
        "A positive value indicates that the claim amount is above the average claim amount in the dataset."
    )
    interpretation["Injury Claim Amount"] = (
        f"{record['num__injury_claim']:.2f} - This value represents the standardized amount claimed for injuries. "
        "A positive value indicates that the claim amount is above the average claim amount in the dataset."
    )
    interpretation["Months as Customer"] = (
        f"{record['num__months_as_customer']:.2f} - This value represents the standardized number of months the individual has been a customer. "
        "A negative value indicates that the customer has been with the company for fewer months than the average customer in the dataset."
    )
    interpretation["Property Claim Amount"] = (
        f"{record['num__property_claim']:.2f} - This value represents the standardized amount claimed for property damage. "
        "A negative value indicates that the claim amount is below the average claim amount in the dataset."
    )
    interpretation["Capital Gains"] = (
        f"{record['num__capital-gains']:.2f} - This value represents the standardized gains from capital investments. "
        "A positive value indicates that the capital gains are significantly above the average gains in the dataset."
    )
    return interpretation

def display_fraud_reasons(voting_model, X_test, fraud_indices, selected_feature_names):
    # Ensure X_test is a DataFrame with the correct number of columns
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=selected_feature_names)

    feature_names = X_test.columns
    feature_importances = get_feature_importances(voting_model, feature_names)
    top_features = feature_importances.head(5).index.tolist()

    st.write("The following features contributed most to the fraud prediction:")

    for feature in top_features:
        st.write(f"- {feature}")

    feature_descriptions = {
        "num__vehicle_claim": "Vehicle Claim Amount",
        "num__injury_claim": "Injury Claim Amount",
        "num__months_as_customer": "Months as Customer",
        "num__property_claim": "Property Claim Amount",
        "num__capital-gains": "Capital Gains"
    }

    st.write("### Reasons for Flagging Records as Fraud")
    for i, record in X_test[fraud_indices].iterrows():
        st.write(f"Record {i}:")
        for feature in top_features:
            readable_feature = feature_descriptions.get(feature, feature)
            st.write(f"- {readable_feature}: {record[feature]}")

        # interpretation
        st.write("#### Interpretation:")
        interpretation = interpret_standardized_values(record)
        for feature, explanation in interpretation.items():
            st.write(f"- {feature}: {explanation}")