import streamlit as st
import pandas as pd
import numpy as np

from data_visualization import visualize_data
from get_top_features import display_fraud_reasons
from data_requirement import display_data_requirements
from data_processing import preprocess_data, validate_data, feature_engineering, display_data_distribution
from model_training import train_and_predict, plot_roc_curve

# Streamlit app
st.title('Fraud Detection Dashboard')
st.write('Read the instructions on the sidebar to get started.')

# Display data requirements
display_data_requirements()

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])


# Update the Streamlit app to include the fraud reasons
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Validate the data
    is_valid, error_message = validate_data(data)
    if not is_valid:
        st.error(f"Data validation failed: {error_message}")
    else:
        data = preprocess_data(data)
        X, y = feature_engineering(data)
        train_acc, test_acc, conf_matrix, class_report, X_test, y_test, y_pred, voting_model, selected_feature_names, pipeline = train_and_predict(X, y)

        # show the selected features
        st.write("### Selected Features")
        st.write(selected_feature_names)

        st.write(f"Training Accuracy: {train_acc}")
        st.write(f"Testing Accuracy: {test_acc}")
        st.write("Confusion Matrix:")
        st.write(conf_matrix)
        st.write("Classification Report:")
        st.write(class_report)

        # Plot ROC curve
        # Ensure X_test is transformed using the same pipeline
        X_test_transformed = pipeline.transform(X_test)

        # Plot ROC curve
        y_pred_prob = voting_model.predict_proba(X_test_transformed)
        plot_roc_curve(y_test, y_pred_prob)

        # Ensure X_test is a DataFrame with the correct number of columns
        if isinstance(X_test, np.ndarray):
            X_test = pd.DataFrame(X_test, columns=selected_feature_names)

        # List records flagged as fraud
        fraud_indices = y_pred == 1
        fraud_records = X_test[fraud_indices]

        test_indices = X_test.index[fraud_indices]
        original_fraud_records = data.iloc[test_indices]

        # Display the count of fraudulent records
        st.write(f"Number of records flagged as fraud: {fraud_records.shape[0]}")

        # Display the original records flagged as fraud
        st.write("Original records flagged as fraud:")
        st.write(original_fraud_records)

        # Display reasons for flagging records as fraud
        display_fraud_reasons(voting_model, X_test_transformed, fraud_indices, selected_feature_names)

        st.write("### Dataset Preview")
        st.write(data.head())

        # Display data distribution
        display_data_distribution(data)

        # Visualize data
        visualize_data(data)