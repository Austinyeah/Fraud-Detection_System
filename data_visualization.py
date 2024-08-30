import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io



def visualize_data(data):
    st.write("### Data Visualization")

    # Select columns to visualize
    columns_to_visualize = st.multiselect("Select columns to visualize", data.columns)

    if columns_to_visualize:
        for column in columns_to_visualize:
            st.write(f"#### {column}")
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            # Histogram
            sns.histplot(data[column], kde=True, ax=ax[0])
            ax[0].set_title(f"Histogram of {column}")

            # Box plot
            sns.boxplot(x=data[column], ax=ax[1])
            ax[1].set_title(f"Box plot of {column}")

            st.pyplot(fig)


def generate_report(conf_matrix, class_report, fraud_records):
    report = io.StringIO()
    report.write("Confusion Matrix:\n")
    report.write(str(conf_matrix))
    report.write("\n\nClassification Report:\n")
    report.write(class_report)
    report.write("\n\nFraudulent Records:\n")

    # Convert fraud_records to a DataFrame
    fraud_records_df = pd.DataFrame(fraud_records)

    fraud_records_df.to_csv(report, index=False)
    return report.getvalue()