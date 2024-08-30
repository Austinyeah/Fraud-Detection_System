import streamlit as st

def display_data_requirements():
    st.sidebar.header('Data Requirements')
    st.sidebar.write("""
    ### Required Columns and Data Types

    Here’s what your dataset should contain:

    | Column Name             | Description                                         | Expected Data Type   |
    |-------------------------|-----------------------------------------------------|-----------------------|
    | `fraud_reported`        | Indicates whether fraud has been reported (e.g., 'Y' for yes, 'N' for no). | Categorical (String)  |
    | `policy_csl`            | Customer service level or policy type.             | Categorical (String)  |
    | `collision_type`        | Type of collision involved in the incident.        | Categorical (String)  |
    | `property_damage`       | Indicates whether there was property damage (e.g., 'Y' for yes, 'N' for no). | Categorical (String)  |
    | `police_report_available` | Indicates if a police report is available (e.g., 'Y' for yes, 'N' for no). | Categorical (String)  |
    | `age`                   | Age of the insured or policyholder.                 | Numerical (Integer)   |
    | `total_claim_amount`    | The total amount claimed.                          | Numerical (Float)     |
    | `months_as_customer`    | Number of months the customer has been with the company. | Numerical (Integer)   |


    ### Optional Columns

    - `policy_number`: Unique identifier for the insurance policy.
    - `policy_bind_date`: Date when the policy was bound.
    - `insured_zip`: ZIP code of the insured.
    - `incident_date`: Date when the incident occurred.
    - `incident_location`: Location of the incident.
    - `auto_make`, `auto_model`, `auto_year`: Details of the vehicle involved.

    **Note:** If any of these columns are missing, the processing may not work correctly. Ensure your dataset follows these guidelines.
    """)