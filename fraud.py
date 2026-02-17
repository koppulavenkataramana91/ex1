import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load df and pipeline
data = pickle.load(open('data.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.title("Insurance Claim Fraud Detection PlatForm")

# Collect inputs
months_as_customer = st.slider("Months as Customer", 1, 240, 120)
age = st.slider("Age of Insured", 18, 100, 45)
policy_deductable = st.selectbox("Policy Deductible", data['policy_deductable'].unique())
policy_annual_premium = st.number_input("Policy Annual Premium", min_value=200.0, max_value=5000.0, value=1200.5, step=50.0)
umbrella_limit = st.number_input("Umbrella Limit", min_value=0, max_value=10000000, value=0, step=1000)
insured_zip = st.text_input("Insured ZIP Code", "452748")
insured_sex = st.radio("Insured Sex", data['insured_sex'].unique())
capital_gains = st.number_input("Capital Gains", min_value=0, max_value=100000, value=5000, step=100)
capital_loss = st.number_input("Capital Loss", min_value=-100000, max_value=0, value=0, step=100)
incident_hour_of_the_day = st.slider("Incident Hour of the Day", 0, 23, 14)

# Auto make flags
auto_make = st.selectbox("Auto Make", ["Dodge","Ford","Honda","Jeep","Mercedes","Nissan","Saab","Suburu","Toyota","Volkswagen"])
auto_flags = {f"auto_make_{brand}": (auto_make == brand) for brand in ["Dodge","Ford","Honda","Jeep","Mercedes","Nissan","Saab","Suburu","Toyota","Volkswagen"]}

if st.button("PREDICT FRAUD"):
    # Build query dict
    query_dict = {
        "months_as_customer": months_as_customer,
        "age": age,
        "policy_deductable": policy_deductable,
        "policy_annual_premium": policy_annual_premium,
        "umbrella_limit": umbrella_limit,
        "insured_zip": int(insured_zip) if insured_zip.isdigit() else 0,
        "insured_sex": insured_sex,
        "capital-gains": capital_gains,
        "capital-loss": capital_loss,
        "incident_hour_of_the_day": incident_hour_of_the_day,
        **auto_flags
    }

    # Create DataFrame with same columns as training data
    query_df = pd.DataFrame([query_dict])
    query_df = query_df.reindex(columns=data.columns, fill_value=0)

    # Predict
    op = pipe.predict(query_df)
    if op[0] == 1:
        st.subheader("⚠️ This claim is predicted as FRAUDULENT")
    else:
        st.subheader("✅ This claim is predicted as LEGITIMATE")
