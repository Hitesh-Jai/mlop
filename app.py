import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------
# Load model
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "loan_model.joblib"

model = joblib.load(MODEL_PATH)

st.set_page_config(page_title="Loan Risk Predictor", layout="centered")
st.title("🏦 Loan Risk Prediction")

# -------------------------
# Form
# -------------------------
with st.form("loan_form"):
    age = st.number_input("Age", min_value=18, max_value=100)
    income = st.number_input("Income", min_value=0)
    
    employment = st.selectbox(
        "Employment Type",
        ["Salaried", "Self-employed"]
    )

    residence = st.selectbox(
        "Residence Type",
        ["Rented", "Owned"]
    )

    credit_score = st.number_input("Credit Score", min_value=300, max_value=900)
    loan_amount = st.number_input("Loan Amount", min_value=0)
    loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60])

    previous_default = st.radio(
        "Previous Default?",
        ["Yes", "No"]
    )

    submit = st.form_submit_button("Predict Risk")

# -------------------------
# Prediction
# -------------------------
if submit:
    input_df = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "EmploymentType": employment,
        "ResidenceType": residence,
        "CreditScore": credit_score,
        "LoanAmount": loan_amount,
        "LoanTerm": loan_term,
        "PreviousDefault": previous_default
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"📊 Predicted Risk Category: **{prediction}**")
