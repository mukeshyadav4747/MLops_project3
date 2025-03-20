import streamlit as st
import joblib
import pandas as pd

# Load the trained model (Ensure the correct path)
model_path = "api/model.pkl"  # Correct path inside the container
model = joblib.load(model_path)

# Streamlit UI
st.title("Machine Learning Model Predictor")

# Input fields
loan_id = st.number_input("Loan_id", value=0.0)
applicant_income = st.number_input("Applicant_income", value=0.0)
loan_amount = st.number_input("Loan_amount", value=0.0)
loan_term = st.number_input("Loan_term", value=0.0)
credit_score = st.number_input("Credit_score", value=0.0)
employment_years = st.number_input("Employment_years", value=0.0)
interest_rate = st.number_input("Interest_rate", value=0.0)
previous_defaults = st.number_input("Previous_defaults", value=0.0)

# Predict button
if st.button("Predict"):
    features = pd.DataFrame([[loan_id, applicant_income,loan_amount,loan_term, credit_score,employment_years,interest_rate,previous_defaults]])
    prediction = model.predict(features)
    st.success(f"Predicted Value: {int(prediction[0])}")
