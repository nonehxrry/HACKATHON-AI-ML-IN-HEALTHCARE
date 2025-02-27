import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF
from transformers import pipeline
from langdetect import detect
import math
import os

# Set page configuration
st.set_page_config(
    page_title="AI Predictive Methods for Healthcare Analysis",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to right, #ffffff, #e6f7ff);
            font-family: 'Arial', sans-serif;
        }
        .header-container {
            background: linear-gradient(to right, #4CAF50, #5ecf5e);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        }
        .header-container h1 {
            font-size: 40px;
        }
        .header-container p {
            font-size: 20px;
            margin-top: 5px;
        }
        footer {
            text-align: center;
            margin-top: 50px;
            font-size: 14px;
            color: #666;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.markdown(
    """
    <div class="header-container">
        <h1>AI Predictive Methods for Healthcare Analysis</h1>
        <p>Revolutionizing healthcare with AI-driven predictive analytics for smarter, faster decisions!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Load the trained model
model_path = 'healthcare_model.pkl'  # Path to the model file
try:
    with st.spinner("Loading model..."):
        model = joblib.load(model_path)  # Corrected: Load the model
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Initialize session state
if "patient_details" not in st.session_state:
    st.session_state["patient_details"] = {
        "full_name": "",
        "age": 30,
        "gender": "Male",
        "blood_pressure": 120,
        "cholesterol": 200,
        "bmi": 25,
        "glucose": 100,
        "smoking_status": "Non-smoker",
        "alcohol_consumption": "None",
        "physical_activity": "Moderate",
        "family_history": "No",
        "symptoms": "",
        "diagnosis": "",
        "treatment": ""
    }

# Navigation menu
step = st.radio(
    "Navigate through the steps:",
    ["Patient Information", "Health Metrics", "Diagnosis & Treatment", "Final Report"]
)

# Step 1: Patient Information
if step == "Patient Information":
    st.markdown("### Step 1: Patient Information")
    st.session_state["patient_details"]["full_name"] = st.text_input("Full Name", st.session_state["patient_details"]["full_name"])
    st.session_state["patient_details"]["age"] = st.number_input("Age", min_value=0, max_value=120, value=st.session_state["patient_details"]["age"])
    st.session_state["patient_details"]["gender"] = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state["patient_details"]["gender"] == "Male" else 1)

# Step 2: Health Metrics
elif step == "Health Metrics":
    st.markdown("### Step 2: Health Metrics")
    st.session_state["patient_details"]["blood_pressure"] = st.number_input("Blood Pressure (mmHg)", min_value=0, value=st.session_state["patient_details"]["blood_pressure"])
    st.session_state["patient_details"]["cholesterol"] = st.number_input("Cholesterol (mg/dL)", min_value=0, value=st.session_state["patient_details"]["cholesterol"])
    st.session_state["patient_details"]["bmi"] = st.number_input("BMI", min_value=0.0, value=float(st.session_state["patient_details"]["bmi"]))
    st.session_state["patient_details"]["glucose"] = st.number_input("Glucose Level (mg/dL)", min_value=0, value=st.session_state["patient_details"]["glucose"])
    st.session_state["patient_details"]["smoking_status"] = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"], index=0 if st.session_state["patient_details"]["smoking_status"] == "Non-smoker" else 1)
    st.session_state["patient_details"]["alcohol_consumption"] = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"], index=["None", "Light", "Moderate", "Heavy"].index(st.session_state["patient_details"]["alcohol_consumption"]))
    st.session_state["patient_details"]["physical_activity"] = st.selectbox("Physical Activity", ["None", "Light", "Moderate", "Heavy"], index=["None", "Light", "Moderate", "Heavy"].index(st.session_state["patient_details"]["physical_activity"]))
    st.session_state["patient_details"]["family_history"] = st.selectbox("Family History of Diseases", ["No", "Yes"], index=0 if st.session_state["patient_details"]["family_history"] == "No" else 1)

# Step 3: Diagnosis & Treatment
elif step == "Diagnosis & Treatment":
    st.markdown("### Step 3: Diagnosis & Treatment")
    st.session_state["patient_details"]["symptoms"] = st.text_area("Symptoms", st.session_state["patient_details"]["symptoms"])

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "age": [st.session_state["patient_details"]["age"]],
        "gender": [1 if st.session_state["patient_details"]["gender"] == "Female" else 0],
        "blood_pressure": [st.session_state["patient_details"]["blood_pressure"]],
        "cholesterol": [st.session_state["patient_details"]["cholesterol"]],
        "bmi": [st.session_state["patient_details"]["bmi"]],
        "glucose": [st.session_state["patient_details"]["glucose"]],
        "smoking_status": [1 if st.session_state["patient_details"]["smoking_status"] == "Smoker" else 0],
        "alcohol_consumption_light": [1 if st.session_state["patient_details"]["alcohol_consumption"] == "Light" else 0],
        "alcohol_consumption_moderate": [1 if st.session_state["patient_details"]["alcohol_consumption"] == "Moderate" else 0],
        "alcohol_consumption_heavy": [1 if st.session_state["patient_details"]["alcohol_consumption"] == "Heavy" else 0],
        "physical_activity_light": [1 if st.session_state["patient_details"]["physical_activity"] == "Light" else 0],
        "physical_activity_moderate": [1 if st.session_state["patient_details"]["physical_activity"] == "Moderate" else 0],
        "physical_activity_heavy": [1 if st.session_state["patient_details"]["physical_activity"] == "Heavy" else 0],
        "family_history": [1 if st.session_state["patient_details"]["family_history"] == "Yes" else 0],
    })

    # Ensure input data matches the model's expected features
    if hasattr(model, 'feature_names_in_'):
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.markdown("### Diagnosis: High Risk of Disease ❌")
            st.error(f"Risk Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.markdown("### Diagnosis: Low Risk of Disease ✅")
            st.success(f"Low Risk Probability: {prediction_proba[0][0]:.2f}")

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", style="BU", size=12)
        pdf.cell(200, 10, txt="Healthcare Analysis Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        # Patient Information
        pdf.cell(200, 10, txt="Patient Information:", ln=True)
        pdf.cell(200, 10, txt=f"Full Name: {st.session_state['patient_details'].get('full_name', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Age: {st.session_state['patient_details'].get('age', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Gender: {st.session_state['patient_details'].get('gender', 'N/A')}", ln=True)
        pdf.ln(10)

        # Health Metrics
        pdf.cell(200, 10, txt="Health Metrics:", ln=True)
        pdf.cell(200, 10, txt=f"Blood Pressure: {st.session_state['patient_details'].get('blood_pressure', 'N/A')} mmHg", ln=True)
        pdf.cell(200, 10, txt=f"Cholesterol: {st.session_state['patient_details'].get('cholesterol', 'N/A')} mg/dL", ln=True)
        pdf.cell(200, 10, txt=f"BMI: {st.session_state['patient_details'].get('bmi', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Glucose Level: {st.session_state['patient_details'].get('glucose', 'N/A')} mg/dL", ln=True)
        pdf.ln(10)

        # Diagnosis Results
        pdf.cell(200, 10, txt="Diagnosis Results:", ln=True)
        pdf.cell(200, 10, txt=f"Diagnosis: {'High Risk of Disease' if prediction[0] == 1 else 'Low Risk of Disease'}", ln=True)
        pdf.cell(200, 10, txt=f"Low Risk Probability: {prediction_proba[0][0]:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"High Risk Probability: {prediction_proba[0][1]:.2f}", ln=True)

        # Save PDF to buffer
        buffer = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        buffer.write(pdf_bytes)
        buffer.seek(0)

        st.download_button(
            label="Download Report as PDF",
            data=buffer,
            file_name="healthcare_analysis_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Step 4: Final Report
elif step == "Final Report":
    st.markdown("### Step 4: Final Report")
    patient_details = st.session_state["patient_details"]

    # Display final report
    st.markdown("### Patient Information")
    st.write(f"**Full Name:** {patient_details['full_name']}")
    st.write(f"**Age:** {patient_details['age']}")
    st.write(f"**Gender:** {patient_details['gender']}")

    st.markdown("### Health Metrics")
    st.write(f"**Blood Pressure:** {patient_details['blood_pressure']} mmHg")
    st.write(f"**Cholesterol:** {patient_details['cholesterol']} mg/dL")
    st.write(f"**BMI:** {patient_details['bmi']}")
    st.write(f"**Glucose Level:** {patient_details['glucose']} mg/dL")
    st.write(f"**Smoking Status:** {patient_details['smoking_status']}")
    st.write(f"**Alcohol Consumption:** {patient_details['alcohol_consumption']}")
    st.write(f"**Physical Activity:** {patient_details['physical_activity']}")
    st.write(f"**Family History of Diseases:** {patient_details['family_history']}")

    st.markdown("### Diagnosis & Treatment")
    st.write(f"**Symptoms:** {patient_details['symptoms']}")
    st.write(f"**Diagnosis:** {patient_details['diagnosis']}")
    st.write(f"**Treatment:** {patient_details['treatment']}")

# Footer
st.markdown(
    """
    <footer>
        <p>© 2025 AI Predictive Methods for Healthcare Analysis. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
