# AIMLInHealthcare.py

import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF

# Load the trained model
model_path = 'healthcare_model.pkl'
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Set page configuration
st.set_page_config(
    page_title="AI Predictive Methods for Healthcare",
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
        <h1>AI Predictive Methods for Healthcare</h1>
        <p>Revolutionizing healthcare with AI-driven predictive analytics for smarter, faster decisions!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state
if "patient_details" not in st.session_state:
    st.session_state["patient_details"] = {
        "age": 30,
        "sex": 1,  # 1 = Male, 0 = Female
        "cp": 0,  # Chest pain type (0-3)
        "trestbps": 120,  # Resting blood pressure
        "chol": 200,  # Serum cholesterol
        "fbs": 0,  # Fasting blood sugar > 120 mg/dL (1 = true, 0 = false)
        "restecg": 0,  # Resting electrocardiographic results (0-2)
        "thalach": 150,  # Maximum heart rate achieved
        "exang": 0,  # Exercise-induced angina (1 = yes, 0 = no)
        "oldpeak": 1.0,  # ST depression induced by exercise
        "slope": 1,  # Slope of the peak exercise ST segment (0-2)
        "ca": 0,  # Number of major vessels (0-3)
        "thal": 2,  # Thalassemia (1-3)
    }

# Navigation menu
step = st.radio(
    "Navigate through the steps:",
    ["Patient Information", "Health Details", "Final Diagnosis"]
)

# Step 1: Patient Information
if step == "Patient Information":
    st.markdown("### Step 1: Patient Information")
    st.session_state["patient_details"]["age"] = st.number_input("Age:", min_value=1, max_value=100, value=st.session_state["patient_details"]["age"])
    st.session_state["patient_details"]["sex"] = st.selectbox("Sex:", ["Male", "Female"], index=0 if st.session_state["patient_details"]["sex"] == 1 else 1)

# Step 2: Health Details
elif step == "Health Details":
    st.markdown("### Step 2: Health Details")
    st.session_state["patient_details"]["cp"] = st.number_input("Chest Pain Type (0-3):", min_value=0, max_value=3, value=st.session_state["patient_details"]["cp"])
    st.session_state["patient_details"]["trestbps"] = st.number_input("Resting Blood Pressure (mmHg):", min_value=0, value=st.session_state["patient_details"]["trestbps"])
    st.session_state["patient_details"]["chol"] = st.number_input("Serum Cholesterol (mg/dL):", min_value=0, value=st.session_state["patient_details"]["chol"])
    st.session_state["patient_details"]["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dL:", ["No", "Yes"], index=st.session_state["patient_details"]["fbs"])
    st.session_state["patient_details"]["restecg"] = st.number_input("Resting Electrocardiographic Results (0-2):", min_value=0, max_value=2, value=st.session_state["patient_details"]["restecg"])
    st.session_state["patient_details"]["thalach"] = st.number_input("Maximum Heart Rate Achieved:", min_value=0, value=st.session_state["patient_details"]["thalach"])
    st.session_state["patient_details"]["exang"] = st.selectbox("Exercise-Induced Angina:", ["No", "Yes"], index=st.session_state["patient_details"]["exang"])
    st.session_state["patient_details"]["oldpeak"] = st.number_input("ST Depression Induced by Exercise:", min_value=0.0, value=st.session_state["patient_details"]["oldpeak"])
    st.session_state["patient_details"]["slope"] = st.number_input("Slope of the Peak Exercise ST Segment (0-2):", min_value=0, max_value=2, value=st.session_state["patient_details"]["slope"])
    st.session_state["patient_details"]["ca"] = st.number_input("Number of Major Vessels (0-3):", min_value=0, max_value=3, value=st.session_state["patient_details"]["ca"])
    st.session_state["patient_details"]["thal"] = st.number_input("Thalassemia (1-3):", min_value=1, max_value=3, value=st.session_state["patient_details"]["thal"])

# Step 3: Final Diagnosis
elif step == "Final Diagnosis":
    st.markdown("### Step 3: Final Diagnosis")
    patient_details = st.session_state["patient_details"]

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "age": [patient_details["age"]],
        "sex": [patient_details["sex"]],
        "cp": [patient_details["cp"]],
        "trestbps": [patient_details["trestbps"]],
        "chol": [patient_details["chol"]],
        "fbs": [patient_details["fbs"]],
        "restecg": [patient_details["restecg"]],
        "thalach": [patient_details["thalach"]],
        "exang": [patient_details["exang"]],
        "oldpeak": [patient_details["oldpeak"]],
        "slope": [patient_details["slope"]],
        "ca": [patient_details["ca"]],
        "thal": [patient_details["thal"]],
    })

    # Prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        if prediction[0] == 1:
            st.markdown("### Diagnosis: High Risk ❌")
            st.error(f"Risk Probability: {prediction_proba[0][1]:.2f}")
        else:
            st.markdown("### Diagnosis: Low Risk ✅")
            st.success(f"Low Risk Probability: {prediction_proba[0][0]:.2f}")

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", style="BU", size=12)  # Set font to bold and underline
        pdf.cell(200, 10, txt="Health Risk Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)  # Reset font to normal
        pdf.ln(10)

        # Health Details
        pdf.cell(200, 10, txt="Health Details:", ln=True)
        pdf.cell(200, 10, txt=f"Age: {patient_details.get('age', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Sex: {'Male' if patient_details.get('sex', 1) == 1 else 'Female'}", ln=True)
        pdf.cell(200, 10, txt=f"Chest Pain Type: {patient_details.get('cp', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Resting Blood Pressure: {patient_details.get('trestbps', 'N/A')} mmHg", ln=True)
        pdf.cell(200, 10, txt=f"Serum Cholesterol: {patient_details.get('chol', 'N/A')} mg/dL", ln=True)
        pdf.cell(200, 10, txt=f"Fasting Blood Sugar > 120 mg/dL: {'Yes' if patient_details.get('fbs', 0) == 1 else 'No'}", ln=True)
        pdf.cell(200, 10, txt=f"Resting Electrocardiographic Results: {patient_details.get('restecg', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Maximum Heart Rate Achieved: {patient_details.get('thalach', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Exercise-Induced Angina: {'Yes' if patient_details.get('exang', 0) == 1 else 'No'}", ln=True)
        pdf.cell(200, 10, txt=f"ST Depression Induced by Exercise: {patient_details.get('oldpeak', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Slope of the Peak Exercise ST Segment: {patient_details.get('slope', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Number of Major Vessels: {patient_details.get('ca', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Thalassemia: {patient_details.get('thal', 'N/A')}", ln=True)
        pdf.ln(10)

        # Prediction Results
        pdf.cell(200, 10, txt="Prediction Results:", ln=True)
        pdf.cell(200, 10, txt=f"Prediction: {'High Risk' if prediction[0] == 1 else 'Low Risk'}", ln=True)
        pdf.cell(200, 10, txt=f"Low Risk Probability: {prediction_proba[0][0]:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"High Risk Probability: {prediction_proba[0][1]:.2f}", ln=True)

        # Save PDF to buffer
        buffer = BytesIO()
        pdf_bytes = pdf.output(dest="S").encode("latin1")  # Encode the PDF content
        buffer.write(pdf_bytes)
        buffer.seek(0)

        st.download_button(
            label="Download Report as PDF",
            data=buffer,
            file_name="health_risk_prediction_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Footer
st.markdown(
    """
    <footer>
        <p>© 2025 AI Predictive Methods for Healthcare. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
