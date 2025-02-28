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
model_path = 'healthcare_model.pkl'  # Corrected: Assign the file path directly
try:
    model = joblib.load
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
    if "patient_details" not in st.session_state:
        st.session_state["patient_details"] = {"bmi": 0.0}  # Initialize BMI as a float

    # Ensure the BMI value is a float
    bmi_value = float(st.session_state["patient_details"]["bmi"])

    # Display BMI input widget
    st.session_state["patient_details"]["bmi"] = st.number_input(
        "BMI",
        min_value=0.0,  # Ensure this is a float
        value=bmi_value  # Use the explicitly converted float value
    )

    # Display the current BMI value
    st.write(f"Current BMI: {st.session_state['patient_details']['bmi']}")
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

    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    # Check if the model has the feature_names_in_ attribute
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        # If feature_names_in_ is not available, manually specify the feature names
        # Replace this list with the actual feature names used during model training
        feature_names = ['age', 'height', 'weight', 'bmi', 'blood_pressure']  # Example feature names

    # Reindex the input data to match the model's expected features
    # Fill missing columns with 0 (or another default value)
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

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
        pdf.set_font("Arial", style="BU", size=12)  # Set font to bold and underline
        pdf.cell(200, 10, txt="Healthcare Analysis Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)  # Reset font to normal
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

        # Save PDF to bytes
        pdf_bytes = pdf.output(dest="S")

        # Download PDF
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
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
        <p> 2025 AI Predictive Methods for Healthcare Analysis. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)

# --- Chatbot in Sidebar ---
st.sidebar.markdown("## 🤖 AI Healthcare Chatbot")

# --- Initialize Chat History & Session Variables ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "bot", "content": "👋 Hello! You can speak or type your question.\n\n**📌 Categories:**\n- Disease Diagnosis 🩺\n- Treatment Recommendations 💊\n- Health Tips 🍎\n- Symptom Checker 🤒\n- Mental Health Support 🧠"}
    ]
if "last_topic" not in st.session_state:
    st.session_state["last_topic"] = None  # Track conversation topic
if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""  # Track the input field value

# --- Smarter Chatbot Response System ---
def chatbot_response(user_message):
    user_message = user_message.lower().strip()

    # Standard Greetings
    greetings = ["hello", "hi", "hey", "how are you"]
    if user_message in greetings:
        return "👋 Hello! How can I assist you today? You can ask about disease diagnosis, treatment, or health tips!"

    # Disease Diagnosis
    if "diagnosis" in user_message or "disease" in user_message:
        st.session_state["last_topic"] = "diagnosis"
        return "🩺 **Disease Diagnosis:**\n- **Symptoms:** Describe your symptoms for a preliminary diagnosis.\n- **Risk Factors:** Provide your health metrics for a detailed analysis."

    # Treatment Recommendations
    if "treatment" in user_message or "medicine" in user_message:
        st.session_state["last_topic"] = "treatment"
        return "💊 **Treatment Recommendations:**\n- **Medications:** Based on your diagnosis, we can recommend medications.\n- **Lifestyle Changes:** Suggestions for diet, exercise, and other lifestyle changes."

    # Health Tips
    if "health tips" in user_message or "healthy living" in user_message:
        st.session_state["last_topic"] = "health tips"
        return "🍎 **Health Tips:**\n- **Diet:** Eat a balanced diet rich in fruits and vegetables.\n- **Exercise:** Regular physical activity is essential.\n- **Sleep:** Ensure 7-9 hours of sleep per night."

    # Symptom Checker
    if "symptom" in user_message or "checker" in user_message:
        st.session_state["last_topic"] = "symptom checker"
        return "🤒 **Symptom Checker:**\n- **Common Symptoms:** Fever, cough, headache, etc.\n- **Severe Symptoms:** Chest pain, difficulty breathing, etc."

    # Mental Health Support
    if "mental health" in user_message or "stress" in user_message:
        st.session_state["last_topic"] = "mental health"
        return "🧠 **Mental Health Support:**\n- **Counseling:** Seek professional help if needed.\n- **Relaxation Techniques:** Practice mindfulness and meditation."

    # Default Response
    return "🤖 Hmm, I don't have an exact answer for that. Try asking about disease diagnosis, treatment, or health tips!"

# --- Display Chat History ---
st.sidebar.markdown("### 💬 Chat History:")
for message in st.session_state["chat_messages"]:
    role = "👤 You" if message["role"] == "user" else "🤖 Bot"
    st.sidebar.markdown(f"**{role}:** {message['content']}")

# --- Text Input Field for Manual Chat ---
user_input = st.sidebar.text_input("💬 Type your question:", value=st.session_state["user_input"], key="chat_input")

# --- Process User Input ---
if st.sidebar.button("🚀 Send"):
    if user_input.strip():
        # Add user input to chat history
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        
        # Get bot response
        bot_reply = chatbot_response(user_input)
        st.session_state["chat_messages"].append({"role": "bot", "content": bot_reply})
        
        # Clear input field by resetting session state
        st.session_state["user_input"] = ""  

# Initialize session state keys if they don't exist
if "bmi_active" not in st.session_state:
    st.session_state["bmi_active"] = False  # Default value

# --- Display BMI Calculator if Triggered ---
if st.session_state["bmi_active"]:
    st.sidebar.markdown("### 🏋️ BMI Calculator")
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=70.0, step=0.1)
    height = st.sidebar.number_input("Height (cm)", min_value=50.0, value=170.0, step=0.1)
    
    if st.sidebar.button("📊 Calculate BMI"):
        height_m = height / 100  # Convert height to meters
        bmi = round(weight / (height_m ** 2), 2)
        st.sidebar.success(f"📌 Your BMI: {bmi}")
        
        # BMI Interpretation
        if bmi < 18.5:
            st.sidebar.warning("Underweight: Consider consulting a nutritionist.")
        elif 18.5 <= bmi < 24.9:
            st.sidebar.success("Normal weight: Keep up the good work!")
        elif 25 <= bmi < 29.9:
            st.sidebar.warning("Overweight: Consider a balanced diet and exercise.")
        else:
            st.sidebar.error("Obese: Please consult a healthcare professional.")
        
        st.session_state["bmi_active"] = False  # Reset BMI trigger        
        # Refresh UI to show cleared input field
        st.rerun()
