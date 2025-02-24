import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF

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

# Load the trained model
model_path = 'healthcare_model.pkl'
try:
    model = joblib.load(model_path)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()

# Initialize session state
if "patient_details" not in st.session_state:
    st.session_state["patient_details"] = {
        "full_name": "",
        "email": "",
        "phone": "",
        "age": 30,
        "gender": "Male",
        "blood_pressure": 120,
        "cholesterol": 200,
        "bmi": 25,
        "glucose": 100,
        "smoking_status": "Non-smoker",
        "alcohol_consumption": "Non-drinker",
        "physical_activity": "Sedentary",
        "family_history": "No",
        "symptoms": "",
        "medical_history": "",
        "test_results": None
    }

# Navigation menu
step = st.radio(
    "Navigate through the steps:",
    ["Patient Information", "Health Details", "Upload Test Results", "Final Diagnosis"]
)

# Step 1: Patient Information
if step == "Patient Information":
    st.markdown("### Step 1: Patient Information")
    st.session_state["patient_details"]["full_name"] = st.text_input("Full Name", st.session_state["patient_details"]["full_name"])
    st.session_state["patient_details"]["email"] = st.text_input("Email Address", st.session_state["patient_details"]["email"])
    st.session_state["patient_details"]["phone"] = st.text_input("Phone Number", st.session_state["patient_details"]["phone"])

# Step 2: Health Details
elif step == "Health Details":
    st.markdown("### Step 2: Health Details")
    st.session_state["patient_details"]["age"] = st.slider("Age:", 1, 100, st.session_state["patient_details"]["age"])
    st.session_state["patient_details"]["gender"] = st.selectbox("Gender:", ["Male", "Female", "Other"])
    st.session_state["patient_details"]["blood_pressure"] = st.number_input("Blood Pressure (mmHg):", min_value=0, step=1, value=st.session_state["patient_details"]["blood_pressure"])
    st.session_state["patient_details"]["cholesterol"] = st.number_input("Cholesterol (mg/dL):", min_value=0, step=1, value=st.session_state["patient_details"]["cholesterol"])
    st.session_state["patient_details"]["bmi"] = st.number_input("BMI:", min_value=0.0, step=0.1, value=st.session_state["patient_details"]["bmi"])
    st.session_state["patient_details"]["glucose"] = st.number_input("Glucose Level (mg/dL):", min_value=0, step=1, value=st.session_state["patient_details"]["glucose"])
    st.session_state["patient_details"]["smoking_status"] = st.selectbox("Smoking Status:", ["Non-smoker", "Ex-smoker", "Smoker"])
    st.session_state["patient_details"]["alcohol_consumption"] = st.selectbox("Alcohol Consumption:", ["Non-drinker", "Occasional", "Regular"])
    st.session_state["patient_details"]["physical_activity"] = st.selectbox("Physical Activity:", ["Sedentary", "Light", "Moderate", "Active"])
    st.session_state["patient_details"]["family_history"] = st.selectbox("Family History of Disease:", ["No", "Yes"])
    st.session_state["patient_details"]["symptoms"] = st.text_area("Symptoms", st.session_state["patient_details"]["symptoms"])
    st.session_state["patient_details"]["medical_history"] = st.text_area("Medical History", st.session_state["patient_details"]["medical_history"])

# Step 3: Upload Test Results
elif step == "Upload Test Results":
    st.markdown("### Step 3: Upload Test Results")
    st.session_state["patient_details"]["test_results"] = st.file_uploader("Upload Test Results")

# Step 4: Final Diagnosis
elif step == "Final Diagnosis":
    st.markdown("### Step 4: Final Diagnosis")
    patient_details = st.session_state["patient_details"]

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        "age": [patient_details["age"]],
        "gender": [1 if patient_details["gender"] == "Female" else 0],
        "blood_pressure": [patient_details["blood_pressure"]],
        "cholesterol": [patient_details["cholesterol"]],
        "bmi": [patient_details["bmi"]],
        "glucose": [patient_details["glucose"]],
        "smoking_status": [1 if patient_details["smoking_status"] == "Smoker" else 0],
        "alcohol_consumption": [1 if patient_details["alcohol_consumption"] == "Regular" else 0],
        "physical_activity": [1 if patient_details["physical_activity"] == "Active" else 0],
        "family_history": [1 if patient_details["family_history"] == "Yes" else 0],
    })

    input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Map prediction to disease
        disease_mapping = {
            0: "Heart Disease",
            1: "Diabetes",
            2: "Cancer",
            3: "Stroke"
        }
        predicted_disease = disease_mapping.get(prediction[0], "Unknown")

        st.markdown(f"### Predicted Disease: {predicted_disease}")
        st.write(f"**Probability:** {prediction_proba[0][prediction[0]]:.2f}")

        # Generate PDF Report
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.set_font("Arial", style="BU", size=12)  # Set font to bold and underline
        pdf.cell(200, 10, txt="Health Risk Prediction Report", ln=True, align="C")
        pdf.set_font("Arial", size=12)  # Reset font to normal
        pdf.ln(10)

        # Patient Information
        pdf.cell(200, 10, txt="Patient Information:", ln=True)
        pdf.cell(200, 10, txt=f"Full Name: {patient_details.get('full_name', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Email: {patient_details.get('email', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Phone: {patient_details.get('phone', 'N/A')}", ln=True)
        pdf.ln(10)

        # Health Details
        pdf.cell(200, 10, txt="Health Details:", ln=True)
        pdf.cell(200, 10, txt=f"Age: {patient_details.get('age', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Gender: {patient_details.get('gender', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Blood Pressure: {patient_details.get('blood_pressure', 'N/A')} mmHg", ln=True)
        pdf.cell(200, 10, txt=f"Cholesterol: {patient_details.get('cholesterol', 'N/A')} mg/dL", ln=True)
        pdf.cell(200, 10, txt=f"BMI: {patient_details.get('bmi', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Glucose Level: {patient_details.get('glucose', 'N/A')} mg/dL", ln=True)
        pdf.cell(200, 10, txt=f"Smoking Status: {patient_details.get('smoking_status', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Alcohol Consumption: {patient_details.get('alcohol_consumption', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Physical Activity: {patient_details.get('physical_activity', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Family History: {patient_details.get('family_history', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Symptoms: {patient_details.get('symptoms', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Medical History: {patient_details.get('medical_history', 'N/A')}", ln=True)
        pdf.ln(10)

        # Prediction Results
        pdf.cell(200, 10, txt="Prediction Results:", ln=True)
        pdf.cell(200, 10, txt=f"Predicted Disease: {predicted_disease}", ln=True)
        pdf.cell(200, 10, txt=f"Probability: {prediction_proba[0][prediction[0]]:.2f}", ln=True)
        pdf.ln(10)

        # Recommendations
        pdf.cell(200, 10, txt="Recommendations:", ln=True)
        pdf.cell(200, 10, txt="- Consult a doctor for further evaluation.", ln=True)
        pdf.cell(200, 10, txt="- Maintain a healthy diet and exercise regularly.", ln=True)
        pdf.ln(10)

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

# --- Chatbot in Sidebar ---
st.sidebar.markdown("## 🤖 AI Healthcare Chatbot")

# --- Initialize Chat History & Session Variables ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "bot", "content": "👋 Hello! You can speak or type your question.\n\n**📌 Categories:**\n- Disease Prediction 🏥\n- Health Tips 💡\n- Symptom Checker 🔍\n- Medical Advice 🩺\n- Test Results 📊"}
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
        return "👋 Hello! How can I assist you today? You can ask about disease prediction, health tips, or medical advice!"

    # Disease Prediction
    disease_topics = ["disease prediction", "predict disease", "health risk"]
    if any(topic in user_message for topic in disease_topics):
        st.session_state["last_topic"] = "disease"
        return "📌 **Disease Prediction:**\n- **Heart Disease** ❤️\n- **Diabetes** 🩸\n- **Cancer** 🎗️\n- **Stroke** 🧠\n\n💡 Ask about a specific disease for details!"

    # Specific Diseases with More Details
    disease_details = {
        "heart disease": """❤️ **Heart Disease Details:**
        - **Risk Factors:** High blood pressure, high cholesterol, smoking, diabetes, obesity.
        - **Symptoms:** Chest pain, shortness of breath, fatigue.
        - **Prevention:** Regular exercise, healthy diet, no smoking.
        - **Treatment:** Medications, surgery, lifestyle changes.
        - 💡 **Tip:** Regular check-ups can help in early detection.""",

        "diabetes": """🩸 **Diabetes Details:**
        - **Risk Factors:** Family history, obesity, sedentary lifestyle.
        - **Symptoms:** Increased thirst, frequent urination, fatigue.
        - **Prevention:** Healthy diet, regular exercise, weight management.
        - **Treatment:** Insulin therapy, medications, lifestyle changes.
        - 💡 **Tip:** Monitor blood sugar levels regularly.""",

        "cancer": """🎗️ **Cancer Details:**
        - **Risk Factors:** Smoking, alcohol, radiation, family history.
        - **Symptoms:** Unexplained weight loss, fatigue, lumps.
        - **Prevention:** Avoid tobacco, limit alcohol, healthy diet.
        - **Treatment:** Surgery, chemotherapy, radiation therapy.
        - 💡 **Tip:** Early detection increases the chances of successful treatment.""",

        "stroke": """🧠 **Stroke Details:**
        - **Risk Factors:** High blood pressure, smoking, diabetes.
        - **Symptoms:** Sudden numbness, confusion, trouble speaking.
        - **Prevention:** Control blood pressure, quit smoking, healthy diet.
        - **Treatment:** Medications, surgery, rehabilitation.
        - 💡 **Tip:** FAST (Face, Arms, Speech, Time) is a key symptom checker."""
    }

    # Check for a specific disease type
    for key, response in disease_details.items():
        if key in user_message:
            st.session_state["last_topic"] = key  # Store last topic
            return response

    # Follow-Up Questions Based on Last Topic
    if st.session_state["last_topic"]:
        if "tell me more" in user_message or "more details" in user_message:
            # Provide additional details based on the last topic
            if st.session_state["last_topic"] == "heart disease":
                return "❤️ **More on Heart Disease:**\n- Regular exercise and a healthy diet can significantly reduce the risk.\n- Early detection through regular check-ups is crucial."
            elif st.session_state["last_topic"] == "diabetes":
                return "🩸 **More on Diabetes:**\n- Managing blood sugar levels through diet and medication is key.\n- Regular monitoring can prevent complications."
            elif st.session_state["last_topic"] == "cancer":
                return "🎗️ **More on Cancer:**\n- Early detection through screenings can save lives.\n- Lifestyle changes can reduce the risk."
            elif st.session_state["last_topic"] == "stroke":
                return "🧠 **More on Stroke:**\n- Recognizing symptoms early can save lives.\n- Rehabilitation can help in recovery."

    # Health Tips
    if "health tips" in user_message or "tips" in user_message:
        return "💡 **Health Tips:**\n- **Eat a balanced diet** 🥗\n- **Exercise regularly** 🏋️‍♂️\n- **Get enough sleep** 😴\n- **Avoid smoking and excessive alcohol** 🚭🍷\n- **Regular check-ups** 🩺"

    # Symptom Checker
    if "symptom checker" in user_message or "symptoms" in user_message:
        return "🔍 **Symptom Checker:**\n- **Chest pain:** Could indicate heart disease.\n- **Increased thirst:** Could indicate diabetes.\n- **Unexplained weight loss:** Could indicate cancer.\n- **Sudden numbness:** Could indicate stroke.\n\n💡 Always consult a doctor for accurate diagnosis."

    # Default Response
    return "🤖 Hmm, I don't have an exact answer for that. Try asking about disease prediction, health tips, or medical advice!"

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
        
        # Refresh UI to show cleared input field
        st.rerun()

# --- Footer ---
st.markdown(
    """
    <footer>
        <p>© 2025 AI Predictive Methods for Healthcare. All rights reserved.</p>
    </footer>
    """,
    unsafe_allow_html=True
)
