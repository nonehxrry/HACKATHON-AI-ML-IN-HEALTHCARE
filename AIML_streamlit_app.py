import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
from fpdf import FPDF

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

model_path = "healthcare_model.pkl"

try:
    model = joblib.load
    if not hasattr(model, "predict"):
        st.error("Loaded file is not a valid ML model. Please check and reload.")
        st.stop()
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Model file not found: {model_path}")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Ensure model is callable before prediction
if not hasattr(model, "predict"):
    st.error("Loaded object is not a valid model. Ensure 'healthcare_model.pkl' is a trained model.")
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
        "treatment": "",
        "diet": "Balanced",
        "sleep_hours": 7,
        "stress_level": "Low",
        "heart_rate": 72,
        "oxygen_saturation": 98,
        "waist_circumference": 80,
        "hip_circumference": 90,
        "fasting_blood_sugar": 90,
        "post_meal_blood_sugar": 120,
        "hba1c": 5.5
    }

# Navigation menu
step = st.radio(
    "Navigate through the steps:",
    ["Patient Information", "Health Metrics", "Lifestyle Habits", "Diabetes Tracker", "Diagnosis & Treatment", "Final Report"]
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
    st.session_state["patient_details"]["blood_pressure"] = st.number_input("Blood Pressure (mmHg)", min_value=0, value=st.session_state["patient_details"].get("blood_pressure", 120))
    st.session_state["patient_details"]["cholesterol"] = st.number_input("Cholesterol (mg/dL)", min_value=0, value=st.session_state["patient_details"].get("cholesterol", 200))
    st.session_state["patient_details"]["bmi"] = st.number_input("BMI", min_value=0.0, value=float(st.session_state["patient_details"].get("bmi", 25.0)))

    st.write(f"Current BMI: {st.session_state['patient_details']['bmi']}")
    st.session_state["patient_details"]["glucose"] = st.number_input("Glucose Level (mg/dL)", min_value=0, value=st.session_state["patient_details"].get("glucose", 100))
    st.session_state["patient_details"]["heart_rate"] = st.number_input("Heart Rate (bpm)", min_value=0, value=st.session_state["patient_details"].get("heart_rate", 72))
    st.session_state["patient_details"]["oxygen_saturation"] = st.number_input("Oxygen Saturation (%)", min_value=0, max_value=100, value=st.session_state["patient_details"].get("oxygen_saturation", 98))

# Step 3: Lifestyle Habits
elif step == "Lifestyle Habits":
    st.markdown("### Step 3: Lifestyle Habits")
    st.session_state["patient_details"]["smoking_status"] = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"], index=0 if st.session_state["patient_details"]["smoking_status"] == "Non-smoker" else 1)
    st.session_state["patient_details"]["alcohol_consumption"] = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"], index=["None", "Light", "Moderate", "Heavy"].index(st.session_state["patient_details"]["alcohol_consumption"]))
    st.session_state["patient_details"]["physical_activity"] = st.selectbox("Physical Activity", ["None", "Light", "Moderate", "Heavy"], index=["None", "Light", "Moderate", "Heavy"].index(st.session_state["patient_details"]["physical_activity"]))
    st.session_state["patient_details"]["family_history"] = st.selectbox("Family History of Diseases", ["No", "Yes"], index=0 if st.session_state["patient_details"]["family_history"] == "No" else 1)
    
    # Ensure the diet selection is valid
    diet_options = ["Balanced", "Unbalanced", "Vegetarian", "Vegan"]
    current_diet = st.session_state["patient_details"].get("diet", "No diet information available")
    if current_diet not in diet_options:
        current_diet = "Balanced"  # Default to Balanced if the current diet is invalid
    st.session_state["patient_details"]["diet"] = st.selectbox("Diet", diet_options, index=diet_options.index(current_diet))
    
    st.session_state["patient_details"]["sleep_hours"] = st.number_input("Sleep Hours", min_value=0, max_value=24, value=st.session_state["patient_details"].get("sleep_hours", 7))
    if "patient_details" not in st.session_state:
        st.session_state["patient_details"] = {}

    # Ensure 'stress_level' exists before accessing it
    if "stress_level" not in st.session_state["patient_details"]:
        st.session_state["patient_details"]["stress_level"] = "Low"  # Default value
    
    # Use the safe default value
    stress_options = ["Low", "Moderate", "High"]
    st.session_state["patient_details"]["stress_level"] = st.selectbox(
        "Stress Level",
        stress_options,
        index=stress_options.index(st.session_state["patient_details"]["stress_level"])
    )
    
    st.write(f"Selected Stress Level: {st.session_state['patient_details']['stress_level']}")
    st.session_state["patient_details"]["waist_circumference"] = st.number_input("Waist Circumference (cm)", min_value=0, value=st.session_state["patient_details"].get("waist_circumference", 80))
    st.session_state["patient_details"]["hip_circumference"] = st.number_input("Hip Circumference (cm)", min_value=0, value=st.session_state["patient_details"].get("hip_circumference", 90))

# Step 4: Diabetes Tracker
elif step == "Diabetes Tracker":
    st.markdown("### Step 4: Diabetes Tracker")
    st.session_state["patient_details"]["fasting_blood_sugar"] = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=0, value=st.session_state["patient_details"].get("fasting_blood_sugar", 90))
    st.session_state["patient_details"]["post_meal_blood_sugar"] = st.number_input("Post-Meal Blood Sugar (mg/dL)", min_value=0, value=st.session_state["patient_details"].get("post_meal_blood_sugar", 120))
    st.session_state["patient_details"]["hba1c"] = st.number_input("HbA1c (%)", min_value=0.0, value=st.session_state["patient_details"].get("hba1c", 5.5))

    if st.button("Analyze Diabetes Risk"):
        fasting_sugar = st.session_state["patient_details"]["fasting_blood_sugar"]
        post_meal_sugar = st.session_state["patient_details"]["post_meal_blood_sugar"]
        hba1c = st.session_state["patient_details"]["hba1c"]

        if fasting_sugar >= 126 or post_meal_sugar >= 200 or hba1c >= 6.5:
            st.error("High risk of diabetes. Consult a doctor immediately.")
        elif 100 <= fasting_sugar < 126 or 140 <= post_meal_sugar < 200 or 5.7 <= hba1c < 6.5:
            st.warning("Prediabetes detected. Lifestyle changes are recommended.")
        else:
            st.success("Normal blood sugar levels. Keep up the healthy lifestyle.")

# Step 5: Diagnosis & Treatment
elif step == "Diagnosis & Treatment":
    st.markdown("### Step 5: Diagnosis & Treatment")
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
        "diet_unbalanced": [1 if st.session_state["patient_details"]["diet"] == "Unbalanced" else 0],
        "diet_vegetarian": [1 if st.session_state["patient_details"]["diet"] == "Vegetarian" else 0],
        "diet_vegan": [1 if st.session_state["patient_details"]["diet"] == "Vegan" else 0],
        "sleep_hours": [st.session_state["patient_details"]["sleep_hours"]],
        "stress_level_moderate": [1 if st.session_state["patient_details"]["stress_level"] == "Moderate" else 0],
        "stress_level_high": [1 if st.session_state["patient_details"]["stress_level"] == "High" else 0],
        "heart_rate": [st.session_state["patient_details"]["heart_rate"]],
        "oxygen_saturation": [st.session_state["patient_details"]["oxygen_saturation"]],
        "waist_circumference": [st.session_state["patient_details"]["waist_circumference"]],
        "hip_circumference": [st.session_state["patient_details"]["hip_circumference"]],
        "fasting_blood_sugar": [st.session_state["patient_details"]["fasting_blood_sugar"]],
        "post_meal_blood_sugar": [st.session_state["patient_details"]["post_meal_blood_sugar"]],
        "hba1c": [st.session_state["patient_details"]["hba1c"]]
    })

    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)

    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        feature_names = input_data.columns

    input_data = input_data.reindex(columns=feature_names, fill_value=0)

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
        pdf.cell(200, 10, txt=f"Heart Rate: {st.session_state['patient_details'].get('heart_rate', 'N/A')} bpm", ln=True)
        pdf.cell(200, 10, txt=f"Oxygen Saturation: {st.session_state['patient_details'].get('oxygen_saturation', 'N/A')} %", ln=True)
        pdf.ln(10)

        # Lifestyle Habits
        pdf.cell(200, 10, txt="Lifestyle Habits:", ln=True)
        pdf.cell(200, 10, txt=f"Smoking Status: {st.session_state['patient_details'].get('smoking_status', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Alcohol Consumption: {st.session_state['patient_details'].get('alcohol_consumption', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Physical Activity: {st.session_state['patient_details'].get('physical_activity', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Family History: {st.session_state['patient_details'].get('family_history', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Diet: {st.session_state['patient_details'].get('diet', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Sleep Hours: {st.session_state['patient_details'].get('sleep_hours', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Stress Level: {st.session_state['patient_details'].get('stress_level', 'N/A')}", ln=True)
        pdf.cell(200, 10, txt=f"Waist Circumference: {st.session_state['patient_details'].get('waist_circumference', 'N/A')} cm", ln=True)
        pdf.cell(200, 10, txt=f"Hip Circumference: {st.session_state['patient_details'].get('hip_circumference', 'N/A')} cm", ln=True)
        pdf.ln(10)

        # Diabetes Tracker
        pdf.cell(200, 10, txt="Diabetes Tracker:", ln=True)
        pdf.cell(200, 10, txt=f"Fasting Blood Sugar: {st.session_state['patient_details'].get('fasting_blood_sugar', 'N/A')} mg/dL", ln=True)
        pdf.cell(200, 10, txt=f"Post-Meal Blood Sugar: {st.session_state['patient_details'].get('post_meal_blood_sugar', 'N/A')} mg/dL", ln=True)
        pdf.cell(200, 10, txt=f"HbA1c: {st.session_state['patient_details'].get('hba1c', 'N/A')} %", ln=True)
        pdf.ln(10)

        # Diagnosis Results
        pdf.cell(200, 10, txt="Diagnosis Results:", ln=True)
        pdf.cell(200, 10, txt=f"Diagnosis: {'High Risk of Disease' if prediction[0] == 1 else 'Low Risk of Disease'}", ln=True)
        pdf.cell(200, 10, txt=f"Low Risk Probability: {prediction_proba[0][0]:.2f}", ln=True)
        pdf.cell(200, 10, txt=f"High Risk Probability: {prediction_proba[0][1]:.2f}", ln=True)

        # Save PDF to buffer
        buffer = BytesIO()
        pdf_output = pdf.output(dest="S").encode("latin1")  # Encode the PDF content
        buffer.write(pdf_output)
        buffer.seek(0)

        st.download_button(
            label="Download Report as PDF",
            data=buffer,
            file_name="healthcare_analysis_report.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Step 6: Final Report
elif step == "Final Report":
    st.markdown("### Step 6: Final Report")
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
    st.write(f"**Heart Rate:** {patient_details['heart_rate']} bpm")
    st.write(f"**Oxygen Saturation:** {patient_details['oxygen_saturation']} %")

    st.markdown("### Lifestyle Habits")
    st.write(f"**Smoking Status:** {patient_details['smoking_status']}")
    st.write(f"**Alcohol Consumption:** {patient_details['alcohol_consumption']}")
    st.write(f"**Physical Activity:** {patient_details['physical_activity']}")
    st.write(f"**Family History of Diseases:** {patient_details['family_history']}")
    st.write(f"**Diet:** {patient_details['diet']}")
    st.write(f"**Sleep Hours:** {patient_details['sleep_hours']}")
    st.write(f"**Stress Level:** {patient_details['stress_level']}")
    st.write(f"**Waist Circumference:** {patient_details['waist_circumference']} cm")
    st.write(f"**Hip Circumference:** {patient_details['hip_circumference']} cm")

    st.markdown("### Diabetes Tracker")
    st.write(f"**Fasting Blood Sugar:** {patient_details['fasting_blood_sugar']} mg/dL")
    st.write(f"**Post-Meal Blood Sugar:** {patient_details['post_meal_blood_sugar']} mg/dL")
    st.write(f"**HbA1c:** {patient_details['hba1c']} %")

    st.markdown("### Diagnosis & Treatment")
    st.write(f"**Symptoms:** {patient_details['symptoms']}")
    st.write(f"**Diagnosis:** {patient_details['diagnosis']}")
    st.write(f"**Treatment:** {patient_details['treatment']}")

# Footer
st.markdown(
    """
    <footer>
        <p>© Team Technosapiens - 2025 - AI Predictive Methods for Healthcare Analysis. All rights reserved.</p>
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
    st.session_state["bmi_active"] = False

# --- Display BMI Calculator if Triggered ---
if st.session_state["bmi_active"]:
    st.sidebar.markdown("### 🏋️ BMI Calculator")
    weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=70.0, step=0.1)
    height = st.sidebar.number_input("Height (cm)", min_value=50.0, value=170.0, step=0.1)

    if st.sidebar.button("📊 Calculate BMI"):
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)
        st.sidebar.success(f"📌 Your BMI: {bmi}")

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
