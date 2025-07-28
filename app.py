import streamlit as st
import pandas as pd
import joblib
import base64

# Load model
model = joblib.load("heart_disease_model.pkl")

# Convert image to base64
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load images
logo_base64 = get_base64("4e211555-a283-42db-80d8-1c995b223b5d.png")
bg_base64 = get_base64("heart.jpg")
high_risk_img_path = "high_risk.png"
low_risk_img_path = "low_risk.png"

# Page config
st.set_page_config(page_title="Heart Risk Analyzer", layout="centered")

# CSS Styling with blurred background like vehicle app
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Great+Vibes&display=swap');

    .blur-bg {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("data:image/jpeg;base64,{bg_base64}");
        background-size: cover;
        background-position: center;
        filter: blur(6px);
        opacity: 0.6;
        z-index: -1;
    }}

    .stApp {{
        background-color: rgba(255, 255, 255, 0.0);
    }}

    section.main > div:first-child {{
        background-color: rgba(255, 255, 255, 0.92);
        padding: 2rem;
        border-radius: 14px;
        box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
        max-width: 750px;
        margin: 3rem auto;
        font-family: "Times New Roman", Times, serif !important;
        color: #000;
    }}

    .custom-header {{
        font-family: 'Playfair Display', serif !important;
        font-size: 20px;
        font-weight: 700;
        color: #c62828;
    }}

    .slogan {{
        font-size: 14px;
        color: #6c6c6c;
        font-style: italic;
        font-family: 'Georgia', serif !important;
        margin-top: -6px;
        margin-left: 4px;
    }}

    .subheader {{
        font-size: 20px;
        font-weight: bold;
        color: green;
        margin-bottom: 20px;
    }}

    .result-box {{
        background-color: #E8F8F5;
        padding: 1rem 1.5rem;
        border-left: 5px solid #117864;
        font-size: 20px;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #145A32;
    }}

    .danger-box {{
        background-color: #FDEDEC;
        border-left: 5px solid #C0392B;
        color: #922B21;
    }}
    </style>
    <div class="blur-bg"></div>

    <div style="position: fixed; top: 10px; left: 10px; display: flex; flex-direction: column;
                align-items: start; padding: 8px 12px; z-index: 999;">
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{logo_base64}" style="height: 60px; border-radius: 8px; margin-right: 10px;">
            <div>
                <div class="custom-header">HeartWise</div>
                <div class="slogan">Know Better. Live Longer.</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Title and Header
st.markdown("<h1 style='text-align:center; color:#154360; font-size:30px;'>Heart Disease Risk Analyzer</h1>", unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter the patient information:</div>', unsafe_allow_html=True)

# Form
with st.form("heart_form"):
    age = st.number_input("Age", 1, 120, 50)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (1–4)", [1, 2, 3, 4])
    trestbps = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    oldpeak = st.number_input("Oldpeak ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope (1–3)", [1, 2, 3])
    submitted = st.form_submit_button("Analyze Risk")

# Prediction
if submitted:
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope]],
        columns=[
            "age", "sex", "chest pain type", "resting bp s",
            "cholesterol", "fasting blood sugar", "resting ecg",
            "max heart rate", "exercise angina", "oldpeak", "ST slope"
        ])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown('<div class="result-box danger-box">High Risk: Heart disease is likely.</div>', unsafe_allow_html=True)
        st.image(high_risk_img_path, caption="High Risk Alert", use_container_width=True)

        st.markdown("### Health Tips")
        st.markdown("""
        - Consult a cardiologist immediately  
        - Eat a heart-friendly diet  
        - Include physical activity  
        - Avoid tobacco, alcohol  
        - Monitor BP & cholesterol
        """)
    else:
        st.markdown('<div class="result-box">Low Risk: Heart disease is unlikely.</div>', unsafe_allow_html=True)
        st.image(low_risk_img_path, caption="All Clear: Low Risk", use_container_width=True)
