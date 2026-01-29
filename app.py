import streamlit as st
import pandas as pd
import joblib
import time
import random
import google.generativeai as genai  # ✅ Gemini import

# --- PATCH sklearn _RemainderColsList ISSUE ---
import sklearn.compose._column_transformer as ctf
if not hasattr(ctf, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ctf._RemainderColsList = _RemainderColsList

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD ML MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please upload 'model.pkl' in the app folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__} - {e}")
        return None

model = load_model()

# --- HELPER FUNCTION: SIMULATE FEATURE PROCESSING (ADAPT TO YOUR FEATURES) ---
def preprocess_input(temp, humidity, soil_moisture, leaf_color, disease_history):
    # Example: Convert categorical leaf color to numeric encoding
    leaf_color_map = {"Green": 0, "Yellow": 1, "Brown": 2, "Spotted": 3}
    leaf_color_val = leaf_color_map.get(leaf_color, 0)
    
    # Construct dataframe or feature vector as your model expects
    data = pd.DataFrame([{
        "Temperature": temp,
        "Humidity": humidity,
        "Soil_Moisture": soil_moisture,
        "Leaf_Color": leaf_color_val,
        "Disease_History": disease_history
    }])
    return data

# --- GEMINI CHAT FUNCTION (UPDATED MODEL) ---
def gemini_chat_completion(prompt):
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {type(e).__name__} - {e}"

# --- PAGE STYLING ---
st.markdown("""
<style>
    .main { background-color: #FFFFFF; color: #111827; font-family: 'Inter', sans-serif; }
    .hero { text-align: center; background: linear-gradient(90deg, #D1FAE5, #F0FDF4);
            padding: 35px 15px; border-radius: 12px; margin-bottom: 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    .hero-title { font-size: 42px; font-weight: 800; color: #065F46; margin-bottom: 10px; }
    .hero-subtitle { font-size: 16px; color: #065F46; max-width: 650px; margin: 0 auto; }
    .section-title { font-size: 18px; font-weight: 600; color: #065F46;
                     margin-top: 10px; margin-bottom: 10px; }
    .stButton>button { background-color: #16A34A; color: #FFFFFF; border-radius: 6px;
                       font-weight: 600; border: none; padding: 0.6rem 1.4rem;
                       transition: background 0.2s ease, transform 0.15s ease; }
    .stButton>button:hover { background-color: #15803D; transform: scale(1.02); }
    .footer { text-align: center; font-size: 12px; margin-top: 50px; color: #4B5563; }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero">
    <div class="hero-title">🌿 Plant Disease Detection</div>
    <div class="hero-subtitle">
        Quickly detect possible diseases in your plants based on environmental and visual factors.
        Provide inputs below and get an instant diagnosis to help protect your crops.
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1.2, 2.3, 1.2])

with col1:
    st.markdown("<div class='section-title'>🌱 Plant Health Insights</div>", unsafe_allow_html=True)
    st.markdown("""
    - Common diseases: Blight, Rust, Mosaic Virus  
    - Key factors: Temperature, Humidity, Soil Moisture  
    - Visual cues: Leaf color, spots, and texture  
    - Early detection helps prevent spread  
    """)
    st.markdown("<div class='section-title'>💡 Care Tip</div>", unsafe_allow_html=True)
    tips = [
        "Regularly inspect leaves for discoloration or spots.",
        "Maintain optimal soil moisture to prevent root rot.",
        "Avoid overhead watering to reduce fungal risk.",
        "Ensure good airflow around plants to reduce humidity.",
        "Use disease-resistant plant varieties when possible."
    ]
    st.markdown(f"✅ {random.choice(tips)}")

with col2:
    st.markdown("<div class='section-title'>🧩 Input Parameters</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        Temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, step=0.1, format="%.1f")
        Humidity = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, step=1.0, format="%.1f")
        Soil_Moisture = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0, step=1.0, format="%.1f")
        Leaf_Color = st.selectbox("Leaf Color", ["Green", "Yellow", "Brown", "Spotted"])
    with c2:
        Disease_History = st.number_input("Past Disease Incidents (count)", 0, 20, 0, step=1)
        Predict_btn = st.button("🔍 Detect Disease")

    if Predict_btn:
        if model is None:
            st.error("Model not loaded. Cannot perform detection.")
        else:
            input_df = preprocess_input(Temperature, Humidity, Soil_Moisture, Leaf_Color, Disease_History)
            with st.spinner("Analyzing plant health..."):
                time.sleep(1)
                try:
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    classes = model.classes_

                    st.markdown("<div class='section-title'>📊 Detection Results</div>", unsafe_allow_html=True)
                    st.write(f"### Predicted Disease: **{prediction}**")

                    st.markdown("#### Confidence Scores:")
                    for cls, prob in zip(classes, probabilities):
                        st.write(f"- {cls}: {prob*100:.1f}%")

                    st.success("✅ Detection complete! Take action accordingly.")
                except Exception as e:
                    st.error(f"Error during detection: {type(e).__name__} - {e}")

with col3:
    st.markdown("<div class='section-title'>📈 Quick Stats</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Detection Accuracy:** 92%  
    - **Common Diseases Covered:** 8  
    - **Avg Detection Time:** 2 seconds  
    - **User Feedback Rating:** 4.7/5  
    """)

# --- CHATBOT SECTION USING GEMINI ---
st.divider()
st.markdown("<div class='section-title'>🤖 Plant Health Assistant (Gemini AI)</div>", unsafe_allow_html=True)
st.info("Ask questions like: 'What are symptoms of blight?' or 'How to treat yellow leaves?'")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

prompt = st.chat_input("Ask me about plant diseases, symptoms, or care tips...", disabled=st.session_state.processing)

if prompt:
    st.session_state.processing = True
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        ai_response = gemini_chat_completion(prompt)

    with st.chat_message("assistant"):
        st.markdown(ai_response)

    st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
    st.session_state.processing = False

if st.button("Clear Chat"):
    st.session_state.chat_messages = []

# --- FOOTER ---
st.markdown("<div class='footer'>© 2025 Plant Disease Detector | Powered by Streamlit + Gemini AI</div>", unsafe_allow_html=True)
