import streamlit as st
import pandas as pd
import joblib
import time
import random
from PIL import Image
import google.generativeai as genai  # ✅ Gemini import

# --- PATCH sklearn _RemainderColsList ISSUE ---
import sklearn.compose._column_transformer as ctf
if not hasattr(ctf, '_RemainderColsList'):
    class _RemainderColsList(list):
        pass
    ctf._RemainderColsList = _RemainderColsList

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EV-Smart-Range-Forecaster",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD ML MODEL ---
@st.cache_resource
def load_model():
    try:
        model = joblib.load("ev_range_predictor_reduced.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Please upload 'ev_range_predictor_reduced.pkl' in the app folder.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {type(e).__name__} - {e}")
        return None

model = load_model()

# --- HELPER FUNCTION FOR ENERGY RATE ---
def energy_rate(speed, terrain, weather, braking, acceleration):
    rate = 0.15
    if speed <= 50:
        rate = 0.12
    elif speed > 80:
        rate = 0.18
    if terrain == "Hilly":
        rate *= 1.2
    if weather == "Hot":
        rate *= 1.1
    if weather == "Cold":
        rate *= 1.15
    rate *= 1 + 0.05 * braking + 0.07 * acceleration
    return rate

# --- GEMINI CHAT FUNCTION (UPDATED MODEL) ---
def gemini_chat_completion(prompt):
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        # ✅ Using advanced Gemini 2.5 model
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {type(e).__name__} - {e}"

# --- PAGE STYLING ---
st.markdown("""
<style>
    .main { background-color: #FFFFFF; color: #111827; font-family: 'Inter', sans-serif; }
    .hero { text-align: center; background: linear-gradient(90deg, #E0F2FE, #F8FAFC);
            padding: 35px 15px; border-radius: 12px; margin-bottom: 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    .hero-title { font-size: 42px; font-weight: 800; color: #0F172A; margin-bottom: 10px; }
    .hero-subtitle { font-size: 16px; color: #475569; max-width: 650px; margin: 0 auto; }
    .section-title { font-size: 18px; font-weight: 600; color: #1E293B;
                     margin-top: 10px; margin-bottom: 10px; }
    .stButton>button { background-color: #2563EB; color: #FFFFFF; border-radius: 6px;
                       font-weight: 600; border: none; padding: 0.6rem 1.4rem;
                       transition: background 0.2s ease, transform 0.15s ease; }
    .stButton>button:hover { background-color: #1E40AF; transform: scale(1.02); }
    .footer { text-align: center; font-size: 12px; margin-top: 50px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero">
    <div class="hero-title">⚡ EV Vehicle Range Predictor 🚗</div>
    <div class="hero-subtitle">
        Estimate your electric vehicle's driving range instantly.  
        Adjust speed, terrain, and weather to see how they affect performance and battery life.
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col1, col2, col3 = st.columns([1.2, 2.3, 1.2])

with col1:
    st.markdown("<div class='section-title'>⚙️ EV Insights</div>", unsafe_allow_html=True)
    st.markdown("""
    - Typical Battery Capacity: **40–75 kWh**  
    - Average Driving Range: **300–500 km**  
    - Charging Time: **30–60 minutes**  
    - Optimal Temperature: **20–25°C**  
    - Efficiency improves with **moderate speeds**
    """)
    st.markdown("<div class='section-title'>💡 Smart Driving Tip</div>", unsafe_allow_html=True)
    tips = [
        "Keep tire pressure optimal to maximize efficiency.",
        "Avoid harsh acceleration for longer range.",
        "Preheat or precool your EV while charging.",
        "Use regenerative braking effectively in traffic.",
        "Plan routes that avoid steep inclines."
    ]
    st.markdown(f"✅ {random.choice(tips)}")

with col2:
    st.markdown("<div class='section-title'>🧩 Input Parameters</div>", unsafe_allow_html=True)

    # --- IMAGE UPLOADER ---
    uploaded_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")

    c1, c2 = st.columns(2)
    with c1:
        SoC = st.number_input("State of Charge (%)", 0.0, 100.0, 80.0, step=1.0, format="%.1f")
        Speed = st.number_input("Speed (Km/h)", 0.0, 200.0, 60.0, step=1.0, format="%.1f")
        Temperature = st.number_input("Temperature (°C)", -20.0, 60.0, 25.0, step=0.1, format="%.1f")
        Terrain = st.selectbox("Terrain Type", ["Flat", "Hilly"])
    with c2:
        Braking = st.number_input("Braking (m/s²)", 0.0, 10.0, 0.5, step=0.1, format="%.2f")
        Acceleration = st.number_input("Acceleration (m/s²)", 0.0, 10.0, 1.0, step=0.1, format="%.2f")
        Weather = st.selectbox("Weather Condition", ["Normal", "Hot", "Cold", "Rainy"])
        Prev_SoC = st.number_input("Previous SoC (%)", 0.0, 100.0, 85.0, step=1.0, format="%.1f")

    predict_btn = st.button("🚀 Predict Range")

    if predict_btn:
        if model is None:
            st.error("Model not loaded. Cannot predict.")
        else:
            input_data = pd.DataFrame([{
                "SoC": SoC,
                "Speed (Km/h)": Speed,
                "Temperature": Temperature,
                "Terrain": Terrain,
                "Braking (m/s²)": Braking,
                "Acceleration (m/s²)": Acceleration,
                "Weather": Weather,
                "Prev_SoC": Prev_SoC
            }])

            with st.spinner("Calculating optimal range..."):
                time.sleep(1)
                try:
                    predicted_SoC = model.predict(input_data)[0]

                    rate = energy_rate(Speed, Terrain, Weather, Braking, Acceleration)
                    battery_capacity_kwh = 40
                    remaining_energy_kwh = (predicted_SoC / 100) * battery_capacity_kwh
                    predicted_range_km = remaining_energy_kwh / rate

                    st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
                    colA, colB = st.columns(2)
                    with colA:
                        st.metric("Predicted SoC (%)", f"{predicted_SoC:.2f}")
                    with colB:
                        st.metric("Estimated Range (km)", f"{predicted_range_km:.1f}")

                    st.markdown(f"""
                    **Remaining Battery Energy:** {remaining_energy_kwh:.2f} kWh  
                    **Energy Consumption Rate:** {rate:.3f} kWh/km
                    """)
                    st.success("✅ Prediction complete! Check metrics above.")
                except Exception as e:
                    st.error(f"Error during prediction: {type(e).__name__} - {e}")

with col3:
    st.markdown("<div class='section-title'>📈 Quick Stats</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Energy Efficiency:** 91%  
    - **Charging Infrastructure:** 82% coverage  
    - **Top Efficient Models:** Model 3, Kona, Leaf  
    - **Avg User Range:** 412 km  
    """)

# --- CHATBOT SECTION USING GEMINI ---
st.divider()
st.markdown("<div class='section-title'>🤖 EV Chat Assistant (Gemini AI)</div>", unsafe_allow_html=True)
st.info("Ask things like: 'What’s my range at 100 km/h in hot weather on hilly terrain?' or 'How does cold weather affect EV efficiency?'")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

prompt = st.chat_input("Ask me about EVs, range, or efficiency...", disabled=st.session_state.processing)

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
st.markdown("<div class='footer'>© 2025 EV Predictor | Powered by Streamlit + Gemini AI</div>", unsafe_allow_html=True)
