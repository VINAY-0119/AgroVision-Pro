import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Image Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("plant_disease_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- CLASS NAMES (example, replace with your classes) ---
CLASS_NAMES = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy",
    "Powdery Mildew",
]

# --- IMAGE PREPROCESSING FUNCTION ---
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# --- GEMINI CHAT FUNCTION ---
def gemini_chat_completion(prompt):
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model_gemini = genai.GenerativeModel("models/gemini-flash-latest")
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {type(e).__name__} - {e}"

# --- PAGE STYLING ---
st.markdown("""
<style>
    .main { background-color: #FAFAFA; color: #111827; font-family: 'Inter', sans-serif; }
    .hero { text-align: center; background: linear-gradient(90deg, #D1FAE5, #E0F2FE);
            padding: 35px 15px; border-radius: 12px; margin-bottom: 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    .hero-title { font-size: 42px; font-weight: 800; color: #065F46; margin-bottom: 10px; }
    .hero-subtitle { font-size: 16px; color: #374151; max-width: 650px; margin: 0 auto; }
    .section-title { font-size: 18px; font-weight: 600; color: #065F46;
                     margin-top: 10px; margin-bottom: 10px; }
    .stButton>button { background-color: #10B981; color: #FFFFFF; border-radius: 6px;
                       font-weight: 600; border: none; padding: 0.6rem 1.4rem;
                       transition: background 0.2s ease, transform 0.15s ease; }
    .stButton>button:hover { background-color: #047857; transform: scale(1.02); }
    .footer { text-align: center; font-size: 12px; margin-top: 50px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero">
    <div class="hero-title">🌿 Plant Disease Image Classifier</div>
    <div class="hero-subtitle">
        Upload a leaf image to get an instant prediction of plant diseases.
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN CONTENT ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown("<div class='section-title'>📤 Upload Leaf Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        if model is None:
            st.error("Model not loaded. Cannot predict.")
        else:
            if st.button("🔍 Predict Disease"):
                with st.spinner("Analyzing image..."):
                    time.sleep(1)  # Simulate processing time

                    img_array = preprocess_image(image)
                    preds = model.predict(img_array)
                    pred_prob = np.max(preds)
                    pred_class = CLASS_NAMES[np.argmax(preds)]

                    st.markdown("<div class='section-title'>📊 Prediction Results</div>", unsafe_allow_html=True)
                    st.success(f"Predicted Disease: **{pred_class}**")
                    st.write(f"Confidence: {pred_prob * 100:.2f}%")

with col2:
    st.markdown("<div class='section-title'>🤖 Plant Disease Chat Assistant (Gemini AI)</div>", unsafe_allow_html=True)
    st.info("Ask about symptoms, disease prevention, or treatment options.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    prompt = st.chat_input("Ask me about plant diseases or care...", disabled=st.session_state.processing)

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
st.markdown("<div class='footer'>© 2025 Plant Disease Classifier | Powered by Streamlit + Gemini AI</div>", unsafe_allow_html=True)
