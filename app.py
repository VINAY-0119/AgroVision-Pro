import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Detector (TFLite)",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD TFLITE MODEL ---
@st.cache_resource
def load_tflite_model(model_path="plant_disease_recog_model_pwp_quantized.tflite"):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# --- RUN INFERENCE ---
def predict_with_tflite(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape for resizing
    input_shape = input_details[0]['shape']

    # Convert and resize the image to expected size, normalize if needed
    img = Image.fromarray(image_array)
    img = img.resize((input_shape[2], input_shape[1]))
    img = np.array(img) / 255.0  # Normalize to [0,1]
    input_data = np.expand_dims(img.astype(np.float32), axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  # Output probabilities

# --- HELPER FUNCTIONS ---
def get_class_label(index):
    # Replace these classes with your actual model classes
    classes = [
        "Healthy",
        "Powdery Mildew",
        "Leaf Rust",
        "Early Blight",
        "Late Blight",
        "Bacterial Spot",
        "Septoria Leaf Spot"
    ]
    if 0 <= index < len(classes):
        return classes[index]
    return "Unknown"

# --- PAGE STYLING ---
st.markdown("""
<style>
    .main { background-color: #FFFFFF; color: #111827; font-family: 'Inter', sans-serif; }
    .hero { text-align: center; background: linear-gradient(90deg, #DCFCE7, #F0FDF4);
            padding: 35px 15px; border-radius: 12px; margin-bottom: 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08); }
    .hero-title { font-size: 42px; font-weight: 800; color: #166534; margin-bottom: 10px; }
    .hero-subtitle { font-size: 16px; color: #4B5563; max-width: 650px; margin: 0 auto; }
    .section-title { font-size: 18px; font-weight: 600; color: #166534;
                     margin-top: 10px; margin-bottom: 10px; }
    .stButton>button { background-color: #22C55E; color: #FFFFFF; border-radius: 6px;
                       font-weight: 600; border: none; padding: 0.6rem 1.4rem;
                       transition: background 0.2s ease, transform 0.15s ease; }
    .stButton>button:hover { background-color: #15803D; transform: scale(1.02); }
    .footer { text-align: center; font-size: 12px; margin-top: 50px; color: #6B7280; }
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("""
<div class="hero">
    <div class="hero-title">🌿 Plant Disease Detector</div>
    <div class="hero-subtitle">
        Upload a clear photo of your plant leaf to identify possible diseases instantly.  
        Get actionable tips to keep your plants healthy.
    </div>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1.5, 3])

with col1:
    st.markdown("<div class='section-title'>💡 Care Tips</div>", unsafe_allow_html=True)
    tips = [
        "Water plants early in the day to reduce fungal growth.",
        "Remove and dispose of diseased leaves promptly.",
        "Use resistant plant varieties when available.",
        "Maintain proper spacing for airflow.",
        "Avoid overhead watering to keep leaves dry."
    ]
    st.markdown(f"✅ {random.choice(tips)}")

    st.markdown("<div class='section-title'>📊 Quick Facts</div>", unsafe_allow_html=True)
    st.markdown("""
    - Plant diseases cause up to 40% crop losses globally.  
    - Early detection can reduce spread significantly.  
    - Most fungal diseases thrive in wet conditions.  
    - Proper sanitation limits infections.
    """)

with col2:
    st.markdown("<div class='section-title'>📷 Upload Leaf Image</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

        if st.button("🌱 Detect Disease"):
            interpreter = load_tflite_model()

            with st.spinner("Analyzing image..."):
                time.sleep(1)
                image_np = np.array(img)
                try:
                    predictions = predict_with_tflite(interpreter, image_np)
                    top_index = np.argmax(predictions)
                    confidence = predictions[top_index]

                    st.markdown("<div class='section-title'>🔍 Prediction Results</div>", unsafe_allow_html=True)
                    st.metric("Disease", get_class_label(top_index))
                    st.metric("Confidence", f"{confidence * 100:.2f}%")

                    if top_index == 0:
                        st.success("Your plant looks healthy! Keep up the good care.")
                    else:
                        st.warning(f"Possible disease detected: {get_class_label(top_index)}")
                        st.info("Consider consulting agricultural experts for treatment options.")

                except Exception as e:
                    st.error(f"Error during prediction: {type(e).__name__} - {e}")

# --- FOOTER ---
st.markdown("<div class='footer'>© 2026 Plant Disease Detector | Powered by Streamlit + TensorFlow Lite</div>", unsafe_allow_html=True)
