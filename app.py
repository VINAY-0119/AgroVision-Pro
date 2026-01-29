import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import google.generativeai as genai
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Image Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- LOAD TFLITE MODEL ---
@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(
            model_path="plant_disease_recog_model_pwp_quantized.tflite"
        )
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

interpreter = load_model()

# --- CLASS NAMES ---
CLASS_NAMES = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy",
    "Powdery Mildew",
]

# --- IMAGE PREPROCESSING ---
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --- PREDICTION ---
def predict(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])
    return preds

# --- GEMINI CHAT ---
def gemini_chat_completion(prompt):
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ Gemini API error: {e}"

# --- HERO ---
st.markdown("""
<div style="text-align:center; padding:30px; background:#ECFDF5; border-radius:12px;">
<h1>🌿 Plant Disease Image Classifier</h1>
<p>Upload a leaf image to detect plant diseases instantly.</p>
</div>
""", unsafe_allow_html=True)

# --- MAIN ---
col1, col2 = st.columns([1.5, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        if interpreter and st.button("🔍 Predict Disease"):
            with st.spinner("Analyzing image..."):
                time.sleep(1)

                img_array = preprocess_image(image)
                preds = predict(interpreter, img_array)

                confidence = float(np.max(preds))
                disease = CLASS_NAMES[int(np.argmax(preds))]

                st.success(f"**Prediction:** {disease}")
                st.write(f"**Confidence:** {confidence * 100:.2f}%")

with col2:
    st.subheader("🤖 Plant Disease Assistant")
    prompt = st.chat_input("Ask about symptoms or treatment")

    if prompt:
        with st.spinner("Thinking..."):
            st.markdown(gemini_chat_completion(prompt))

# --- FOOTER ---
st.markdown(
    "<div style='text-align:center; color:#6B7280; margin-top:40px;'>"
    "© 2025 Plant Disease Classifier | Streamlit + TFLite"
    "</div>",
    unsafe_allow_html=True
)
