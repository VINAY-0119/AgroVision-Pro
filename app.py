import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
import tflite_runtime.interpreter as tflite
import google.generativeai as genai
import time
import logging

# --- CONSTANTS ---
CLASS_NAMES = [
    "Apple Scab",
    "Black Rot",
    "Cedar Apple Rust",
    "Healthy",
    "Powdery Mildew",
]

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Plant Disease Image Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- HELPER FUNCTIONS ---

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

def preprocess_image(image, interpreter, target_size=(224, 224)):
    input_details = interpreter.get_input_details()
    input_dtype = input_details[0]['dtype']

    image = image.convert("RGB").resize(target_size)

    if input_dtype == np.uint8:
        img = np.array(image, dtype=np.uint8)
    elif input_dtype == np.float32:
        img = np.array(image, dtype=np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    img = np.expand_dims(img, axis=0)
    return img

def predict(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])
    scale, zero_point = output_details[0]['quantization']
    if scale and scale > 0:
        preds = scale * (preds - zero_point)

    probs = softmax(preds)
    return probs

def gemini_chat_completion(prompt, model):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return "⚠️ Sorry, I couldn't process your question right now. Please try again later."

# --- LOAD MODELS ---

@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(model_path="plant_disease_recog_model_pwp_quantized.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        st.error(f"❌ Model loading failed: {e}")
        return None

@st.cache_resource
def load_gemini():
    try:
        genai.configure(api_key=st.secrets["gemini"]["api_key"])
        return genai.GenerativeModel("models/gemini-flash-latest")
    except Exception as e:
        logging.error(f"Gemini model loading failed: {e}")
        st.warning("⚠️ Gemini AI model not loaded. Chat assistant is disabled.")
        return None

interpreter = load_model()
gemini_model = load_gemini()

# --- UI LAYOUT ---

st.markdown("""
<div style="text-align:center; padding:30px; background:#ECFDF5; border-radius:12px;">
<h1>🌿 Plant Disease Image Classifier</h1>
<p>Upload a leaf image to detect plant diseases instantly.</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

disease = None
confidence = None
preds = None

with col1:
    uploaded_file = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        except UnidentifiedImageError:
            st.error("⚠️ The uploaded file is not a valid image.")
            image = None

        predict_button = st.button(
            "🔍 Predict Disease",
            disabled=uploaded_file is None or interpreter is None or image is None
        )

        if predict_button and image is not None:
            if interpreter is None:
                st.error("Model is not loaded.")
            else:
                with st.spinner("Analyzing image..."):
                    time.sleep(1)  # simulate delay

                    try:
                        img_array = preprocess_image(image, interpreter)
                        preds = predict(interpreter, img_array)

                        confidence = float(np.max(preds))
                        disease = CLASS_NAMES[int(np.argmax(preds))]

                        st.success(f"**Prediction:** {disease}")
                        st.write(f"**Confidence:** {confidence * 100:.2f}%")

                        st.write("### Other class probabilities:")
                        for i, cls in enumerate(CLASS_NAMES):
                            st.write(f"{cls}: {preds[0][i] * 100:.2f}%")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

with col2:
    st.subheader("🤖 Plant Disease Assistant")
    st.write("Ask me about symptoms, treatment, or prevention for the detected plant disease.")
    prompt = st.chat_input("Type your question here...")

    if prompt:
        if gemini_model is None:
            st.warning("Chat assistant is currently unavailable.")
        else:
            with st.spinner("Thinking..."):
                base_context = ""
                if disease is not None and confidence is not None:
                    base_context = f"The detected disease is {disease} with a confidence of {confidence * 100:.1f}%. "

                full_prompt = base_context + "User question: " + prompt
                response = gemini_chat_completion(full_prompt, gemini_model)
                st.markdown(response)

# --- FOOTER ---
st.markdown(
    "<div style='text-align:center; color:#6B7280; margin-top:40px;'>"
    "© 2025 Plant Disease Classifier | Streamlit + TFLite"
    "</div>",
    unsafe_allow_html=True
)
