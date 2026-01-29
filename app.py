import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.special

# Load TFLite model with caching
@st.cache_resource
def load_tflite_model(model_path='plant_disease_recog_model_pwp_quantized.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Replace these class names with your exact 39 classes
CLASS_NAMES = [
    'Apple Scab',
    'Apple Black Rot',
    'Apple Cedar Rust',
    'Apple Healthy',
    'Blueberry Healthy',
    'Cherry Powdery Mildew',
    'Cherry Healthy',
    'Corn Cercospora Leaf Spot',
    'Corn Common Rust',
    'Corn Northern Leaf Blight',
    'Corn Healthy',
    'Grape Black Rot',
    'Grape Esca (Black Measles)',
    'Grape Leaf Blight (Isariopsis Leaf Spot)',
    'Grape Healthy',
    'Peach Bacterial Spot',
    'Peach Healthy',
    'Pepper Bell Bacterial Spot',
    'Pepper Bell Healthy',
    'Potato Early Blight',
    'Potato Late Blight',
    'Potato Healthy',
    'Raspberry Healthy',
    'Soybean Healthy',
    'Squash Powdery Mildew',
    'Strawberry Leaf Scorch',
    'Strawberry Healthy',
    'Tomato Bacterial Spot',
    'Tomato Early Blight',
    'Tomato Late Blight',
    'Tomato Leaf Mold',
    'Tomato Septoria Leaf Spot',
    'Tomato Spider Mites Two-Spotted Spider Mite',
    'Tomato Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus',
    'Tomato Healthy',
    'Class 38 Placeholder',
    'Class 39 Placeholder'
]

def preprocess_image(image: Image.Image):
    # Resize to model input shape (width, height)
    width = input_details[0]['shape'][2]
    height = input_details[0]['shape'][1]
    image = image.resize((width, height))

    img_array = np.array(image).astype(input_details[0]['dtype'])

    # Normalize if float32 input
    if input_details[0]['dtype'] == np.float32:
        img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image: Image.Image):
    input_data = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]

    # Apply softmax if output not probabilities
    probs = scipy.special.softmax(output)

    predicted_index = np.argmax(probs)

    # Safety check
    if predicted_index >= len(CLASS_NAMES):
        raise ValueError("Prediction index out of bounds!")

    predicted_label = CLASS_NAMES[predicted_index]
    confidence = probs[predicted_index] * 100

    return predicted_label, confidence, probs

st.title("Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Disease"):
        label, confidence, probs = predict(image)

        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # Warn on low confidence
        if confidence < 50:
            st.warning("Low confidence in prediction; result may be unreliable.")

        # Show top 3 predictions
        top_k = 3
        top_indices = np.argsort(probs)[-top_k:][::-1]
        st.write("Top predictions:")
        for i in top_indices:
            st.write(f"{CLASS_NAMES[i]}: {probs[i]*100:.2f}%")
