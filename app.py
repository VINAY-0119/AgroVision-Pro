import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_tflite_model(model_path='plant_disease_recog_model_pwp_quantized.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image: Image.Image, input_shape):
    image = image.resize((input_shape[1], input_shape[2]))
    img_array = np.array(image)
    
    if input_details[0]['dtype'] == np.uint8:
        if img_array.dtype != np.uint8:
            img_array = img_array.astype(np.uint8)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    else:
        img_array = img_array.astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

def predict(image: Image.Image):
    input_shape = input_details[0]['shape']
    input_data = preprocess_image(image, input_shape)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

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

st.title("Plant Disease Detection (TFLite Model)")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Disease"):
        preds = predict(image)
        if preds.ndim == 2:
            preds = preds[0]
        
        st.write(f"Raw model output: {preds}")
        st.write(f"Output shape: {preds.shape}")
        
        predicted_index = np.argmax(preds)
        st.write(f"Predicted index: {predicted_index}")
        st.write(f"Number of classes: {len(CLASS_NAMES)}")
        
        if predicted_index >= len(CLASS_NAMES):
            st.error("Prediction index exceeds number of class labels! Please check CLASS_NAMES.")
        else:
            predicted_label = CLASS_NAMES[predicted_index]
            confidence = preds[predicted_index] * 100
            st.success(f"Prediction: **{predicted_label}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
