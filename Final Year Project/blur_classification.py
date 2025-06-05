import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Set up the Streamlit app
st.title("Blur Classification App")
st.write("This app classifies images as blurred or sharp.")

# Function to load and preprocess the image
def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Load the trained model
@st.cache_resource
def load_trained_model():
    model_path = "/Users/admin/Desktop/Acadmcs/Final Year Project/blur_detection_cnn_model.h5"  # Update this with the actual path
    model = load_model(model_path)
    return model

model = load_trained_model()

# Debug model input shape
st.write(f"Model expects input shape: {model.input_shape}")

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    target_size = model.input_shape[1:3]  # Extract required input dimensions (e.g., (224, 224))
    preprocessed_image = preprocess_image(image, target_size=target_size)

    # Debug input image shape
    st.write(f"Input image shape after preprocessing: {preprocessed_image.shape}")

    # Make prediction
    try:
        prediction = model.predict(preprocessed_image)
        class_labels = ["Blurred", "Sharp"]  # Update if needed based on your model
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display the result
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
    except ValueError as e:
        st.error(f"Error during prediction: {e}")


