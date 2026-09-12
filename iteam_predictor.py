import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ANN.h5")

loaded_model = load_model()

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

st.title("Fashion MNIST Image Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Convert RGB image to grayscale
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Resize to model input size
    resized = cv2.resize(gray, (28, 28))

    # Normalize exactly like training data
    resized = resized.astype("float32") / 255.0

    # Add batch dimension
    sample = np.expand_dims(resized, axis=0)

    # Prediction
    prediction = loaded_model.predict(sample, verbose=0)

    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.subheader("Prediction")

    st.success(f"Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")