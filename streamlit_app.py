import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# App title
st.title("🖼️ CIFAR-10 Image Classifier")
st.write("Upload a 32x32 image to get a prediction from the trained CNN model.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).resize((32, 32))
    st.image(image, caption="Uploaded Image", use_column_width=False)

    # Preprocess the image
    img_array = np.array(image) / 255.0
    if img_array.shape != (32, 32, 3):
        st.error("Please upload a valid 32x32 RGB image.")
    else:
        img_input = img_array.reshape(1, 32, 32, 3)

        # Predict
        prediction = model.predict(img_input)
        predicted_class = class_names[np.argmax(prediction)]

        # Show result
        st.success(f"🧠 Prediction: **{predicted_class}**")
