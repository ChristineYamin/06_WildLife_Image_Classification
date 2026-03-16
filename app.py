import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

#Page Config
st.set_page_config(page_title="Wildlife Image Classifier" , page_icon="🐾")
st.title("🐾 Wildlife Image Classification App")
st.write("Upload a wildlife image and the model will predict the animal class.")

# Class Labels
class_names = [
    "bear",
    "cheetah",
    "chimpanzee",
    "crocodile",
    "deer",
    "eagle",
    "elephant",
    "fox",
    "giraffe",
    "kangaroo",
    "snake",
    "wolf"
]

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_mobilenet_model.keras")
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224,224))
    image_array = np.array(image)

    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array.astype(np.float32))
    return image_array

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg","jpeg","png"]

)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    