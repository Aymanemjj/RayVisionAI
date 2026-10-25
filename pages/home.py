import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

MODEL_PATH = "./artifacts/ft_model.keras"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = (180, 180)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


st.title("Chest X-ray classifier")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file)
    st.image(img, use_container_width=True)

    prob = float(get_model().predict(preprocess(img))[0][0])
    label = CLASS_NAMES[int(prob > 0.5)]


    st.subheader(f"Prediction: {label}")