import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pickle

st.title("Image Classifier")
st.text("Upload the Image")

model = pickle.load(open("img_model4.p", "rb"))

uploaded_file = st.file_uploader("Choose an Image: ", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="uploaded_image")

if st.button("PREDICT"):
    CATEGORIES = ['Sunflower', 'Rugby Ball Leather', 'Ice Cream Cone']
    st.write("Result...")
    flat_data = []
    
    # Convert PIL image to NumPy array
    img = np.array(img)
    
    # Resize using OpenCV
    img_resized = cv2.resize(img, (150, 150))

    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)

    y_out = model.predict(flat_data)
    st.title(f"Predicted Output: {CATEGORIES[y_out[0]]}")
