import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('brain_tumor_detection_model.h5')

categories = ["glioma", "meningioma", "notumor", "pituitary"]

def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Brain Tumor Detection App")
st.write("Upload an MRI image to detect the type of brain tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    
    img_array = preprocess_image(img)
    
    predictions = model.predict(img_array)
    
    predicted_class = np.argmax(predictions, axis=1)
    
    predicted_label = categories[predicted_class[0]]
    
    st.markdown(f"<h2 style='text-align: center; color: #FF6347;'>Prediction: {predicted_label}</h2>", unsafe_allow_html=True)
    st.write("The model predicts the type of tumor based on the uploaded MRI scan.")

    
