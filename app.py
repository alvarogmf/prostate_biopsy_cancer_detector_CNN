import streamlit as st
import packages.Reporting.reporting as rp
from PIL import Image
import cv2
import skimage.io
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import tifffile as tiff


st.title("Prostate Biopsy Cancer Detector")
st.header("Prostrate cancer detector through Deep Learning")
st.text("Upload a Prostate Biopsy to determine if the sample has cancer or not")


uploaded_file = st.file_uploader("Choose a Biopsy Image ...", type='tiff')
if uploaded_file is not None:
    image = tiff.imread(uploaded_file)
    new_img = cv2.resize(image, (int(image.shape[1]/image.shape[0]*512),512))
    st.image(new_img)
    model = tf.keras.models.load_model(os.path.join('model/first_model_changing_isup_grade.h5'))
    new_image = cv2.resize(image,(258, 258))
    to_predict = np.array(new_image).reshape(-1, 258, 258, 3)
    prediction = model.predict(to_predict)

    if prediction == 0:
        st.write("This Biopsy has Cancer")
    elif prediction == 1:
        st.write("This Biopsy doesn't have Cancer")
    else:
        st.write("There was an error with the detection")