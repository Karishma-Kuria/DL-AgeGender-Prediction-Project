import streamlit as st
import tensorflow as tf
from tensorflow import keras
import streamlit as st
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('/content/DL-AgeGender-Prediction-Project/models/a_g_best.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Gender Age Prediction
         """
         )

file = st.file_uploader("Please upload a file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    
        size = (198,198)    
        image1 = ImageOps.fit(image_data, size)
        image1 = np.asarray(image1)
        image2 = image1.astype(np.float32)
        img = np.expand_dims(image2, axis = 0)
        img1 = np.copy(img)
        img1 = np.divide(img1, 255, out=img1, casting="unsafe")
        age_pred, gender_pred = model.predict(img1)
        max=-1
        count=0

        for i in age_pred[0]:
          if i>max:
            max = i
            temp = count
          count+=1
        print(temp)
        if temp==0:
          st.header('0-24 yrs old')
        if temp==1:
          st.header('25-49 yrs old')
        if temp==2:
          st.header('50-74 yrs old')
        if temp==3:
          st.header('75-99 yrs old')
        if temp==4:
          st.header('91-124 yrs old')

        if gender_pred[0][0]>gender_pred[0][1]:
          st.header('Male')
        else:
          st.header('Female')
        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    file.getvalue()
    print(file)
    st.image(image, use_column_width=True)
    import_and_predict(image, model)