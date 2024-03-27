import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

# @st.cache(suppress_st_warning=True,allow_output_mutation=True)
def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (100,100),Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

model = tf.keras.models.load_model(r'my_model2.h5')

st.write("""
         # ***Glaucoma detector using CNN and deep learning***
         """
         )

st.write("This is a simple image classification web app to predict glaucoma through fundus image of eye")

file = st.file_uploader("Please upload an image(jpg) file", type=["jpg"])

if file is None:
    st.text("You haven't uploaded a jpg image file")
else:
    imageI = Image.open(file)
    prediction = import_and_predict(imageI, model)
    pred = prediction[0][0]
    if(pred > 0.5):
        st.write("""
                 ## **Prediction:** Your eye is Healthy. Great!!
                 """,""" ## **Amount of prediction%: **""",pred
                 )
        st.balloons()
    else:
        st.write("""
                 # Prediction: You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible.
                 """
                 )
        st.write(""" 
                 ## **Amount of prediction %:**""",pred)
        st.write(""" ### Here are some symptoms of glaucoma that users can watch out for: 

##### Gradual Loss of Peripheral Vision: One of the earliest signs of glaucoma is often the gradual loss of peripheral (side) vision. This may go unnoticed at first but can progress over time if left untreated.

##### Blurred Vision: Individuals with glaucoma may experience blurred or hazy vision, especially in low light conditions.

##### Halos Around Lights: Glaucoma can cause halos or rainbow-colored rings to appear around lights, particularly at night.

##### Intense Eye Pain: In acute cases of angle-closure glaucoma, individuals may experience sudden, severe eye pain, along with headaches and nausea.

##### Redness in the Eye: Glaucoma can cause the eye to appear red or bloodshot due to increased pressure within the eye.

##### Nausea or Vomiting: In acute angle-closure glaucoma, individuals may experience nausea or vomiting along with severe eye pain.

##### Vision Loss: As glaucoma progresses, it can lead to permanent vision loss or blindness if left untreated. This typically starts with peripheral vision loss and can advance to tunnel vision or complete blindness.

It's essential to note that early-stage glaucoma may not present any noticeable symptoms, which is why regular eye exams, especially for individuals over 40 or with a family history of glaucoma, are critical for early detection and treatment. If users experience any of the above symptoms, they should seek immediate medical attention from an eye care professional for a comprehensive eye examination. Early diagnosis and treatment can help slow or prevent further vision loss caused by glaucoma.  """)

