import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#import image
from tensorflow.keras.preprocessing import image
#import opencv

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url(http://www.bifmc.org/wp-content/uploads/2016/10/self-breast-examination-760x504.jpg);
background-size: 90%;
background-position: top;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./breast_CNN_model.h5')
	return model


def predict_class(image, model):
    
    image = tf.cast(image, tf.float32)
    #image = image.img_to_array(image)
   # image=cv2.imread(image)
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    return prediction


model = load_model()
#st.title('Alzhiemer Disease Prediction')
new_title = '<p style="font-family:sans-serif; color:Green; font-size: 50px; font-weight:bold;">BREAST CANCER DETECTION</p>'
st.markdown(new_title, unsafe_allow_html=True)

file = st.file_uploader("Upload an image ")

if file is None:
	st.text('Waiting for upload....')

else:
	slot = st.empty()
	slot.text('Running inference....')

	test_image = tf.keras.preprocessing.image.load_img(file,target_size=(128,128))
    
    
    

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['bengin', 'maligant', 'normal']

	result = class_names[np.argmax(pred)]

	output = 'The Stage is ' + result

	slot.text('Done')

	st.success(output)