"""
Streamlit App: Live Digit Recognizer
--------------------------------------
Yeh app ek canvas dikhata hai jaha user digit draw kar sakta hai,
aur trained CNN model use karke live prediction deta hai.

Chalane ka tarika (terminal mein, isi folder se):
    streamlit run app.py
"""

import streamlit as st
import numpy as np
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognizer")
st.write("Neeche canvas pe koi bhi digit (0-9) draw karo, aur model uska prediction dikhayega.")

# ---------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# ---------------------------------------------------
# @st.cache_resource ka matlab: model ek hi baar load hoga, baar baar nahi
# (warna har draw ke saath model reload hota rehta, slow ho jaata)
@st.cache_resource
def load_digit_model():
    return load_model("digit_model.h5")

model = load_digit_model()

# ---------------------------------------------------
# 3. DRAWING CANVAS
# ---------------------------------------------------
# Yeh ek 280x280 wala black canvas banata hai jaha user white brush se draw karega
# (MNIST format: black background, white digit - isliye seedha wahi rakh rahe hain,
#  taaki humein invert na karna pade baad mein)
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,          # brush ki thickness - MNIST jaisi bold strokes ke liye
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ---------------------------------------------------
# 4. PREDICTION LOGIC
# ---------------------------------------------------
if canvas_result.image_data is not None:
    # Canvas se image nikalo (RGBA format mein aati hai)
    img = canvas_result.image_data

    # Grayscale mein convert karo (sirf 1 channel chahiye, RGB nahi)
    img_pil = Image.fromarray(img.astype('uint8')).convert('L')

    # 28x28 mein resize karo (MNIST ka size)
    img_resized = img_pil.resize((28, 28))
    img_array = np.array(img_resized)

    # Check karo canvas khaali toh nahi hai (agar sab black hai, kuch draw nahi hua)
    if img_array.max() > 0:
        # Normalize karo (0-1 range)
        img_normalized = img_array.astype('float32') / 255.0

        # Model ke liye shape sahi karo: (1, 28, 28, 1)
        img_final = img_normalized.reshape(1, 28, 28, 1)

        # Prediction lo
        prediction = model.predict(img_final)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # ---------------------------------------------------
        # 5. RESULTS DIKHAO
        # ---------------------------------------------------
        st.subheader(f"Prediction: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")

        # Bar chart se saari probabilities dikhao
        st.write("Saari digits ki probability:")
        prob_dict = {str(i): float(prediction[0][i]) for i in range(10)}
        st.bar_chart(prob_dict)
    else:
        st.info("Canvas pe koi digit draw karo prediction dekhne ke liye.")
