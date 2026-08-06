"""
Streamlit App: Live Digit Recognizer (v2 - with reset button + polished UI)
------------------------------------------------------------------------------
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
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered")

st.title("🔢 Handwritten Digit Recognizer")
st.markdown(
    "A CNN trained on MNIST (99.2% test accuracy) that recognizes handwritten "
    "digits in real time. Draw a single digit (0-9) below."
)

# ---------------------------------------------------
# 2. LOAD THE TRAINED MODEL
# ---------------------------------------------------
@st.cache_resource
def load_digit_model():
    return load_model("digit_model.h5")

model = load_digit_model()

# ---------------------------------------------------
# 3. RESET BUTTON LOGIC
# ---------------------------------------------------
# Trick: streamlit-drawable-canvas doesn't have a direct "clear" API,
# so we force a full canvas reset by changing its `key` every time
# the reset button is pressed. A new key = a brand new blank canvas.
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Reset Canvas", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()

# ---------------------------------------------------
# 4. DRAWING CANVAS
# ---------------------------------------------------
with col1:
    st.caption("Draw here:")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,          # bold stroke - matches MNIST training style
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",  # changes on reset -> new blank canvas
)

# ---------------------------------------------------
# 5. PREDICTION LOGIC
# ---------------------------------------------------
if canvas_result.image_data is not None:
    img = canvas_result.image_data
    img_pil = Image.fromarray(img.astype('uint8')).convert('L')
    img_resized = img_pil.resize((28, 28))
    img_array = np.array(img_resized)

    if img_array.max() > 0:
        img_normalized = img_array.astype('float32') / 255.0
        img_final = img_normalized.reshape(1, 28, 28, 1)

        prediction = model.predict(img_final, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # ---------------------------------------------------
        # 6. RESULTS DISPLAY
        # ---------------------------------------------------
        st.divider()
        result_col1, result_col2 = st.columns([1, 2])

        with result_col1:
            st.metric(label="Prediction", value=str(predicted_digit))
            st.metric(label="Confidence", value=f"{confidence:.1f}%")

        with result_col2:
            st.caption("Probability across all digits:")
            prob_dict = {str(i): float(prediction[0][i]) for i in range(10)}
            st.bar_chart(prob_dict)

        if confidence < 50:
            st.warning(
                "Low confidence — try drawing the digit bigger and bolder, "
                "centered in the canvas."
            )
    else:
        st.info("👆 Draw a digit above to see the prediction.")

# ---------------------------------------------------
# 7. SIDEBAR - PROJECT INFO
# ---------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        - Trained on the **MNIST** dataset (60,000 images)
        - **CNN** architecture with Dropout regularization
        - **99.2%** test accuracy
        - Built with TensorFlow/Keras + Streamlit

        **Note:** Works best with a single, bold, centered digit.
        Multi-digit numbers are not yet supported.
        """
    )
    st.markdown("[View source code on GitHub](https://github.com/bitmesra25-ui/mnist-digit-recognizer)")
