"""
Streamlit App: Handwritten Digit Recognizer (v3 - multi-digit support)
--------------------------------------------------------------------------
Draws a canvas where the user can write one OR multiple digits.
Uses OpenCV contour detection to segment individual digits, then
runs each through the trained CNN and combines results into a number.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image

# ---------------------------------------------------
# 1. PAGE SETUP
# ---------------------------------------------------
st.set_page_config(page_title="Digit Recognizer", page_icon="🔢", layout="centered")

st.title("🔢 Handwritten Digit Recognizer")
st.markdown(
    "A CNN trained on MNIST (99.2% test accuracy). Draw **one or more digits** "
    "below and it will recognize the full number."
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
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Reset Canvas", use_container_width=True):
        st.session_state.canvas_key += 1
        st.rerun()

# ---------------------------------------------------
# 4. DRAWING CANVAS (wider now, to fit multiple digits)
# ---------------------------------------------------
with col1:
    st.caption("Draw here (leave a gap between digits):")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=560,
    height=280,
    drawing_mode="freedraw",
    key=f"canvas_{st.session_state.canvas_key}",
)

# ---------------------------------------------------
# 5. DIGIT SEGMENTATION + PREDICTION
# ---------------------------------------------------
def segment_and_predict(canvas_image, model, min_area=100):
    """
    Takes the raw canvas RGBA image, finds each individual digit using
    OpenCV contours, crops + preprocesses each one, and predicts it.
    Returns: predicted number (str), list of (digit_crop, predicted_digit, confidence)
    """
    # Convert canvas RGBA image to grayscale
    gray = cv2.cvtColor(canvas_image.astype('uint8'), cv2.COLOR_RGBA2GRAY)

    # Canvas is already black background / white strokes (MNIST style),
    # so pixels > 0 ARE the digit already - no inversion needed.
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] * b[3] > min_area]
    boxes = sorted(boxes, key=lambda b: b[0])  # left to right

    results = []
    predicted_number = ""

    for (x, y, w, h) in boxes:
        digit_crop = thresh[y:y+h, x:x+w]

        # Make square (preserve aspect ratio)
        size = max(w, h)
        square = np.zeros((size, size), dtype=np.uint8)
        y_off, x_off = (size - h) // 2, (size - w) // 2
        square[y_off:y_off+h, x_off:x_off+w] = digit_crop

        # Add padding (like MNIST digits have margin)
        padded_size = int(size * 1.4)
        padded = np.zeros((padded_size, padded_size), dtype=np.uint8)
        offset = (padded_size - size) // 2
        padded[offset:offset+size, offset:offset+size] = square

        # Resize to 28x28 and normalize
        digit_img = Image.fromarray(padded).resize((28, 28))
        digit_array = np.array(digit_img).astype('float32') / 255.0
        digit_input = digit_array.reshape(1, 28, 28, 1)

        pred = model.predict(digit_input, verbose=0)
        predicted_digit = int(np.argmax(pred))
        confidence = float(np.max(pred) * 100)

        predicted_number += str(predicted_digit)
        results.append((digit_array, predicted_digit, confidence))

    return predicted_number, results


# Newer versions of streamlit-drawable-canvas raise RuntimeError if image_data
# is read before the canvas has sent any data (e.g. on first page load).
# So we read it safely and treat that case as "nothing drawn yet".
try:
    image_data = canvas_result.image_data
except RuntimeError:
    image_data = None

if image_data is not None:
    if image_data[:, :, :3].max() > 0:  # canvas is not empty
        predicted_number, results = segment_and_predict(image_data, model)

        if predicted_number:
            st.divider()
            st.subheader(f"Predicted number: {predicted_number}")

            st.caption(f"Detected {len(results)} digit(s):")
            cols = st.columns(len(results))
            for i, (digit_img, digit, conf) in enumerate(results):
                with cols[i]:
                    st.image(digit_img, width=80, clamp=True)
                    st.write(f"**{digit}** ({conf:.0f}%)")

            avg_conf = np.mean([r[2] for r in results])
            if avg_conf < 50:
                st.warning(
                    "Low confidence — try drawing bigger, bolder digits with "
                    "clear gaps between them."
                )
        else:
            st.info("Couldn't detect a digit — try drawing bigger and bolder.")
    else:
        st.info("👆 Draw one or more digits above to see the prediction.")

# ---------------------------------------------------
# 6. SIDEBAR - PROJECT INFO
# ---------------------------------------------------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        - Trained on the **MNIST** dataset (60,000 images)
        - **CNN** architecture with Dropout regularization
        - **99.2%** test accuracy on single digits
        - **Multi-digit support** via OpenCV contour segmentation
        - Built with TensorFlow/Keras, OpenCV + Streamlit

        **Tips for best results:**
        - Draw bold, thick strokes
        - Leave a gap between digits
        - Keep digits roughly the same size
        """
    )
    st.markdown("[View source code on GitHub](https://github.com/bitmesra25-ui/mnist-digit-recognizer)")
