import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2

# 1. Load your trained model
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_my_model()

st.title("🔢 MNIST Digit Recognizer")
st.write("Draw a digit (0-9) in the box below and let the AI guess it!")

# 2. Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# 3. Process the drawing when the user is done
if canvas_result.image_data is not None:
    # Resize drawing to 28x28 (MNIST size)
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Normalize
    img_normalized = img_gray / 255.0
    # Reshape for model (1, 28, 28, 1)
    img_final = np.expand_dims(img_normalized, axis=(0, -1))

    if st.button('Predict'):
        prediction = model.predict(img_final)
        result = np.argmax(prediction)
        confidence = np.max(prediction)
        
        st.header(f"Result: {result}")
        st.write(f"Confidence: {confidence*100:.2f}%")
