import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import tensorflow as tf

# ── CONFIG ──────────────────────────────────────────────
IMG_SIZE   = 64
CHARACTERS = ["a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au"]
DEVANAGARI = ["अ", "आ", "इ", "ई", "उ", "ऊ", "ए", "ऐ", "ओ", "औ"]
# ────────────────────────────────────────────────────────

# ── LOAD MODEL (cached so it only loads once) ───────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("hindi_vowel_model.keras")

model = load_model()

# ── UI ───────────────────────────────────────────────────
st.title("Hindi Vowel Recognizer")
st.write("Draw a Hindi vowel below and the model will predict it.")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=3,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Get the drawn image
        img = canvas_result.image_data.astype("uint8")

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

        # Find the bounding box of the actual drawing (non-black pixels)
        coords = cv2.findNonZero(img)

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)

            # Crop tightly around the drawn character
            cropped = img[y:y+h, x:x+w]

            # Make the crop SQUARE by padding the shorter side to match the longer side
            size = max(w, h)
            square = np.zeros((size, size), dtype=np.uint8)
            y_offset = (size - h) // 2
            x_offset = (size - w) // 2
            square[y_offset:y_offset+h, x_offset:x_offset+w] = cropped

            # Add extra padding around the square so character isn't edge-to-edge
            pad = int(size * 0.6)
            square = cv2.copyMakeBorder(
                square, pad, pad, pad, pad,
                cv2.BORDER_CONSTANT, value=0
            )

            img = square
        # If nothing was drawn, img stays as-is (will predict garbage, that's fine)

        # Invert colors: training data was dark ink on white background,
        # canvas is white stroke on black background
        img = cv2.bitwise_not(img)

        # Resize to model's expected input size
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # Normalize
        img = img.astype("float32") / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # DEBUG: save what the model actually sees
        debug_img = (img.reshape(IMG_SIZE, IMG_SIZE) * 255).astype("uint8")
        cv2.imwrite("debug_input.png", debug_img)

        # Predict
        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)
        confidence = predictions[0][predicted_index] * 100

        st.subheader(f"Prediction: {DEVANAGARI[predicted_index]} ({CHARACTERS[predicted_index]})")
        st.write(f"Confidence: {confidence:.1f}%")

        # Show all probabilities
        st.write("All probabilities:")
        for i, char in enumerate(DEVANAGARI):
            st.write(f"{char} ({CHARACTERS[i]}): {predictions[0][i]*100:.1f}%")
    else:
        st.write("Please draw something first.")