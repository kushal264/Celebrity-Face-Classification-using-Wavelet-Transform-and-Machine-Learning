import streamlit as st
import cv2
import numpy as np
import json
import pickle
import pywt
import pandas as pd

# ---------- WAVELET TRANSFORM ----------
def w2d(img, mode="haar", level=1):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = np.float32(img_gray) / 255.0
    coeffs = pywt.wavedec2(img_gray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    img_wv = pywt.waverec2(coeffs_H, mode)
    img_wv = np.uint8(img_wv * 255)
    return img_wv

# ---------- LOAD MODEL & CLASSES ----------
@st.cache_resource
def load_all():
    with open("server/artifacts/class_dictionary.json", "r") as f:
        class_dict = json.load(f)
    num_to_class = {v: k for k, v in class_dict.items()}
    
    with open("server/artifacts/saved_model.pkl", "rb") as f:
        model = pickle.load(f)

    return num_to_class, model

num_to_class, model = load_all()

# ---------- PREDICTION ----------
def predict(img):
    img_resized = cv2.resize(img, (32, 32))
    img_wv = w2d(img_resized)
    combined = np.concatenate((img_resized.flatten(), img_wv.flatten())).reshape(1, -1)
    
    predicted_class = model.predict(combined)[0]
    probabilities = model.predict_proba(combined)[0]
    
    return num_to_class[predicted_class], probabilities


# ----------------- UI DESIGN ------------------

st.set_page_config(page_title="Celebrity Classifier", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center;'>Celebrity Face Classifier</h1>
    """,
    unsafe_allow_html=True
)

# ----------- TOP ROW: Circular Celebrity Images ----------
col1, col2, col3, col4, col5 = st.columns(5)

celebs = list(num_to_class.values())
img_paths = [
    "ui/messi.jpg",
    "ui/sharapova.jpg",
    "ui/federer.jpg",
    "ui/serena.jpg",
    "ui/ronaldo.jpg"
]

cols = [col1, col2, col3, col4, col5]

for col, name, path in zip(cols, celebs, img_paths):
    col.image(path, width=130, caption=name.upper(), use_column_width=False)
    col.markdown("<br>", unsafe_allow_html=True)


# ------- MIDDLE SECTION: Upload / Prediction / Table --------

left, mid, right = st.columns([1.2, 1, 1.2])

# 1️⃣ Upload box
uploaded_file = left.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    left.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", width=200)
    
    if left.button("Classify", use_container_width=True):
        name, prob = predict(img)
        
        # Show prediction result
        mid.image(img, caption=name, width=220)
        mid.markdown(f"<h3 style='text-align:center;'>{name}</h3>", unsafe_allow_html=True)
        
        # Probability table
        prob_table = pd.DataFrame({
            "Player": celebs,
            "Probability Score": prob
        })
        right.table(prob_table)
