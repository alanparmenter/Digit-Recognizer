import os
import streamlit as st
import numpy as np
import psycopg2
import pandas as pd
import torch
import cv2
import streamlit_drawable_canvas as dc
from datetime import datetime

st.title("Digit Recognizer")

# Load trained PyTorch model
from recognizer import load_model
model = load_model('digit_model.pt')

def log_prediction(predicted_digit, true_value=None):
    conn = psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USER"),
        password=os.getenv("DATABASE_PASSWORD"),
        host=os.getenv("DATABASE_HOST"),
    )
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO predictions (date_time_stamp, predicted_digit, true_value) VALUES (NOW(), %s, %s)',
                    (predicted_digit, true_value)
                )
    finally:
        conn.close()
def centre_image(img):
    iy, ix = np.nonzero(img)
    cy, cx = np.mean(iy), np.mean(ix)
    rows, cols = img.shape
    shiftx = int(np.round(cols/2.0 - cx))
    shifty = int(np.round(rows/2.0 - cy))
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    return cv2.warpAffine(img, M, (cols, rows))

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

col1, col2 = st.columns(2)

with col1:
    canvas_result = dc.st_canvas(
        display_toolbar=False,
        fill_color="rgba(0,0,0,0)",
        stroke_width=10,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=200,
        height=200,
        drawing_mode="freedraw",
        key=f"f{st.session_state.canvas_key}",
    )
with col2:
    true_value = st.number_input("True Label:", min_value=0, max_value=9, step=1)
    if st.button("Submit"):
        try:
            img_array = np.array(canvas_result.image_data, dtype=np.uint8)
            img_gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
            img_resized = cv2.resize(img_gray, (28, 28))
            img_centred = centre_image(img_resized)
            img_tensor = torch.tensor(img_centred / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, prediction].item()
            log_prediction(prediction, true_value)
            st.write(f"Prediction: {prediction}")
            st.write(f"Confidence: {confidence:.2f}")
            st.session_state.canvas_key += 1 
        except Exception as e:
            st.write(e)
def get_prediction_dataframe():
    conn = psycopg2.connect(
        dbname=os.getenv("DATABASE_NAME"),
        user=os.getenv("DATABASE_USER"),
        password=os.getenv("DATABASE_PASSWORD"),
        host=os.getenv("DATABASE_HOST"),
    )
    try:
        df = pd.read_sql(
            "SELECT date_time_stamp AS timestamp, predicted_digit AS predicted, true_value AS labelled FROM predictions",
            conn
        )
        return df
    finally:
        conn.close()
df = get_prediction_dataframe()
st.dataframe(df, use_container_width=True)
