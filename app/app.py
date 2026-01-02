import streamlit as st
import joblib
import numpy as np

import os
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

st.title("🎓 Student Performance Predictor")
st.write("Predict a student's final score using ML")

# User inputs
study_hours = st.slider("Study Hours", 1, 4, 2)
prev_score_1 = st.number_input("Previous Score 1 (G1)", 0, 20, 10)
prev_score_2 = st.number_input("Previous Score 2 (G2)", 0, 20, 10)
absences = st.number_input("Number of Absences", 0, 100, 5)

if st.button("Predict Final Score"):
    input_data = np.array([[study_hours, prev_score_1, prev_score_2, absences]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Final Score: {prediction[0]:.2f}")

    st.info(
        "Prediction is based on study hours, previous academic performance "
        "and attendance using a trained ML regression model."
    )

    if prev_score_1 + prev_score_2 == 0:
        st.warning(
            "Previous scores are very low — prediction may be unreliable."
        )
import pandas as pd

features = ["Study Hours", "G1", "G2", "Absences"]
importance = model.feature_importances_

st.bar_chart(pd.Series(importance, index=features))
st.markdown("---")
st.caption("Built by Shaik Raihan Basha | ML + Streamlit Project")
