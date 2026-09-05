"""
Machine Failure Predictor — Streamlit app

Loads the Random Forest trained in train_model.py (model.pkl) and lets
someone enter sensor readings to get a live failure prediction.

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Machine Failure Predictor", page_icon="⚙️", layout="centered")

MODEL_PATH = "model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_missing = False
except FileNotFoundError:
    model_missing = True

st.title("Machine Failure Predictor")
st.write(
    "Enter a machine's current sensor readings to estimate its failure risk. "
    "This uses a Random Forest trained on the AI4I 2020 dataset."
)

if model_missing:
    st.error(
        f"Couldn't find `{MODEL_PATH}` in this folder. Run `train_model.py` "
        "first so it can save the trained model here, then restart this app."
    )
    st.stop()

st.subheader("Sensor readings")
st.caption(
    "The model was trained on five readings together. Torque and rotational "
    "speed carry the most weight in its predictions — temperature and tool "
    "wear still matter, so all five are needed for an accurate result."
)

col1, col2 = st.columns(2)
with col1:
    air_temp = st.number_input("Air temperature (K)", value=300.0, step=0.1, format="%.1f")
    process_temp = st.number_input("Process temperature (K)", value=310.0, step=0.1, format="%.1f")
    rot_speed = st.number_input("Rotational speed (rpm)", value=1500, step=10)
with col2:
    torque = st.number_input("Torque (Nm)", value=40.0, step=0.5, format="%.1f")
    tool_wear = st.number_input("Tool wear (min)", value=100, step=1)

if st.button("Predict", type="primary"):
    X = np.array([[air_temp, process_temp, rot_speed, torque, tool_wear]])
    prediction = model.predict(X)[0]
    failure_probability = model.predict_proba(X)[0][1]

    if prediction == 1:
        st.error(f"Failure predicted — estimated risk {failure_probability:.1%}")
    else:
        st.success(f"No failure predicted — estimated risk {failure_probability:.1%}")

    st.progress(min(float(failure_probability), 1.0))

st.divider()
st.caption(
    "On held-out test data this model catches about 60% of real failures "
    "at 89% precision (few false alarms, but it does miss some). Treat this "
    "as a learning demo, not a substitute for real maintenance monitoring."
)
