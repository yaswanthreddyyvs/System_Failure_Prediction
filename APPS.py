import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("XGBoost_model.joblib")
scaler = joblib.load("scaler.joblib")  # comment out if not using

st.set_page_config(page_title="System Failure Prediction", layout="centered")
st.title("🔧 System Failure Prediction App")

st.markdown("Enter the parameters below to predict failure status:")

# Input fields
type_map = {"L": 1, "M": 2, "H": 3}
product_type = st.selectbox("Product Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature [K]", value=298.0)
process_temp = st.number_input("Process Temperature [K]", value=308.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=50.0)
tool_wear = st.number_input("Tool Wear [min]", value=0.0)

# Feature engineering (automated)
tor_toolwear = torque * tool_wear                  # 6
temp_diff = process_temp - air_temp                # 7
power_rps = rot_speed * torque * 0.0166            # 8

# Show engineered features in UI
st.write("**Engineered Features**")
st.write(f"Tor*Toolwear: {tor_toolwear:.2f}")
st.write(f"Temp Difference: {temp_diff:.2f}")
st.write(f"Power [rps]: {power_rps:.2f}")

# Convert to model input format (9 features total)
features = np.array([
    type_map[product_type],   # 0: Type
    air_temp,                 # 1
    process_temp,             # 2
    rot_speed,                # 3
    torque,                   # 4
    tool_wear,                # 5
    tor_toolwear,             # 6
    temp_diff,                # 7
    power_rps                 # 8
]).reshape(1, -1)

# Apply scaling
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict"):
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if pred == 1:
        st.error(f"Failure Detected (Confidence: {prob:.2%})")
    else:
        st.success(f"No Failure (Confidence: {1 - prob:.2%})")
