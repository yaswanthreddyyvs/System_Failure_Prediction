import streamlit as st
import numpy as np
import joblib
import os


BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "RandomForest_model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
 

st.set_page_config(page_title="System Failure Prediction", layout="centered")
st.title("System Failure Prediction App")

st.markdown("Enter the parameters below to predict failure status:")


type_map = {"L": 1, "M": 2, "H": 3}
product_type = st.selectbox("Product Type", ["L", "M", "H"])
air_temp = st.number_input("Air Temperature [K]", value=298.0)
process_temp = st.number_input("Process Temperature [K]", value=308.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1500.0)
torque = st.number_input("Torque [Nm]", value=50.0)
tool_wear = st.number_input("Tool Wear [min]", value=0.0)

tor_toolwear = torque * tool_wear                  
temp_diff = process_temp - air_temp                
power_rps = rot_speed * torque * 0.0166            


st.write("**Engineered Features**")
st.write(f"Tor*Toolwear: {tor_toolwear:.2f}")
st.write(f"Temp Difference: {temp_diff:.2f}")
st.write(f"Power [rps]: {power_rps:.2f}")


features = np.array([
    type_map[product_type],   
    air_temp,                 
    process_temp,             
    rot_speed,               
    torque,                   
    tool_wear,               
    tor_toolwear,             
    temp_diff,               
    power_rps                 
]).reshape(1, -1)


features_scaled = scaler.transform(features)


if st.button("Predict"):
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    if pred == 1:
        st.error(f"Failure Detected (Confidence: {prob:.2%})")
    else:
        st.success(f"No Failure (Confidence: {1 - prob:.2%})")
