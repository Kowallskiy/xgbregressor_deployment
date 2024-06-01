import joblib
import numpy as np
import streamlit as st

model = joblib.load('xgb_model.pkl')

def predict(input_data):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction[0]

st.title("XGBRegressor Model")

st.write("Enter input data:")

