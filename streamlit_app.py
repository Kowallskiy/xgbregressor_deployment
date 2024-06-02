import joblib
import numpy as np
import streamlit as st

model = joblib.load('xgb_model.pkl')

def predict(input_data):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction[0]

st.title("XGBRegressor Model")

st.write("Enter input data:")

input_data = []
'''
'fullyVaccinated', 'imputed_total_cases', 'imputed_total_deaths',
'new_vaccinations_smoothed', 'partiallyVaccinated', 'stringency_index',
'test24hours', 'totalTests', 'totalVaccinations',
'total_tests_per_thousand', 'vaccinated24hours', 'rfh', 'r3h', 'day', 'day_of_week', 'month'
'''
input_value_1 = st.number_input("Fully Vaccinated", min_value=9327654, step=1)
input_value_2 = st.number_input('Total Deaths', min_value=4072, step=1)
input_value_3 = st.number_input("New Vaccinations", min_value=0, step=1)
input_value_4 = st.number_input("Partially Vaccinated", min_value=4663827, step=1)
input_value_5 = st.number_input("Stringency Index", min_value=0)
input_value_6 = st.number_input("Tests in the last 24 hours", min_value=0)
input_value_7 = st.number_input("Total Tests", min_value=4166833, step=1)
input_value_8 = st.number_input("Total Vaccinations", min_value=9982068, step=1)
input_value_9 = st.number_input("Total Tests per 1000", min_value=0)
input_value_10 = st.number_input("Vaccinated in the last 24 hours", min_value=0)
input_value_11 = st.number_input("rfh", min_value=0)
input_value_12 = st.number_input("r3h", min_value=0)
input_value_13 = st.number_input("Day (out of 365)", min_value=1, max_value=365, step=1)
input_value_14 = st.number_input("Day of the week", min_value=0, max_value=6, step=1)
input_value_15 = st.number_input("Month", min_value=1, max_value=12, step=1)

input_data.extend([input_value_1, input_value_2, input_value_3, input_value_4, input_value_5, input_value_6, input_value_7,
                  input_value_8, input_value_9, input_value_10, input_value_11, input_value_12, input_value_13, input_value_14,
                  input_value_15])

st.write(f"Input Data: {input_data}")
st.write(f"Input Data Shape: {np.array(input_data).reshape(1, -1).shape}")

if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f"Prediction: {prediction}")