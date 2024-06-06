import joblib
import numpy as np
import streamlit as st

model = joblib.load('xgb_model.pkl')

def predict(input_data):
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction[0]

def make_stationary(users_input, last_known_value):
        transformed_value = users_input - last_known_value
        return transformed_value

st.title("XGBRegressor Model")

st.write("The model predicts total confirmed COVID-19 cases in Zambia")

input_data = []

'''
The features:

'fullyVaccinated', 'new_deaths_smoothed',  'new_people_vaccinated_smoothed', 'new_vaccinations_smoothed', 
'partiallyVaccinated', 'stringency_index', 'test24hours', 'totalTests', 'totalVaccinations', 'vaccinated24hours', 
'rfh', 'r3h', 'month', 'day_of_week'.
'''

st.write("Enter input data:")

input_value_1 = st.number_input("Fully Vaccinated", min_value=9327654, step=1)
input_value_1 = make_stationary(input_value_1, last_known_value=9327654)
input_value_2 = st.number_input('New Deaths', min_value=0, step=1)
input_value_3 = st.number_input("New People Vaccinated", min_value=0, step=1)
input_value_3 = make_stationary(input_value_3, last_known_value=959)
input_value_4 = st.number_input("New Vaccinations", min_value=0, step=1)
input_value_5 = st.number_input("Partially Vaccinated", min_value=4663827, step=1)
input_value_5 = make_stationary(input_value_5, last_known_value=4663827)
input_value_6 = st.number_input("Stringency Index", min_value=0.0, step=0.01)
input_value_6 = make_stationary(input_value_6, last_known_value=13.89)
input_value_7 = st.number_input("Tests in 24 hours", min_value=0, step=1)
input_value_8 = st.number_input("Total Tests", min_value=4166833, step=1)
input_value_8 = make_stationary(input_value_8, last_known_value=4166833)
input_value_9 = st.number_input("Total Vaccinations", min_value=9982068, step=1)
input_value_9 = make_stationary(input_value_9, last_known_value=9982068)
input_value_10 = st.number_input("Vaccinated in the last 24 hours", min_value=0)
input_value_11 = st.number_input("rfh", min_value=0.0, step=0.01)
input_value_12 = st.number_input("r3h", min_value=0.0, step=0.01)
input_value_13 = st.number_input("Day of the week (0 to 6)", min_value=0, max_value=6, step=1)
input_value_14 = st.number_input("Month (1 to 12)", min_value=1, max_value=12, step=1)

input_data.extend([input_value_1, input_value_2, input_value_3, input_value_4, input_value_5, input_value_6, input_value_7,
                  input_value_8, input_value_9, input_value_10, input_value_11, input_value_12, input_value_13, input_value_14])

if st.button("Predict"):
    prediction = predict(input_data)
    st.write(f"Prediction: {int(prediction)}")