import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np
import streamlit as st

## Load the trained model, scaler pickle, OHE
model = load_model('model.h5')

## Load the encoder and scaler
with open('label_encoder_gender.pk1','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pk1','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pk1','rb') as file:
    scaler = pickle.load(file)

## streamlit app
st.title('Customer Churn Prediction')

## User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
credit_score = st.number_input('Credit Score')
age = st.number_input('Age', min_value=18, max_value=100, value=35)
tenure = st.number_input('Tenure',0,10)
balance = st.number_input('Balance')
num_of_products = st.number_input('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary')

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

## Combine OHE columns with input data
input_df = pd.concat([input_data.drop('Geography',axis=1),geo_encoded_df],axis=1)


# Ensure all columns expected by the scaler are present
expected_cols = scaler.feature_names_in_
input_df = input_df.reindex(columns=expected_cols, fill_value=0)

input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_probability = prediction[0][0]

print(f"Prediction Probability is {prediction_probability}")
st.write(f"Prediction Probability is {prediction_probability}")

if prediction_probability > 0.5:
    print('The customer is likely to churn.')
    st.write('The customer is likely to churn.')
else:
    print('The customer is not likely to churn.')
    st.write('The customer is not likely to churn.')