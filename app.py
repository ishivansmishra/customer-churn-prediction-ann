import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -------------------------------
# Load trained model (SAFE)
# -------------------------------
model = tf.keras.models.load_model('model.h5', compile=False)

# -------------------------------
# Load encoders & scaler
# -------------------------------
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title('Customer Churn Prediction')

# -------------------------------
# User Inputs (WITH UNIQUE KEYS)
# -------------------------------
geography = st.selectbox(
    'Geography',
    onehot_encoder_geo.categories_[0],
    key='geo'
)

gender = st.selectbox(
    'Gender',
    label_encoder_gender.classes_,
    key='gender'
)

age = st.slider('Age', 18, 92, key='age')

balance = st.number_input('Balance', key='balance')

credit_score = st.number_input('Credit Score', key='credit_score')

estimated_salary = st.number_input('Estimated Salary', key='salary')

tenure = st.slider('Tenure', 0, 10, key='tenure')

num_of_products = st.slider('Number of Products', 1, 4, key='products')

has_cr_card = st.selectbox('Has Credit Card', [0, 1], key='card')

is_active_member = st.selectbox('Is Active Member', [0, 1], key='active')

# -------------------------------
# Prepare Input Data
# -------------------------------
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# -------------------------------
# One-Hot Encode Geography
# -------------------------------
geo_encoded = onehot_encoder_geo.transform(
    pd.DataFrame([[geography]], columns=['Geography'])
).toarray()

geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# -------------------------------
# Combine Data
# -------------------------------
input_data = pd.concat(
    [input_data.reset_index(drop=True), geo_encoded_df],
    axis=1
)

# -------------------------------
# Ensure correct column order
# -------------------------------
input_data = input_data[scaler.feature_names_in_]

# -------------------------------
# Scale Input
# -------------------------------
input_data_scaled = scaler.transform(input_data)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# -------------------------------
# Output
# -------------------------------
st.subheader('Prediction Result')

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.error('The customer is likely to churn ❌')
else:
    st.success('The customer is not likely to churn ✅')