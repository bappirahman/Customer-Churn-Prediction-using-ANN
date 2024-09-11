import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

st.title('Churn Prediction')

credit_score = st.slider('Credit Score', 350, 850)
gender = st.radio('Gender', ('Male', 'Female'))
age = st.slider('Age', 18, 100)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance', 0.0, 500000.0)
num_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.radio('Has Credit Card', ('Yes', 'No'))
is_active_member = st.radio('Is Active Member', ('Yes', 'No'))
estimated_salary = st.number_input('Estimated Salary', 0.0, 500000.0)
geography = st.selectbox('Geography', ('France', 'Germany', 'Spain'))

model = tf.keras.models.load_model('./assets/churn_model.h5')

with open('./assets/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
with open('./assets/label_encoder.pkl', 'rb') as le_file:
    le = pickle.load(le_file)
    
with open('./assets/one_hot_encoder.pkl', 'rb') as ohe_file:
    ohe = pickle.load(ohe_file)

if st.button('Predict'):
    input_data = {
        'CreditScore': [credit_score],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
        'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
    }

    # Process input data
    input_df = pd.DataFrame(input_data)
    geo_ohe = ohe.transform([input_df['Geography']])
    geo_features = ohe.get_feature_names_out(['Geography'])
    geo_df = pd.DataFrame(geo_ohe.toarray(), columns=geo_features)
    
    # Merge processed data
    input_df.drop(['Geography'], axis=1, inplace=True)
    input_df = pd.concat([input_df, geo_df], axis=1)
    input_df['Gender'] = le.transform(input_df['Gender'])
    input_df = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_df)
    st.write(input_df)
    # Display prediction
    st.write('Customer will most likely leave' if prediction > 0.5 else 'Customer will most likely stay')
