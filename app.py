import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

# Load Trained Models
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder = pickle.load(file=file)

with open('onehhot_encoder_country.pkl','rb') as file:
    oh_encoder = pickle.load(file=file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file=file)

## Streamlit app
st.title("Customer Churn Prediction")

#User input
credit_score = st.number_input('credit_score')
country = st.selectbox('country',oh_encoder.categories_[0])
gender = st.selectbox('gender',label_encoder.classes_)
age = st.slider('age',18,90)
tenure = st.slider('Tenure',0,10)
balance = st.number_input('balance') 
products_number = st.slider('products_number',1,4)
credit_card = st.selectbox('credit_card',[0,1])
active_member = st.selectbox('active_member',[0,1])
estimated_salary = st.number_input('estimated_salary')

input_data = pd.DataFrame({
    "credit_score"    : [credit_score],
    "gender"          : [label_encoder.transform([gender])[0]],
    "age"             : [age],
    "tenure"          : [tenure],
    "balance"         : [balance],
    "products_number" : [products_number],
    "credit_card"     : [credit_card],
    "active_member"   : [active_member],
    "estimated_salary": [estimated_salary]
})

st.write("Here is the dataframe")
st.write(input_data)

ohe_df = pd.DataFrame(oh_encoder.transform([[country]]),columns=oh_encoder.get_feature_names_out(["country"]))

st.write("Here is the dataframe")
st.write(ohe_df)

conct_data = pd.concat([input_data,ohe_df], axis=1)
input_sclaed = scaler.transform(conct_data)

prediction = model.predict(input_sclaed)
st.write(f'churn probability: {prediction[0][0]*100:2f}')