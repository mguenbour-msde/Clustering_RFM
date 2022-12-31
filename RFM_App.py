import streamlit as st
import pandas as pd
import pickle


st.write("""
# MSDE4 : ML Course
## Projet ML RFM App

This app predicts the customer category
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Recency = st.sidebar.slider('Recency', 1.0, 360.0, 310.0)
    Frequency = st.sidebar.slider('Frequency', 1, 100, 1)
    Monetary_value = st.sidebar.slider('Monetory_value', 1.0, 30000.0, 295.0)
    data = {'Recency': Recency,
            'Frequency': Frequency,
            'Monetary_value': Monetary_value}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

model_rfm=pickle.load(open("model_rfm.pkl", "rb"))
prediction = model_rfm.predict(df)

st.subheader('Class labels and their corresponding index number')
st.write(pd.DataFrame(model_rfm.classes_))

st.subheader('Prediction')
st.write(prediction)