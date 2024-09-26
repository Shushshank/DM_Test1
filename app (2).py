import streamlit as st
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

# Load the trained model
with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Revenue Prediction App')

# Create input fields for user to enter features
st.header('Enter Store Details')
avg_order_value = st.number_input('Average Order Value', min_value=0.0)
total_orders = st.number_input('Total Orders', min_value=0)
avg_customers_per_month = st.number_input('Average Customers Per Month', min_value=0)
avg_order_frequency = st.number_input('Average Order Frequency', min_value=0.0)

# Create a button to trigger the prediction
if st.button('Predict Monthly Revenue'):
    # Create a DataFrame with the user's input
    input_data = pd.DataFrame({
        'avg_order_value': [avg_order_value],
        'total_orders': [total_orders],
        'avg_customers_per_month': [avg_customers_per_month],
        'avg_order_frequency': [avg_order_frequency]
    })

    # Make the prediction using the loaded model
    predicted_revenue = model.predict(input_data)[0][0]

    # Display the predicted revenue
    st.subheader('Predicted Monthly Revenue:')
    st.write(f'{predicted_revenue:.2f}')
