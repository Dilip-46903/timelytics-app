import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('delivery_time_model.pkl')

st.title("📦 Delivery Time Prediction")

# User inputs
product_category = st.selectbox("Select Product Category", ['Electronics', 'Clothing', 'Books'])
customer_location = st.selectbox("Select Customer Location", ['New York', 'Los Angeles', 'Chicago'])
shipping_method = st.selectbox("Select Shipping Method", ['Standard', 'Express', 'Overnight'])

# Input DataFrame
input_data = pd.DataFrame({
    'product_category': [product_category],
    'customer_location': [customer_location],
    'shipping_method': [shipping_method]
})

# Encode like training
input_encoded = pd.get_dummies(input_data)

# Align with training columns
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[model_columns]

# Predict
if st.button("Predict Delivery Time"):
    prediction = model.predict(input_encoded)
    st.success(f"🚚 Estimated Delivery Time: {prediction[0]:.2f} days")
