import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('delivery_time_model.pkl')

# App title
st.title("📦 Timelytics - Order Delivery Time Prediction")

# Input form
st.header("Enter Order Details:")

product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Home", "Books", "Other"])
customer_location = st.selectbox("Customer Location", ["Urban", "Suburban", "Rural"])
shipping_method = st.selectbox("Shipping Method", ["Standard", "Express", "Same Day"])

# You can add more inputs if needed
order_value = st.number_input("Order Value ($)", min_value=1.0, step=1.0)

# Convert categorical inputs to numerical (you must use same encoding as during training)
def preprocess_inputs(product_category, customer_location, shipping_method, order_value):
    category_map = {"Electronics": 0, "Clothing": 1, "Home": 2, "Books": 3, "Other": 4}
    location_map = {"Urban": 0, "Suburban": 1, "Rural": 2}
    shipping_map = {"Standard": 0, "Express": 1, "Same Day": 2}

    features = [
        category_map[product_category],
        location_map[customer_location],
        shipping_map[shipping_method],
        order_value
    ]
    return np.array(features).reshape(1, -1)

# Prediction button
if st.button("Predict Delivery Time"):
    inputs = preprocess_inputs(product_category, customer_location, shipping_method, order_value)
    prediction = model.predict(inputs)
    st.success(f"📅 Expected Delivery Time: {prediction[0]:.2f} days")
