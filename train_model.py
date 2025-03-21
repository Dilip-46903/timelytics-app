import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample data (replace with your real dataset)
data = {
    'product_category': ['Electronics', 'Clothing', 'Books', 'Electronics', 'Books'],
    'customer_location': ['New York', 'Los Angeles', 'Chicago', 'Chicago', 'New York'],
    'shipping_method': ['Standard', 'Express', 'Overnight', 'Standard', 'Express'],
    'delivery_time': [5, 2, 1, 4, 3]
}

df = pd.DataFrame(data)

# Encode categorical features
X = df[['product_category', 'customer_location', 'shipping_method']]
X_encoded = pd.get_dummies(X)
y = df['delivery_time']

# Train model
model = RandomForestRegressor()
model.fit(X_encoded, y)

# Save model
joblib.dump(model, 'delivery_time_model.pkl')
print("Model saved as delivery_time_model.pkl")
