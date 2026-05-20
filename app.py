import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("kc_house_data.csv")

# Features and target
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'floors']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit title
st.title("🏠 House Price Prediction Dashboard")

st.write("Predict house prices using Machine Learning")

# Inputs
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
sqft_living = st.slider("Square Feet Living", 500, 10000, 1800)
floors = st.slider("Floors", 1, 5, 1)

# Create input dataframe
sample = pd.DataFrame({
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'sqft_living': [sqft_living],
    'floors': [floors]
})

# Predict
prediction = model.predict(sample)

# Button
if st.button("Predict Price"):
    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

# Graph
st.subheader("House Price Distribution")

fig, ax = plt.subplots()

ax.hist(df['price'], bins=30)

ax.set_xlabel("Price")
ax.set_ylabel("Count")

st.pyplot(fig)