import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Page setup
st.set_page_config(page_title="🌿 Crop Yield Predictor", layout="wide")

# Header
st.title("🌾 Crop Yield Forecasting")
st.markdown("Predict the yield of major crops based on key farming inputs.")

# Load and prepare data
try:
    data = pd.read_csv("Crop_Yield.csv")
    data = pd.get_dummies(data, columns=["Crop"])
except Exception as e:
    st.error("❌ Error loading `Crop_Yield.csv`. Please ensure it exists and is correctly formatted.")
    st.stop()

X = data.drop("Yield", axis=1)
y = data["Yield"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Get crop list
crop_types = [col.replace("Crop_", "") for col in X.columns if col.startswith("Crop_")]

# Sidebar for user inputs
st.sidebar.header("🧾 Enter Farming Conditions")
rainfall = st.sidebar.slider("🌧️ Rainfall (mm)", 400, 1300, 800)
temperature = st.sidebar.slider("🌡️ Temperature (°C)", 20, 35, 27)
area = st.sidebar.number_input("🌾 Area of Land (hectares)", min_value=0.1, value=1.0)
fertilizer = st.sidebar.number_input("🧪 Fertilizer Used (kg)", min_value=0, value=100)
selected_crop = st.sidebar.selectbox("🌽 Select Crop", crop_types)

# Build input vector
input_data = [rainfall, temperature, area, fertilizer]
for crop in crop_types:
    input_data.append(1 if crop == selected_crop else 0)

input_df = pd.DataFrame([input_data], columns=X.columns)

# Prediction trigger
if st.sidebar.button("🔍 Predict Yield"):
    predicted_yield = model.predict(input_df)[0]
    st.success(f"🌱 Estimated Yield for **{selected_crop}**: **{predicted_yield:.2f} tons/hectare**")

    # Comparison chart
    st.subheader("📊 Yield Comparison Across Crops")
    comparison_data = []

    for crop in crop_types:
        test_input = [rainfall, temperature, area, fertilizer]
        for c in crop_types:
            test_input.append(1 if c == crop else 0)
        test_df = pd.DataFrame([test_input], columns=X.columns)
        pred = model.predict(test_df)[0]
        comparison_data.append((crop, pred))

    comparison_df = pd.DataFrame(comparison_data, columns=["Crop", "Predicted Yield"])
    comparison_df = comparison_df.sort_values(by="Predicted Yield", ascending=True)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(comparison_df["Crop"], comparison_df["Predicted Yield"], color="seagreen")
    ax.set_xlabel("Yield (tons/hectare)")
    ax.set_title("📈 Predicted Yield by Crop")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🧠 Built by Auwal Adam for the 3MTT Knowledge Showcase")
