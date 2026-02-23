import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("🌾 Siyana Mandi Wheat Price Predictor")

# 1. Load Files
try:
    model = joblib.load('wheat_model.joblib')
    jan_ref = joblib.load('jan_ref.joblib')
    df = pd.read_csv('wheat_prices_siyana.csv')
    st.success("✅ Model and Data loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# 2. User Input
jan_price = st.number_input("Enter January 2026 Price (per quintal):", min_value=1000, value=2400)

if st.button("Predict 2026 Peak Windows"):
    # Scaling Logic
    scaling_factor = jan_price / jan_ref
    
    # Simple Forecast logic (Example)
    st.subheader("Results for 2026")
    st.write(f"Using Scaling Factor: {scaling_factor:.2f}")
    
    # Add your prediction and plotting code here...
    # st.pyplot(plt)
