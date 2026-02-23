import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

def analyze_and_forecast():
    # --- 1. UI HEADER ---
    st.title("🌾 Siyana Mandi Wheat Price Predictor")
    st.markdown("### 2026 Monthly Market Maximum Price Forecast")
    
    # --- 2. DATA LOADING ---
    df = pd.read_csv('wheat_prices_siyana.csv')
    df['Price'] = df['Price (Rs/Qtl)'].str.replace('Rs. ', '', regex=False).str.replace(',', '', regex=False).astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['MonthName'] = df['Date'].dt.month_name()
    df['DayOfYear'] = df['Date'].dt.dayofyear

    # --- 3. STREAMLIT INPUT (Replaces input()) ---
    jan_input_price = st.number_input("Enter current market price for January 2026 (Rs/Qtl):", 
                                      min_value=1000.0, 
                                      value=2400.0, 
                                      step=10.0)

    avg_jan_hist = df[df['MonthName'] == 'January']['Price'].mean()
    scale_factor = jan_input_price / avg_jan_hist

    # --- 4. ML LOGIC ---
    X = df[['DayOfYear']]
    y = df['Price'] * scale_factor
    model = DecisionTreeRegressor(max_depth=10)
    model.fit(X, y)

    timeline_2026 = pd.date_range(start='2026-01-01', end='2026-06-30')
    predict_df = pd.DataFrame({
        'Date': timeline_2026,
        'DayOfYear': timeline_2026.dayofyear,
        'Month': timeline_2026.month_name()
    })
    predict_df['Predicted_Price'] = model.predict(predict_df[['DayOfYear']])

    # --- 5. UI OUTPUT (Replaces print()) ---
    if st.button("Generate Forecast"):
        st.write("---")
        st.subheader("Predicted Peak Windows for 2026")
        
        # Creating a list to store data for a nice table
        results = []

        for month in ['January', 'February', 'March', 'April', 'May', 'June']:
            month_data = predict_df[predict_df['Month'] == month]
            if not month_data.empty:
                max_price = month_data['Predicted_Price'].max()
                max_dates = month_data[month_data['Predicted_Price'] == max_price]['Date']

                start_window = max_dates.min().strftime('%d %B')
                end_window = max_dates.max().strftime('%d %B')
                
                results.append({
                    "Month": month,
                    "Predicted Max Price (Rs)": round(max_price, 2),
                    "Expected Peak Window": f"{start_window} to {end_window}"
                })

        # Display as a professional table
        st.table(pd.DataFrame(results))

        # Optional: Add a chart because everyone loves charts!
        st.line_chart(predict_df.set_index('Date')['Predicted_Price'])

if __name__ == "__main__":
    analyze_and_forecast()
