# app.py
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

# Import our custom modules
from data_fetcher import fetch_historical_data
from model_trainer import build_and_train_lstm_model, predict_future_prices # Added 's' to prices
from num1 import preprocess_data # Import preprocess_data for initial display/scaling if needed

# --- Streamlit App Configuration ---
st.set_page_config(layout="wide", page_title="Crypto Price Predictor")

st.title("📈 Cryptocurrency Price Predictor")
st.markdown("Predicting the next 30 days of cryptocurrency prices using LSTM Neural Networks.")

# --- User Inputs ---
st.sidebar.header("Configuration")
coin_options = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Cardano": "cardano",
    "Solana": "solana",
    "Dogecoin": "dogecoin"
}
selected_coin_name = st.sidebar.selectbox("Select Cryptocurrency", list(coin_options.keys()))
selected_coin_id = coin_options[selected_coin_name]

# Days of historical data to fetch for training
days_to_fetch = st.sidebar.slider("Historical Data Days (for training)", 180, 730, 365, step=30) # 0.5 to 2 years

# Model Hyperparameters
look_back = st.sidebar.slider("Look Back Period (Days for LSTM input)", 30, 90, 60, step=5)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, step=10)
batch_size = st.sidebar.slider("Batch Size", 16, 64, 32, step=16)

# --- Main App Logic ---
if st.sidebar.button("Predict Price"):
    st.header(f"Predicting {selected_coin_name} Price")
    st.info("Fetching historical data and training the model. This may take a moment...")

    # 1. Fetch Data
    with st.spinner(f"Fetching {selected_coin_name} historical data for {days_to_fetch} days..."):
        df_historical = fetch_historical_data(selected_coin_id, days=days_to_fetch)

    if df_historical is None or df_historical.empty:
        st.error(f"Could not fetch historical data for {selected_coin_id}. Please try again or check the coin ID.")
    else:
        st.success("Historical data fetched successfully!")
        st.subheader("Historical Data Overview")
        st.write(df_historical.tail()) # Show last few entries

        # 2. Train Model
        with st.spinner("Training LSTM model..."):
            # Ensure df_historical has enough data for training
            if len(df_historical) < look_back + 1:
                st.error(f"Not enough historical data ({len(df_historical)} days) for the selected 'Look Back Period' ({look_back} days). Please fetch more historical data or reduce the look back period.")
            else:
                model, scaler, raw_training_data_for_pred = build_and_train_lstm_model(
                    df_historical,
                    look_back=look_back,
                    epochs=epochs,
                    batch_size=batch_size
                )

                if model is None:
                    st.error("Model training failed. Please check your inputs or data.")
                else:
                    st.success("Model trained successfully!")

                    # 3. Make Predictions
                    with st.spinner("Generating 30-day price predictions..."):
                        future_predictions = predict_future_prices(
                            model,
                            raw_training_data_for_pred, # Pass the raw data, scaler will handle transform
                            scaler,
                            look_back=look_back,
                            future_days=30
                        )

                    # 4. Visualize Results
                    st.subheader("Price Prediction for the Next 30 Days")

                    # Generate future dates
                    last_historical_date = df_historical.index[-1]
                    future_dates = [last_historical_date + timedelta(days=i) for i in range(1, 31)]
                    predicted_df = pd.DataFrame({'Date': future_dates, 'Predicted Price (USD)': future_predictions})
                    predicted_df.set_index('Date', inplace=True)

                    st.write(predicted_df.head())

                    # Combine for plotting
                    combined_df = pd.concat([df_historical['Price'], predicted_df['Predicted Price (USD)']]).rename('Price')

                    fig, ax = plt.subplots(figsize=(14, 7))
                    ax.plot(df_historical.index, df_historical['Price'], label=f'{selected_coin_name} Historical Price', color='blue')
                    ax.plot(predicted_df.index, predicted_df['Predicted Price (USD)'], label=f'{selected_coin_name} Predicted Price', color='red', linestyle='--')
                    ax.axvline(last_historical_date, color='grey', linestyle=':', label='Prediction Start')

                    ax.set_title(f'{selected_coin_name} Price: Historical vs. 30-Day Prediction', fontsize=16)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Price (USD)', fontsize=12)
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig)

                    st.warning("Disclaimer: Cryptocurrency prices are highly volatile. This model is for educational and experimental purposes only and should not be used for financial advice.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Your Name/Company")
