import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from num1 import preprocess_data, create_sequences_for_prediction
import pandas as pd
from datetime import timedelta

def build_and_train_lstm_model(data, look_back=60, epochs=50, batch_size=32):
    if len(data) < look_back + 1:
        print(f"Insufficient data to train LSTM with look_back={look_back}. Need at least {look_back + 1} data points.")
        return None, None, None

    training_data = data['Price'].values.reshape(-1, 1)
    X_train, y_train, scaler = preprocess_data(training_data, look_back)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    print(f"Starting LSTM model training for {epochs} epochs...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    print("LSTM model training completed.")

    return model, scaler, training_data


def predict_future_prices(model, initial_data_for_prediction, scaler, look_back, future_days=30):
    scaled_initial_input = scaler.transform(initial_data_for_prediction[-look_back:])
    current_batch = scaled_initial_input.reshape(1, look_back, 1)

    predicted_prices = []

    for _ in range(future_days):
        predicted_scaled_price = model.predict(current_batch, verbose=0)[0]
        predicted_prices.append(scaler.inverse_transform(predicted_scaled_price.reshape(-1, 1))[0][0])

        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1, 0] = predicted_scaled_price

    return predicted_prices


if __name__ == '__main__':
    from data_fetcher import fetch_historical_data
    import matplotlib.pyplot as plt

    coin_id = 'bitcoin'
    days_to_fetch = 365 * 2
    df = fetch_historical_data(coin_id, days=days_to_fetch)

    if df is not None:
        model, scaler, raw_training_data_for_pred = build_and_train_lstm_model(df, epochs=10, look_back=60)

        if model:
            future_predictions = predict_future_prices(model, raw_training_data_for_pred, scaler, look_back=60, future_days=30)
            print("\nFuture 30-day predictions:")
            print(future_predictions[:5], "...")

            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Price'], label='Historical Prices')
            last_date = df.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
            plt.plot(future_dates, future_predictions, label='Predicted Prices', linestyle='--')
            plt.title(f"{coin_id.capitalize()} Price Prediction")
            plt.xlabel("Date")
            plt.ylabel("Price (USD)")
            plt.legend()
            plt.grid(True)
            plt.show()
