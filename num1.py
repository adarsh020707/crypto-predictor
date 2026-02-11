# utils.py
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data, look_back=60):
    """
    Scales data and creates sequences for LSTM model.

    Args:
        data (np.array): The numerical data (e.g., 'Price' column).
        look_back (int): Number of previous time steps to use as input features.

    Returns:
        tuple: (X, y, scaler) - X (features), y (target), MinMaxScaler object.
    """
    # Reshape data for scaling (expecting 2D array)
    data = data.reshape(-1, 1)

    # Scale the data to be between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences for LSTM
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler

def create_sequences_for_prediction(data, look_back, scaler):
    """
    Prepares the last 'look_back' data points for making future predictions.

    Args:
        data (np.array): The historical price data used for training (already scaled).
        look_back (int): Number of previous time steps the model was trained on.
        scaler (MinMaxScaler): The scaler used for the training data.

    Returns:
        np.array: Reshaped input for the LSTM model.
    """
    last_look_back_data = data[-look_back:]
    # Reshape for prediction (1 sample, look_back features, 1 timestep)
    return last_look_back_data.reshape(1, look_back, 1)

if __name__ == '__main__':
    # Example usage:
    sample_data = np.array([i for i in range(100)]) # Dummy data
    X_train, y_train, scaler = preprocess_data(sample_data.reshape(-1, 1), look_back=10)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # Simulate data for next prediction
    last_data_point_for_pred = sample_data[-10:].reshape(-1, 1) # last 10 points
    scaled_last_data = scaler.transform(last_data_point_for_pred)
    prediction_input = create_sequences_for_prediction(scaled_last_data, look_back=10, scaler=scaler)
    print("Prediction Input Shape:", prediction_input.shape)