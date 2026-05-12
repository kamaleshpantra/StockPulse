import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLLogic:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.scalers = {}

    def build_model(self, input_shape):
        """Define the LSTM model architecture."""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def prepare_sequences(self, df, sequence_length=60):
        """Preprocess data and create sequences for LSTM."""
        features = df[['Close', 'avg_sentiment', 'post_count']].values
        target = df['trend'].values

        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(scaled_features) - sequence_length):
            X.append(scaled_features[i:i + sequence_length])
            y.append(target[i + sequence_length])

        return np.array(X), np.array(y), scaler

    def train_and_save(self, ticker, df, epochs=50, batch_size=32):
        """Train model for a specific ticker and save it."""
        X, y, scaler = self.prepare_sequences(df)
        if len(X) == 0:
            logger.error(f"Not enough data to train model for {ticker}")
            return None

        # Split data (80/20)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = self.build_model((X.shape[1], X.shape[2]))
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)

        # Save model and scaler
        model_path = os.path.join(self.model_dir, f"{ticker}_model.h5")
        model.save(model_path)
        
        # In a real scenario, we'd save the scaler using joblib or pickle
        self.scalers[ticker] = scaler
        
        logger.info(f"Model saved for {ticker} at {model_path}")
        return model

    def predict_trend(self, ticker, df, sequence_length=60):
        """Predict the trend for the next day."""
        model_path = os.path.join(self.model_dir, f"{ticker}_model.h5")
        if not os.path.exists(model_path):
            logger.error(f"No model found for {ticker}")
            return None

        model = load_model(model_path)
        
        # Preprocess features
        features = df[['Close', 'avg_sentiment', 'post_count']].values
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features) # Note: Ideally use saved scaler

        if len(scaled_features) < sequence_length:
            return None

        latest_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, 3)
        prediction_prob = model.predict(latest_sequence, verbose=0)[0][0]
        
        return {
            "probability": float(prediction_prob),
            "trend": "UP" if prediction_prob >= 0.5 else "DOWN"
        }
