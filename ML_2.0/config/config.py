"""
Configuration settings for ML 2.0 Trading Predictor
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Data processing settings
PREDICTION_HORIZON = 10  # Number of candles to predict ahead (10 x 15min = 150min)
LOOKBACK_WINDOW = 60  # Number of past candles to use for prediction
TEST_SIZE = 0.2  # Percentage of data to use for testing
VALIDATION_SIZE = 0.1  # Percentage of training data to use for validation
RANDOM_SEED = 42  # For reproducibility

# Features to use from raw data
PRICE_FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
TECHNICAL_INDICATORS = [
    'sma_20', 'ema_20', 'rsi_14', 'macd', 'bollinger_upper', 
    'bollinger_lower', 'atr_14', 'obv'
]

# Neural Network hyperparameters
NN_CONFIG = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 10,  # Early stopping patience
    'hidden_layers': [128, 64, 32],
    'dropout_rate': 0.2,
    'activation': 'relu',
    'optimizer': 'adam',
    'loss': 'mean_squared_error'
}

# Gradient Boosting hyperparameters
GB_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'subsample': 0.8,
    'random_state': RANDOM_SEED
}

# Decision Tree hyperparameters
DT_CONFIG = {
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_SEED
}

# UI settings
TERMINAL_WIDTH = 80  # For formatting terminal output
SHOW_PROGRESS_BAR = True
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL 