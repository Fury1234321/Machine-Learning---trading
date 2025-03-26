"""
Data processing utilities for ML 2.0 Trading Predictor
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List, Optional
import sys
import os

# Add the project root to path to allow imports from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LOOKBACK_WINDOW, PREDICTION_HORIZON, PRICE_FEATURES, TECHNICAL_INDICATORS

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load trading data from CSV file
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names to uppercase first letter (e.g., 'close' -> 'Close')
        df.columns = [col.lower() for col in df.columns]  # First convert all to lowercase
        
        # Map common date column names to standard 'date'
        date_columns = ['date', 'datetime', 'time', 'timestamp']
        for col in df.columns:
            if col in date_columns:
                df.rename(columns={col: 'date'}, inplace=True)
                break
        
        # Map price column names to standard format
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert date column to datetime if it exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
        print(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the dataframe
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Simple Moving Average
    result['sma_20'] = result['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average
    result['ema_20'] = result['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = result['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    result['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = result['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = result['Close'].ewm(span=26, adjust=False).mean()
    result['macd'] = ema_12 - ema_26
    
    # Bollinger Bands
    sma_20 = result['Close'].rolling(window=20).mean()
    std_20 = result['Close'].rolling(window=20).std()
    result['bollinger_upper'] = sma_20 + (std_20 * 2)
    result['bollinger_lower'] = sma_20 - (std_20 * 2)
    
    # Average True Range (ATR)
    high_low = result['High'] - result['Low']
    high_close = np.abs(result['High'] - result['Close'].shift())
    low_close = np.abs(result['Low'] - result['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    result['atr_14'] = true_range.rolling(14).mean()
    
    # On-Balance Volume (OBV)
    result['obv'] = np.where(result['Close'] > result['Close'].shift(), 
                             result['Volume'], 
                             np.where(result['Close'] < result['Close'].shift(), 
                                     -result['Volume'], 0)).cumsum()
    
    # Drop NaN values that result from the rolling windows
    result.dropna(inplace=True)
    
    return result

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable - price movement after PREDICTION_HORIZON candles
    
    Args:
        df: DataFrame with price data
        
    Returns:
        DataFrame with target variable
    """
    # Future price after PREDICTION_HORIZON periods
    future_close = df['Close'].shift(-PREDICTION_HORIZON)
    
    # Target is the price movement (1 if price goes up, 0 if it goes down)
    df['target'] = (future_close > df['Close']).astype(int)
    
    # Drop rows where target is NaN (the last PREDICTION_HORIZON rows)
    df.dropna(subset=['target'], inplace=True)
    
    return df

def prepare_data(df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], Dict[str, MinMaxScaler]]:
    """
    Prepare data for model training and testing
    
    Args:
        df: DataFrame with price data and features
        
    Returns:
        Tuple of (data_dict, scalers_dict)
        data_dict contains X_train, X_test, y_train, y_test
        scalers_dict contains the fitted scalers for later use
    """
    # Create features and target
    df = add_technical_indicators(df)
    df = create_target(df)
    
    # Get the features and target
    features = PRICE_FEATURES + TECHNICAL_INDICATORS
    
    # Check that all features exist in the dataframe
    missing_features = [feature for feature in features if feature not in df.columns]
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Use only available features
        features = [feature for feature in features if feature in df.columns]
        
    X = df[features].values
    y = df['target'].values
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences for time series
    X_seq, y_seq = create_sequences(X_scaled, y)
    
    # Split into train and test sets (time-based split)
    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    data_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': features
    }
    
    scalers_dict = {
        'feature_scaler': scaler
    }
    
    return data_dict, scalers_dict

def create_sequences(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series prediction
    
    Args:
        X: Feature array
        y: Target array
        
    Returns:
        Tuple of (X_seq, y_seq) where each element in X_seq is a sequence of LOOKBACK_WINDOW elements
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - LOOKBACK_WINDOW):
        X_seq.append(X[i:i + LOOKBACK_WINDOW])
        y_seq.append(y[i + LOOKBACK_WINDOW])
    
    return np.array(X_seq), np.array(y_seq)

def load_and_prepare_data(file_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, MinMaxScaler]]:
    """
    Load data from file and prepare it for model training
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (data_dict, scalers_dict)
    """
    df = load_data(file_path)
    if df.empty:
        return {}, {}
    
    try:
        return prepare_data(df)
    except Exception as e:
        print(f"An error occurred during data preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}, {} 