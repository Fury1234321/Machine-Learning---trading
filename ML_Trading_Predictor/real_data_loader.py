#!/usr/bin/env python3
"""
Real Stock Data Loader

This module provides functionality to load, validate, and preprocess real stock data.
It replaces the synthetic data generation approach with real market data processing.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz
from typing import Optional, List, Dict, Union, Tuple
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class RealDataLoader:
    """Class for loading and processing real stock data."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing real stock data files
        """
        # Get script directory if data_dir not provided
        if data_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(script_dir, 'data')
        else:
            self.data_dir = data_dir
            
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # List of available data files
        self.available_files = self._scan_data_files()
        
    def _scan_data_files(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Scan the data directory for available stock data files.
        
        Returns:
            Dictionary mapping tickers to timeframes to file paths
        """
        result = {}
        
        # Look for CSV files in data directory
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        
        for file_path in csv_files:
            filename = os.path.basename(file_path)
            
            # Skip synthetic files
            if "generated" in filename.lower():
                continue
                
            # Try to extract ticker and timeframe from filename
            parts = filename.replace(".csv", "").split("_")
            
            if len(parts) >= 2:
                ticker = parts[0]
                timeframe = parts[1]
                
                # Initialize nested dictionary if needed
                if ticker not in result:
                    result[ticker] = {}
                if timeframe not in result[ticker]:
                    result[ticker][timeframe] = []
                    
                result[ticker][timeframe].append(file_path)
            else:
                logging.warning(f"Could not parse ticker and timeframe from filename: {filename}")
        
        return result
    
    def list_available_data(self) -> Dict[str, List[str]]:
        """
        Get a list of available tickers and timeframes.
        
        Returns:
            Dictionary mapping tickers to available timeframes
        """
        result = {}
        for ticker, timeframes in self.available_files.items():
            result[ticker] = list(timeframes.keys())
        return result
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate the stock data for common issues.
        
        Args:
            df: DataFrame containing stock data
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for required columns
        required_columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                return False, f"Missing required column: {col}"
        
        # Check for empty dataframe
        if len(df) == 0:
            return False, "DataFrame is empty"
        
        # Check for missing values in critical columns
        critical_cols = ['Open', 'High', 'Low', 'Close']
        missing_values = df[critical_cols].isnull().sum().sum()
        if missing_values > 0:
            missing_pct = (missing_values / (len(df) * len(critical_cols))) * 100
            if missing_pct > 5:  # More than 5% missing values is problematic
                return False, f"Too many missing values in price data: {missing_pct:.2f}%"
            
        # Check for sequential timestamps
        if not pd.api.types.is_datetime64_dtype(df['Datetime']):
            try:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
            except:
                return False, "Could not convert 'Datetime' column to datetime"
                
        # Sort and check for gaps
        df_sorted = df.sort_values('Datetime')
        time_diffs = df_sorted['Datetime'].diff().dropna()
        
        # Check if timestamps are unique
        if len(df_sorted) != len(df_sorted['Datetime'].unique()):
            return False, "Duplicate timestamps detected"
            
        # Validate price data
        if (df['Low'] > df['High']).any():
            return False, "Found instances where Low price is greater than High price"
            
        if (df['Open'] > df['High']).any() or (df['Close'] > df['High']).any():
            return False, "Found instances where Open or Close is greater than High price"
            
        if (df['Open'] < df['Low']).any() or (df['Close'] < df['Low']).any():
            return False, "Found instances where Open or Close is less than Low price"
            
        # Check for outliers in price data
        price_mean = df['Close'].mean()
        price_std = df['Close'].std()
        upper_limit = price_mean + 10 * price_std
        lower_limit = price_mean - 10 * price_std
        
        outliers = ((df['Close'] > upper_limit) | (df['Close'] < lower_limit)).sum()
        if outliers > 0:
            logging.warning(f"Detected {outliers} potential price outliers")
            
        # Check for zero or negative prices
        if (df['Close'] <= 0).any():
            return False, "Found non-positive price values"
            
        # Check for zero or negative volume
        if (df['Volume'] < 0).any():
            return False, "Found negative volume values"
            
        return True, "Data validation passed"
    
    def load_data(self, ticker: str, timeframe: str, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load stock data from file.
        
        Args:
            ticker: Stock ticker symbol
            timeframe: Data timeframe (e.g., '15m', '1h', '1d')
            file_path: Optional specific file path to load
            
        Returns:
            DataFrame containing stock data
        """
        # Determine file path if not provided
        if file_path is None:
            if ticker in self.available_files and timeframe in self.available_files[ticker]:
                # Use the first available file for this ticker and timeframe
                file_path = self.available_files[ticker][timeframe][0]
            else:
                raise ValueError(f"No data file found for {ticker} with timeframe {timeframe}")
        
        # Load data
        logging.info(f"Loading data for {ticker} ({timeframe}) from {file_path}")
        df = pd.read_csv(file_path)
        
        # Validate data
        is_valid, message = self.validate_data(df)
        if not is_valid:
            logging.error(f"Data validation failed: {message}")
            raise ValueError(f"Invalid data: {message}")
        else:
            logging.info(f"Data validation passed for {ticker} ({timeframe})")
        
        # Convert Datetime to datetime object if it's a string
        if isinstance(df['Datetime'].iloc[0], str):
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        
        # Set Datetime as index
        df.set_index('Datetime', inplace=True)
        
        # Add timezone if not present
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Sort by datetime
        df = df.sort_index()
        
        # Add ticker column if needed
        if 'ticker' not in df.columns:
            df['ticker'] = ticker
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess stock data by adding technical indicators.
        
        Args:
            df: DataFrame containing stock data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Copy dataframe
        processed_df = df.copy()
        
        # Add basic technical indicators
        # 1. Simple Moving Averages
        for window in [5, 10, 20, 50, 200]:
            processed_df[f'SMA_{window}'] = processed_df['Close'].rolling(window=window).mean()
        
        # 2. Exponential Moving Averages
        for window in [5, 10, 20, 50, 200]:
            processed_df[f'EMA_{window}'] = processed_df['Close'].ewm(span=window, adjust=False).mean()
        
        # 3. RSI (Relative Strength Index)
        delta = processed_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        processed_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (Moving Average Convergence Divergence)
        ema12 = processed_df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = processed_df['Close'].ewm(span=26, adjust=False).mean()
        processed_df['MACD'] = ema12 - ema26
        processed_df['MACD_Signal'] = processed_df['MACD'].ewm(span=9, adjust=False).mean()
        processed_df['MACD_Hist'] = processed_df['MACD'] - processed_df['MACD_Signal']
        
        # 5. Bollinger Bands
        processed_df['BB_Middle'] = processed_df['Close'].rolling(window=20).mean()
        std = processed_df['Close'].rolling(window=20).std()
        processed_df['BB_Upper'] = processed_df['BB_Middle'] + 2 * std
        processed_df['BB_Lower'] = processed_df['BB_Middle'] - 2 * std
        processed_df['BB_Width'] = (processed_df['BB_Upper'] - processed_df['BB_Lower']) / processed_df['BB_Middle']
        
        # 6. ATR (Average True Range)
        high_low = processed_df['High'] - processed_df['Low']
        high_close = (processed_df['High'] - processed_df['Close'].shift()).abs()
        low_close = (processed_df['Low'] - processed_df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        processed_df['ATR'] = true_range.rolling(14).mean()
        
        # 7. Volume indicators
        processed_df['Volume_ROC'] = processed_df['Volume'].pct_change(5)
        processed_df['Volume_SMA_20'] = processed_df['Volume'].rolling(window=20).mean()
        processed_df['Volume_SMA_ratio'] = processed_df['Volume'] / processed_df['Volume_SMA_20']
        
        # 8. Price Rate of Change
        for period in [1, 5, 10, 20]:
            processed_df[f'Price_ROC_{period}'] = processed_df['Close'].pct_change(period) * 100
        
        # 9. Commodity Channel Index (CCI)
        typical_price = (processed_df['High'] + processed_df['Low'] + processed_df['Close']) / 3
        # Mean Absolute Deviation
        def mean_abs_dev(x):
            return np.abs(x - x.mean()).mean()
        
        tp_sma = typical_price.rolling(window=20).mean()
        meandev = typical_price.rolling(window=20).apply(mean_abs_dev, raw=True)
        processed_df['CCI'] = (typical_price - tp_sma) / (0.015 * meandev)
        
        # 10. Stochastic Oscillator
        low_14 = processed_df['Low'].rolling(window=14).min()
        high_14 = processed_df['High'].rolling(window=14).max()
        processed_df['%K'] = ((processed_df['Close'] - low_14) / (high_14 - low_14)) * 100
        processed_df['%D'] = processed_df['%K'].rolling(window=3).mean()
        
        # Fill NaN values (resulting from rolling windows)
        processed_df = processed_df.bfill().ffill()
        
        return processed_df
    
    def create_multi_index_df(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Create a MultiIndex DataFrame for compatibility with ML models.
        
        Args:
            df: DataFrame containing stock data
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with MultiIndex columns
        """
        # Check if already has MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            return df
            
        # Create new column names with MultiIndex
        new_columns = pd.MultiIndex.from_product([[ticker], df.columns])
        
        # Convert to MultiIndex DataFrame
        df_multi = pd.DataFrame(df.values, index=df.index, columns=new_columns)
        
        return df_multi

def load_real_data(ticker: str, timeframe: str, date_range: str = '3m') -> pd.DataFrame:
    """
    Load and prepare real stock data for the ML model.
    
    Args:
        ticker: Stock ticker symbol
        timeframe: Data timeframe (e.g., '15m', '1h', '1d')
        date_range: How much historical data to use
        
    Returns:
        Processed DataFrame ready for ML model
    """
    # Create data loader
    loader = RealDataLoader()
    
    # Load data
    try:
        data = loader.load_data(ticker, timeframe)
    except ValueError as e:
        logging.error(f"Error loading data: {str(e)}")
        raise
    
    # Filter data based on date range
    end_date = data.index.max()
    
    if date_range.endswith('d'):
        days = int(date_range[:-1])
        start_date = end_date - timedelta(days=days)
    elif date_range.endswith('w'):
        weeks = int(date_range[:-1])
        start_date = end_date - timedelta(weeks=weeks)
    elif date_range.endswith('m'):
        months = int(date_range[:-1])
        start_date = end_date - timedelta(days=months * 30)
    elif date_range.endswith('y'):
        years = int(date_range[:-1])
        start_date = end_date - timedelta(days=years * 365)
    else:
        raise ValueError(f"Invalid date range format: {date_range}")
    
    # Filter data
    data = data[data.index >= start_date]
    
    # Preprocess data
    processed_data = loader.preprocess_data(data)
    
    # Create MultiIndex DataFrame
    multi_index_data = loader.create_multi_index_df(processed_data, ticker)
    
    return multi_index_data

if __name__ == "__main__":
    # Test the data loader
    loader = RealDataLoader()
    
    # Print available data
    available_data = loader.list_available_data()
    print("Available data:")
    for ticker, timeframes in available_data.items():
        print(f"  {ticker}: {', '.join(timeframes)}")
    
    # Load sample data if available
    for ticker in available_data:
        for timeframe in available_data[ticker]:
            try:
                data = loader.load_data(ticker, timeframe)
                processed_data = loader.preprocess_data(data)
                print(f"Loaded {ticker} ({timeframe}): {len(data)} rows, {len(processed_data.columns)} features")
            except Exception as e:
                print(f"Error loading {ticker} ({timeframe}): {str(e)}") 