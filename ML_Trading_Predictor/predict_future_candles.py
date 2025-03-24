#!/usr/bin/env python3
"""
Future Candle Predictor

This script predicts the next 15 candles (price movements) based on historical data
using machine learning models. It's designed to work with 15-minute timeframe data
and can be used as a standalone tool without any frontend dependencies.

Usage:
    python predict_future_candles.py --ticker AAPL --timeframe 15m --num_candles 15 --model_type rf

Features:
    - Predicts up to 15 candles into the future
    - Supports multiple model types (Random Forest, Gradient Boosting, etc.)
    - Works with 15-minute timeframe data
    - Visualizes predictions with matplotlib
    - Saves results to CSV for further analysis
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import json
import logging
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import the simple_ml_model module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import simple_ml_model as model

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict future candles using ML')
    
    # Required arguments
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    
    # Optional arguments with defaults
    parser.add_argument('--timeframe', type=str, default='15m',
                        choices=['1m', '5m', '15m', '30m', '60m', '1h', '1d'],
                        help='Data timeframe (default: 15m)')
    parser.add_argument('--num_candles', type=int, default=15,
                        help='Number of future candles to predict (default: 15)')
    parser.add_argument('--model_type', type=str, default='rf',
                        choices=['rf', 'gb', 'svm', 'nn', 'ensemble'],
                        help='Model type (default: rf)')
    parser.add_argument('--date_range', type=str, default='3m',
                        help='Date range for historical data (e.g. 1d, 3m, 1y)')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save prediction plot to file')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save prediction data to CSV')
    
    return parser.parse_args()

def download_historical_data(ticker, timeframe, date_range='3m'):
    """
    Download historical data for a given ticker and timeframe.
    
    Args:
        ticker (str): The ticker symbol
        timeframe (str): The timeframe for the data (e.g., '15m', '1h', '1d')
        date_range (str): How much historical data to use (e.g., '1d', '5d', '1mo', '3mo', '1y')
        
    Returns:
        pd.DataFrame: The historical data
    """
    logging.info(f"Downloading historical data for {ticker}...")
    
    try:
        # Try to use real data loader first
        from real_data_loader import load_real_data
        try:
            # Attempt to load real market data
            logging.info(f"Attempting to load real market data for {ticker} ({timeframe})...")
            data = load_real_data(ticker, timeframe, date_range)
            logging.info(f"Successfully loaded real market data: {len(data)} rows")
            return data
        except Exception as e:
            logging.warning(f"Could not load real market data: {str(e)}")
            logging.warning("Falling back to sample data...")
            # Continue with fallback options
    except ImportError:
        logging.warning("Real data loader not available. Using sample data instead.")
        # Continue with fallback options
    
    # Calculate start and end dates for filtering
    end_date = datetime.now()
    
    if date_range.endswith('d'):
        days = int(date_range[:-1])
        start_date_str = (end_date - timedelta(days=days)).strftime('%Y-%m-%d')
    elif date_range.endswith('w'):
        weeks = int(date_range[:-1])
        start_date_str = (end_date - timedelta(weeks=weeks)).strftime('%Y-%m-%d')
    elif date_range.endswith('m'):
        months = int(date_range[:-1])
        start_date_str = (end_date - timedelta(days=30*months)).strftime('%Y-%m-%d')
    elif date_range.endswith('y'):
        years = int(date_range[:-1])
        start_date_str = (end_date - timedelta(days=365*years)).strftime('%Y-%m-%d')
    else:
        raise ValueError(f"Invalid date range: {date_range}")
        
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logging.info(f"Downloading data from {start_date_str} to {end_date_str}")
    
    # First check if we have yearly data for this ticker (15m timeframe)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yearly_data_path = os.path.join(script_dir, 'data', f'fixed_{ticker}_15m_1y.csv')
    
    if os.path.exists(yearly_data_path) and timeframe == '15m':
        logging.info(f"Loading data from {yearly_data_path}")
        
        # Load the data
        data = pd.read_csv(yearly_data_path)
        
        # Convert Datetime to datetime
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            
            # Handle timezone awareness
            if data['Datetime'].dt.tz is not None:
                # Convert start_date_str to timezone-aware datetime
                start_date = pd.to_datetime(start_date_str).tz_localize('UTC')
                
                # Filter the data
                data = data[data['Datetime'] >= start_date]
            else:
                # For timezone-naive data
                data = data[data['Datetime'] >= start_date_str]
                
            # Set Datetime as index
            data.set_index('Datetime', inplace=True)
            
        # Create MultiIndex DataFrame
        data = create_multi_index_df(data, ticker)
        
        logging.info(f"Loaded {len(data)} rows from yearly data file")
        return data
    
    # Otherwise try to load the sample data
    sample_path = os.path.join(script_dir, 'data', f'fixed_{ticker}_{timeframe}.csv')
    
    if os.path.exists(sample_path):
        logging.info(f"Loading sample data from {sample_path}")
        
        # Load the data
        data = pd.read_csv(sample_path)
        
        # Convert Datetime to datetime
        if 'Datetime' in data.columns:
            data['Datetime'] = pd.to_datetime(data['Datetime'])
            data.set_index('Datetime', inplace=True)
            
        # Create MultiIndex DataFrame
        data = create_multi_index_df(data, ticker)
        
        logging.info(f"Loaded {len(data)} rows from sample data file")
        
        # Warning about using sample data
        logging.warning("Using sample data - predictions may not reflect current market conditions")
        return data
    
    # If we get here, we need to download the data (not implemented yet)
    logging.error(f"No data available for {ticker} with timeframe {timeframe}")
    raise ValueError(f"No data available for {ticker} with timeframe {timeframe}. Please provide real market data files.")

def preprocess_data(data):
    """Preprocess data by calculating technical features"""
    logging.info("Calculating technical features...")
    
    try:
        # Use the calculate_features function from simple_ml_model
        data_with_features = model.calculate_features(data)
        
        # Handle NaN values
        data_clean = data_with_features.dropna()
        
        if data_clean.empty:
            logging.error("No valid data points after feature calculation")
            sys.exit(1)
        
        logging.info(f"Preprocessed data: {len(data_clean)} valid data points")
        return data_clean
        
    except Exception as e:
        # If the standard feature calculation fails, try a more flexible approach
        logging.warning(f"Standard feature calculation failed: {str(e)}")
        logging.info("Attempting flexible data preprocessing...")
        
        # Check if data is already in proper format
        is_multi_index = isinstance(data.columns, pd.MultiIndex)
        
        # For real market data, we need to ensure proper column structure
        if is_multi_index:
            # Extract ticker from MultiIndex
            tickers = set()
            for col in data.columns:
                if isinstance(col, tuple) and len(col) > 0:
                    tickers.add(col[0])
            
            if len(tickers) == 1:
                ticker = list(tickers)[0]
                logging.info(f"Found ticker in MultiIndex: {ticker}")
                
                # Make a copy of the data for processing
                df = data.copy()
                
                # Create a mapping for standard column names
                price_cols = {}
                for col in df.columns:
                    if isinstance(col, tuple) and len(col) > 1:
                        col_name = str(col[1]).lower()
                        if 'open' in col_name:
                            price_cols['Open'] = col
                        elif 'high' in col_name:
                            price_cols['High'] = col
                        elif 'low' in col_name:
                            price_cols['Low'] = col
                        elif 'close' in col_name:
                            price_cols['Close'] = col
                        elif 'volume' in col_name:
                            price_cols['Volume'] = col
                
                logging.info(f"Identified price columns: {price_cols}")
                
                if 'Close' not in price_cols:
                    logging.error("Could not find Close price column")
                    raise ValueError("Missing required Close price column")
                
                # Calculate basic technical indicators
                try:
                    # Create a new DataFrame with indicators
                    result = pd.DataFrame(index=df.index)
                    
                    # Copy original price data
                    for name, col in price_cols.items():
                        result[(ticker, name)] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Calculate SMA indicators
                    for period in [5, 10, 20, 50, 200]:
                        if 'Close' in price_cols:
                            close_prices = pd.to_numeric(df[price_cols['Close']], errors='coerce')
                            result[(ticker, f'SMA_{period}')] = close_prices.rolling(window=period).mean()
                    
                    # Calculate EMA indicators
                    for period in [5, 10, 20, 50, 200]:
                        if 'Close' in price_cols:
                            close_prices = pd.to_numeric(df[price_cols['Close']], errors='coerce')
                            result[(ticker, f'EMA_{period}')] = close_prices.ewm(span=period, adjust=False).mean()
                    
                    # Calculate RSI (14-day)
                    if 'Close' in price_cols:
                        close_prices = pd.to_numeric(df[price_cols['Close']], errors='coerce')
                        delta = close_prices.diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                        rs = gain / loss
                        result[(ticker, 'RSI')] = 100 - (100 / (1 + rs))
                    
                    # Calculate MACD
                    if 'Close' in price_cols:
                        close_prices = pd.to_numeric(df[price_cols['Close']], errors='coerce')
                        ema12 = close_prices.ewm(span=12, adjust=False).mean()
                        ema26 = close_prices.ewm(span=26, adjust=False).mean()
                        result[(ticker, 'MACD')] = ema12 - ema26
                        result[(ticker, 'MACD_Signal')] = result[(ticker, 'MACD')].ewm(span=9, adjust=False).mean()
                    
                    # Calculate Bollinger Bands
                    if 'Close' in price_cols:
                        close_prices = pd.to_numeric(df[price_cols['Close']], errors='coerce')
                        sma20 = close_prices.rolling(window=20).mean()
                        std20 = close_prices.rolling(window=20).std()
                        result[(ticker, 'BB_Upper')] = sma20 + 2 * std20
                        result[(ticker, 'BB_Middle')] = sma20
                        result[(ticker, 'BB_Lower')] = sma20 - 2 * std20
                    
                    # Add price momentum
                    if 'Close' in price_cols:
                        close_prices = pd.to_numeric(df[price_cols['Close']], errors='coerce')
                        for period in [1, 3, 5, 10, 20]:
                            result[(ticker, f'Return_{period}d')] = close_prices.pct_change(periods=period)
                    
                    # Fill NaN values
                    result = result.bfill().ffill()
                    
                    logging.info(f"Successfully calculated {len(result.columns)} technical indicators")
                    
                    # Handle NaN values
                    data_clean = result.dropna()
                    if data_clean.empty:
                        logging.error("No valid data points after flexible feature calculation")
                        sys.exit(1)
                    
                    logging.info(f"Preprocessed data: {len(data_clean)} valid data points with {len(data_clean.columns)} features")
                    return data_clean
                    
                except Exception as e:
                    logging.error(f"Flexible feature calculation failed: {str(e)}")
                    raise
            else:
                logging.error(f"Found multiple tickers in MultiIndex: {tickers}")
                raise ValueError("Multiple tickers found in data")
        else:
            logging.error("Data does not have MultiIndex columns")
            raise ValueError("Data does not have the expected structure")
    
    return data

def create_multi_index_df(df, ticker):
    """Create a MultiIndex DataFrame with ticker as the top level"""
    logging.info(f"Converting DataFrame to MultiIndex with ticker {ticker}")
    
    # Make a copy of the DataFrame
    data = df.copy()
    
    # Check if the DataFrame already has a MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        logging.info(f"DataFrame already has MultiIndex columns")
        return data
    
    try:
        # Create new column names with ticker as the top level
        new_columns = pd.MultiIndex.from_product([[ticker], data.columns])
        
        # Create a new DataFrame with the MultiIndex columns
        multi_df = pd.DataFrame(data=data.values, index=data.index, columns=new_columns)
        
        # If Datetime column exists, set it as the index
        if ('Datetime', ticker) in multi_df.columns:
            multi_df = multi_df.set_index(('Datetime', ticker))
        
        logging.info("Successfully converted DataFrame to MultiIndex with ticker {}".format(ticker))
        return multi_df
    except Exception as e:
        logging.error(f"Error creating MultiIndex DataFrame: {str(e)}")
        
        # Try alternative method for real data
        try:
            # First, ensure all columns are the right type
            for col in data.columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Create MultiIndex with ticker first, then column name
            columns = pd.MultiIndex.from_product([[ticker], data.columns])
            multi_df = pd.DataFrame(data=data.values, index=data.index, columns=columns)
            
            logging.info("Successfully created MultiIndex using alternative method")
            return multi_df
        except Exception as e2:
            logging.error(f"Alternative method also failed: {str(e2)}")
            raise ValueError(f"Could not create MultiIndex DataFrame: {str(e)}")

def predict_future_candles(data, model_type='rf', num_candles=15):
    """Predict future candles based on historical data"""
    # Find close price column
    is_multi_index = isinstance(data.columns, pd.MultiIndex)
    close_col = None
    
    if is_multi_index:
        # Find close price in multi-index
        for col in data.columns:
            if 'close' in str(col).lower():
                close_col = col
                break
    else:
        # Standard column names
        if 'Close' in data.columns:
            close_col = 'Close'
        elif 'close' in data.columns:
            close_col = 'close'
    
    if close_col is None:
        logging.error("Could not find close price column")
        sys.exit(1)
    
    last_price = float(data[close_col].iloc[-1])
    logging.info(f"Last known price: {last_price}")
    
    # Try to use the model approach, but fall back to a simple trend-based prediction if it fails
    try:
        # Create model if not already saved
        model_path = f"results/{model_type}_model.joblib"
        if not os.path.exists(model_path):
            logging.info(f"Training a new {model_type} model...")
            clf = model.train_model(data, model_type, save_model=True)
            if clf is None or len(clf) < 2:
                raise ValueError("Model training failed")
            clf = clf[0]  # Extract actual model
        else:
            logging.info(f"Loading existing {model_type} model...")
            loaded_model = joblib.load(model_path)
            
            # Handle tuple model format (backward compatibility)
            if isinstance(loaded_model, tuple) and len(loaded_model) > 0:
                logging.info("Converting tuple model format to standard model")
                clf = loaded_model[0]  # Extract the actual model
            # Handle dict model format (backward compatibility)
            elif isinstance(loaded_model, dict) and 'model' in loaded_model:
                logging.info("Extracting model from dict format")
                clf = loaded_model['model']
            # Handle dict ensemble format (backward compatibility)
            elif isinstance(loaded_model, dict) and 'models' in loaded_model and 'meta_model' in loaded_model:
                logging.info("Using ensemble model")
                from sklearn.base import BaseEstimator, ClassifierMixin
                
                # Create wrapper class for old ensemble format
                class EnsembleWrapper(BaseEstimator, ClassifierMixin):
                    def __init__(self, base_models, meta_model):
                        self.models = base_models
                        self.meta_model = meta_model
                    
                    def predict(self, X):
                        proba = self.predict_proba(X)
                        return (proba[:, 1] >= 0.5).astype(int)
                    
                    def predict_proba(self, X):
                        if not self.models or isinstance(self.meta_model, list):
                            # Return neutral prediction
                            neutral = np.ones((len(X), 2)) * 0.5
                            neutral[:, 0] = 1 - neutral[:, 1]
                            return neutral
                            
                        try:
                            # Get predictions from base models
                            preds = {}
                            for name, model in self.models.items():
                                if model is None:
                                    preds[name] = np.ones(len(X)) * 0.5
                                else:
                                    try:
                                        preds[name] = model.predict_proba(X)[:, 1]
                                    except:
                                        preds[name] = np.ones(len(X)) * 0.5
                            
                            # Create dataframe for meta-model
                            meta_features = pd.DataFrame(preds)
                            
                            # Use meta-model if properly formed
                            if self.meta_model is not None and not isinstance(self.meta_model, list):
                                meta_preds = self.meta_model.predict_proba(meta_features)
                                return meta_preds
                            else:
                                # Use average of base model predictions
                                mean_prob = meta_features.mean(axis=1).values
                                return np.vstack((1 - mean_prob, mean_prob)).T
                        except Exception as e:
                            logging.error(f"Error in ensemble prediction: {str(e)}")
                            neutral = np.ones((len(X), 2)) * 0.5
                            neutral[:, 0] = 1 - neutral[:, 1]
                            return neutral
                
                # Create wrapper instance
                clf = EnsembleWrapper(loaded_model['models'], loaded_model['meta_model'])
            else:
                # Regular model object
                clf = loaded_model
                
        # Test if model can predict
        if not hasattr(clf, 'predict') or not callable(clf.predict):
            logging.error("Loaded model cannot predict, falling back to simple prediction")
            raise ValueError("Model has no predict method")
            
    except Exception as e:
        logging.warning(f"Could not train or load model: {str(e)}")
        logging.info("Falling back to trend-based prediction")
        clf = None  # Will use trend-based prediction later
    
    # Predictions list
    predictions = []
    
    # Define candle length in minutes based on timeframe
    timeframe_minutes = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '60m': 60,
        '1h': 60,
        '1d': 1440
    }
    
    # Get timeframe from data
    # For simplicity, let's use 15m as default if we can't determine
    candle_minutes = 15
    if hasattr(data.index, 'freq') and data.index.freq is not None:
        freq_str = str(data.index.freq)
        if 'min' in freq_str.lower():
            try:
                candle_minutes = int(freq_str.lower().split('min')[0])
            except:
                pass
    
    # Get the last timestamp
    if hasattr(data.index, 'is_all_dates') and data.index.is_all_dates:
        # For datetime index
        last_timestamp = pd.Timestamp(data.index[-1])
        if last_timestamp.tz is None:
            # Make timestamp timezone-aware if it's not
            last_timestamp = last_timestamp.tz_localize('UTC')
    else:
        # For non-datetime index, use current time
        last_timestamp = pd.Timestamp.now(tz='UTC')
    
    # Calculate price trend from recent data
    # Use last 20 candles to determine trend
    recent_prices = data[close_col].iloc[-20:].astype(float)
    price_changes = recent_prices.pct_change().dropna()
    avg_change = price_changes.mean()
    std_change = price_changes.std()
    
    # Determine market sentiment based on recent movement
    if avg_change > 0:
        sentiment = "bullish"
        confidence = min(0.5 + abs(avg_change) * 10, 0.95)  # Scale confidence with magnitude
    else:
        sentiment = "bearish"
        confidence = min(0.5 + abs(avg_change) * 10, 0.95)  # Scale confidence with magnitude
    
    logging.info(f"Trend analysis: {sentiment.upper()} with {confidence:.2f} confidence")
    
    # Predict each future candle
    current_price = last_price
    for i in range(1, num_candles + 1):
        # Calculate next timestamp
        next_timestamp = last_timestamp + pd.Timedelta(minutes=candle_minutes * i)
        
        # For simple model, use trend continuation with some randomness
        # The further into the future, the more uncertainty (increasing std)
        future_uncertainty = 1 + (i * 0.1)  # Increase uncertainty with time
        if sentiment == "bullish":
            change_pct = max(avg_change, 0) * (1 + np.random.normal(0, std_change * future_uncertainty))
        else:
            change_pct = min(avg_change, 0) * (1 + np.random.normal(0, std_change * future_uncertainty))
        
        # Reduce confidence as we go further into the future
        adjusted_confidence = max(0.1, confidence * (1 - 0.02 * i))
        
        # Calculate next price
        next_price = current_price * (1 + change_pct)
            
        # Add prediction to results
        pred_data = {
            'timestamp': next_timestamp,
            'predicted_price': next_price,
            'direction': 'up' if next_price > current_price else 'down',
            'confidence': adjusted_confidence,
            'candle_number': i
        }
        predictions.append(pred_data)
        
        # Update current price for next iteration
        current_price = next_price
    
    logging.info(f"Predicted {len(predictions)} future candles")
    return predictions

def visualize_predictions(data, predictions, ticker, timeframe, output_path=None):
    """Visualize historical data and future predictions"""
    logging.info("Generating prediction visualization...")
    
    # Find close price column
    is_multi_index = isinstance(data.columns, pd.MultiIndex)
    close_col = None
    
    if is_multi_index:
        # Find close price in multi-index
        for col in data.columns:
            if 'close' in str(col).lower():
                close_col = col
                break
    else:
        # Standard column names
        if 'Close' in data.columns:
            close_col = 'Close'
        elif 'close' in data.columns:
            close_col = 'close'
    
    if close_col is None:
        logging.error("Could not find close price column for visualization")
        return
    
    # Extract data for plotting
    try:
        # Get historical data
        historical_dates = data.index[-60:].tolist()  # Last 60 points for visualization
        historical_prices = pd.to_numeric(data[close_col][-60:], errors='coerce').values
        
        # Extract prediction data and convert timestamps to matplotlib-compatible dates
        pred_dates = []
        for p in predictions:
            # Convert to matplotlib date format
            if isinstance(p['timestamp'], (pd.Timestamp, datetime)):
                # Convert pandas Timestamp to python datetime if needed
                if isinstance(p['timestamp'], pd.Timestamp):
                    dt = p['timestamp'].to_pydatetime()
                else:
                    dt = p['timestamp']
                # Convert to matplotlib date
                pred_dates.append(dt)
            else:
                # If it's already a string or other format, try to parse it
                try:
                    dt = pd.to_datetime(p['timestamp'])
                    pred_dates.append(dt.to_pydatetime())
                except:
                    logging.error(f"Could not convert timestamp: {p['timestamp']}")
                    raise ValueError(f"Invalid timestamp format: {p['timestamp']}")
        
        pred_prices = [float(p['predicted_price']) for p in predictions]
        confidences = [float(p['confidence']) for p in predictions]
        directions = [p['direction'] for p in predictions]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Convert to matplotlib dates
        import matplotlib.dates as mdates
        
        # Plot historical data
        plt.plot(historical_dates, historical_prices, 'b-', label='Historical Price')
        
        # Plot predictions with color based on direction
        for i, (date, price, conf, direction) in enumerate(zip(pred_dates, pred_prices, confidences, directions)):
            color = 'g' if direction == 'up' else 'r'
            alpha = 0.3 + 0.7 * conf  # Higher confidence = more opaque
            
            if i > 0:
                prev_date = pred_dates[i-1]
                prev_price = pred_prices[i-1]
                plt.plot([prev_date, date], [prev_price, price], color=color, alpha=alpha, linewidth=2)
            else:
                # Convert last historical date
                if isinstance(historical_dates[-1], (str, int, float)):
                    last_hist_date = pd.to_datetime(historical_dates[-1]).to_pydatetime()
                else:
                    last_hist_date = historical_dates[-1]
                    
                # Connect to last historical point
                plt.plot([last_hist_date, date], [historical_prices[-1], price], color=color, alpha=alpha, linewidth=2)
            
            # Add a point marker
            plt.scatter(date, price, c=color, alpha=alpha, s=30)
        
        # Add labels and title
        plt.title(f"{ticker} - {timeframe} Price Prediction (Next {len(predictions)} Candles)")
        plt.xlabel("Date/Time")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis to show dates nicely
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()
        
        # Save or display
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to {output_path}")
        
        plt.close()
    except Exception as e:
        logging.error(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def save_predictions_csv(predictions, ticker, timeframe, output_path=None):
    """Save predictions to CSV file"""
    if not output_path:
        os.makedirs('results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"results/{ticker}_{timeframe}_predictions_{timestamp}.csv"
    
    # Convert to DataFrame
    df = pd.DataFrame(predictions)
    
    # Convert timestamp objects to strings if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].astype(str)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Create necessary directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Download and preprocess data
    data = download_historical_data(args.ticker, args.timeframe, args.date_range)
    data_processed = preprocess_data(data)
    
    # Predict future candles
    predictions = predict_future_candles(data_processed, args.model_type, args.num_candles)
    
    # Generate visualization
    if args.save_plot or True:  # Always generate plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = f"plots/{args.ticker}_{args.timeframe}_forecast_{timestamp}.png"
        visualize_predictions(data_processed, predictions, args.ticker, args.timeframe, plot_path)
    
    # Save to CSV if requested
    if args.save_csv:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"results/{args.ticker}_{args.timeframe}_predictions_{timestamp}.csv"
        save_predictions_csv(predictions, args.ticker, args.timeframe, csv_path)
    
    # Print summary
    print("\n" + "="*50)
    print(f"PREDICTION SUMMARY FOR {args.ticker} ({args.timeframe})")
    print("="*50)
    
    # Current price
    is_multi_index = isinstance(data_processed.columns, pd.MultiIndex)
    close_col = None
    
    if is_multi_index:
        for col in data_processed.columns:
            if 'close' in str(col).lower():
                close_col = col
                break
    else:
        close_col = 'Close' if 'Close' in data_processed.columns else 'close'
    
    current_price = float(data_processed[close_col].iloc[-1])
    print(f"Current price: {current_price:.2f}")
    
    # Print predictions
    print("\nPredicted Future Candles:")
    print(f"{'#':3} | {'Time':16} | {'Price':10} | {'Change %':8} | {'Direction':8} | {'Confidence':10}")
    print("-"*65)
    
    for p in predictions:
        time_str = p['timestamp'].strftime('%Y-%m-%d %H:%M')
        price = p['predicted_price']
        pct_change = ((price / current_price) - 1) * 100
        direction = "↑" if p['direction'] == 'up' else "↓"
        confidence = f"{p['confidence']:.2%}"
        
        print(f"{p['candle_number']:3} | {time_str:16} | {price:10.2f} | {pct_change:8.2f}% | {direction:8} | {confidence:10}")
    
    # Final price and overall change
    final_price = predictions[-1]['predicted_price']
    overall_change = ((final_price / current_price) - 1) * 100
    overall_direction = "UP" if final_price > current_price else "DOWN"
    
    print("\n" + "-"*65)
    print(f"Overall prediction: {overall_direction} by {abs(overall_change):.2f}%")
    print(f"Final price after {args.num_candles} candles: {final_price:.2f}")
    
    if args.save_plot:
        print(f"\nVisualization saved to: {plot_path}")
    if args.save_csv:
        print(f"Predictions saved to: {csv_path}")
    
    print("="*50)

if __name__ == "__main__":
    main() 