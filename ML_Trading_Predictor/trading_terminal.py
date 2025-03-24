#!/usr/bin/env python3
"""
Trading Terminal - Enhanced User Experience

A terminal-based user interface for the ML Trading Predictor with improved
visuals, progress bars, and a clear workflow for training and using ML models.

Features:
- Welcome screen with system status
- Progress bars for data loading and model training
- Stock data visualization in terminal
- Model training and prediction with clear visual feedback
- Trading action recommendations based on predictions

Usage:
    python trading_terminal.py
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import logging
import json
import tqdm
import colorama
from colorama import Fore, Back, Style
from tabulate import tabulate
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Initialize colorama
colorama.init(autoreset=True)

# Set up logging with custom formatter to add colors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Import the ML model modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import simple_ml_model as model
import predict_future_candles
try:
    from real_data_loader import RealDataLoader
    HAS_REAL_DATA_LOADER = True
except ImportError:
    HAS_REAL_DATA_LOADER = False

# Constants
MODELS = {
    'rf': 'Random Forest',
    'gb': 'Gradient Boosting',
    'svm': 'Support Vector Machine',
    'nn': 'Neural Network',
    'ensemble': 'Ensemble Model'
}

TIMEFRAMES = ['15m', '1h', '1d', '1wk']
DEFAULT_TICKER = 'AAPL'
DEFAULT_TIMEFRAME = '15m'
MODEL_DIR = 'results'
DATA_DIR = 'data'

def print_header():
    """Print a fancy header for the terminal."""
    terminal_width = os.get_terminal_size().columns
    
    print("\n" + "=" * terminal_width)
    print(Fore.CYAN + Style.BRIGHT + "🚀 ML TRADING PREDICTOR TERMINAL 🚀".center(terminal_width))
    print("=" * terminal_width + "\n")

def print_footer():
    """Print a footer for the terminal."""
    terminal_width = os.get_terminal_size().columns
    print("\n" + "=" * terminal_width)
    print(Fore.CYAN + "© 2025 ML Trading Predictor".center(terminal_width))
    print("=" * terminal_width + "\n")

def print_section(title):
    """Print a section header."""
    terminal_width = os.get_terminal_size().columns
    print("\n" + "-" * terminal_width)
    print(Fore.GREEN + Style.BRIGHT + f"📊 {title} 📊".center(terminal_width))
    print("-" * terminal_width + "\n")

def check_system_status():
    """Check system status and display it to the user."""
    print_section("SYSTEM STATUS")
    
    # Check if data directory exists
    data_status = Fore.GREEN + "AVAILABLE" if os.path.exists(DATA_DIR) else Fore.RED + "MISSING"
    
    # Check if models exist
    models_info = []
    for model_type in MODELS.keys():
        model_path = os.path.join(MODEL_DIR, f"{model_type}_model.joblib")
        if os.path.exists(model_path):
            # Get model creation time
            timestamp = datetime.fromtimestamp(os.path.getmtime(model_path))
            age = (datetime.now() - timestamp).days
            
            # Add trained data points info if available
            model_meta_path = os.path.join(MODEL_DIR, f"{model_type}_model_meta.json")
            if os.path.exists(model_meta_path):
                try:
                    with open(model_meta_path, 'r') as f:
                        meta = json.load(f)
                    data_points = meta.get('data_points', 'unknown')
                    training_time = meta.get('training_time', 'unknown')
                    status = f"{Fore.GREEN}TRAINED{Style.RESET_ALL} ({age} days ago, {data_points:,} points, {training_time:.2f}s)"
                except:
                    status = f"{Fore.GREEN}TRAINED{Style.RESET_ALL} ({age} days ago)"
            else:
                status = f"{Fore.GREEN}TRAINED{Style.RESET_ALL} ({age} days ago)"
        else:
            status = f"{Fore.RED}UNTRAINED{Style.RESET_ALL}"
        models_info.append([MODELS[model_type], status])
    
    # Check data files
    data_files = []
    yearly_data_exists = False
    real_data_exists = False
    synthetic_data_exists = False
    
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".csv"):
                file_path = os.path.join(DATA_DIR, file)
                size_mb = os.path.getsize(file_path) / 1048576  # Convert to MB
                
                # Count rows in data file
                try:
                    row_count = sum(1 for _ in open(file_path)) - 1  # Subtract header row
                except:
                    row_count = "?"
                
                if "1y" in file:
                    yearly_data_exists = True
                    
                # Check if this is real data or synthetic
                if "fixed_" in file or "generated" in file:
                    data_type = f"{Fore.YELLOW}SYNTHETIC{Style.RESET_ALL}"
                    synthetic_data_exists = True
                else:
                    data_type = f"{Fore.GREEN}REAL{Style.RESET_ALL}"
                    real_data_exists = True
                    
                timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                age = (datetime.now() - timestamp).days
                data_files.append([file, f"{size_mb:.2f} MB", f"{row_count:,} points", data_type, f"{age} days ago"])
    
    # Print status tables
    print(f"📂 Data Directory: {data_status}")
    print(f"🗄️  Model Storage: {Fore.GREEN + 'AVAILABLE' if os.path.exists(MODEL_DIR) else Fore.RED + 'MISSING'}")
    print(f"📊 Real Data: {Fore.GREEN + 'AVAILABLE' if real_data_exists else Fore.RED + 'MISSING'}")
    print(f"🔄 Synthetic Data: {Fore.GREEN + 'AVAILABLE' if synthetic_data_exists else Fore.YELLOW + 'NOT FOUND'}")
    print(f"📅 Full Year Data: {Fore.GREEN + 'AVAILABLE' if yearly_data_exists else Fore.YELLOW + 'NOT FOUND'}")
    print(f"📡 Real Data Loader: {Fore.GREEN + 'ENABLED' if HAS_REAL_DATA_LOADER else Fore.RED + 'DISABLED'}")
    
    print("\n📊 Trained Models Status:")
    print(tabulate(models_info, headers=["Model Type", "Status"], tablefmt="pretty"))
    
    if data_files:
        print("\n📈 Available Data Files:")
        # Sort data files by size (largest first) for better view
        data_files.sort(key=lambda x: x[0])
        for i, file_info in enumerate(data_files):
            file_info[0] = f"[{i+1}] {file_info[0]}"
        
        print(tabulate(data_files[:10], headers=["Filename", "Size", "Data Points", "Type", "Age"], tablefmt="pretty"))
        if len(data_files) > 10:
            print(f"...and {len(data_files) - 10} more files")
    
    real_data_info = None
    if HAS_REAL_DATA_LOADER:
        try:
            loader = RealDataLoader()
            available_data = loader.list_available_data()
            if available_data:
                print("\n🌐 Available Real Market Data:")
                for ticker, timeframes in available_data.items():
                    print(f"  {Fore.CYAN}{ticker}{Style.RESET_ALL}: {', '.join(timeframes)}")
                real_data_info = available_data
        except Exception as e:
            print(f"\n⚠️ {Fore.RED}Error scanning real market data: {str(e)}")
    
    return {
        'data_available': os.path.exists(DATA_DIR),
        'models': {model_type: os.path.exists(os.path.join(MODEL_DIR, f"{model_type}_model.joblib")) 
                  for model_type in MODELS.keys()},
        'yearly_data': yearly_data_exists,
        'real_data': real_data_exists,
        'synthetic_data': synthetic_data_exists,
        'has_real_data_loader': HAS_REAL_DATA_LOADER,
        'real_data_info': real_data_info,
        'data_files': data_files
    }

def display_stock_info(data, ticker):
    """Display stock information in a nice format."""
    print_section(f"STOCK OVERVIEW: {ticker}")
    
    # Find close price column
    close_col = None
    is_multi_index = isinstance(data.columns, pd.MultiIndex)
    
    if is_multi_index:
        for col in data.columns:
            if 'close' in str(col).lower():
                close_col = col
                break
    else:
        close_col = 'Close' if 'Close' in data.columns else 'close'
    
    if close_col is None:
        print(f"{Fore.RED}Error: Could not find close price column")
        return
    
    # Extract data
    try:
        prices = data[close_col].astype(float)
    except Exception as e:
        print(f"{Fore.RED}Error converting prices to float: {str(e)}")
        print(f"{Fore.YELLOW}Attempting to fix data format...")
        
        # Try to fix the data format
        if is_multi_index:
            # Print the first few column names to debug
            print(f"Column structure: {data.columns.names}")
            print(f"First few columns: {list(data.columns)[:5]}")
            
            # Try to extract using more flexible approach
            for col in data.columns:
                col_name = col[-1] if isinstance(col, tuple) else col
                if 'close' in str(col_name).lower():
                    try:
                        prices = pd.to_numeric(data[col], errors='coerce')
                        close_col = col
                        print(f"{Fore.GREEN}Found valid price column: {col}")
                        break
                    except:
                        continue
        
        if close_col is None:
            print(f"{Fore.RED}Could not find a valid price column. Cannot display stock info.")
            return {}
    
    # Calculate statistics
    last_price = float(prices.iloc[-1])
    start_price = float(prices.iloc[0])
    highest_price = float(prices.max())
    lowest_price = float(prices.min())
    
    # Calculate statistics
    pct_change = ((last_price / start_price) - 1) * 100
    volatility = prices.pct_change().std() * 100
    
    # Determine trend
    short_ma = prices.rolling(window=5).mean().iloc[-1]
    long_ma = prices.rolling(window=20).mean().iloc[-1]
    trend = "BULLISH 📈" if short_ma > long_ma else "BEARISH 📉"
    trend_color = Fore.GREEN if short_ma > long_ma else Fore.RED
    
    # Format dates
    start_date = data.index[0]
    end_date = data.index[-1]
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
        end_date = end_date.strftime('%Y-%m-%d')
    
    # Print information
    print(f"🔶 {Fore.YELLOW}Ticker:{Style.RESET_ALL} {ticker}")
    print(f"📅 {Fore.YELLOW}Period:{Style.RESET_ALL} {start_date} to {end_date}")
    print(f"📊 {Fore.YELLOW}Data Points:{Style.RESET_ALL} {len(data):,}")
    print(f"💰 {Fore.YELLOW}Current Price:{Style.RESET_ALL} ${last_price:.2f}")
    print(f"📊 {Fore.YELLOW}Price Range:{Style.RESET_ALL} ${lowest_price:.2f} - ${highest_price:.2f}")
    
    change_color = Fore.GREEN if pct_change >= 0 else Fore.RED
    print(f"📈 {Fore.YELLOW}Period Change:{Style.RESET_ALL} {change_color}{pct_change:.2f}%")
    print(f"📊 {Fore.YELLOW}Volatility:{Style.RESET_ALL} {volatility:.2f}%")
    print(f"🔮 {Fore.YELLOW}Current Trend:{Style.RESET_ALL} {trend_color}{trend}")
    
    # Generate a small ASCII chart
    display_ascii_chart(prices, trend_color)
    
    return {
        'last_price': last_price,
        'trend': trend,
        'change': pct_change
    }

def display_ascii_chart(prices, color=Fore.WHITE):
    """Display a simple ASCII chart of prices."""
    terminal_width = min(os.get_terminal_size().columns - 10, 80)
    chart_height = 7
    
    # Sample the data to fit the width
    if len(prices) > terminal_width:
        indices = np.linspace(0, len(prices)-1, terminal_width, dtype=int)
        sampled_prices = prices.iloc[indices].values
    else:
        sampled_prices = prices.values
    
    # Normalize to chart height
    min_price = min(sampled_prices)
    max_price = max(sampled_prices)
    if max_price == min_price:  # Avoid division by zero
        normalized = [chart_height // 2] * len(sampled_prices)
    else:
        normalized = [int((p - min_price) / (max_price - min_price) * chart_height) 
                     for p in sampled_prices]
    
    # Draw the chart
    chart = []
    for i in range(chart_height, -1, -1):
        line = ""
        for val in normalized:
            if val >= i:
                line += "▓"
            else:
                line += " "
        chart.append(line)
    
    print("\n" + color + "Price Chart:")
    for line in chart:
        print(color + line)
    
    # Add price range labels
    print(f"{color}${min_price:.2f}" + " " * (terminal_width - 16) + f"${max_price:.2f}")
    print(Style.RESET_ALL)

def load_stock_data(ticker='AAPL', timeframe='15m', date_range='3m', data_file_idx=None):
    """Load stock data with progress bar."""
    print_section(f"LOADING DATA: {ticker}")
    
    # Create progress bar
    bar = tqdm.tqdm(total=100, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    # Update progress
    bar.update(10)
    bar.set_description("Checking data sources")
    time.sleep(0.5)
    
    # Check for real data first
    if HAS_REAL_DATA_LOADER:
        try:
            loader = RealDataLoader()
            available_data = loader.list_available_data()
            
            if ticker in available_data and timeframe in available_data[ticker]:
                print(f"{Fore.GREEN}✅ Real market data available for {ticker} ({timeframe})")
                use_real_data = True
            else:
                print(f"{Fore.YELLOW}⚠️ No real market data found for {ticker} ({timeframe})")
                use_real_data = False
        except Exception as e:
            print(f"{Fore.RED}❌ Error checking real data: {str(e)}")
            use_real_data = False
    else:
        use_real_data = False
        print(f"{Fore.YELLOW}⚠️ Real data loader not available, using fallback options")
    
    # Let the user choose a data file if data_file_idx is provided
    data_path = None
    if data_file_idx is not None and os.path.exists(DATA_DIR):
        data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
        data_files.sort()
        if 0 < data_file_idx <= len(data_files):
            data_path = os.path.join(DATA_DIR, data_files[data_file_idx-1])
            print(f"{Fore.GREEN}Using selected data file: {data_files[data_file_idx-1]}")
            # If user selected a file, disable real data loading
            use_real_data = False
    
    # Default data paths for fallback
    yearly_data_path = os.path.join(DATA_DIR, f'fixed_{ticker}_15m_1y.csv')
    default_data_path = os.path.join(DATA_DIR, f'fixed_{ticker}_{timeframe}.csv')
    
    # Update progress
    bar.update(10)
    bar.set_description("Reading data files")
    time.sleep(0.5)
    
    try:
        # Try to load real market data if available
        if use_real_data:
            print(f"{Fore.CYAN}Loading real market data for {ticker} ({timeframe})...")
            
            # Load the data
            data = loader.load_data(ticker, timeframe)
            print(f"{Fore.GREEN}✅ Loaded real market data: {len(data):,} data points")
            
            bar.update(30)
            bar.set_description("Preprocessing data")
            time.sleep(0.5)
            
            # Preprocess the data
            processed_data = loader.preprocess_data(data)
            
            # Create MultiIndex DataFrame
            data_processed = loader.create_multi_index_df(processed_data, ticker)
            
            print(f"{Fore.GREEN}✅ Processed {len(data_processed):,} rows with {len(processed_data.columns)} indicators")
            
        # Load data using custom approach if a specific file is chosen
        elif data_path and os.path.exists(data_path):
            print(f"{Fore.CYAN}Loading from {data_path}")
            
            # Read the CSV file using pandas
            data = pd.read_csv(data_path)
            bar.update(20)
            
            # Check if this is real data
            is_real_data = "fixed_" not in os.path.basename(data_path) and "generated" not in os.path.basename(data_path)
            
            # Convert 'Datetime' column to datetime and set as index
            if 'Datetime' in data.columns:
                data['Datetime'] = pd.to_datetime(data['Datetime'])
                data.set_index('Datetime', inplace=True)
            
            if is_real_data:
                # For real data, ensure we have OHLCV columns with proper types
                print(f"{Fore.CYAN}Processing real market data...")
                
                # Ensure column names are standardized
                rename_map = {}
                for col in data.columns:
                    if col.lower() == 'open':
                        rename_map[col] = 'Open'
                    elif col.lower() == 'high':
                        rename_map[col] = 'High'
                    elif col.lower() == 'low':
                        rename_map[col] = 'Low'
                    elif col.lower() == 'close':
                        rename_map[col] = 'Close'
                    elif col.lower() == 'volume':
                        rename_map[col] = 'Volume'
                    elif col.lower() == 'ticker' or col.lower() == 'symbol':
                        rename_map[col] = 'ticker'
                
                if rename_map:
                    data.rename(columns=rename_map, inplace=True)
                
                # Ensure data has required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    print(f"{Fore.YELLOW}Warning: Missing columns in real data: {', '.join(missing_cols)}")
                    # Try to infer missing columns
                    if 'Close' not in data.columns and 'Price' in data.columns:
                        data['Close'] = data['Price']
                    if 'Open' not in data.columns and 'Close' in data.columns:
                        data['Open'] = data['Close']
                    if 'High' not in data.columns and 'Close' in data.columns:
                        data['High'] = data['Close']
                    if 'Low' not in data.columns and 'Close' in data.columns:
                        data['Low'] = data['Close']
                    if 'Volume' not in data.columns:
                        data['Volume'] = 0
                
                # Convert price columns to float
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in data.columns:
                        try:
                            data[col] = pd.to_numeric(data[col], errors='coerce')
                        except Exception as e:
                            print(f"{Fore.RED}Error converting {col} to numeric: {str(e)}")
                
                # Check if ticker column exists and handle appropriately
                if 'ticker' in data.columns:
                    # Use the ticker from the data if available
                    unique_tickers = data['ticker'].unique()
                    if len(unique_tickers) == 1:
                        data_ticker = unique_tickers[0]
                        print(f"{Fore.CYAN}Using ticker from data: {data_ticker}")
                    else:
                        print(f"{Fore.YELLOW}Multiple tickers found in data: {unique_tickers}")
                        print(f"{Fore.YELLOW}Using specified ticker: {ticker}")
                        data_ticker = ticker
                    
                    # Remove ticker column before creating MultiIndex
                    data = data.drop(columns=['ticker'])
                else:
                    data_ticker = ticker
                
                # Create multi-index with correct structure for real data
                columns = pd.MultiIndex.from_product([[data_ticker], data.columns])
                data_multi = pd.DataFrame(data=data.values, index=data.index, columns=columns)
                
                data = data_multi
            else:
                # For synthetic data, use the standard approach
                # Create multi-index if needed
                data = predict_future_candles.create_multi_index_df(data, ticker)
            
            print(f"{Fore.GREEN}✅ Loaded {len(data):,} data points")
            
            bar.update(20)
            bar.set_description("Preprocessing data")
            time.sleep(0.5)
            
            # Preprocess data
            data_processed = predict_future_candles.preprocess_data(data)
            
        else:
            # Fall back to predict_future_candles function
            print(f"{Fore.YELLOW}Using automatic data loading fallback...")
            
            # Choose the right data source for fallback
            if os.path.exists(yearly_data_path) and timeframe == '15m':
                print(f"{Fore.GREEN}Found yearly data file: {os.path.basename(yearly_data_path)}")
            elif os.path.exists(default_data_path):
                print(f"{Fore.YELLOW}Found standard data file: {os.path.basename(default_data_path)}")
            else:
                print(f"{Fore.RED}No suitable data files found - will attempt to download")
                
            # Use predict_future_candles' download_historical_data function
            data = predict_future_candles.download_historical_data(ticker, timeframe, date_range)
            bar.update(20)
            
            bar.set_description("Preprocessing data")
            time.sleep(0.5)
            
            # Preprocess data
            data_processed = predict_future_candles.preprocess_data(data)
        
        bar.update(40)
        bar.set_description("Preparing visualization")
        time.sleep(0.5)
        
        # Close the progress bar
        bar.update(10)
        bar.close()
        
        return data_processed
        
    except Exception as e:
        bar.close()
        print(f"\n{Fore.RED}Error loading data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def train_model(data, model_type='rf'):
    """Train a model with progress bar and visual feedback."""
    print_section(f"TRAINING MODEL: {MODELS[model_type]}")
    
    model_path = os.path.join(MODEL_DIR, f"{model_type}_model.joblib")
    model_meta_path = os.path.join(MODEL_DIR, f"{model_type}_model_meta.json")
    
    # Check if model already exists
    if os.path.exists(model_path):
        model_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))).days
        
        # Load metadata if available
        model_meta = {}
        if os.path.exists(model_meta_path):
            try:
                with open(model_meta_path, 'r') as f:
                    model_meta = json.load(f)
            except:
                pass
        
        data_points = model_meta.get('data_points', 'unknown')
        training_time = model_meta.get('training_time', 'unknown')
        
        if isinstance(data_points, (int, float)):
            data_points_str = f"{data_points:,}"
        else:
            data_points_str = str(data_points)
            
        if isinstance(training_time, (int, float)):
            training_time_str = f"{training_time:.2f}s"
        else:
            training_time_str = str(training_time)
        
        print(f"⚠️  {Fore.YELLOW}Model already exists (trained {model_age} days ago)")
        print(f"    {Fore.YELLOW}Training data: {data_points_str} points")
        print(f"    {Fore.YELLOW}Training time: {training_time_str}")
        
        # Ask user if they want to retrain
        response = input(f"Do you want to retrain the model? (y/N): ").lower().strip()
        if response != 'y':
            print(f"{Fore.GREEN}Using existing {MODELS[model_type]} model")
            try:
                clf = joblib.load(model_path)
                return clf, model_meta
            except Exception as e:
                print(f"{Fore.RED}Error loading model: {str(e)}")
                print(f"{Fore.YELLOW}Will retrain model")
    
    print(f"🧠 Training {MODELS[model_type]} model with {len(data):,} data points...")
    
    # Create a progress bar
    bar = tqdm.tqdm(total=100, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    
    try:
        # Feature selection
        bar.update(10)
        bar.set_description("Selecting features")
        time.sleep(0.5)
        
        # Create target
        bar.update(15)
        bar.set_description("Creating target variable")
        time.sleep(0.5)
        
        # Split data
        bar.update(15)
        bar.set_description("Splitting data for training")
        time.sleep(0.5)
        
        # Train model with timing
        bar.update(10)
        bar.set_description(f"Training {MODELS[model_type]}")
        
        start_time = time.time()
        clf = model.train_model(data, model_type, save_model=False)  # Don't save yet
        training_time = time.time() - start_time
        
        # Save model and metadata
        joblib.dump(clf, model_path)
        
        # Create and save metadata
        model_meta = {
            'data_points': len(data),
            'training_time': training_time,
            'features': data.shape[1] if hasattr(data, 'shape') else 'unknown',
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(model_meta_path, 'w') as f:
            json.dump(model_meta, f)
        
        # Evaluate model
        bar.update(40)
        bar.set_description("Evaluating model performance")
        time.sleep(0.5)
        
        # Close the progress bar
        bar.update(10)
        bar.close()
        
        print(f"\n{Fore.GREEN}✅ {MODELS[model_type]} model successfully trained and saved!")
        print(f"📊 {Fore.YELLOW}Training data:{Style.RESET_ALL} {len(data):,} points")
        print(f"⏱️  {Fore.YELLOW}Training time:{Style.RESET_ALL} {training_time:.2f} seconds")
        
        return clf, model_meta
        
    except Exception as e:
        bar.close()
        print(f"\n{Fore.RED}Error training model: {str(e)}")
        return None, {}

def generate_trading_signals(predictions, current_price):
    """Generate trading signals based on predictions."""
    # Extract prediction data
    pred_prices = [p['predicted_price'] for p in predictions]
    directions = [p['direction'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    # Calculate overall direction and confidence
    up_predictions = sum(1 for d in directions if d == 'up')
    down_predictions = len(directions) - up_predictions
    
    # Weighted average of confidences
    weighted_confidence = sum(conf if dir == 'up' else (1-conf) 
                             for dir, conf in zip(directions, confidences)) / len(directions)
    
    # Calculate expected return
    final_price = pred_prices[-1]
    expected_return = ((final_price / current_price) - 1) * 100
    
    # Generate trading signal
    if up_predictions > down_predictions and weighted_confidence > 0.5:
        signal = "BUY"
        signal_color = Fore.GREEN
    elif down_predictions > up_predictions and weighted_confidence > 0.5:
        signal = "SELL"
        signal_color = Fore.RED
    else:
        signal = "HOLD"
        signal_color = Fore.YELLOW
    
    # Calculate strength (1-5 stars)
    strength = int(weighted_confidence * 5)
    
    # Entry and exit points
    entry_price = current_price
    exit_price = final_price
    
    # Stop loss (3% below entry for buy, 3% above for sell)
    if signal == "BUY":
        stop_loss = entry_price * 0.97
    elif signal == "SELL":
        stop_loss = entry_price * 1.03
    else:
        stop_loss = None
    
    # Risk-reward ratio
    if stop_loss is not None:
        risk = abs(entry_price - stop_loss)
        reward = abs(exit_price - entry_price)
        risk_reward = reward / risk if risk > 0 else 0
    else:
        risk_reward = None
    
    return {
        'signal': signal,
        'signal_color': signal_color,
        'strength': strength,
        'confidence': weighted_confidence,
        'expected_return': expected_return,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'stop_loss': stop_loss,
        'risk_reward': risk_reward
    }

def display_predictions(predictions, ticker, timeframe, trading_signals):
    """Display predictions with trading signals."""
    print_section(f"PREDICTIONS & TRADING SIGNAL: {ticker}")
    
    # Extract prediction data
    timestamps = [p['timestamp'] for p in predictions]
    prices = [p['predicted_price'] for p in predictions]
    directions = [p['direction'] for p in predictions]
    confidences = [p['confidence'] for p in predictions]
    
    # Current price
    current_price = trading_signals['entry_price']
    
    # Print trading signal
    signal = trading_signals['signal']
    signal_color = trading_signals['signal_color']
    strength = "★" * trading_signals['strength'] + "☆" * (5 - trading_signals['strength'])
    
    print(f"🔮 {Fore.YELLOW}Trading Signal:{Style.RESET_ALL} {signal_color}{signal} {strength}")
    print(f"💰 {Fore.YELLOW}Current Price:{Style.RESET_ALL} ${current_price:.2f}")
    print(f"🎯 {Fore.YELLOW}Target Price:{Style.RESET_ALL} ${trading_signals['exit_price']:.2f}")
    
    if trading_signals['stop_loss']:
        print(f"🛑 {Fore.YELLOW}Suggested Stop Loss:{Style.RESET_ALL} ${trading_signals['stop_loss']:.2f}")
    
    if trading_signals['risk_reward']:
        print(f"⚖️  {Fore.YELLOW}Risk-Reward Ratio:{Style.RESET_ALL} 1:{trading_signals['risk_reward']:.2f}")
    
    # Format expected return with color
    return_color = Fore.GREEN if trading_signals['expected_return'] >= 0 else Fore.RED
    print(f"📈 {Fore.YELLOW}Expected Return:{Style.RESET_ALL} {return_color}{trading_signals['expected_return']:.2f}%")
    
    # Print confidence
    conf_color = Fore.GREEN if trading_signals['confidence'] > 0.6 else (
        Fore.YELLOW if trading_signals['confidence'] > 0.5 else Fore.RED)
    print(f"🎲 {Fore.YELLOW}Signal Confidence:{Style.RESET_ALL} {conf_color}{trading_signals['confidence']:.2f}")
    
    # Print prediction table
    print(f"\n{Fore.CYAN}Predicted Price Movement:")
    
    table_data = []
    for i, (ts, price, direction, conf) in enumerate(zip(timestamps, prices, directions, confidences)):
        direction_arrow = "↑" if direction == "up" else "↓"
        direction_color = Fore.GREEN if direction == "up" else Fore.RED
        change_pct = ((price / current_price) - 1) * 100
        
        table_data.append([
            i+1,
            ts.strftime("%Y-%m-%d %H:%M"),
            f"${price:.2f}",
            f"{change_pct:+.2f}%",
            f"{direction_color}{direction_arrow}",
            f"{conf:.2f}"
        ])
    
    headers = ["#", "Time", "Price", "Change", "Dir", "Conf"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    
    # Visualize price path as ASCII
    display_price_path(current_price, prices, directions)
    
    return trading_signals

def display_price_path(current_price, predicted_prices, directions):
    """Display a simple ASCII visualization of the predicted price path."""
    print(f"\n{Fore.CYAN}Price Path Visualization:")
    
    # Define width and prepare ranges
    width = 50
    all_prices = [current_price] + predicted_prices
    min_price = min(all_prices) * 0.99
    max_price = max(all_prices) * 1.01
    price_range = max_price - min_price
    
    # Create visualization
    lines = []
    
    # Add price labels on the right
    top_label = f"${max_price:.2f}"
    bottom_label = f"${min_price:.2f}"
    middle_label = f"${(min_price + max_price) / 2:.2f}"
    
    # Print visualization
    path = [int((p - min_price) / price_range * 10) for p in all_prices]
    
    # Create the visualization with bars
    for row in range(10, -1, -1):
        line = ""
        for col, p in enumerate(path):
            if p == row:
                # Current position
                if col == 0:
                    line += Fore.YELLOW + "⬤"
                else:
                    # Use different colors for up/down
                    if directions[col-1] == "up":
                        line += Fore.GREEN + "▲"
                    else:
                        line += Fore.RED + "▼"
            elif col > 0 and path[col-1] > row and p < row:
                # Line going down
                line += Fore.RED + "│"
            elif col > 0 and path[col-1] < row and p > row:
                # Line going up
                line += Fore.GREEN + "│"
            elif p > row:
                # Space above
                line += " "
            else:
                # Space below
                line += " "
        
        # Add price labels
        if row == 10:
            line += f" {top_label}"
        elif row == 5:
            line += f" {middle_label}"
        elif row == 0:
            line += f" {bottom_label}"
            
        print(line)
    
    # Add time indicators
    timeline = Fore.CYAN + "Now" + " " * (width - 8) + "Future"
    print(timeline)
    print(Style.RESET_ALL)

def main():
    """Main function to run the terminal interface."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ML Trading Terminal')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER, help='Ticker symbol (default: AAPL)')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, 
                        choices=TIMEFRAMES, help='Data timeframe (default: 15m)')
    parser.add_argument('--model_type', type=str, default='rf',
                        choices=list(MODELS.keys()), help='Model type (default: rf)')
    parser.add_argument('--date_range', type=str, default='3m',
                        help='Date range for historical data (default: 3m)')
    parser.add_argument('--num_candles', type=int, default=5,
                        help='Number of future candles to predict (default: 5)')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Print header
    print_header()
    
    # Welcome message
    print(f"{Fore.CYAN}Welcome to the ML Trading Predictor Terminal!")
    print(f"{Fore.CYAN}This tool will help you make data-driven trading decisions using machine learning.\n")
    
    # Check system status
    status = check_system_status()
    
    # Let user select data file if available
    data_file_idx = None
    if status['data_available'] and len(status.get('data_files', [])) > 0:
        print(f"\n{Fore.CYAN}Would you like to select a specific data file for training/prediction?")
        response = input(f"Enter file number or press Enter to use recommended file: ").strip()
        
        if response and response.isdigit():
            data_file_idx = int(response)
            if data_file_idx > len(status.get('data_files', [])):
                print(f"{Fore.RED}Invalid file number. Using recommended file.")
                data_file_idx = None
    
    # Load stock data
    data = load_stock_data(args.ticker, args.timeframe, args.date_range, data_file_idx)
    if data is None:
        print(f"{Fore.RED}Failed to load data. Exiting.")
        return
    
    # Display stock information
    stock_info = display_stock_info(data, args.ticker)
    
    # Train model if not already done
    if not status['models'][args.model_type]:
        print(f"\n{Fore.YELLOW}⚠️ {MODELS[args.model_type]} model not found. Training required.")
    
    # Confirm model training
    if not status['models'][args.model_type]:
        print(f"\n{Fore.CYAN}Ready to train {MODELS[args.model_type]} model.")
        input(f"{Fore.CYAN}Press Enter to continue...")
        model_obj, model_meta = train_model(data, args.model_type)
    else:
        # Ask if user wants to train anyway
        print(f"\n{Fore.GREEN}✅ {MODELS[args.model_type]} model already trained.")
        response = input(f"Do you want to use the existing model? (Y/n): ").lower().strip()
        if response == 'n':
            model_obj, model_meta = train_model(data, args.model_type)
        else:
            model_path = os.path.join(MODEL_DIR, f"{args.model_type}_model.joblib")
            try:
                model_obj = joblib.load(model_path)
                print(f"{Fore.GREEN}✅ Successfully loaded existing {MODELS[args.model_type]} model")
                
                # Load metadata if available
                model_meta = {}
                model_meta_path = os.path.join(MODEL_DIR, f"{args.model_type}_model_meta.json")
                if os.path.exists(model_meta_path):
                    try:
                        with open(model_meta_path, 'r') as f:
                            model_meta = json.load(f)
                        print(f"📊 {Fore.YELLOW}Training data:{Style.RESET_ALL} {model_meta.get('data_points', 'unknown'):,} points")
                    except:
                        pass
            except Exception as e:
                print(f"{Fore.RED}Error loading model: {str(e)}")
                print(f"{Fore.YELLOW}Will train new model")
                model_obj, model_meta = train_model(data, args.model_type)
    
    # Make predictions
    print_section("GENERATING PREDICTIONS")
    print(f"🔮 Predicting {args.num_candles} future candles for {args.ticker}...")
    
    try:
        # Use predict_future_candles function
        with tqdm.tqdm(total=100, ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}') as bar:
            bar.update(20)
            bar.set_description("Processing data")
            time.sleep(0.5)
            
            bar.update(30)
            bar.set_description("Running prediction model")
            time.sleep(0.5)
            
            # Get predictions
            predictions = predict_future_candles.predict_future_candles(data, args.model_type, args.num_candles)
            
            bar.update(30)
            bar.set_description("Generating trading signals")
            time.sleep(0.5)
            
            # Generate trading signals
            current_price = stock_info['last_price']
            trading_signals = generate_trading_signals(predictions, current_price)
            
            bar.update(20)
        
        # Display predictions and trading signals
        display_predictions(predictions, args.ticker, args.timeframe, trading_signals)
        
        # Save visualization if requested
        save_response = input(f"\n{Fore.CYAN}Save prediction visualization? (y/N): ").lower().strip()
        if save_response == 'y':
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = f"plots/{args.ticker}_{args.timeframe}_forecast_{timestamp}.png"
            predict_future_candles.visualize_predictions(data, predictions, args.ticker, args.timeframe, plot_path)
            print(f"{Fore.GREEN}✅ Visualization saved to: {plot_path}")
        
    except Exception as e:
        print(f"{Fore.RED}Error generating predictions: {str(e)}")
    
    # Print footer
    print_footer()

if __name__ == "__main__":
    main() 