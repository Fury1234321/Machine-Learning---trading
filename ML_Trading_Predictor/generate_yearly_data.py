#!/usr/bin/env python3
"""
Generate 1 Year of 15-Minute Data for AAPL

This script creates approximately 1 year of 15-minute AAPL data by expanding the existing
sample data. This is useful for testing and demonstrating the ML models without requiring
external API access or paid data sources.

The script will:
1. Read existing 15-minute data
2. Expand it to generate a full year by replicating and adjusting patterns
3. Save as both raw and fixed format CSV files
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import logging
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_existing_data(file_path):
    """Load existing 15-minute data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def expand_data_to_one_year(df, ticker="AAPL"):
    """Expand the data to cover approximately one year by replicating patterns."""
    logging.info("Expanding data to cover one year...")
    
    # Copy the original dataframe
    original_df = df.copy()
    
    # Convert Datetime to datetime object if it's a string
    if isinstance(original_df['Datetime'].iloc[0], str):
        original_df['Datetime'] = pd.to_datetime(original_df['Datetime'])
    
    # Get the timezone from the existing data
    has_timezone = original_df['Datetime'].dt.tz is not None
    if has_timezone:
        tzinfo = original_df['Datetime'].dt.tz
        logging.info(f"Data has timezone: {tzinfo}")
    else:
        # Add UTC timezone if none exists
        tzinfo = pytz.UTC
        logging.info("Data is timezone-naive, adding UTC timezone")
        original_df['Datetime'] = original_df['Datetime'].dt.tz_localize(tzinfo)
    
    # Sort by datetime to ensure chronological order
    original_df = original_df.sort_values('Datetime')
    
    # Calculate average daily range to create realistic price movements
    daily_high = original_df.groupby(original_df['Datetime'].dt.date)['High'].max()
    daily_low = original_df.groupby(original_df['Datetime'].dt.date)['Low'].min()
    avg_daily_range = (daily_high - daily_low).mean()
    
    # Calculate volatility for realistic price movements
    returns = original_df['Close'].pct_change().dropna()
    volatility = returns.std()
    
    # Determine starting and ending dates for the one year period
    # Use timezone-aware datetime
    end_date = datetime.now(tzinfo).replace(hour=20, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)
    
    logging.info(f"Generating data from {start_date} to {end_date}")
    
    # Create empty list to hold expanded dataframes
    expanded_segments = []
    
    # Use the original data as the first segment
    original_range = (original_df['Datetime'].max() - original_df['Datetime'].min()).total_seconds()
    
    # Calculate how many segments we need to cover a year
    # Assuming market hours and typical 15-minute candles
    # Trading hours per day: ~6.5 hours (26 15-minute candles)
    # Trading days per year: ~252 days
    # So we need approximately 252 * 26 = 6552 candles
    target_candles = 6552
    
    # How many original patterns do we need to replicate to reach target
    segment_count = max(int(target_candles / len(original_df)), 1)
    
    logging.info(f"Will generate {segment_count} segments to reach ~{target_candles} candles")
    
    # Starting price is the last close price in original data
    current_price = original_df['Close'].iloc[-1]
    
    # Add first segment
    expanded_segments.append(original_df)
    
    # Generate additional segments to fill the year
    for i in range(1, segment_count):
        logging.info(f"Generating segment {i+1}/{segment_count}")
        
        # Create a copy of the original dataframe for this segment
        segment = original_df.copy()
        
        # Update the datetime for this segment
        time_shift = timedelta(seconds=original_range * i)
        segment['Datetime'] = segment['Datetime'] + time_shift
        
        # Filter out dates beyond our end date
        segment = segment[segment['Datetime'] <= end_date]
        if len(segment) == 0:
            break
            
        # Adjust prices to create realistic movement patterns
        # Get last close price from previous segment
        last_close = expanded_segments[-1]['Close'].iloc[-1]
        
        # Generate more realistic price movements
        # Random walk with drift
        drift = np.random.normal(0.0001, volatility)  # Small upward bias
        price_factor = 1 + drift
        
        # Add seasonal patterns for more realism - simulate market cycles
        cycle_factor = np.sin(np.pi * i / segment_count) * volatility * 10
        
        # Apply price adjustments based on last close with continuity
        first_price = segment['Open'].iloc[0]
        price_shift = last_close / first_price
        
        # Apply price adjustments with continuity between segments
        segment['Open'] = segment['Open'] * price_shift * price_factor * (1 + cycle_factor)
        segment['High'] = segment['High'] * price_shift * price_factor * (1 + cycle_factor * 1.05)  # Highs slightly more volatile
        segment['Low'] = segment['Low'] * price_shift * price_factor * (1 + cycle_factor * 0.95)   # Lows slightly less volatile
        segment['Close'] = segment['Close'] * price_shift * price_factor * (1 + cycle_factor)
        
        # Ensure High is always highest, Low is always lowest
        segment['High'] = segment[['Open', 'High', 'Close']].max(axis=1)
        segment['Low'] = segment[['Open', 'Low', 'Close']].min(axis=1)
        
        # Add some random noise to volume with occasional volume spikes
        volume_factor = np.random.uniform(0.8, 1.2)
        # Add occasional volume spikes (1 in 20 chance)
        if np.random.random() < 0.05:
            volume_factor *= np.random.uniform(1.5, 3.0)
        segment['Volume'] = segment['Volume'] * volume_factor
        
        # Add segment to our collection
        expanded_segments.append(segment)
    
    # Combine all segments
    expanded_df = pd.concat(expanded_segments, ignore_index=True)
    
    # Sort by datetime and remove duplicates
    expanded_df = expanded_df.sort_values('Datetime').drop_duplicates(subset=['Datetime'])
    
    # Filter to get exactly one year
    expanded_df = expanded_df[expanded_df['Datetime'] >= start_date]
    
    # Add ticker column if needed
    if 'ticker' not in expanded_df.columns:
        expanded_df['ticker'] = ticker
    
    # Ensure we have the right number of data points
    logging.info(f"Generated {len(expanded_df)} rows covering 1 year")
    # Check if we have too few rows
    if len(expanded_df) < 5000:
        logging.warning(f"Generated only {len(expanded_df)} rows, which is less than expected for a full year of 15m data.")
        logging.warning("Consider using a larger source data file.")
    
    return expanded_df

def create_fixed_version(df):
    """Create a 'fixed' version of the dataset with technical indicators."""
    logging.info("Creating fixed version with technical indicators...")
    
    # Copy dataframe
    fixed_df = df.copy()
    
    # Add basic technical indicators
    # 1. Simple Moving Averages
    for window in [5, 10, 20, 50, 200]:
        fixed_df[f'SMA_{window}'] = fixed_df['Close'].rolling(window=window).mean()
    
    # 2. Exponential Moving Averages
    for window in [5, 10, 20, 50, 200]:
        fixed_df[f'EMA_{window}'] = fixed_df['Close'].ewm(span=window, adjust=False).mean()
    
    # 3. RSI (Relative Strength Index)
    delta = fixed_df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    fixed_df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. MACD (Moving Average Convergence Divergence)
    ema12 = fixed_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = fixed_df['Close'].ewm(span=26, adjust=False).mean()
    fixed_df['MACD'] = ema12 - ema26
    fixed_df['MACD_Signal'] = fixed_df['MACD'].ewm(span=9, adjust=False).mean()
    fixed_df['MACD_Hist'] = fixed_df['MACD'] - fixed_df['MACD_Signal']
    
    # 5. Bollinger Bands
    fixed_df['BB_Middle'] = fixed_df['Close'].rolling(window=20).mean()
    std = fixed_df['Close'].rolling(window=20).std()
    fixed_df['BB_Upper'] = fixed_df['BB_Middle'] + 2 * std
    fixed_df['BB_Lower'] = fixed_df['BB_Middle'] - 2 * std
    fixed_df['BB_Width'] = (fixed_df['BB_Upper'] - fixed_df['BB_Lower']) / fixed_df['BB_Middle']
    
    # 6. ATR (Average True Range)
    high_low = fixed_df['High'] - fixed_df['Low']
    high_close = (fixed_df['High'] - fixed_df['Close'].shift()).abs()
    low_close = (fixed_df['Low'] - fixed_df['Close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    fixed_df['ATR'] = true_range.rolling(14).mean()
    
    # 7. Volume indicators
    fixed_df['Volume_ROC'] = fixed_df['Volume'].pct_change(5)
    fixed_df['Volume_SMA_20'] = fixed_df['Volume'].rolling(window=20).mean()
    fixed_df['Volume_SMA_ratio'] = fixed_df['Volume'] / fixed_df['Volume_SMA_20']
    
    # 8. Price Rate of Change
    for period in [1, 5, 10, 20]:
        fixed_df[f'Price_ROC_{period}'] = fixed_df['Close'].pct_change(period) * 100
    
    # 9. Commodity Channel Index (CCI)
    typical_price = (fixed_df['High'] + fixed_df['Low'] + fixed_df['Close']) / 3
    # Mean Absolute Deviation - replace mad() which is deprecated
    def mean_abs_dev(x):
        return np.abs(x - x.mean()).mean()
    
    tp_sma = typical_price.rolling(window=20).mean()
    meandev = typical_price.rolling(window=20).apply(mean_abs_dev, raw=True)
    fixed_df['CCI'] = (typical_price - tp_sma) / (0.015 * meandev)
    
    # 10. Stochastic Oscillator
    low_14 = fixed_df['Low'].rolling(window=14).min()
    high_14 = fixed_df['High'].rolling(window=14).max()
    fixed_df['%K'] = ((fixed_df['Close'] - low_14) / (high_14 - low_14)) * 100
    fixed_df['%D'] = fixed_df['%K'].rolling(window=3).mean()
    
    # Fill NaN values (resulting from rolling windows)
    # Use backward fill first, then forward fill for any remaining NaNs
    fixed_df = fixed_df.bfill().ffill()
    
    return fixed_df

def main():
    parser = argparse.ArgumentParser(description='Generate 1 year of 15-minute AAPL data')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Ticker symbol (default: AAPL)')
    args = parser.parse_args()
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Input file path - use the fixed version which has better structure
    input_file = os.path.join(data_dir, f'fixed_{args.ticker}_15m.csv')
    
    # Output file paths
    output_raw_file = os.path.join(data_dir, f'{args.ticker}_15m_1y.csv')
    output_fixed_file = os.path.join(data_dir, f'fixed_{args.ticker}_15m_1y.csv')
    
    # Load existing data
    original_df = load_existing_data(input_file)
    
    # Expand data to 1 year
    expanded_df = expand_data_to_one_year(original_df, args.ticker)
    
    # Save expanded raw data
    expanded_df.to_csv(output_raw_file, index=False)
    logging.info(f"Saved raw 1-year data to {output_raw_file}")
    
    # Create and save fixed version
    fixed_df = create_fixed_version(expanded_df)
    fixed_df.to_csv(output_fixed_file, index=False)
    logging.info(f"Saved fixed 1-year data to {output_fixed_file}")
    
    print(f"\n==============================================================")
    print(f"Successfully generated 1 year of 15-minute data for {args.ticker}")
    print(f"==============================================================")
    print(f"Raw data: {output_raw_file}")
    print(f"Fixed data: {output_fixed_file}")
    print(f"Total data points: {len(expanded_df):,}")
    print(f"Date range: {expanded_df['Datetime'].min()} to {expanded_df['Datetime'].max()}")
    print(f"Features added: {len(fixed_df.columns) - len(expanded_df.columns)}")
    print(f"==============================================================")

if __name__ == "__main__":
    main() 