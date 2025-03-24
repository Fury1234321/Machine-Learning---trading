#!/usr/bin/env python3
"""
Direct ML Model Runner
Runs the trading ML model with command line arguments and outputs JSON
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64

# Add path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import simple_ml_model as model
# Import the model selector module
import model_selector

def parse_args():
    parser = argparse.ArgumentParser(description='Run ML Trading Model')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker symbol')
    parser.add_argument('--start_date', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--timeframe', type=str, default='1d', 
                        choices=['1m', '5m', '15m', '30m', '60m', '1h', '1d', '1wk', '1mo'],
                        help='Data timeframe (default: 1d)')
    parser.add_argument('--model_type', type=str, default=None,
                        choices=['rf', 'gb', 'svm', 'nn', 'ensemble'],
                        help='Model type (default: None, will prompt for selection)')
    parser.add_argument('--save_plot', action='store_true', help='Save plot to file')
    parser.add_argument('--output_format', type=str, default='json',
                        choices=['json', 'csv', 'text'],
                        help='Output format (default: json)')
    parser.add_argument('--skip_model_selection', action='store_true', 
                        help='Skip interactive model selection and use default (rf)')
    
    return parser.parse_args()

def run_model(args):
    # Set default dates if not provided
    if not args.end_date:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    if not args.start_date:
        # Default to 1 year ago
        start_date = datetime.now() - timedelta(days=365)
        args.start_date = start_date.strftime('%Y-%m-%d')
    
    # Model selection
    model_type = args.model_type
    
    # If model_type is not specified and interactive selection is not skipped,
    # run the model selector
    if model_type is None and not args.skip_model_selection:
        try:
            # Run interactive model selection
            print(f"Selecting model for {args.ticker} analysis...")
            model_type = model_selector.select_model()
            
            # If user cancels selection, default to random forest
            if model_type is None:
                print("No model selected. Defaulting to Random Forest (rf).")
                model_type = 'rf'
        except Exception as e:
            print(f"Error during model selection: {str(e)}")
            print("Defaulting to Random Forest (rf).")
            model_type = 'rf'
    elif model_type is None and args.skip_model_selection:
        # Default to rf if selection is skipped
        model_type = 'rf'
    
    # Display selected model information
    if model_type in model_selector.MODEL_INFO:
        model_info = model_selector.MODEL_INFO[model_type]
        print(f"\nUsing {model_info['name']} model for prediction.")
        print(f"Best for: {model_info['best_for']}\n")
    
    # Download data
    print(f"Downloading data for {args.ticker} from {args.start_date} to {args.end_date}...", file=sys.stderr)
    data = model.download_data(args.ticker, args.start_date, args.end_date, args.timeframe)
    
    if data.empty:
        return {"error": f"No data available for {args.ticker} in the specified date range."}
    
    # Calculate features
    print("Calculating technical features...", file=sys.stderr)
    data_with_features = model.calculate_features(data)
    
    # Remove NaN values
    data_with_features = data_with_features.dropna()
    
    if data_with_features.empty:
        return {"error": "No valid data points after feature calculation."}
    
    # Basic statistics
    latest_price = data_with_features['Close'].iloc[-1]
    daily_change = latest_price - data_with_features['Close'].iloc[-2] if len(data_with_features) > 1 else 0
    daily_change_pct = (daily_change / data_with_features['Close'].iloc[-2]) * 100 if len(data_with_features) > 1 else 0
    
    returns = data_with_features['Close'].pct_change().dropna()
    avg_return = returns.mean() * 100
    volatility = returns.std() * 100
    
    first_price = data_with_features['Close'].iloc[0]
    period_return = ((latest_price - first_price) / first_price) * 100
    
    # RSI status
    rsi_value = data_with_features['RSI_14'].iloc[-1] if 'RSI_14' in data_with_features.columns else None
    if rsi_value is not None:
        if rsi_value > 70:
            rsi_status = "Overbought"
        elif rsi_value < 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Neutral"
    else:
        rsi_status = "Unknown"
    
    # Price vs SMAs
    sma_status = []
    if 'SMA20' in data_with_features.columns:
        sma20 = data_with_features['SMA20'].iloc[-1]
        sma_status.append(f"{'Above' if latest_price > sma20 else 'Below'} SMA20")
    
    if 'SMA50' in data_with_features.columns:
        sma50 = data_with_features['SMA50'].iloc[-1]
        sma_status.append(f"{'Above' if latest_price > sma50 else 'Below'} SMA50")
    
    if 'SMA200' in data_with_features.columns:
        sma200 = data_with_features['SMA200'].iloc[-1]
        sma_status.append(f"{'Above' if latest_price > sma200 else 'Below'} SMA200")
    
    # Create ML prediction
    try:
        print("Running machine learning predictions...", file=sys.stderr)
        
        # Get last 60 days of data for prediction
        prediction_data = data_with_features.tail(60).copy()
        
        # Ensure the results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Create a temporary model if needed
        model_path = f"results/{model_type}_model.joblib"
        if not os.path.exists(model_path):
            print(f"Training a new {model_type} model...", file=sys.stderr)
            model.train_model(prediction_data, model_type, save_model=True)
        
        # Make prediction
        prediction, confidence = model.predict_next_day(prediction_data, model_type=model_type)
        # Process numeric prediction value (1 for bullish, -1 for bearish, 0 for unknown/error)
        try:
            # prediction should already be numeric but convert to int to be sure
            prediction = int(prediction)
            if prediction > 0:
                prediction_text = "Bullish"
            elif prediction < 0:
                prediction_text = "Bearish"
            else:
                prediction_text = "Neutral"
        except (ValueError, TypeError):
            prediction_text = "Error"
            confidence = 0.0
            print(f"Error in prediction: prediction '{prediction}' is not a numeric value", file=sys.stderr)
        
    except Exception as e:
        prediction_text = "Error"
        confidence = 0.0
        print(f"Error in prediction: {str(e)}", file=sys.stderr)
    
    # Generate plot
    plot_path = None
    if args.save_plot:
        try:
            print("Generating plot...", file=sys.stderr)
            plt.figure(figsize=(12, 8))
            
            # Create a plot with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price and SMAs
            ax1.plot(data_with_features.index, data_with_features['Close'], label='Price', linewidth=2)
            
            if 'SMA20' in data_with_features.columns:
                ax1.plot(data_with_features.index, data_with_features['SMA20'], label='SMA 20', linestyle='--')
            
            if 'SMA50' in data_with_features.columns:
                ax1.plot(data_with_features.index, data_with_features['SMA50'], label='SMA 50', linestyle='--')
            
            if 'SMA200' in data_with_features.columns:
                ax1.plot(data_with_features.index, data_with_features['SMA200'], label='SMA 200', linestyle='--')
            
            ax1.set_title(f"{args.ticker} Price Chart")
            ax1.set_ylabel('Price')
            ax1.grid(True)
            ax1.legend()
            
            # Plot RSI
            if 'RSI_14' in data_with_features.columns:
                ax2.plot(data_with_features.index, data_with_features['RSI_14'], label='RSI', color='purple')
                ax2.axhline(y=70, color='r', linestyle='--')
                ax2.axhline(y=30, color='g', linestyle='--')
                ax2.set_ylabel('RSI')
                ax2.grid(True)
                ax2.set_ylim(0, 100)
            
            ax2.set_xlabel('Date')
            
            plt.tight_layout()
            
            # Save plot
            os.makedirs('plots', exist_ok=True)
            plot_path = f"plots/{args.ticker}_{args.start_date}_{args.end_date}.png"
            plt.savefig(plot_path)
            plt.close()
            
            print(f"Plot saved to {plot_path}", file=sys.stderr)
            
        except Exception as e:
            print(f"Error generating plot: {str(e)}", file=sys.stderr)
    
    # Get the model info for the result
    model_info_display = {}
    if model_type in model_selector.MODEL_INFO:
        info = model_selector.MODEL_INFO[model_type]
        model_info_display = {
            "name": info["name"],
            "description": info["description"],
            "pros": info["pros"][:3],  # Include top 3 pros
            "cons": info["cons"][:3],  # Include top 3 cons
            "best_for": info["best_for"]
        }
    
    # Prepare results
    result = {
        "ticker": args.ticker,
        "latest_price": float(latest_price),
        "daily_change": float(daily_change),
        "daily_change_pct": float(daily_change_pct),
        "avg_return": float(avg_return),
        "volatility": float(volatility),
        "period_return": float(period_return),
        "rsi": float(rsi_value) if rsi_value is not None else None,
        "rsi_status": rsi_status,
        "price_vs_sma": sma_status,
        "prediction": prediction_text,
        "confidence": float(confidence) if isinstance(confidence, (int, float)) else 0,
        "model_used": model_type,
        "model_info": model_info_display,
        "plot_path": plot_path
    }
    
    return result

def main():
    args = parse_args()
    result = run_model(args)
    
    # Output results
    if args.output_format == 'json':
        # Convert all numpy/pandas types to native Python types for JSON serialization
        for key, value in result.items():
            if isinstance(value, (np.integer, np.floating)):
                result[key] = value.item()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
        
        print(json.dumps(result, indent=2))
    
    elif args.output_format == 'csv':
        # Flatten any nested structures
        flat_result = {}
        for key, value in result.items():
            if isinstance(value, list):
                flat_result[key] = ",".join(str(v) for v in value)
            elif isinstance(value, dict):
                # Skip nested dictionaries in CSV output
                continue
            else:
                flat_result[key] = value
        
        # Print CSV header and row
        headers = list(flat_result.keys())
        print(",".join(headers))
        print(",".join(str(flat_result[h]) for h in headers))
    
    else:  # text output
        for key, value in result.items():
            if key == "model_info" and isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, list):
                        print(f"  {k}:")
                        for item in v:
                            print(f"    - {item}")
                    else:
                        print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    main() 