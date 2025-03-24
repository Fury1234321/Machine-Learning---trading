#!/usr/bin/env python3
"""
ML Trading Predictor - Main Interactive Interface

This is the main entry point for the ML Trading Predictor application.
It provides an interactive guided experience for users of all experience levels.

Simply run:
    python main.py

And follow the on-screen prompts to navigate through the application.
"""

import os
import sys
import time
import subprocess
import argparse
import colorama
from colorama import Fore, Style, Back
import inquirer
from tabulate import tabulate

# Initialize colorama
colorama.init(autoreset=True)

# Add path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import model_selector

# Constants for UI
TITLE = "ML Trading Predictor"
VERSION = "1.0.0"
DEFAULT_TICKER = "AAPL"
DEFAULT_TIMEFRAME = "15m"
DEFAULT_NUM_CANDLES = 5
DEFAULT_MODEL = "rf"

def print_header():
    """Display the application header with title and version"""
    terminal_width = os.get_terminal_size().columns
    print("\n" + "=" * terminal_width)
    print(Fore.CYAN + Style.BRIGHT + f"🚀 {TITLE} v{VERSION} 🚀".center(terminal_width))
    print("=" * terminal_width + "\n")

def print_section(title):
    """Print a section header"""
    print(Fore.YELLOW + f"\n=== {title} ===\n")

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import inquirer
        import tabulate
        import tqdm
        import pandas
        import numpy
        import matplotlib
        import sklearn
        import yfinance
    except ImportError as e:
        print(Fore.RED + f"Missing dependency: {str(e)}")
        print(Fore.YELLOW + "Installing required dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(Fore.GREEN + "Dependencies installed. Please restart the application.")
        sys.exit(0)

def select_timeframe():
    """Guide the user to select a timeframe"""
    print_section("Timeframe Selection")
    
    # Show information about timeframes
    print(Fore.CYAN + "Timeframe determines the granularity of the data:")
    print("- Shorter timeframes (1m, 5m, 15m) are better for intraday trading")
    print("- Longer timeframes (1d, 1wk) are better for swing or position trading")
    print(Fore.YELLOW + "Note: The model is currently optimized for 15-minute data\n")
    
    questions = [
        inquirer.List('timeframe',
                     message="Select a timeframe",
                     choices=[
                         ('1 minute', '1m'),
                         ('5 minutes', '5m'),
                         ('15 minutes (recommended)', '15m'),
                         ('30 minutes', '30m'),
                         ('1 hour', '1h'),
                         ('1 day', '1d'),
                         ('1 week', '1wk'),
                         ('1 month', '1mo'),
                     ],
                     default='15m'),
    ]
    answers = inquirer.prompt(questions)
    return answers['timeframe']

def select_model():
    """Guide the user to select a machine learning model"""
    print_section("Model Selection")
    
    # Show information about model selection
    print(Fore.CYAN + "The ML model determines how predictions are generated:")
    print("- Different models have different strengths and weaknesses")
    print("- You can view detailed pros and cons for each model type\n")
    
    # Initial selection - use the model selector or quick selection
    questions = [
        inquirer.List('selection_method',
                     message="How would you like to select a model?",
                     choices=[
                         ('Interactive model selector with pros/cons', 'interactive'),
                         ('Quick selection', 'quick'),
                     ]),
    ]
    answers = inquirer.prompt(questions)
    
    if answers['selection_method'] == 'interactive':
        # Use the model_selector module
        print(Fore.GREEN + "\nStarting interactive model selector...")
        selected_model = model_selector.select_model()
        if not selected_model:
            print(Fore.YELLOW + "No model selected. Defaulting to Random Forest (rf).")
            selected_model = DEFAULT_MODEL
        return selected_model
    else:
        # Quick selection
        questions = [
            inquirer.List('model',
                         message="Select a model type",
                         choices=[
                             ('Random Forest (general purpose)', 'rf'),
                             ('Gradient Boosting (high accuracy)', 'gb'),
                             ('Support Vector Machine (trend classification)', 'svm'),
                             ('Neural Network (complex patterns)', 'nn'),
                             ('Ensemble (combines multiple models)', 'ensemble'),
                         ],
                         default='rf'),
        ]
        answers = inquirer.prompt(questions)
        return answers['model']

def select_prediction_options():
    """Guide the user to select prediction options"""
    print_section("Prediction Options")
    
    # Number of candles to predict
    questions = [
        inquirer.List('num_candles',
                     message="How many future periods would you like to predict?",
                     choices=[
                         ('1 period', 1),
                         ('3 periods', 3),
                         ('5 periods (recommended)', 5),
                         ('10 periods', 10),
                         ('15 periods', 15),
                     ],
                     default=5),
    ]
    answers = inquirer.prompt(questions)
    num_candles = answers['num_candles']
    
    # Additional options
    questions = [
        inquirer.Checkbox('options',
                        message="Select additional options",
                        choices=[
                            ('Save visualization plot', 'save_plot'),
                            ('Save predictions to CSV', 'save_csv'),
                            ('Show detailed model metrics', 'show_metrics'),
                        ],
                        default=['save_plot']),
    ]
    answers2 = inquirer.prompt(questions)
    
    return {
        'num_candles': num_candles,
        'save_plot': 'save_plot' in answers2['options'],
        'save_csv': 'save_csv' in answers2['options'],
        'show_metrics': 'show_metrics' in answers2['options']
    }

def select_mode():
    """Guide the user to select the application mode"""
    print_section("Mode Selection")
    
    questions = [
        inquirer.List('mode',
                     message="What would you like to do?",
                     choices=[
                         ('Trading Terminal - Interactive visualization and predictions', 'terminal'),
                         ('Predict Future Candles - Generate price predictions', 'predict'),
                         ('Compare Models - Evaluate different ML models', 'compare'),
                         ('Basic Prediction - Simple price direction forecast', 'basic'),
                         ('Run Demo - Start the guided demo', 'demo'),
                     ]),
    ]
    answers = inquirer.prompt(questions)
    return answers['mode']

def run_trading_terminal(ticker, timeframe, model_type, num_candles, options):
    """Run the trading terminal with the specified options"""
    cmd = [sys.executable, "trading_terminal.py",
           "--ticker", ticker,
           "--timeframe", timeframe,
           "--model_type", model_type,
           "--num_candles", str(num_candles)]
    
    print(Fore.CYAN + "\nStarting Trading Terminal with command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd)

def run_predict_future(ticker, timeframe, model_type, num_candles, options):
    """Run the predict_future_candles.py script with the specified options"""
    cmd = [sys.executable, "predict_future_candles.py",
           "--ticker", ticker,
           "--timeframe", timeframe,
           "--model_type", model_type,
           "--num_candles", str(num_candles)]
    
    if options['save_plot']:
        cmd.append("--save_plot")
    
    if options['save_csv']:
        cmd.append("--save_csv")
    
    print(Fore.CYAN + "\nPredicting future candles with command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd)

def run_compare_models(ticker, timeframe):
    """Run the compare_models.py script"""
    # Ask user for specific compare options
    print_section("Model Comparison Options")
    
    # Models to compare
    questions = [
        inquirer.Checkbox('models',
                        message="Select models to compare",
                        choices=[
                            ('Random Forest', 'rf'),
                            ('Gradient Boosting', 'gb'),
                            ('Support Vector Machine', 'svm'),
                            ('Neural Network', 'nn'),
                            ('Ensemble', 'ensemble'),
                        ],
                        default=['rf', 'gb', 'ensemble']),
    ]
    models_answer = inquirer.prompt(questions)
    models = ','.join(models_answer['models'])
    
    # Number of folds
    questions = [
        inquirer.List('folds',
                     message="Number of cross-validation folds",
                     choices=['2', '3', '5'],
                     default='3'),
    ]
    folds_answer = inquirer.prompt(questions)
    folds = folds_answer['folds']
    
    # Run the comparison
    cmd = [sys.executable, "compare_models.py",
           "--ticker", ticker,
           "--timeframe", timeframe,
           "--models", models,
           "--folds", folds]
    
    print(Fore.CYAN + "\nComparing models with command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd)

def run_basic_prediction(ticker, timeframe, model_type):
    """Run the basic prediction script"""
    cmd = [sys.executable, "run_ml_model.py",
           "--ticker", ticker,
           "--timeframe", timeframe,
           "--model_type", model_type,
           "--output_format", "text"]
    
    print(Fore.CYAN + "\nRunning basic prediction with command:")
    print(" ".join(cmd))
    
    subprocess.run(cmd)

def run_demo():
    """Run the demo script"""
    cmd = [sys.executable, "demo.py"]
    
    print(Fore.CYAN + "\nStarting demo...")
    subprocess.run(cmd)

def main():
    """Main function to run the interactive interface"""
    # Parse command line arguments for direct mode (non-interactive)
    parser = argparse.ArgumentParser(description='ML Trading Predictor')
    parser.add_argument('--non-interactive', action='store_true', 
                        help='Run in non-interactive mode with defaults')
    parser.add_argument('--ticker', type=str, default=DEFAULT_TICKER,
                        help=f'Ticker symbol (default: {DEFAULT_TICKER})')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                        help=f'Data timeframe (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--model_type', type=str, default=DEFAULT_MODEL,
                        help=f'Model type (default: {DEFAULT_MODEL})')
    parser.add_argument('--mode', type=str, default='terminal',
                        choices=['terminal', 'predict', 'compare', 'basic', 'demo'],
                        help='Application mode (default: terminal)')
    
    args = parser.parse_args()
    
    # Check dependencies
    check_dependencies()
    
    # Display header
    print_header()
    
    # Welcome message
    print(Fore.GREEN + f"Welcome to {TITLE}!")
    print("This interactive interface will guide you through the process of")
    print("analyzing stocks and generating predictions using machine learning.\n")
    
    # If non-interactive mode, use command line arguments
    if args.non_interactive:
        print(Fore.YELLOW + "Running in non-interactive mode with default settings...")
        ticker = args.ticker
        timeframe = args.timeframe
        model_type = args.model_type
        mode = args.mode
        num_candles = DEFAULT_NUM_CANDLES
        options = {
            'num_candles': num_candles,
            'save_plot': True,
            'save_csv': False,
            'show_metrics': False
        }
    else:
        # Interactive mode - guide user through options
        mode = select_mode()
        
        # If demo mode, run it immediately
        if mode == 'demo':
            run_demo()
            return
        
        # Use default ticker (AAPL) instead of prompting user
        ticker = DEFAULT_TICKER
        print(Fore.BLUE + f"Using default ticker: {ticker}")
        
        # Select timeframe
        timeframe = select_timeframe()
        
        # Select model (unless in compare mode)
        if mode != 'compare':
            model_type = select_model()
        else:
            model_type = DEFAULT_MODEL  # Not used in compare mode
        
        # Get prediction options (for terminal and predict modes)
        if mode in ['terminal', 'predict']:
            options = select_prediction_options()
            num_candles = options['num_candles']
        else:
            num_candles = DEFAULT_NUM_CANDLES
            options = {
                'num_candles': num_candles,
                'save_plot': True,
                'save_csv': False,
                'show_metrics': False
            }
    
    # Run the selected mode
    if mode == 'terminal':
        run_trading_terminal(ticker, timeframe, model_type, num_candles, options)
    elif mode == 'predict':
        run_predict_future(ticker, timeframe, model_type, num_candles, options)
    elif mode == 'compare':
        run_compare_models(ticker, timeframe)
    elif mode == 'basic':
        run_basic_prediction(ticker, timeframe, model_type)
    elif mode == 'demo':
        run_demo()

if __name__ == "__main__":
    main() 