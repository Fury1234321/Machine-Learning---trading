#!/usr/bin/env python3
"""
ML Trading Predictor Demo

This script runs the enhanced terminal interface with preset parameters to showcase
the visualization capabilities and workflow for new users.

Usage:
    python demo.py
"""

import os
import subprocess
import sys
import time
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

def print_header():
    """Print a fancy header for the demo."""
    terminal_width = os.get_terminal_size().columns
    
    print("\n" + "=" * terminal_width)
    print(Fore.CYAN + Style.BRIGHT + "🚀 ML TRADING PREDICTOR DEMO 🚀".center(terminal_width))
    print("=" * terminal_width + "\n")
    
    print(Fore.GREEN + "This demo will showcase the enhanced terminal interface with preset parameters.")
    print(Fore.GREEN + "The demo will run the following commands:")
    print(Fore.YELLOW + "1. Interactive model selection with pros/cons analysis")
    print(Fore.YELLOW + "2. Check system status and available models")
    print(Fore.YELLOW + "3. Load Apple (AAPL) 15-minute data")
    print(Fore.YELLOW + "4. Train or load the selected model")
    print(Fore.YELLOW + "5. Generate predictions for the next 5 candles")
    print(Fore.YELLOW + "6. Display trading signals and visualization")
    
    print("\n" + Fore.CYAN + "Starting demo in 3 seconds...")
    time.sleep(3)

def run_model_selector():
    """Run the interactive model selector and return the selected model."""
    print(Fore.CYAN + "\n=== Model Selection Phase ===")
    print(Fore.GREEN + "First, let's choose the best machine learning model for our trading strategy.")
    print(Fore.YELLOW + "The model selector will show pros and cons of each model type.\n")
    
    # Import the model selector
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import model_selector
    
    # Run the model selector to get user's choice
    print(Fore.WHITE + "Starting model selector...")
    selected_model = model_selector.select_model()
    
    # Default to Random Forest if no selection was made
    if not selected_model:
        print(Fore.YELLOW + "\nNo model selected. Defaulting to Random Forest (rf).")
        selected_model = "rf"
    
    # Get the model info to display
    model_info = model_selector.get_model_info(selected_model)
    print(Fore.GREEN + f"\nYou've selected: {model_info['name']} ({selected_model})")
    print(Fore.CYAN + f"Best for: {model_info['best_for']}")
    
    print(Fore.YELLOW + "\nContinuing to trading terminal with selected model...\n")
    time.sleep(2)
    
    return selected_model

def main():
    """Run the demo script."""
    # Print header
    print_header()
    
    # Check for required dependencies
    try:
        import tqdm
        import tabulate
    except ImportError:
        print(Fore.RED + "Error: Required dependencies not found.")
        print(Fore.YELLOW + "Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tqdm", "colorama", "tabulate"])
    
    # Run model selection first
    selected_model = run_model_selector()
    
    # Run the trading terminal with the selected model
    cmd = [sys.executable, "trading_terminal.py", 
           "--ticker", "AAPL", 
           "--timeframe", "15m", 
           "--num_candles", "5", 
           "--model_type", selected_model]
    
    print(Fore.CYAN + "\nExecuting command: " + " ".join(cmd))
    print(Fore.CYAN + "\nStarting enhanced terminal interface...\n")
    
    # Run the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 