#!/usr/bin/env python3
"""
ML 2.0 Trading Predictor - Main Entry Point

This is the main entry point for the ML 2.0 Trading Predictor application.
The application allows training and managing different types of machine learning models
for predicting price movements in financial data.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import modules
from ui.terminal_ui import run as run_terminal_ui
from scripts.ai_agent_debugger import run as run_debugger

def setup_environment():
    """Set up the application environment"""
    # Ensure required directories exist
    for directory in ['data', 'logs', 'models', 'models/neural_network', 
                      'models/gradient_boosting', 'models/tree']:
        os.makedirs(os.path.join(current_dir, directory), exist_ok=True)
    
    # Set up logging
    from utils.logger import setup_logging
    setup_logging()

def main():
    """Main function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ML 2.0 Trading Predictor')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode using AI agent debugger')
    args = parser.parse_args()
    
    # Set up the environment
    setup_environment()
    
    # Run the appropriate UI
    if args.debug:
        run_debugger()
    else:
        run_terminal_ui()

if __name__ == "__main__":
    main() 