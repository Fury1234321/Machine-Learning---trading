#!/bin/bash
# ML 2.0 Trading Predictor - Startup Script for Unix/Linux
# To start the application on macOS, open Terminal and run:
# chmod +x start.sh  # Make it executable (only needed once)
# ./start.sh         # Run the application
# Determine script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ASCII Art Title
echo "============================================================"
echo "  __  __ _      ____    ___    _____               _"
echo " |  \/  | |    |___ \  / _ \  |_   _|             | |"
echo " | \  / | |      __) || | | |   | |  _ __ __ _  __| | ___ _ __"
echo " | |\/| | |     |__ < | | | |   | | | '__/ _\` |/ _\` |/ _ \ '__|"
echo " | |  | | |____ ___) || |_| |  _| |_| | | (_| | (_| |  __/ |"
echo " |_|  |_|______|____/  \___/  |_____|_|  \__,_|\__,_|\___|_|"
echo "                                                            "
echo "============================================================"
echo "             Machine Learning Trading Predictor             "
echo "============================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Please make sure python3-venv is installed and try again."
        exit 1
    fi
    
    echo "Installing dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies."
        echo "Please check the requirements.txt file and try again."
        exit 1
    fi
else
    # Activate virtual environment
    source venv/bin/activate
fi

# Start ML 2.0 Trading Predictor
echo "Starting ML 2.0 Trading Predictor..."
python3 main.py

# Deactivate virtual environment on exit
deactivate 