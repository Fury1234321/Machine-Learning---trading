#!/bin/bash
# ML Trading Predictor - Startup Script for macOS
# To run this file, open Terminal and enter:
# chmod +x start.sh && ./start.sh

# Change to the directory containing this script
cd "$(dirname "$0")"

# Display startup banner
echo "====================================="
echo "🚀 Starting ML Trading Predictor... 🚀"
echo "====================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "Please install Python 3 from python.org and try again."
    echo ""
    echo "Press any key to exit..."
    read -n 1
    exit 1
fi

# Check if we need to set up virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Setting up virtual environment for first run..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    echo "📥 Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Start the application
echo "🚀 Launching ML Trading Predictor..."
python main.py

# Keep terminal window open if there was an error
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️ Application closed with an error."
    echo "Press any key to exit..."
    read -n 1
fi 