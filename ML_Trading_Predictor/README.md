# ML Trading Predictor

A standalone machine learning framework for predicting future price movements of financial assets using various ML models.

## 🚀 Quick Start

### macOS Users

For the best experience on macOS:

1. **Double-click** the "**Start ML Trading Predictor.command**" file
   - If you get a security warning, go to System Preferences > Security & Privacy, and click "Open Anyway"
   - You only need to do this once

If the above doesn't work:
1. Open Terminal (Applications > Utilities > Terminal)
2. Type `chmod +x "` then drag the "Start ML Trading Predictor.command" file into Terminal and press Enter
3. Now you can double-click the command file to start the application

### Windows Users

Simply double-click the `start.bat` file.

### Manual Start (Advanced)

Or run this command to start the interactive interface:

```bash
python main.py
```

The interactive interface will guide you through all options with simple menus.

## ✨ Features

- **Streamlined Interface**: Quick start with simplified workflow and sensible defaults
- **Interactive Guided Experience**: User-friendly menus that guide you through all options
- **Enhanced Terminal Visuals**: Progress bars, ASCII charts, and trading signals
- **Model Selection Helper**: Choose the right ML model with detailed pros/cons analysis
- **Multiple Timeframes**: Support from 1-minute to monthly data (optimized for 15-minute)
- **Price Predictions**: Forecast up to 15 future price candles
- **Trading Signals**: Get clear BUY/SELL/HOLD recommendations
- **Model Comparison**: Evaluate different ML approaches for your trading style

## 📋 Installation

1. Download or clone this repository
2. Start the application using the instructions for your platform:

   **macOS**:
   - Double-click "Start ML Trading Predictor.command"
   - If it opens in a text editor instead:
     1. Open Terminal (Applications > Utilities > Terminal)
     2. Run: `chmod +x "/path/to/Start ML Trading Predictor.command"`
     3. Try double-clicking again

   **Windows**:
   - Double-click "start.bat"

Or install dependencies manually:
```bash
pip install -r requirements.txt
python main.py
```

## 📊 Available Modes

The application uses AAPL (Apple) stock data by default and offers these modes:

### 1. Trading Terminal
Interactive visualization with real-time trading signals and predictions.

### 2. Predict Future Candles
Generate and visualize multiple future price predictions.

### 3. Compare Models
Find the best performing ML model for your specific trading needs.

### 4. Basic Prediction
Quick and simple price direction forecast.

### 5. Demo
Guided tour of the system's capabilities.

## 🧠 ML Models

Choose from multiple machine learning approaches:

- **Random Forest**: General-purpose trading prediction
- **Gradient Boosting**: Higher accuracy for complex patterns
- **Support Vector Machine**: Effective trend classification
- **Neural Network**: Complex pattern recognition
- **Ensemble**: Combines multiple models for robust predictions

## 📝 Advanced Usage

While the interactive interface is recommended, you can also use direct command line options:

```bash
# Non-interactive mode with defaults
python main.py --non-interactive

# Specify parameters directly
python main.py --non-interactive --ticker MSFT --timeframe 1h --model_type ensemble --mode predict
```

### Direct Script Access

For advanced users, access individual component scripts directly:

```bash
# Trading Terminal
python trading_terminal.py --ticker AAPL --timeframe 15m --num_candles 5 --model_type rf

# Predict Future Candles
python predict_future_candles.py --ticker AAPL --timeframe 15m --num_candles 5 --model_type rf --save_plot

# Compare Models
python compare_models.py --ticker AAPL --timeframe 15m --horizon 1 --models rf,gb --folds 3

# Basic Prediction
python run_ml_model.py --ticker AAPL --timeframe 15m
```

## 📂 Project Structure

- `main.py` - Interactive interface and main entry point
- `Start ML Trading Predictor.command` - macOS launcher (double-click to start)
- `start.bat` - Windows startup script
- `start.sh` - Legacy shell script for Linux/macOS terminals
- `trading_terminal.py` - Enhanced visualization and prediction terminal
- `simple_ml_model.py` - Core ML functionality and technical indicators
- `predict_future_candles.py` - Future candle prediction engine
- `model_selector.py` - Interactive model selection with pros/cons analysis
- `compare_models.py` - Model comparison and evaluation tool
- `run_ml_model.py` - Basic prediction script
- `demo.py` - Guided demo of system capabilities

## 📈 Technical Details

### Indicators Used

The system calculates and uses over 50 technical indicators, including:

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Various Moving Averages (SMA, EMA)
- Volume indicators
- Momentum indicators
- Volatility measures

### Model Training

Models are automatically trained on historical data and saved for reuse:

- Feature selection and engineering
- Time-series cross-validation
- Model evaluation metrics 
- Confidence scoring
- Ensemble methods for improved accuracy

### Data Sources

The system uses locally cached data files (optimized for 15-minute timeframe):

- Stock data is downloaded from Yahoo Finance
- Option to use pre-cached datasets for faster operation

## 🛠️ Performance Tips

- The first run will train models, which may take some time
- Subsequent runs will use saved models, which is much faster
- For best results, use at least 3 months of historical data
- The system is optimized for 15-minute timeframe data

## 🔧 Troubleshooting

### macOS Issues

1. **File opens in text editor instead of executing**:
   - Open Terminal and run: `chmod +x "/path/to/Start ML Trading Predictor.command"`
   - Then double-click the file again

2. **"Unidentified developer" warning**:
   - Right-click (or Control+click) the file and select "Open"
   - Click "Open" in the dialog that appears
   - In future, you can double-click the file normally

3. **Python not found**:
   - Install Python from [python.org](https://www.python.org/downloads/) 

## Recent Updates

### Model Format Compatibility (2025-03-24)

The following improvements have been made to ensure model format compatibility and reliability:

1. **Enhanced Model Loading**: The application now robustly handles different model formats, including legacy formats, ensuring backward compatibility.

2. **Ensemble Model Improvements**: Fixed issues with ensemble model saving and loading. Ensemble models now have proper predict/predict_proba methods.

3. **Error Handling**: Added comprehensive error handling throughout the model training and prediction pipeline, with graceful fallbacks to trend-based prediction when ML models are unavailable.

4. **Model Format Standardization**: Standardized the model saving format to ensure consistency across different model types.

5. **Resilient Predictions**: The system is now more resilient against data format issues and missing features, automatically adapting to ensure consistent predictions.

These changes enhance the reliability and stability of the ML Trading Predictor, particularly when working with saved models or when training new models. 