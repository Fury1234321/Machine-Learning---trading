# ML 2.0 Trading Predictor

A streamlined machine learning framework for predicting financial market movements, designed with modularity and simplicity in mind.

## Overview

ML 2.0 Trading Predictor is designed to train and evaluate various machine learning models for predicting price movements in financial markets. The project focuses on predicting the direction of price movement 10 candles (time periods) ahead.

Key features:
- Modular architecture for easy extension and maintenance
- Support for multiple model types (Neural Networks, Gradient Boosting, Decision Trees)
- Simple terminal-based user interface
- Comprehensive data preprocessing pipeline
- Model management system for saving and loading trained models
- Automated testing and debugging tools for AI agents

## Project Structure

```
ML_2.0/
├── config/              # Configuration settings
├── data/                # Data directory
│   └── samples/         # Sample data files
├── logs/                # Log files directory
├── models/              # Trained models directory
│   ├── neural_network/  # Neural network models
│   ├── gradient_boosting/  # Gradient boosting models
│   └── tree/            # Decision tree models
├── scripts/             # Utility scripts
├── ui/                  # User interface code
├── utils/               # Utility functions
├── main.py              # Main application entry point
├── start.bat            # Windows startup script
├── start.sh             # Unix/Linux startup script
├── requirements.txt     # Python dependencies
├── CHANGELOG.md         # Changelog
└── README.md            # This file
```

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required packages (installed automatically by startup scripts)

### Installation

1. Clone the repository or download the source code
2. Run the appropriate startup script:
   - Windows: `start.bat`
   - Unix/Linux: `./start.sh`

The startup script will:
- Create a virtual environment if one doesn't exist
- Install required dependencies
- Launch the application

### Usage

The terminal-based interface provides the following options:

1. **View Available Models**: See all trained models along with their performance metrics
2. **Train a New Model**: Train a new machine learning model with the following options:
   - Neural Network (Deep Learning for complex patterns)
   - Gradient Boosting (High accuracy, handles non-linear patterns)
   - Decision Tree (Simple, interpretable model)

When training a model, you'll select a data file and the application will:
1. Preprocess the data (add technical indicators, normalize, etc.)
2. Split the data into training and testing sets
3. Train the selected model type
4. Evaluate performance and display metrics
5. Save the trained model for future use

### Adding Your Own Data

Place your CSV data files in the `data/` directory. Files should contain at minimum:
- Date column (datetime format)
- OHLC price data (Open, High, Low, Close)
- Volume data

Example CSV format:
```
Date,Open,High,Low,Close,Volume
2023-01-01 09:30:00,128.12,130.71,126.83,129.42,630659
2023-01-01 09:45:00,127.81,130.40,126.52,129.11,1183108
...
```

## Model Types

### Neural Network (LSTM-based)
- Best for: Identifying complex patterns in time series data
- Architecture: LSTM layers with dropout and batch normalization
- Input: Sequences of price and technical indicators
- Output: Binary classification (price direction)

### Gradient Boosting
- Best for: High accuracy predictions with robustness to outliers
- Features: Hyperparameter-tuned ensemble of decision trees
- Strengths: Handles non-linear relationships well

### Decision Tree
- Best for: Simple, interpretable models
- Features: Configurable depth and split criteria
- Strengths: Easy to understand and visualize decisions

## Debugging

The project includes an AI agent debugger that can be run with:
```
python main.py --debug
```

This tool automatically tests:
- Environment setup
- Data processing functionality
- Model training for all supported model types

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was inspired by the need for a streamlined, modular approach to financial machine learning
- Special thanks to contributors and the open source community 