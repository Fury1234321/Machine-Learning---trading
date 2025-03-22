# Machine Learning Trading Model

A simple machine learning model for predicting NASDAQ 100 index movements.

## Features

- Downloads historical NASDAQ 100 data using yfinance
- Calculates technical indicators:
  - Returns and lagged returns
  - Simple Moving Averages (10-day and 50-day)
  - Relative Strength Index (RSI)
- Trains a Random Forest Classifier model
- Provides performance metrics and visualizations
- Makes predictions for the next trading day

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- yfinance

## Usage

Simply run the script:

```bash
python simple_ml_model.py
```

The script will:
1. Download NASDAQ 100 data from 2020
2. Create features and split data into training/testing sets
3. Train the model and evaluate performance
4. Generate visualizations saved to the `results` directory
5. Make a prediction for the next trading day

## Performance

The model achieves approximately 53% accuracy on test data, which is slightly better than random guessing for market direction prediction.

## Disclaimer

This model is for educational purposes only and should not be used for actual trading decisions. Financial markets are complex and unpredictable, and past performance does not guarantee future results. 