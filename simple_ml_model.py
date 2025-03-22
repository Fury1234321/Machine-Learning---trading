#!/usr/bin/env python3
"""
Simple ML Trading Model

This script:
1. Downloads NASDAQ 100 data
2. Creates basic technical features (returns, SMAs, RSI)
3. Trains a RandomForest classifier to predict market direction
4. Evaluates model performance 
5. Makes a prediction for the next trading day
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Create results directory
os.makedirs('results', exist_ok=True)

print("Simple ML Trading Model")

# 1. DOWNLOAD DATA
print("Downloading NASDAQ 100 data...")
ticker = "^NDX"
data = yf.download(ticker, start="2020-01-01")
print(f"Downloaded {len(data)} rows of data for {ticker}")

# 2. FEATURE ENGINEERING - Keep it simple
print("Creating features...")
# Returns
data['Return'] = data['Close'].pct_change()
data['Return_Lag1'] = data['Return'].shift(1)
data['Return_Lag2'] = data['Return'].shift(2)

# Simple moving averages
data['SMA10'] = data['Close'].rolling(window=10).mean()
data['SMA50'] = data['Close'].rolling(window=50).mean()

# SMA Crossover
data['SMA_Ratio'] = data['SMA10'] / data['SMA50']

# Simple RSI (14-day)
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
ma_up = up.rolling(window=14).mean()
ma_down = down.rolling(window=14).mean()
data['RSI'] = 100 - (100 / (1 + ma_up / ma_down))

# Target: 1 if price goes up next day, 0 if down
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Drop NaN values
data = data.dropna()
print(f"Data after feature creation: {len(data)} rows")

# 3. TRAIN/TEST SPLIT
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]
print(f"Training data: {len(train_data)} rows")
print(f"Testing data: {len(test_data)} rows")

# 4. PREPARE FEATURES
feature_columns = ['Return', 'Return_Lag1', 'Return_Lag2', 'SMA_Ratio', 'RSI']

X_train = train_data[feature_columns]
y_train = train_data['Target']
X_test = test_data[feature_columns]
y_test = test_data['Target']

# 5. TRAIN MODEL
print("Training machine learning model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. MAKE PREDICTIONS
print("Making predictions...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# 7. EVALUATE MODEL
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\n----- Model Performance -----")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print("\nClassification Report (Test Data):")
print(classification_report(y_test, test_predictions))

# 8. FEATURE IMPORTANCE
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n----- Feature Importance -----")
print(feature_importance)

# 9. VISUALIZE RESULTS
plt.figure(figsize=(12, 8))

# Plot 1: Predicted vs Actual on test data
plt.subplot(2, 1, 1)
plt.plot(test_data.index, test_data['Close'], label='NASDAQ 100')
up_days = test_data[test_predictions == 1].index
down_days = test_data[test_predictions == 0].index
plt.scatter(up_days, test_data.loc[up_days, 'Close'], color='green', marker='^', label='Predicted Up')
plt.scatter(down_days, test_data.loc[down_days, 'Close'], color='red', marker='v', label='Predicted Down')
plt.title('Model Predictions on Test Data')
plt.legend()

# Plot 2: Feature importance
plt.subplot(2, 1, 2)
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')

plt.tight_layout()
plt.savefig('results/ml_model_results.png')
print("Chart saved to results/ml_model_results.png")

# 10. LATEST PREDICTION
latest_data = data.iloc[-1:][feature_columns]
latest_prediction = model.predict(latest_data)[0]
latest_proba = model.predict_proba(latest_data)[0]

# Extract values to avoid Series formatting issues - using iloc[0] to avoid the warning
last_date = data.index[-1].strftime('%Y-%m-%d')
last_close = data['Close'].iloc[-1].item()  # Using .item() to get the scalar value
last_return = data['Return'].iloc[-1].item()
last_rsi = data['RSI'].iloc[-1].item()

print("\n----- Latest Prediction -----")
print(f"Date: {last_date}")
print(f"Closing Price: ${last_close:.2f}")
print(f"Current Return: {last_return:.4f}")
print(f"Current RSI: {last_rsi:.2f}")

if latest_prediction == 1:
    print(f"Prediction: UP with {latest_proba[1]:.2%} confidence")
else:
    print(f"Prediction: DOWN with {latest_proba[0]:.2%} confidence")

print("\nModel saved to results/ml_model_results.png")
print("Done!") 