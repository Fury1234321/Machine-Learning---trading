# ML 2.0 Trading Predictor - User Interface Guide

## 1. Application Overview

ML 2.0 Trading Predictor features a streamlined terminal-based user interface designed for clarity and efficiency. This guide explains how to navigate the interface and make the most of its features.

## 2. Getting Started

### 2.1 Launching the Application
- **macOS/Linux**: Run `./start.sh` or double-click `START_HERE.command`
- **Windows**: Double-click `start.bat`

The startup script will:
- Create a virtual environment if needed
- Install required dependencies
- Launch the main interface

### 2.2 Main Menu
The main menu offers three primary options:
1. **View available models** - Browse existing trained models
2. **Train a new model** - Create and train a new machine learning model
3. **Exit** - Close the application

## 3. Viewing Available Models

When selecting "View available models", the system will:
1. Scan model directories for all trained models
2. Display a table with key information:
   - Model name
   - Model type (Neural Network, Gradient Boosting, Decision Tree)
   - Training date
   - Accuracy metrics (accuracy percentage, F1 score)

Models are sorted by training date with the most recent models appearing first.

## 4. Training a New Model

### 4.1 Model Selection
When you choose "Train a new model", you'll be prompted to select a model type:
- **Neural Network** - Deep learning model for complex patterns
- **Gradient Boosting** - High-accuracy ensemble model
- **Decision Tree** - Simple, interpretable model
- **Back to main menu** - Return to the main menu

Each option includes a brief description of its strengths and use cases.

### 4.2 Data Selection
After selecting a model type, you'll be prompted to choose a data file:
- The system scans the `data/` directory and its subdirectories for CSV files
- Each file is listed with its name and size in MB
- Selecting a file will use it for training
- "Back to main menu" option returns to the main menu

### 4.3 Training Parameter Customization
Before training begins, you can customize key training parameters:
- **Number of epochs** - Enter the desired number of training epochs/iterations
  - Leave blank to use default settings from the configuration file
  - For Neural Networks: Actual training epochs
  - For Gradient Boosting: Number of estimators (trees)
  - For Decision Trees: Used to adjust the max_depth parameter

### 4.4 Training Progress Visualization
During training, all models provide detailed real-time feedback:

#### Neural Network Training Display
- Green progress bar showing current epoch/total epochs
- Training loss and accuracy metrics updated per epoch
- Validation metrics when available
- Elapsed time and estimated time remaining

#### Gradient Boosting Training Display
- Green progress bar showing estimator building progress
- Current/total estimators count
- Training accuracy updated during training
- Elapsed time and estimated time remaining

#### Decision Tree Training Display
- Green progress bar showing training progress
- Max depth and other hyperparameters
- Training and testing accuracy
- Elapsed time for training

#### Training Completion
Once training is complete:
- Final metrics (accuracy, precision, recall, F1 score) are displayed
- The model is automatically saved
- The model immediately appears in the available models list

### 4.5 Training Interruption
Training can be interrupted safely by pressing `Ctrl+C`. The system will:
- Stop the training process
- Save the model in its current state if possible
- Return to the main menu

## 5. UI Elements and Color Coding

### 5.1 Color Scheme
The interface uses color coding for clarity:
- **Cyan**: Section headers and titles
- **Green**: Success messages, progress bars, and metrics
- **Yellow**: Secondary headers and section dividers
- **Red**: Error messages and warnings
- **Blue**: Information messages and instructions

### 5.2 Progress Bars
Progress bars follow a consistent format across all model types:
```
Training: |████████████████████--------------------| 50.0% Complete. Metrics: ...
```

The progress bar includes:
- Visual representation of progress (filled and unfilled segments)
- Percentage complete
- Relevant metrics (varies by model type)
- Time information (elapsed and estimated remaining)

## 6. Best Practices

### 6.1 Data Preparation
- Ensure CSV files have the proper format with OHLC price data and volume
- Larger datasets provide better training but take longer to process
- Sample files are provided in the `data/samples/` directory

### 6.2 Model Selection Tips
- **Neural Networks**: Best for complex patterns, but take longer to train
- **Gradient Boosting**: Good balance of accuracy and training speed
- **Decision Trees**: Fastest to train, but may have lower accuracy

### 6.3 Training Parameter Guidelines
- Start with default epochs and adjust based on results
- For Neural Networks: 50-100 epochs is typically sufficient
- For Gradient Boosting: 100-200 estimators is a good starting point
- Large epoch counts increase training time but may improve accuracy

## 7. Troubleshooting

### 7.1 Common Issues
- **Slow training**: Reduce the number of epochs or use a smaller dataset
- **Low accuracy**: Try a different model type or increase the number of epochs
- **Data errors**: Ensure CSV files follow the expected format

### 7.2 Error Messages
The system provides detailed error messages to help diagnose issues. Common errors include:
- Missing data files
- Invalid data format
- Memory issues with large datasets

## 8. Customization and Extension

This interface is designed to be extended. When adding features:
1. Follow the established color scheme and formatting
2. Maintain consistent navigation patterns
3. Update the CHANGELOG.md file with your changes
4. Ensure all features have appropriate error handling

## 9. Version Compatibility

This guide is for ML 2.0 Trading Predictor v1.0.2 and later versions. 