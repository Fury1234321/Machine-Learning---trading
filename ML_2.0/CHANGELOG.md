# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2023-03-27

### Added
- User customizable epoch count for all model types before training starts
- Standardized green progress bars with real-time metrics for all models
- Enhanced training visualization with detailed metrics across all model types
- Live training feedback including accuracy, elapsed time and ETA
- Consistent UI feedback for all training processes

### Improved
- Gradient Boosting model now shows incremental progress during training
- Decision Tree model now provides better visual feedback during training
- More detailed terminal output with colored success/error messages
- Improved training parameter customization

## [1.0.1] - 2023-03-26

### Fixed
- Improved data processing to handle different column name formats (uppercase/lowercase)
- Fixed Neural Network model to properly handle TensorFlow input shapes
- Added error handling to model training processes
- Added feature name tracking across all model types
- Ensured directories are created before saving models

### Added
- Added macOS-specific START_HERE.command file with clear instructions
- Enhanced input validation in data processing

## [1.0.0] - 2023-03-26

### Added
- Initial release of ML 2.0 Trading Predictor
- Core architecture with modular components
- Support for three model types:
  - Neural Networks (LSTM-based)
  - Gradient Boosting
  - Decision Trees
- Data preprocessing pipeline with feature engineering
- Terminal-based user interface
- Model training and management functionality
- AI agent debugger for automated testing
- Cross-platform startup scripts for Windows and Unix/Linux 