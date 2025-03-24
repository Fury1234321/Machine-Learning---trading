# Changelog

All notable changes to ML Trading Predictor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-03-24

### Fixed
- Model compatibility issues when loading saved models
- Ensemble model format standardization to ensure proper predict methods
- Error handling for model loading and prediction failures
- Graceful fallback to trend-based prediction when models are unavailable
- Backward compatibility with older model formats

### Added
- Comprehensive error handling throughout the model training pipeline
- Improved model format detection and conversion
- Resilient predictions against data format issues
- README documentation of recent model format improvements

## [1.0.0] - 2025-03-22

### Added
- Initial release
- Interactive terminal interface with visual feedback
- Multiple ML models (Random Forest, Gradient Boosting, SVM, Neural Network, Ensemble)
- Future candle prediction with visualization
- Model comparison functionality
- Trading signals (Buy/Sell/Hold) with confidence levels
- Technical indicator calculation (50+ indicators)
- Streamlined user experience with quick start options
- Default AAPL stock data for demo purposes 