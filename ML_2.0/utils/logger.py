"""
Logging utilities for ML 2.0 Trading Predictor
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LOGS_DIR, LOG_LEVEL

def setup_logging():
    """Set up logging for the application"""
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Set up log file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"ml_trading_{timestamp}.log"
    
    # Configure logging
    log_level = getattr(logging, LOG_LEVEL)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log startup message
    logging.info(f"ML 2.0 Trading Predictor started at {datetime.now().isoformat()}")
    logging.info(f"Logging to {log_file}")
    
    return logging.getLogger()

def get_logger(name):
    """
    Get a logger with the specified name
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class ModelTrainingLogger:
    """
    Logger specifically for model training progress
    
    This class provides methods to log training progress,
    performance metrics, and other training-related information.
    """
    
    def __init__(self, model_name, model_type):
        """
        Initialize the model training logger
        
        Args:
            model_name: Name of the model being trained
            model_type: Type of model (neural_network, gradient_boosting, etc.)
        """
        self.logger = logging.getLogger(f"model_training.{model_type}.{model_name}")
        self.model_name = model_name
        self.model_type = model_type
        
        # Set up a separate file handler for this model
        log_file = LOGS_DIR / f"{model_type}_{model_name}_training.log"
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Log training start
        self.logger.info(f"Started training {model_type} model: {model_name}")
        
    def log_epoch(self, epoch, metrics):
        """
        Log metrics for a training epoch
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch}: {metrics_str}")
        
    def log_evaluation(self, metrics):
        """
        Log final evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        self.logger.info("Final evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value}")
            
    def log_training_complete(self, training_time, model_path):
        """
        Log completion of model training
        
        Args:
            training_time: Training time in seconds
            model_path: Path where the model was saved
        """
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Model saved to {model_path}")
        
    def log_error(self, error):
        """
        Log a training error
        
        Args:
            error: Error message or exception
        """
        self.logger.error(f"Training error: {error}") 