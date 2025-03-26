"""
Neural Network model implementation for ML 2.0 Trading Predictor
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import datetime
import pickle
from typing import Dict, Tuple, Any, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import NN_CONFIG, LOOKBACK_WINDOW, MODELS_DIR, PRICE_FEATURES, TECHNICAL_INDICATORS

class NeuralNetworkModel:
    """Neural Network model for predicting price direction"""
    
    def __init__(self):
        self.model = None
        self.config = NN_CONFIG
        self.feature_count = len(PRICE_FEATURES) + len(TECHNICAL_INDICATORS)
        self.model_name = f"nn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_path = MODELS_DIR / "neural_network" / f"{self.model_name}.h5"
        self.metadata_path = MODELS_DIR / "neural_network" / f"{self.model_name}_metadata.pkl"
        
    def build_model(self, input_shape: Tuple[int, int] = None) -> None:
        """
        Build the neural network model architecture
        
        Args:
            input_shape: Optional tuple of (sequence_length, feature_count)
                         If not provided, uses (LOOKBACK_WINDOW, self.feature_count)
        """
        if input_shape is None:
            input_shape = (LOOKBACK_WINDOW, self.feature_count)
        else:
            # Update feature count based on the provided shape
            self.feature_count = input_shape[1]
            
        model = Sequential()
        
        # First use Input layer to define the input shape
        model.add(Input(shape=input_shape))
        
        # LSTM layer to process the time series data
        model.add(LSTM(
            units=self.config['hidden_layers'][0],
            return_sequences=True
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout_rate']))
        
        # Additional LSTM layer
        model.add(LSTM(
            units=self.config['hidden_layers'][1],
            return_sequences=False
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        model.add(Dense(
            units=self.config['hidden_layers'][2],
            activation=self.config['activation']
        ))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Output layer (sigmoid for binary classification)
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=self.config['optimizer'],
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Neural Network model built successfully")
        
    def train(self, data_dict: Dict[str, np.ndarray], scalers_dict: Dict[str, Any], 
              early_stopping: bool = True) -> Dict[str, Any]:
        """
        Train the neural network model
        
        Args:
            data_dict: Dictionary containing X_train, X_test, y_train, y_test
            scalers_dict: Dictionary containing the fitted scalers
            early_stopping: Whether to use early stopping
            
        Returns:
            Dictionary with training history and metrics
        """
        try:
            X_train = data_dict['X_train']
            y_train = data_dict['y_train']
            X_test = data_dict['X_test']
            y_test = data_dict['y_test']
            
            # Get actual input shape from data
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            # Either build the model or verify compatibility
            if self.model is None:
                self.build_model(input_shape)
            else:
                # Check if the existing model is compatible with the data
                expected_shape = self.model.input_shape[1:]
                if expected_shape != input_shape:
                    print(f"Rebuilding model: Input shape mismatch. Expected {expected_shape}, got {input_shape}")
                    self.build_model(input_shape)
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            callbacks = []
            
            # Early stopping to prevent overfitting
            if early_stopping:
                callbacks.append(EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                ))
                
            # Save best model during training
            callbacks.append(ModelCheckpoint(
                filepath=str(self.model_path),
                monitor='val_loss',
                save_best_only=True
            ))
            
            print(f"Training Neural Network model with {X_train.shape[0]} samples, shape: {input_shape}")
            
            # Train the model
            start_time = time.time()
            history = self.model.fit(
                X_train, y_train,
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_split=0.2,
                callbacks=callbacks,
                verbose=1
            )
            training_time = time.time() - start_time
            
            # Evaluate on test set
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            
            # Predictions for more detailed metrics
            y_pred_prob = self.model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'training_time': training_time
            }
            
            # Store feature names if available
            feature_names = data_dict.get('feature_names', [])
            
            # Save model metadata
            metadata = {
                'model_name': self.model_name,
                'config': self.config,
                'feature_count': self.feature_count,
                'feature_names': feature_names,
                'metrics': metrics,
                'training_date': datetime.datetime.now().isoformat(),
                'training_time': training_time,
                'scalers': scalers_dict,
                'input_shape': input_shape
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            return {
                'history': history.history,
                'metrics': metrics,
                'model_name': self.model_name
            }
        except Exception as e:
            print(f"Error training neural network model: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features (should be properly shaped and scaled)
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train() first.")
            
        return self.model.predict(X)
    
    def save(self) -> None:
        """
        Save the trained model to disk
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def load(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model file
            metadata_path: Path to the model metadata file
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.model_path = model_path
            
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.config = metadata.get('config', self.config)
                    self.feature_count = metadata.get('feature_count', self.feature_count)
                    self.model_name = metadata.get('model_name', self.model_name)
                    
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    @staticmethod
    def list_available_models() -> pd.DataFrame:
        """
        List all available trained neural network models
        
        Returns:
            DataFrame with model information
        """
        model_dir = MODELS_DIR / "neural_network"
        if not os.path.exists(model_dir):
            return pd.DataFrame(columns=['model_name', 'training_date', 'accuracy', 'f1'])
            
        models = []
        for file in os.listdir(model_dir):
            if file.endswith('_metadata.pkl'):
                try:
                    with open(os.path.join(model_dir, file), 'rb') as f:
                        metadata = pickle.load(f)
                        models.append({
                            'model_name': metadata.get('model_name', 'Unknown'),
                            'training_date': metadata.get('training_date', 'Unknown'),
                            'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                            'f1': metadata.get('metrics', {}).get('f1', 0),
                        })
                except Exception as e:
                    print(f"Error loading model metadata {file}: {e}")
                    
        return pd.DataFrame(models) 