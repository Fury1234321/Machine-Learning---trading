"""
Decision Tree model implementation for ML 2.0 Trading Predictor
"""

import os
import sys
import numpy as np
import pandas as pd
import time
import datetime
import pickle
from typing import Dict, Tuple, Any, Optional
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.config import DT_CONFIG, MODELS_DIR

class DecisionTreeModel:
    """Decision Tree model for predicting price direction"""
    
    def __init__(self):
        self.model = None
        self.config = DT_CONFIG
        self.model_name = f"dt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_path = MODELS_DIR / "tree" / f"{self.model_name}.pkl"
        self.metadata_path = MODELS_DIR / "tree" / f"{self.model_name}_metadata.pkl"
        self.feature_names = []
        
    def build_model(self) -> None:
        """
        Build the decision tree model
        """
        self.model = DecisionTreeClassifier(
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            random_state=self.config['random_state']
        )
        
        print("Decision Tree model built successfully")
        
    def preprocess_data(self, data_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data for decision tree model
        
        Args:
            data_dict: Dictionary containing X_train, X_test, y_train, y_test
            
        Returns:
            Tuple of (X_train_flat, X_test_flat, y_train, y_test)
        """
        # For decision tree, we need to flatten the 3D sequences to 2D
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # Store feature names if available
        if 'feature_names' in data_dict:
            self.feature_names = data_dict['feature_names']
        
        # Reshape from (samples, sequence_length, features) to (samples, sequence_length * features)
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        return X_train_flat, X_test_flat, y_train, y_test
        
    def train(self, data_dict: Dict[str, np.ndarray], scalers_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the decision tree model
        
        Args:
            data_dict: Dictionary containing X_train, X_test, y_train, y_test
            scalers_dict: Dictionary containing the fitted scalers
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Ensure directories exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            if self.model is None:
                self.build_model()
                
            # Preprocess data for decision tree
            X_train_flat, X_test_flat, y_train, y_test = self.preprocess_data(data_dict)
            
            print(f"Training Decision Tree model with {X_train_flat.shape[0]} samples and {X_train_flat.shape[1]} features")
            print(f"Max depth: {self.config['max_depth']}")
            
            # Use ANSI escape codes for colored output
            GREEN = '\033[92m'
            ENDC = '\033[0m'
            BOLD = '\033[1m'
            
            # Show a simulated progress bar
            # Decision trees don't train in epochs, but we'll show progress
            # during the training process to be consistent with other models
            
            # Start training with progress simulation
            start_time = time.time()
            
            # Show initial progress bar at 0%
            bar_length = 40
            bar = '-' * bar_length
            print(f"\r{GREEN}Training: |{bar}| 0.0% Complete. Initializing...{ENDC}", end='')
            
            # Simulate 10 steps of progress
            steps = 10
            for i in range(steps):
                # Sleep for a short time to simulate computation
                time.sleep(0.1)
                
                # Calculate progress for display
                progress = (i + 1) / steps
                percent = progress * 100
                
                # Generate progress bar
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                # Print progress bar with metrics
                print(f"\r{GREEN}Training: |{bar}| {percent:.1f}% Complete. "
                      f"Preparing training... "
                      f"Elapsed: {time.time() - start_time:.1f}s{ENDC}", end='')
                
            # Now fit the model
            # Save the real training start time
            actual_train_start = time.time()
            self.model.fit(X_train_flat, y_train)
            train_time = time.time() - actual_train_start
            
            # Show 100% progress bar
            bar = '█' * bar_length
            print(f"\r{GREEN}Training: |{bar}| 100.0% Complete. "
                 f"Training finished in {train_time:.2f}s{ENDC}")
            
            # Total time including simulation
            training_time = time.time() - start_time
            
            # Evaluate on train and test sets
            train_pred = self.model.predict(X_train_flat)
            train_acc = accuracy_score(y_train, train_pred) * 100
            
            # Final evaluation on test set
            y_pred = self.model.predict(X_test_flat)
            test_acc = accuracy_score(y_test, y_pred) * 100
            
            # Print training metrics
            print(f"{GREEN}Train accuracy: {train_acc:.2f}%{ENDC}")
            print(f"{GREEN}Test accuracy: {test_acc:.2f}%{ENDC}")
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'training_time': training_time
            }
            
            # Print final metrics with colors
            print(f"{GREEN}{BOLD}Training complete!{ENDC}")
            print(f"{GREEN}Test F1 score: {metrics['f1']*100:.2f}%{ENDC}")
            
            # Feature importance
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = self.model.feature_importances_
            
            # Input shape information
            input_shape = {
                'original': (data_dict['X_train'].shape[1], data_dict['X_train'].shape[2]),
                'flattened': X_train_flat.shape[1]
            }
            
            # Save model metadata
            metadata = {
                'model_name': self.model_name,
                'config': self.config,
                'metrics': metrics,
                'training_date': datetime.datetime.now().isoformat(),
                'training_time': training_time,
                'feature_importance': feature_importance,
                'feature_names': self.feature_names,
                'input_shape': input_shape,
                'scalers': scalers_dict
            }
            
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
                
            # Save model
            self.save()
                
            return {
                'metrics': metrics,
                'model_name': self.model_name,
                'feature_importance': feature_importance
            }
        except Exception as e:
            print(f"Error training decision tree model: {str(e)}")
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
            
        # Flatten the input if it's 3D
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)
            
        return self.model.predict_proba(X)[:, 1]  # Return probability of class 1
    
    def save(self) -> None:
        """
        Save the trained model to disk
        """
        if self.model is None:
            raise ValueError("No model to save. Build and train a model first.")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        print(f"Model saved to {self.model_path}")
        
    def load(self, model_path: str, metadata_path: Optional[str] = None) -> None:
        """
        Load a trained model from disk
        
        Args:
            model_path: Path to the saved model file
            metadata_path: Path to the model metadata file
        """
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        self.model_path = model_path
        
        if metadata_path and os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.config = metadata.get('config', self.config)
                self.model_name = metadata.get('model_name', self.model_name)
                self.feature_names = metadata.get('feature_names', [])
                
        print(f"Model loaded from {model_path}")
        
    @staticmethod
    def list_available_models() -> pd.DataFrame:
        """
        List all available trained decision tree models
        
        Returns:
            DataFrame with model information
        """
        model_dir = MODELS_DIR / "tree"
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