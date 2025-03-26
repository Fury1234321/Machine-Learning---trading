"""
AI Agent Debugger for ML 2.0 Trading Predictor

This script runs automated tests to verify the functionality of the ML 2.0 Trading Predictor
and provides detailed information for AI agents to debug and improve the application.
"""

import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODELS_DIR, LOGS_DIR
from utils.data_processor import load_data, load_and_prepare_data
from models.neural_network.model import NeuralNetworkModel
from models.gradient_boosting.model import GradientBoostingModel
from models.tree.model import DecisionTreeModel

class AIAgentDebugger:
    """
    AI Agent Debugger for diagnosing and testing the ML 2.0 Trading Predictor
    """
    
    def __init__(self):
        self.results = {
            "environment_check": {},
            "data_processing": {},
            "model_training": {},
            "overall_status": "Not Started"
        }
        
    def run_all_tests(self):
        """Run all tests and collect results"""
        try:
            print("Starting AI Agent Debugger for ML 2.0 Trading Predictor\n")
            
            # Check environment
            print("Checking environment...")
            self.check_environment()
            
            # Check data processing
            print("\nChecking data processing...")
            self.check_data_processing()
            
            # Check model training
            print("\nChecking model training...")
            self.check_model_training()
            
            # Determine overall status
            self._evaluate_overall_status()
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            print(f"\nError in AI Agent Debugger: {e}")
            traceback.print_exc()
            self.results["overall_status"] = "Error"
            
        return self.results
            
    def check_environment(self):
        """Check the application environment"""
        results = {}
        
        # Check required directories
        for dir_name in ["data", "models", "logs", 
                         "models/neural_network", 
                         "models/gradient_boosting", 
                         "models/tree"]:
            dir_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / dir_name
            exists = os.path.exists(dir_path)
            results[f"{dir_name}_directory"] = {"exists": exists, "status": "OK" if exists else "Error"}
            
            if not exists:
                print(f"  [ERROR] {dir_name} directory does not exist")
            else:
                print(f"  [OK] {dir_name} directory exists")
        
        # Check for sample data
        sample_data_exists = False
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    sample_data_exists = True
                    break
                    
        results["sample_data"] = {"exists": sample_data_exists, "status": "OK" if sample_data_exists else "Warning"}
        
        if not sample_data_exists:
            print(f"  [WARNING] No sample data files found in {DATA_DIR}")
        else:
            print(f"  [OK] Sample data files found")
        
        # Check Python imports
        try:
            # Check key dependencies
            import tensorflow
            import sklearn
            import pandas
            import numpy
            import inquirer
            results["dependencies"] = {"status": "OK"}
            print("  [OK] All required Python dependencies are installed")
        except ImportError as e:
            results["dependencies"] = {"status": "Error", "message": str(e)}
            print(f"  [ERROR] Missing Python dependencies: {e}")
        
        self.results["environment_check"] = results
        
    def check_data_processing(self):
        """Check data processing functionality"""
        results = {}
        
        # Get sample data file
        sample_files = []
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    sample_files.append(os.path.join(root, file))
        
        if not sample_files:
            results["data_loading"] = {"status": "Error", "message": "No data files found"}
            print("  [ERROR] No data files found for testing")
            self.results["data_processing"] = results
            return
        
        # Test data loading
        try:
            sample_file = sample_files[0]
            df = load_data(sample_file)
            
            if df is not None and not df.empty:
                results["data_loading"] = {"status": "OK", "rows": len(df)}
                print(f"  [OK] Successfully loaded {len(df)} rows from {os.path.basename(sample_file)}")
            else:
                results["data_loading"] = {"status": "Error", "message": "Failed to load data"}
                print(f"  [ERROR] Failed to load data from {os.path.basename(sample_file)}")
        except Exception as e:
            results["data_loading"] = {"status": "Error", "message": str(e)}
            print(f"  [ERROR] Exception when loading data: {e}")
        
        # Test data preparation
        try:
            data_dict, scalers_dict = load_and_prepare_data(sample_file)
            
            if data_dict and 'X_train' in data_dict:
                results["data_preparation"] = {
                    "status": "OK", 
                    "train_samples": data_dict['X_train'].shape[0],
                    "test_samples": data_dict['X_test'].shape[0]
                }
                print(f"  [OK] Successfully prepared data: {data_dict['X_train'].shape[0]} training samples, {data_dict['X_test'].shape[0]} test samples")
            else:
                results["data_preparation"] = {"status": "Error", "message": "Failed to prepare data"}
                print("  [ERROR] Failed to prepare data")
        except Exception as e:
            results["data_preparation"] = {"status": "Error", "message": str(e)}
            print(f"  [ERROR] Exception when preparing data: {e}")
        
        self.results["data_processing"] = results
        
    def check_model_training(self):
        """Check model training functionality"""
        results = {}
        
        # Skip if data processing failed
        if self.results["data_processing"].get("data_preparation", {}).get("status") != "OK":
            results["status"] = "Skipped"
            print("  [SKIPPED] Model training tests due to data processing failure")
            self.results["model_training"] = results
            return
        
        # Get sample data
        sample_files = []
        for root, _, files in os.walk(DATA_DIR):
            for file in files:
                if file.endswith('.csv'):
                    sample_files.append(os.path.join(root, file))
                    
        if not sample_files:
            results["status"] = "Error"
            print("  [ERROR] No data files found for model training tests")
            self.results["model_training"] = results
            return
            
        sample_file = sample_files[0]
        data_dict, scalers_dict = load_and_prepare_data(sample_file)
        
        # Dictionary to store model training results
        model_results = {}
        
        # Check a small subset of data for faster testing
        subset_size = min(500, len(data_dict['X_train']))
        small_data_dict = {
            'X_train': data_dict['X_train'][:subset_size],
            'y_train': data_dict['y_train'][:subset_size],
            'X_test': data_dict['X_test'][:100],
            'y_test': data_dict['y_test'][:100]
        }
        
        # Test decision tree model (fastest to train)
        print("  Testing Decision Tree model...")
        try:
            dt_model = DecisionTreeModel()
            dt_model.build_model()
            training_result = dt_model.train(small_data_dict, scalers_dict)
            
            if training_result and 'metrics' in training_result:
                model_results["decision_tree"] = {
                    "status": "OK",
                    "metrics": training_result['metrics']
                }
                print(f"    [OK] Successfully trained Decision Tree model with accuracy: {training_result['metrics']['accuracy']:.4f}")
            else:
                model_results["decision_tree"] = {"status": "Error", "message": "Training failed"}
                print("    [ERROR] Failed to train Decision Tree model")
        except Exception as e:
            model_results["decision_tree"] = {"status": "Error", "message": str(e)}
            print(f"    [ERROR] Exception when training Decision Tree model: {e}")
        
        # Test gradient boosting model
        print("  Testing Gradient Boosting model...")
        try:
            gb_model = GradientBoostingModel()
            gb_model.build_model()
            training_result = gb_model.train(small_data_dict, scalers_dict)
            
            if training_result and 'metrics' in training_result:
                model_results["gradient_boosting"] = {
                    "status": "OK",
                    "metrics": training_result['metrics']
                }
                print(f"    [OK] Successfully trained Gradient Boosting model with accuracy: {training_result['metrics']['accuracy']:.4f}")
            else:
                model_results["gradient_boosting"] = {"status": "Error", "message": "Training failed"}
                print("    [ERROR] Failed to train Gradient Boosting model")
        except Exception as e:
            model_results["gradient_boosting"] = {"status": "Error", "message": str(e)}
            print(f"    [ERROR] Exception when training Gradient Boosting model: {e}")
        
        # Test neural network model with a very small number of epochs
        print("  Testing Neural Network model (with limited epochs)...")
        try:
            from config.config import NN_CONFIG
            test_config = NN_CONFIG.copy()
            test_config['epochs'] = 2  # Just test 2 epochs to save time
            
            nn_model = NeuralNetworkModel()
            nn_model.config = test_config
            nn_model.build_model()
            training_result = nn_model.train(small_data_dict, scalers_dict)
            
            if training_result and 'metrics' in training_result:
                model_results["neural_network"] = {
                    "status": "OK",
                    "metrics": training_result['metrics']
                }
                print(f"    [OK] Successfully tested Neural Network model with accuracy: {training_result['metrics']['accuracy']:.4f}")
            else:
                model_results["neural_network"] = {"status": "Error", "message": "Training failed"}
                print("    [ERROR] Failed to test Neural Network model")
        except Exception as e:
            model_results["neural_network"] = {"status": "Error", "message": str(e)}
            print(f"    [ERROR] Exception when testing Neural Network model: {e}")
        
        # Check if any models were successfully trained
        if any(result.get("status") == "OK" for result in model_results.values()):
            results["status"] = "OK"
        else:
            results["status"] = "Error"
            
        results["models"] = model_results
        self.results["model_training"] = results
    
    def _evaluate_overall_status(self):
        """Evaluate the overall status based on test results"""
        # Check if any critical errors occurred
        if (self.results["environment_check"].get("dependencies", {}).get("status") == "Error" or
            self.results["data_processing"].get("data_loading", {}).get("status") == "Error" or
            self.results["model_training"].get("status") == "Error"):
            self.results["overall_status"] = "Failed"
        elif (self.results["environment_check"].get("sample_data", {}).get("status") == "Warning"):
            self.results["overall_status"] = "Warning"
        else:
            self.results["overall_status"] = "Passed"
    
    def _print_summary(self):
        """Print a summary of the test results"""
        print("\n" + "="*80)
        print("SUMMARY OF AI AGENT DEBUGGER RESULTS")
        print("="*80)
        
        # Environment check
        env_status = all(item.get("status") == "OK" for item in self.results["environment_check"].values())
        print(f"Environment Check: {'OK' if env_status else 'Issues Found'}")
        
        # Data processing
        data_status = all(item.get("status") == "OK" for item in self.results["data_processing"].values())
        print(f"Data Processing: {'OK' if data_status else 'Issues Found'}")
        
        # Model training
        if self.results["model_training"].get("status") == "Skipped":
            print("Model Training: Skipped")
        else:
            model_results = self.results["model_training"].get("models", {})
            successful_models = sum(1 for model in model_results.values() if model.get("status") == "OK")
            total_models = len(model_results)
            print(f"Model Training: {successful_models}/{total_models} models trained successfully")
        
        # Overall status
        print(f"\nOverall Status: {self.results['overall_status']}")
        
        if self.results['overall_status'] != "Passed":
            print("\nRecommendations for AI agents:")
            
            if self.results["environment_check"].get("dependencies", {}).get("status") == "Error":
                print("- Fix dependency issues. Check requirements.txt and verify all imports.")
                
            if self.results["environment_check"].get("sample_data", {}).get("status") == "Warning":
                print("- Add sample data files to the data directory for testing.")
                
            if self.results["data_processing"].get("data_loading", {}).get("status") == "Error":
                print("- Debug data loading issues. Check file paths and data format.")
                
            if self.results["data_processing"].get("data_preparation", {}).get("status") == "Error":
                print("- Fix data preparation. Check feature engineering and preprocessing steps.")
                
            if self.results["model_training"].get("status") == "Error":
                print("- Debug model training issues. Check model architecture and training parameters.")
        
        print("="*80)

def run():
    """Run the AI agent debugger"""
    debugger = AIAgentDebugger()
    return debugger.run_all_tests()

if __name__ == "__main__":
    run() 