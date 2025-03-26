"""
Terminal-based user interface for ML 2.0 Trading Predictor
"""

import os
import sys
import time
import inquirer
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DATA_DIR, MODELS_DIR, TERMINAL_WIDTH, SHOW_PROGRESS_BAR
from utils.data_processor import load_data
from models.neural_network.model import NeuralNetworkModel
from models.gradient_boosting.model import GradientBoostingModel
from models.tree.model import DecisionTreeModel

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-' * len(text)}{Colors.ENDC}")

def print_section(text: str):
    """Print a formatted section header"""
    print(f"\n{Colors.YELLOW}{text}{Colors.ENDC}")
    print(f"{Colors.YELLOW}{'-' * len(text)}{Colors.ENDC}")

def print_success(text: str):
    """Print a success message"""
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")

def print_error(text: str):
    """Print an error message"""
    print(f"{Colors.RED}{text}{Colors.ENDC}")

def print_info(text: str):
    """Print an info message"""
    print(f"{Colors.BLUE}{text}{Colors.ENDC}")

def progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50, fill: str = '█'):
    """
    Display a progress bar in the terminal
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Bar length
        fill: Bar fill character
    """
    if not SHOW_PROGRESS_BAR:
        return
        
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    # Print new line on complete
    if iteration == total:
        print()

def get_data_files() -> List[str]:
    """
    Get list of available data files
    
    Returns:
        List of data file paths
    """
    data_files = []
    
    # Check main data directory
    if os.path.exists(DATA_DIR):
        for file in os.listdir(DATA_DIR):
            if file.endswith(".csv"):
                data_files.append(str(DATA_DIR / file))
    
    # Check samples directory
    samples_dir = DATA_DIR / "samples"
    if os.path.exists(samples_dir):
        for file in os.listdir(samples_dir):
            if file.endswith(".csv"):
                data_files.append(str(samples_dir / file))
    
    return data_files

def model_selection_menu() -> str:
    """
    Display model selection menu
    
    Returns:
        Selected model type
    """
    questions = [
        inquirer.List('model_type',
                     message="Select the type of model to train",
                     choices=[
                         ('Neural Network (Deep Learning for complex patterns)', 'neural_network'),
                         ('Gradient Boosting (High accuracy, handles non-linear patterns)', 'gradient_boosting'),
                         ('Decision Tree (Simple, interpretable model)', 'tree'),
                         ('Back to main menu', 'back')
                     ],
                     ),
    ]
    
    answer = inquirer.prompt(questions)
    return answer['model_type']

def data_selection_menu() -> str:
    """
    Display data selection menu
    
    Returns:
        Selected data file path
    """
    data_files = get_data_files()
    
    if not data_files:
        print_error("No data files found. Please add CSV files to the data directory.")
        return ""
    
    # Format choices with file sizes
    choices = []
    for file_path in data_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        
        # Load first few rows to get data points count
        try:
            df = pd.read_csv(file_path, nrows=5)
            choices.append((f"{file_name} ({file_size:.2f} MB)", file_path))
        except:
            choices.append((f"{file_name} ({file_size:.2f} MB) - Error loading", file_path))
    
    choices.append(('Back to main menu', 'back'))
    
    questions = [
        inquirer.List('data_file',
                     message="Select a data file to use",
                     choices=choices,
                     ),
    ]
    
    answer = inquirer.prompt(questions)
    return answer['data_file']

def show_available_models(model_type: str = None) -> pd.DataFrame:
    """
    Display available trained models
    
    Args:
        model_type: Optional type of models to show ('neural_network', 'gradient_boosting', 'tree')
        
    Returns:
        DataFrame with model information
    """
    models_df = pd.DataFrame(columns=['model_name', 'model_type', 'training_date', 'accuracy', 'f1'])
    
    # Get neural network models
    if model_type in [None, 'neural_network']:
        nn_models = NeuralNetworkModel.list_available_models()
        if not nn_models.empty:
            nn_models['model_type'] = 'Neural Network'
            models_df = pd.concat([models_df, nn_models])
    
    # Get gradient boosting models
    if model_type in [None, 'gradient_boosting']:
        gb_models = GradientBoostingModel.list_available_models()
        if not gb_models.empty:
            gb_models['model_type'] = 'Gradient Boosting'
            models_df = pd.concat([models_df, gb_models])
    
    # Get decision tree models
    if model_type in [None, 'tree']:
        dt_models = DecisionTreeModel.list_available_models()
        if not dt_models.empty:
            dt_models['model_type'] = 'Decision Tree'
            models_df = pd.concat([models_df, dt_models])
    
    if models_df.empty:
        if model_type:
            print_info(f"No {model_type.replace('_', ' ').title()} models found.")
        else:
            print_info("No trained models found.")
        return models_df
    
    # Sort by training date (most recent first)
    models_df = models_df.sort_values('training_date', ascending=False)
    
    # Display models
    print_section("Available Models")
    
    # Format for display
    display_df = models_df.copy()
    
    # Truncate model name if too long
    display_df['model_name'] = display_df['model_name'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    
    # Format dates
    display_df['training_date'] = pd.to_datetime(display_df['training_date']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Format metrics as percentages
    display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x*100:.2f}%")
    display_df['f1'] = display_df['f1'].apply(lambda x: f"{x*100:.2f}%")
    
    print(display_df.to_string(index=False))
    print()
    
    return models_df

def train_model(model_type: str, data_file: str) -> Dict[str, Any]:
    """
    Train a new model
    
    Args:
        model_type: Type of model to train ('neural_network', 'gradient_boosting', 'tree')
        data_file: Path to data file
        
    Returns:
        Dictionary with training results
    """
    from utils.data_processor import load_and_prepare_data
    
    print_header(f"Training {model_type.replace('_', ' ').title()} Model")
    print_info(f"Data file: {os.path.basename(data_file)}")
    
    # Load and prepare data
    print_info("Loading and preparing data...")
    data_dict, scalers_dict = load_and_prepare_data(data_file)
    
    if not data_dict:
        print_error("Failed to prepare data.")
        return {}
    
    print_success(f"Data prepared successfully. Training set size: {data_dict['X_train'].shape[0]} samples")
    
    # Initialize and train model based on type
    if model_type == 'neural_network':
        model = NeuralNetworkModel()
        print_info("Building neural network model...")
        model.build_model()
        
        # Train with progress reporting
        print_info("Training neural network model...")
        result = model.train(data_dict, scalers_dict)
        
    elif model_type == 'gradient_boosting':
        model = GradientBoostingModel()
        print_info("Building gradient boosting model...")
        model.build_model()
        
        # Train model
        print_info("Training gradient boosting model...")
        result = model.train(data_dict, scalers_dict)
        
    elif model_type == 'tree':
        model = DecisionTreeModel()
        print_info("Building decision tree model...")
        model.build_model()
        
        # Train model
        print_info("Training decision tree model...")
        result = model.train(data_dict, scalers_dict)
    
    else:
        print_error(f"Unknown model type: {model_type}")
        return {}
    
    # Display results
    print_success("Training completed successfully!")
    print_section("Model Performance")
    
    metrics = result.get('metrics', {})
    for metric, value in metrics.items():
        if metric in ['accuracy', 'precision', 'recall', 'f1']:
            print(f"{metric.capitalize()}: {value*100:.2f}%")
        elif metric == 'training_time':
            print(f"Training time: {value:.2f} seconds")
    
    return result

def main_menu():
    """Display the main menu and handle user interactions"""
    while True:
        clear_screen()
        print_header("ML 2.0 Trading Predictor")
        
        questions = [
            inquirer.List('action',
                         message="What would you like to do?",
                         choices=[
                             ('View available models', 'view_models'),
                             ('Train a new model', 'train_model'),
                             ('Exit', 'exit')
                         ],
                         ),
        ]
        
        answer = inquirer.prompt(questions)
        
        if answer['action'] == 'exit':
            print_info("Goodbye!")
            break
            
        elif answer['action'] == 'view_models':
            clear_screen()
            show_available_models()
            input("\nPress Enter to continue...")
            
        elif answer['action'] == 'train_model':
            # Model selection submenu
            model_type = model_selection_menu()
            
            if model_type == 'back':
                continue
                
            # Data selection submenu
            data_file = data_selection_menu()
            
            if data_file == 'back':
                continue
                
            if data_file:
                # Train the selected model
                train_model(model_type, data_file)
                input("\nPress Enter to continue...")

def run():
    """Run the terminal UI"""
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        print_error(f"An error occurred: {e}")
        
if __name__ == "__main__":
    run() 