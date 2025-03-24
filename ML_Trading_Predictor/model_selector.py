#!/usr/bin/env python3
"""
Model Selector Module for ML Trading Predictor

This module provides a command-line interface for selecting ML models
with detailed pros and cons for each model type.
"""

import os
import sys
import pandas as pd
from tabulate import tabulate

# Model information dictionary containing pros and cons
MODEL_INFO = {
    'rf': {
        'name': 'Random Forest',
        'description': 'Ensemble learning method using multiple decision trees',
        'pros': [
            'Handles non-linear relationships well',
            'Less prone to overfitting than single decision trees',
            'Can handle high-dimensional data',
            'Provides feature importance metrics',
            'Works well with categorical and numerical features'
        ],
        'cons': [
            'Can be computationally expensive with large datasets',
            'Less interpretable than simple models',
            'May struggle with very highly correlated features',
            'Tends to bias towards features with more levels'
        ],
        'best_for': 'General-purpose trading prediction with multiple technical indicators'
    },
    'gb': {
        'name': 'Gradient Boosting',
        'description': 'Sequential ensemble method that builds trees to correct errors',
        'pros': [
            'Often achieves higher accuracy than Random Forest',
            'Works well with complex relationships in data',
            'Can capture subtle patterns in price movements',
            'Provides feature importance metrics',
            'Can be fine-tuned with many hyperparameters'
        ],
        'cons': [
            'More prone to overfitting than Random Forest',
            'Requires more careful parameter tuning',
            'Computationally intensive, especially with large datasets',
            'Sequential nature makes it harder to parallelize'
        ],
        'best_for': 'Capturing complex market patterns when you have time to tune parameters'
    },
    'svm': {
        'name': 'Support Vector Machine',
        'description': 'Finds optimal hyperplane to separate classes',
        'pros': [
            'Works well in high-dimensional spaces',
            'Effective with clear margin of separation',
            'Memory efficient as it uses subset of training points',
            'Versatile through different kernel functions',
            'Less prone to overfitting in high dimensional spaces'
        ],
        'cons': [
            'Not suitable for large datasets (slow training)',
            'Sensitive to feature scaling',
            'Less interpretable than tree-based models',
            'Parameter selection can be challenging',
            'Doesn\'t directly provide probability estimates'
        ],
        'best_for': 'Trend classification with well-defined market regimes'
    },
    'nn': {
        'name': 'Neural Network',
        'description': 'Multi-layer perceptron for complex pattern recognition',
        'pros': [
            'Can model highly complex non-linear patterns',
            'Adaptable to various types of data and problems',
            'Able to learn hierarchical feature representations',
            'Can update incrementally with new data',
            'Works well with large datasets'
        ],
        'cons': [
            'Black box with limited interpretability',
            'Requires large amounts of data for best performance',
            'Sensitive to feature scaling',
            'Risk of overfitting with small datasets',
            'Computationally intensive to train'
        ],
        'best_for': 'Complex pattern recognition in large datasets with many features'
    },
    'ensemble': {
        'name': 'Ensemble of Models',
        'description': 'Combines multiple models (RF, GB, SVM) for better predictions',
        'pros': [
            'Usually more accurate than any single model',
            'More robust to different market conditions',
            'Reduces risk of overfitting',
            'Smooths out prediction errors',
            'Combines strengths of multiple approaches'
        ],
        'cons': [
            'Computationally most expensive option',
            'Less interpretable than single models',
            'More complex to implement and maintain',
            'May not always outperform the best single model',
            'Slower prediction time'
        ],
        'best_for': 'Production systems where robustness and accuracy are critical'
    }
}

def display_model_comparison():
    """Display a table comparing all available models"""
    headers = ["Model", "Description", "Best For"]
    table_data = []
    
    for key, info in MODEL_INFO.items():
        table_data.append([
            info['name'],
            info['description'],
            info['best_for']
        ])
    
    print("\n=== ML Model Comparison ===\n")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print("\nUse 'select' command to see details for a specific model")

def display_model_details(model_key):
    """Display detailed information about a specific model"""
    if model_key not in MODEL_INFO:
        print(f"Model '{model_key}' not found. Available models: {', '.join(MODEL_INFO.keys())}")
        return None
    
    info = MODEL_INFO[model_key]
    
    print(f"\n=== {info['name']} ({model_key}) ===\n")
    print(f"Description: {info['description']}")
    print("\nPros:")
    for pro in info['pros']:
        print(f"  ✓ {pro}")
    
    print("\nCons:")
    for con in info['cons']:
        print(f"  ✗ {con}")
    
    print(f"\nBest For: {info['best_for']}")
    return model_key

def select_model():
    """Interactive command-line interface for model selection"""
    selected_model = None
    
    print("\n=== ML Trading Model Selector ===")
    print("This tool helps you select the best machine learning model for your trading strategy")
    
    while True:
        print("\nCommands:")
        print("  list      - Show all available models")
        print("  select X  - View details for model X (rf, gb, svm, nn, ensemble)")
        print("  choose X  - Select model X and continue")
        print("  exit      - Exit without selecting")
        
        cmd = input("\nEnter command: ").strip().lower()
        
        if cmd == "list":
            display_model_comparison()
        
        elif cmd.startswith("select "):
            model_key = cmd.split(" ")[1].lower()
            display_model_details(model_key)
        
        elif cmd.startswith("choose "):
            model_key = cmd.split(" ")[1].lower()
            if model_key in MODEL_INFO:
                selected_model = model_key
                print(f"\nSelected model: {MODEL_INFO[model_key]['name']} ({model_key})")
                return selected_model
            else:
                print(f"Invalid model: {model_key}. Available models: {', '.join(MODEL_INFO.keys())}")
        
        elif cmd == "exit":
            print("Exiting model selector without selection")
            return None
        
        else:
            print("Invalid command. Try 'list', 'select X', 'choose X', or 'exit'")

def get_model_info(model_key):
    """Return the information for a specific model"""
    if model_key in MODEL_INFO:
        return MODEL_INFO[model_key]
    return None

def get_all_models_info():
    """Return information about all available models"""
    return MODEL_INFO

if __name__ == "__main__":
    # If run directly, start the interactive selector
    selected = select_model()
    if selected:
        print(f"You selected {MODEL_INFO[selected]['name']}. Model key: {selected}")
        print("Use this model key in your training scripts.") 