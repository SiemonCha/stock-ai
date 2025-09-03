"""
Visualization and plotting utilities for Stock AI
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_prediction_plot(actual: np.ndarray, predicted: np.ndarray, 
                          dates: Optional[List] = None, 
                          title: str = "Stock Price Prediction",
                          save_path: Optional[str] = None) -> None:
    """Create prediction comparison plot"""
    
    plt.figure(figsize=(12, 6))
    
    x_axis = dates if dates is not None else range(len(actual))
    
    plt.plot(x_axis, actual, label='Actual', color='blue', linewidth=2)
    plt.plot(x_axis, predicted, label='Predicted', color='red', linewidth=2, linestyle='--')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if dates is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_performance_chart(metrics: dict, title: str = "Model Performance",
                           save_path: Optional[str] = None) -> None:
    """Create performance metrics chart"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # RMSE and MAE
    axes[0, 0].bar(['RMSE', 'MAE'], [metrics.get('rmse', 0), metrics.get('mae', 0)], 
                   color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('Error Metrics')
    axes[0, 0].set_ylabel('Error Value')
    
    # Directional Accuracy
    accuracy = metrics.get('directional_accuracy', 0) * 100
    axes[0, 1].pie([accuracy, 100-accuracy], labels=['Correct', 'Incorrect'], 
                   colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
    axes[0, 1].set_title('Directional Accuracy')
    
    # Training History (if available)
    if 'history' in metrics and metrics['history']:
        history = metrics['history']
        axes[1, 0].plot(history.get('loss', []), label='Training Loss')
        axes[1, 0].plot(history.get('val_loss', []), label='Validation Loss')
        axes[1, 0].set_title('Training History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Training History\nNot Available', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Feature importance placeholder
    axes[1, 1].text(0.5, 0.5, 'Feature Importance\nPlot', 
                    ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_error_distribution(errors: np.ndarray, title: str = "Prediction Error Distribution",
                             save_path: Optional[str] = None) -> None:
    """Create error distribution plot"""
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.4f}')
    plt.axvline(np.median(errors), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.4f}')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Std: {np.std(errors):.4f}\\nMin: {np.min(errors):.4f}\\nMax: {np.max(errors):.4f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_training_history(history: dict, save_path: str = 'plots/training_history.png') -> None:
    """Plot training history"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    if 'loss' in history and 'val_loss' in history:
        ax1.plot(history['loss'], label='Training Loss', color='blue')
        ax1.plot(history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # MAE plot
    if 'mae' in history and 'val_mae' in history:
        ax2.plot(history['mae'], label='Training MAE', color='blue')
        ax2.plot(history['val_mae'], label='Validation MAE', color='red')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()