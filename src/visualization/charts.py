"""
Stock AI Visualization Tools
Creates clean, professional charts for analysis and reporting
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_prediction_chart(actual_prices: np.ndarray, 
                          predicted_prices: np.ndarray,
                          dates: List[str],
                          symbol: str,
                          confidence_intervals: Optional[Dict] = None,
                          save_path: Optional[str] = None) -> None:
    """Create prediction vs actual price chart"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual vs predicted
    ax.plot(dates, actual_prices, label='Actual Price', color='blue', linewidth=2)
    ax.plot(dates, predicted_prices, label='Predicted Price', color='red', linewidth=2, linestyle='--')
    
    # Add confidence intervals if available
    if confidence_intervals:
        ax.fill_between(dates, 
                       confidence_intervals['lower'], 
                       confidence_intervals['upper'],
                       alpha=0.3, color='red', label='95% Confidence Interval')
    
    ax.set_title(f'{symbol} - Stock Price Prediction', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_performance_dashboard(metrics: Dict[str, float],
                               symbol: str,
                               save_path: Optional[str] = None) -> None:
    """Create performance metrics dashboard"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Metric names and values
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Plot 1: Key metrics bar chart
    key_metrics = ['RMSE', 'MAE', 'Directional Accuracy']
    key_values = [metrics.get(m.lower().replace(' ', '_'), 0) for m in key_metrics]
    
    bars1 = ax1.bar(key_metrics, key_values, color=['red', 'orange', 'green'])
    ax1.set_title('Key Performance Metrics', fontweight='bold')
    ax1.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars1, key_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 2: Risk metrics
    risk_metrics = ['Sharpe Ratio', 'Max Drawdown', 'VaR']
    risk_values = [metrics.get(m.lower().replace(' ', '_'), 0) for m in risk_metrics]
    
    bars2 = ax2.bar(risk_metrics, risk_values, color=['blue', 'purple', 'brown'])
    ax2.set_title('Risk Metrics', fontweight='bold')
    ax2.set_ylabel('Value')
    
    for bar, value in zip(bars2, risk_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Plot 3: Model comparison (if available)
    if 'model_comparison' in metrics:
        comparison_data = metrics['model_comparison']
        models = list(comparison_data.keys())
        accuracies = [comparison_data[m].get('accuracy', 0) for m in models]
        
        ax3.bar(models, accuracies, color='skyblue')
        ax3.set_title('Model Comparison', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Model Comparison\nNot Available', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Model Comparison', fontweight='bold')
    
    # Plot 4: Training progress (if available)
    if 'training_history' in metrics:
        history = metrics['training_history']
        epochs = range(1, len(history['loss']) + 1)
        
        ax4.plot(epochs, history['loss'], label='Training Loss', color='red')
        if 'val_loss' in history:
            ax4.plot(epochs, history['val_loss'], label='Validation Loss', color='blue')
        
        ax4.set_title('Training Progress', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, 'Training History\nNot Available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Training Progress', fontweight='bold')
    
    plt.suptitle(f'{symbol} - Performance Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_regime_analysis_chart(regime_data: Dict[str, Any],
                               symbol: str,
                               save_path: Optional[str] = None) -> None:
    """Create market regime analysis visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Regime distribution
    regime_labels = list(regime_data.get('regime_distribution', {}).keys())
    regime_counts = list(regime_data.get('regime_distribution', {}).values())
    
    if regime_labels:
        colors = plt.cm.Set3(np.linspace(0, 1, len(regime_labels)))
        wedges, texts, autotexts = ax1.pie(regime_counts, labels=regime_labels, 
                                          autopct='%1.1f%%', colors=colors)
        ax1.set_title('Market Regime Distribution', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Regime Data', ha='center', va='center')
        ax1.set_title('Market Regime Distribution', fontweight='bold')
    
    # Plot 2: Regime performance
    if 'characteristics' in regime_data:
        characteristics = regime_data['characteristics']
        regime_names = list(characteristics.keys())
        returns = [characteristics[r].get('avg_return', 0) * 100 for r in regime_names]
        volatilities = [characteristics[r].get('volatility', 0) * 100 for r in regime_names]
        
        ax2.scatter(volatilities, returns, s=100, alpha=0.7)
        for i, name in enumerate(regime_names):
            ax2.annotate(f'Regime {name}', (volatilities[i], returns[i]))
        
        ax2.set_xlabel('Volatility (%)')
        ax2.set_ylabel('Average Return (%)')
        ax2.set_title('Risk-Return by Regime', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No Characteristic Data', ha='center', va='center')
        ax2.set_title('Risk-Return by Regime', fontweight='bold')
    
    # Plot 3: Transition matrix heatmap
    if 'transition_matrix' in regime_data:
        transition_matrix = np.array(regime_data['transition_matrix'])
        
        sns.heatmap(transition_matrix, annot=True, fmt='.2f', ax=ax3, 
                   cmap='Blues', square=True)
        ax3.set_title('Regime Transition Probabilities', fontweight='bold')
        ax3.set_xlabel('To Regime')
        ax3.set_ylabel('From Regime')
    else:
        ax3.text(0.5, 0.5, 'No Transition Data', ha='center', va='center')
        ax3.set_title('Regime Transition Matrix', fontweight='bold')
    
    # Plot 4: Strategy recommendations
    if 'strategy_recommendation' in regime_data:
        strategy = regime_data['strategy_recommendation']
        
        metrics = ['Position Size', 'Stop Loss', 'Take Profit']
        values = [
            strategy.get('position_size_multiplier', 0),
            strategy.get('stop_loss', 0) * 100,
            strategy.get('take_profit', 0) * 100
        ]
        
        bars = ax4.bar(metrics, values, color=['green', 'red', 'blue'])
        ax4.set_title('Strategy Recommendations', fontweight='bold')
        ax4.set_ylabel('Value (%)')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No Strategy Data', ha='center', va='center')
        ax4.set_title('Strategy Recommendations', fontweight='bold')
    
    plt.suptitle(f'{symbol} - Market Regime Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_portfolio_analysis(portfolio_data: Dict[str, Any],
                            save_path: Optional[str] = None) -> None:
    """Create portfolio optimization analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Asset allocation
    if 'weights' in portfolio_data:
        symbols = list(portfolio_data['weights'].keys())
        weights = list(portfolio_data['weights'].values())
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        wedges, texts, autotexts = ax1.pie(weights, labels=symbols, 
                                          autopct='%1.1f%%', colors=colors)
        ax1.set_title('Portfolio Allocation', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Allocation Data', ha='center', va='center')
        ax1.set_title('Portfolio Allocation', fontweight='bold')
    
    # Plot 2: Efficient frontier (if available)
    if 'efficient_frontier' in portfolio_data:
        frontier = portfolio_data['efficient_frontier']
        ax2.plot(frontier['volatility'], frontier['returns'], 'b-', linewidth=2)
        ax2.set_xlabel('Volatility')
        ax2.set_ylabel('Expected Return')
        ax2.set_title('Efficient Frontier', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Efficient Frontier\nNot Available', 
                ha='center', va='center')
        ax2.set_title('Efficient Frontier', fontweight='bold')
    
    # Plot 3: Risk metrics
    if 'risk_metrics' in portfolio_data:
        risk_data = portfolio_data['risk_metrics']
        metrics = list(risk_data.keys())
        values = list(risk_data.values())
        
        bars = ax3.bar(metrics, values, color='red', alpha=0.7)
        ax3.set_title('Portfolio Risk Metrics', fontweight='bold')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    else:
        ax3.text(0.5, 0.5, 'No Risk Data', ha='center', va='center')
        ax3.set_title('Portfolio Risk Metrics', fontweight='bold')
    
    # Plot 4: Correlation matrix
    if 'correlation_matrix' in portfolio_data:
        corr_matrix = portfolio_data['correlation_matrix']
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', ax=ax4, 
                   cmap='RdYlBu_r', square=True, center=0)
        ax4.set_title('Asset Correlation Matrix', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Correlation Data', ha='center', va='center')
        ax4.set_title('Asset Correlation Matrix', fontweight='bold')
    
    plt.suptitle('Portfolio Optimization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

def create_training_summary(training_results: Dict[str, Any],
                          save_path: Optional[str] = None) -> None:
    """Create training summary visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training success rate
    successful = training_results.get('successful', 0)
    failed = training_results.get('failed', 0)
    total = successful + failed
    
    if total > 0:
        sizes = [successful, failed]
        labels = ['Successful', 'Failed']
        colors = ['green', 'red']
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors)
        ax1.set_title(f'Training Success Rate ({successful}/{total})', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Training Data', ha='center', va='center')
        ax1.set_title('Training Success Rate', fontweight='bold')
    
    # Plot 2: Performance distribution
    if 'results' in training_results:
        rmse_values = [r.get('rmse', 0) for r in training_results['results'] 
                      if r.get('status') == 'success']
        
        if rmse_values:
            ax2.hist(rmse_values, bins=min(10, len(rmse_values)), 
                    color='skyblue', alpha=0.7)
            ax2.set_xlabel('RMSE')
            ax2.set_ylabel('Frequency')
            ax2.set_title('RMSE Distribution', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Performance Data', ha='center', va='center')
            ax2.set_title('RMSE Distribution', fontweight='bold')
    
    # Plot 3: Training time by symbol
    if 'results' in training_results:
        symbols = [r.get('symbol', '') for r in training_results['results']]
        times = [r.get('training_time', 0) for r in training_results['results']]
        
        if symbols and times:
            ax3.bar(symbols, times, color='orange', alpha=0.7)
            ax3.set_xlabel('Symbol')
            ax3.set_ylabel('Training Time (seconds)')
            ax3.set_title('Training Time by Symbol', fontweight='bold')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Timing Data', ha='center', va='center')
            ax3.set_title('Training Time by Symbol', fontweight='bold')
    
    # Plot 4: Summary metrics
    summary = training_results.get('summary', {})
    if summary:
        metric_names = ['Avg RMSE', 'Avg MAE', 'Total Time', 'Workers']
        metric_values = [
            summary.get('avg_rmse', 0),
            summary.get('avg_mae', 0),
            summary.get('total_training_time', 0) / 3600,  # Convert to hours
            summary.get('worker_count', 0)
        ]
        
        bars = ax4.bar(metric_names, metric_values, color=['red', 'orange', 'blue', 'green'])
        ax4.set_title('Summary Metrics', fontweight='bold')
        ax4.set_ylabel('Value')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    else:
        ax4.text(0.5, 0.5, 'No Summary Data', ha='center', va='center')
        ax4.set_title('Summary Metrics', fontweight='bold')
    
    plt.suptitle('Training Summary Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

# Convenience function for creating all charts
def create_complete_analysis(symbol: str,
                           prediction_data: Optional[Dict] = None,
                           performance_data: Optional[Dict] = None,
                           regime_data: Optional[Dict] = None,
                           portfolio_data: Optional[Dict] = None,
                           training_data: Optional[Dict] = None,
                           save_dir: Optional[str] = None) -> None:
    """Create complete analysis with all available charts"""
    
    charts_created = []
    
    if prediction_data:
        path = f"{save_dir}/{symbol}_predictions.png" if save_dir else None
        create_prediction_chart(**prediction_data, symbol=symbol, save_path=path)
        charts_created.append("Prediction Chart")
    
    if performance_data:
        path = f"{save_dir}/{symbol}_performance.png" if save_dir else None
        create_performance_dashboard(performance_data, symbol=symbol, save_path=path)
        charts_created.append("Performance Dashboard")
    
    if regime_data:
        path = f"{save_dir}/{symbol}_regimes.png" if save_dir else None
        create_regime_analysis_chart(regime_data, symbol=symbol, save_path=path)
        charts_created.append("Regime Analysis")
    
    if portfolio_data:
        path = f"{save_dir}/portfolio_analysis.png" if save_dir else None
        create_portfolio_analysis(portfolio_data, save_path=path)
        charts_created.append("Portfolio Analysis")
    
    if training_data:
        path = f"{save_dir}/training_summary.png" if save_dir else None
        create_training_summary(training_data, save_path=path)
        charts_created.append("Training Summary")
    
    print(f"Created {len(charts_created)} charts: {', '.join(charts_created)}")