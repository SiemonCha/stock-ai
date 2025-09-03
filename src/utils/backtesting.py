import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os

class BacktestingFramework:
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def simple_strategy_backtest(self, 
                                predictions: pd.DataFrame, 
                                actual_prices: pd.DataFrame,
                                symbol: str,
                                threshold: float = 0.02) -> Dict:
        """
        Simple buy/sell strategy based on predictions
        Buy if predicted increase > threshold, sell if predicted decrease > threshold
        """
        capital = self.initial_capital
        position = 0
        trades = []
        portfolio_value = []
        
        for i in range(len(predictions)):
            current_price = actual_prices.iloc[i]
            predicted_price = predictions.iloc[i]
            predicted_return = (predicted_price - current_price) / current_price
            
            # Trading logic
            if predicted_return > threshold and position == 0:
                # Buy signal
                shares = capital / current_price
                position = shares
                capital = 0
                trades.append({
                    'date': actual_prices.index[i],
                    'action': 'BUY',
                    'price': current_price,
                    'shares': shares,
                    'value': shares * current_price
                })
                
            elif predicted_return < -threshold and position > 0:
                # Sell signal
                capital = position * current_price
                trades.append({
                    'date': actual_prices.index[i],
                    'action': 'SELL',
                    'price': current_price,
                    'shares': position,
                    'value': position * current_price
                })
                position = 0
            
            # Calculate current portfolio value
            if position > 0:
                portfolio_value.append(position * current_price)
            else:
                portfolio_value.append(capital)
        
        # Final portfolio value
        if position > 0:
            final_value = position * actual_prices.iloc[-1]
        else:
            final_value = capital
        
        # Calculate metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        portfolio_returns = np.diff(portfolio_value) / portfolio_value[:-1]
        sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if len(portfolio_returns) > 1 else 0
        
        max_drawdown = self.calculate_max_drawdown(portfolio_value)
        
        buy_hold_return = (actual_prices.iloc[-1] - actual_prices.iloc[0]) / actual_prices.iloc[0]
        
        results = {
            'symbol': symbol,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'final_value': final_value,
            'trades': trades,
            'portfolio_value': portfolio_value
        }
        
        self.results[symbol] = results
        return results
    
    def calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """
        Calculate maximum drawdown
        """
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def calculate_directional_accuracy(self, predictions: pd.DataFrame, actual_prices: pd.DataFrame) -> float:
        """
        Calculate directional accuracy (up/down prediction accuracy)
        """
        actual_directions = []
        predicted_directions = []
        
        for i in range(1, len(predictions)):
            actual_change = actual_prices.iloc[i] - actual_prices.iloc[i-1]
            predicted_change = predictions.iloc[i] - actual_prices.iloc[i-1]
            
            actual_directions.append(1 if actual_change > 0 else 0)
            predicted_directions.append(1 if predicted_change > 0 else 0)
        
        correct_predictions = sum(1 for a, p in zip(actual_directions, predicted_directions) if a == p)
        return correct_predictions / len(actual_directions)
    
    def plot_backtest_results(self, symbol: str, save_path: str = None):
        """
        Plot backtesting results
        """
        if symbol not in self.results:
            raise ValueError(f"No results found for {symbol}")
        
        results = self.results[symbol]
        portfolio_value = results['portfolio_value']
        
        plt.figure(figsize=(12, 8))
        
        # Portfolio value over time
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_value, label='Strategy Portfolio')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        plt.title(f'{symbol} - Portfolio Value Over Time')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        
        # Performance metrics
        plt.subplot(2, 1, 2)
        metrics = ['Total Return', 'Buy & Hold Return', 'Excess Return', 'Sharpe Ratio', 'Max Drawdown']
        values = [
            results['total_return'] * 100,
            results['buy_hold_return'] * 100,
            results['excess_return'] * 100,
            results['sharpe_ratio'],
            results['max_drawdown'] * 100
        ]
        
        bars = plt.bar(metrics, values)
        plt.title(f'{symbol} - Performance Metrics')
        plt.ylabel('Value (%)')
        plt.xticks(rotation=45)
        
        # Color bars
        for i, bar in enumerate(bars):
            if i < 3:  # Returns
                bar.set_color('green' if values[i] > 0 else 'red')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, symbol: str) -> str:
        """
        Generate a text report of backtesting results
        """
        if symbol not in self.results:
            raise ValueError(f"No results found for {symbol}")
        
        results = self.results[symbol]
        
        report = f"""
Backtesting Report for {symbol}
{'='*50}

Portfolio Performance:
- Initial Capital: ${self.initial_capital:,.2f}
- Final Value: ${results['final_value']:,.2f}
- Total Return: {results['total_return']:.2%}
- Buy & Hold Return: {results['buy_hold_return']:.2%}
- Excess Return: {results['excess_return']:.2%}

Risk Metrics:
- Sharpe Ratio: {results['sharpe_ratio']:.2f}
- Maximum Drawdown: {results['max_drawdown']:.2%}

Trading Activity:
- Number of Trades: {results['num_trades']}
- Average Trade Value: ${np.mean([trade['value'] for trade in results['trades']]):,.2f}

Strategy Performance:
{'Outperformed' if results['excess_return'] > 0 else 'Underperformed'} buy-and-hold by {abs(results['excess_return']):.2%}
        """
        
        return report
    
    def compare_symbols(self, symbols: List[str]) -> pd.DataFrame:
        """
        Compare backtesting results across multiple symbols
        """
        comparison_data = []
        
        for symbol in symbols:
            if symbol in self.results:
                results = self.results[symbol]
                comparison_data.append({
                    'Symbol': symbol,
                    'Total Return (%)': results['total_return'] * 100,
                    'Buy & Hold Return (%)': results['buy_hold_return'] * 100,
                    'Excess Return (%)': results['excess_return'] * 100,
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Max Drawdown (%)': results['max_drawdown'] * 100,
                    'Number of Trades': results['num_trades'],
                    'Final Value ($)': results['final_value']
                })
        
        return pd.DataFrame(comparison_data)