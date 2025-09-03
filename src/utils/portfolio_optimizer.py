import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.weights = None
        self.expected_returns = None
        self.covariance_matrix = None
        
    def calculate_expected_returns(self, price_data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Calculate expected returns for each asset
        """
        returns_data = {}
        
        for symbol, data in price_data.items():
            returns = data['close'].pct_change().dropna()
            returns_data[symbol] = returns.mean() * 252  # Annualized
        
        self.expected_returns = pd.Series(returns_data)
        return self.expected_returns
    
    def calculate_covariance_matrix(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate covariance matrix of returns
        """
        returns_data = {}
        
        for symbol, data in price_data.items():
            returns = data['close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data).dropna()
        self.covariance_matrix = returns_df.cov() * 252  # Annualized
        
        return self.covariance_matrix
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate portfolio return and volatility
        """
        portfolio_return = np.sum(self.expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(self.covariance_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return portfolio_return, portfolio_volatility
    
    def sharpe_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate Sharpe ratio (negative for minimization)
        """
        portfolio_return, portfolio_volatility = self.portfolio_performance(weights)
        return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
    
    def optimize_sharpe_ratio(self, symbols: List[str]) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio
        """
        n_assets = len(symbols)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            self.sharpe_ratio,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = self.portfolio_performance(optimal_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            self.weights = dict(zip(symbols, optimal_weights))
            
            return {
                'weights': self.weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'message': 'Optimization failed'
            }
    
    def optimize_min_volatility(self, symbols: List[str]) -> Dict:
        """
        Optimize portfolio for minimum volatility
        """
        n_assets = len(symbols)
        
        def portfolio_volatility(weights):
            return self.portfolio_performance(weights)[1]
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_guess = np.array([1.0 / n_assets] * n_assets)
        
        # Optimization
        result = minimize(
            portfolio_volatility,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility = self.portfolio_performance(optimal_weights)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            self.weights = dict(zip(symbols, optimal_weights))
            
            return {
                'weights': self.weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'message': 'Optimization failed'
            }
    
    def efficient_frontier(self, symbols: List[str], num_portfolios: int = 100) -> Dict:
        """
        Generate efficient frontier
        """
        n_assets = len(symbols)
        
        # Calculate bounds for target returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Constraints
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.expected_returns * x) - target_return}
            )
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Initial guess
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            # Minimize volatility for target return
            def portfolio_volatility(weights):
                return self.portfolio_performance(weights)[1]
            
            result = minimize(
                portfolio_volatility,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = result.x
                portfolio_return, portfolio_volatility = self.portfolio_performance(weights)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                
                efficient_portfolios.append({
                    'return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'weights': dict(zip(symbols, weights))
                })
        
        return {
            'efficient_portfolios': efficient_portfolios,
            'symbols': symbols
        }
    
    def plot_efficient_frontier(self, efficient_frontier_data: Dict, save_path: str = None):
        """
        Plot the efficient frontier
        """
        portfolios = efficient_frontier_data['efficient_portfolios']
        
        returns = [p['return'] for p in portfolios]
        volatilities = [p['volatility'] for p in portfolios]
        sharpe_ratios = [p['sharpe_ratio'] for p in portfolios]
        
        plt.figure(figsize=(12, 8))
        
        # Efficient frontier
        plt.subplot(2, 1, 1)
        scatter = plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.grid(True, alpha=0.3)
        
        # Individual asset positions
        for i, symbol in enumerate(efficient_frontier_data['symbols']):
            asset_return = self.expected_returns[symbol]
            asset_volatility = np.sqrt(self.covariance_matrix.iloc[i, i])
            plt.plot(asset_volatility, asset_return, 'ro', markersize=8, label=symbol)
        
        plt.legend()
        
        # Weights distribution for max Sharpe ratio portfolio
        plt.subplot(2, 1, 2)
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_portfolio = portfolios[max_sharpe_idx]
        
        weights = list(max_sharpe_portfolio['weights'].values())
        symbols = list(max_sharpe_portfolio['weights'].keys())
        
        plt.pie(weights, labels=symbols, autopct='%1.1f%%')
        plt.title(f'Optimal Portfolio Allocation (Sharpe Ratio: {max_sharpe_portfolio["sharpe_ratio"]:.2f})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def rebalance_portfolio(self, 
                          current_positions: Dict[str, float], 
                          target_weights: Dict[str, float],
                          total_value: float) -> Dict[str, float]:
        """
        Calculate trades needed to rebalance portfolio
        """
        trades = {}
        
        for symbol, target_weight in target_weights.items():
            target_value = total_value * target_weight
            current_value = current_positions.get(symbol, 0)
            trade_value = target_value - current_value
            trades[symbol] = trade_value
        
        return trades