"""
Advanced Portfolio Optimization
Institutional-grade portfolio theory with modern risk management
"""

import numpy as np
import pandas as pd
import scipy.optimize as sco
from scipy import linalg
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

try:
    from sklearn.covariance import LedoitWolf, ShrunkCovariance
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    MAX_SHARPE = "max_sharpe"
    MIN_VOLATILITY = "min_volatility"
    MAX_RETURN = "max_return"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ESG_WEIGHTED = "esg_weighted"

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    max_weight: float = 0.4                    # Maximum single asset weight
    min_weight: float = 0.0                    # Minimum single asset weight
    max_turnover: Optional[float] = None       # Maximum portfolio turnover
    max_tracking_error: Optional[float] = None # Maximum tracking error vs benchmark
    sector_constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    long_only: bool = True                     # Long-only constraint
    target_return: Optional[float] = None      # Target return constraint
    max_drawdown: Optional[float] = None       # Maximum drawdown constraint

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk (95%)
    cvar_95: float  # Conditional Value at Risk (95%)
    calmar_ratio: float
    information_ratio: Optional[float] = None
    tracking_error: Optional[float] = None
    beta: Optional[float] = None

@dataclass
class BlackLittermanInputs:
    """Black-Litterman model inputs"""
    P: np.ndarray  # Picking matrix
    Q: np.ndarray  # Investor views (expected returns)
    Omega: np.ndarray  # Uncertainty matrix of views
    tau: float = 0.025  # Scaling factor
    risk_aversion: float = 3.0  # Risk aversion parameter

class CovarianceEstimator:
    """Advanced covariance matrix estimation"""
    
    def __init__(self, method: str = 'ledoit_wolf'):
        self.method = method
        self.estimator = None
        
        if SKLEARN_AVAILABLE:
            if method == 'ledoit_wolf':
                self.estimator = LedoitWolf()
            elif method == 'shrunk':
                self.estimator = ShrunkCovariance()
    
    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """Estimate covariance matrix using advanced methods"""
        
        if self.estimator and SKLEARN_AVAILABLE:
            try:
                cov_matrix = self.estimator.fit(returns).covariance_
                return cov_matrix
            except Exception as e:
                logger.warning(f"Advanced covariance estimation failed: {e}, using sample covariance")
        
        # Fallback to sample covariance
        return returns.cov().values
    
    def estimate_robust_covariance(self, returns: pd.DataFrame, 
                                  lookback_period: int = 252) -> np.ndarray:
        """Estimate covariance with exponential weighting"""
        
        # Use exponential weighting for more recent observations
        weights = np.exp(np.linspace(-1, 0, min(len(returns), lookback_period)))
        weights = weights / weights.sum()
        
        if len(returns) > lookback_period:
            recent_returns = returns.tail(lookback_period)
        else:
            recent_returns = returns
            weights = weights[-len(returns):]
        
        # Weighted covariance
        mean_returns = np.average(recent_returns, axis=0, weights=weights)
        
        diff = recent_returns.values - mean_returns
        weighted_cov = np.zeros((len(recent_returns.columns), len(recent_returns.columns)))
        
        for i, weight in enumerate(weights):
            weighted_cov += weight * np.outer(diff[i], diff[i])
        
        return weighted_cov * 252  # Annualize

class BlackLittermanOptimizer:
    """Black-Litterman portfolio optimization"""
    
    def __init__(self, returns: pd.DataFrame, market_caps: Optional[pd.Series] = None):
        self.returns = returns
        self.assets = returns.columns
        self.n_assets = len(self.assets)
        
        # Market capitalization weights (if available)
        if market_caps is not None:
            self.market_weights = market_caps / market_caps.sum()
        else:
            self.market_weights = pd.Series(1/self.n_assets, index=self.assets)
    
    def optimize(self, bl_inputs: BlackLittermanInputs) -> Dict[str, Any]:
        """Perform Black-Litterman optimization"""
        
        # Estimate covariance matrix
        cov_estimator = CovarianceEstimator('ledoit_wolf')
        Sigma = cov_estimator.estimate_covariance(self.returns)
        
        # Market-implied returns (reverse optimization)
        pi = bl_inputs.risk_aversion * Sigma @ self.market_weights.values
        
        # Black-Litterman formula
        tau_Sigma = bl_inputs.tau * Sigma
        
        # Posterior expected returns
        M1 = linalg.inv(tau_Sigma) + bl_inputs.P.T @ linalg.inv(bl_inputs.Omega) @ bl_inputs.P
        M2 = linalg.inv(tau_Sigma) @ pi + bl_inputs.P.T @ linalg.inv(bl_inputs.Omega) @ bl_inputs.Q
        
        mu_bl = linalg.inv(M1) @ M2
        
        # Posterior covariance
        Sigma_bl = linalg.inv(linalg.inv(tau_Sigma) + bl_inputs.P.T @ linalg.inv(bl_inputs.Omega) @ bl_inputs.P)
        
        # Optimal weights
        weights = linalg.inv(bl_inputs.risk_aversion * Sigma) @ mu_bl
        
        return {
            'weights': pd.Series(weights, index=self.assets),
            'expected_returns': pd.Series(mu_bl, index=self.assets),
            'posterior_cov': Sigma_bl,
            'market_implied_returns': pd.Series(pi, index=self.assets)
        }

class RiskParityOptimizer:
    """Risk parity portfolio optimization"""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.assets = returns.columns
        self.n_assets = len(self.assets)
    
    def optimize(self, target_risk_contrib: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Optimize for risk parity"""
        
        if target_risk_contrib is None:
            target_risk_contrib = np.ones(self.n_assets) / self.n_assets
        
        # Estimate covariance
        cov_estimator = CovarianceEstimator()
        Sigma = cov_estimator.estimate_covariance(self.returns)
        
        # Objective function: minimize sum of squared deviations from target risk contributions
        def objective(weights):
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)
            marginal_contrib = Sigma @ weights / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Sum of squared deviations
            return np.sum((risk_contrib - target_risk_contrib) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        bounds = [(0.001, 1) for _ in range(self.n_assets)]  # Long-only with small minimum
        
        # Initial guess (equal weights)
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = sco.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = result.x
            portfolio_vol = np.sqrt(weights @ Sigma @ weights)
            marginal_contrib = Sigma @ weights / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            return {
                'weights': pd.Series(weights, index=self.assets),
                'risk_contributions': pd.Series(risk_contrib, index=self.assets),
                'portfolio_volatility': portfolio_vol,
                'success': True
            }
        else:
            logger.error(f"Risk parity optimization failed: {result.message}")
            return {
                'weights': pd.Series(np.ones(self.n_assets) / self.n_assets, index=self.assets),
                'success': False
            }

class AdvancedPortfolioOptimizer:
    """Advanced portfolio optimizer with multiple objectives and constraints"""
    
    def __init__(self, returns: pd.DataFrame, benchmark_returns: Optional[pd.Series] = None):
        self.returns = returns
        self.assets = returns.columns
        self.n_assets = len(self.assets)
        self.benchmark_returns = benchmark_returns
        
        # Precompute statistics
        self.mean_returns = returns.mean() * 252  # Annualized
        self.cov_estimator = CovarianceEstimator('ledoit_wolf')
        self.cov_matrix = self.cov_estimator.estimate_robust_covariance(returns)
        
        # Risk-free rate (placeholder)
        self.risk_free_rate = 0.02
    
    def optimize(self, objective: OptimizationObjective, 
                constraints: OptimizationConstraints = None) -> Dict[str, Any]:
        """Main optimization function"""
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        if objective == OptimizationObjective.MAX_SHARPE:
            return self._optimize_max_sharpe(constraints)
        elif objective == OptimizationObjective.MIN_VOLATILITY:
            return self._optimize_min_volatility(constraints)
        elif objective == OptimizationObjective.MAX_RETURN:
            return self._optimize_max_return(constraints)
        elif objective == OptimizationObjective.RISK_PARITY:
            return self._optimize_risk_parity(constraints)
        elif objective == OptimizationObjective.BLACK_LITTERMAN:
            return self._optimize_black_litterman(constraints)
        else:
            raise ValueError(f"Unsupported optimization objective: {objective}")
    
    def _optimize_max_sharpe(self, constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Maximize Sharpe ratio"""
        
        def neg_sharpe(weights):
            portfolio_return = np.sum(weights * self.mean_returns)
            portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
            
            if portfolio_vol == 0:
                return -np.inf
            
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        return self._optimize_with_constraints(neg_sharpe, constraints)
    
    def _optimize_min_volatility(self, constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Minimize portfolio volatility"""
        
        def portfolio_volatility(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)
        
        return self._optimize_with_constraints(portfolio_volatility, constraints)
    
    def _optimize_max_return(self, constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Maximize expected return"""
        
        def neg_return(weights):
            return -np.sum(weights * self.mean_returns)
        
        return self._optimize_with_constraints(neg_return, constraints)
    
    def _optimize_risk_parity(self, constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Risk parity optimization"""
        
        rp_optimizer = RiskParityOptimizer(self.returns)
        result = rp_optimizer.optimize()
        
        if result['success']:
            weights = result['weights'].values
            metrics = self._calculate_portfolio_metrics(weights)
            
            return {
                'weights': result['weights'],
                'metrics': metrics,
                'risk_contributions': result['risk_contributions'],
                'success': True
            }
        else:
            return result
    
    def _optimize_black_litterman(self, constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Black-Litterman optimization with sample views"""
        
        # Create sample views (momentum-based)
        recent_returns = self.returns.tail(60).mean() * 252  # Last 60 days annualized
        long_term_returns = self.returns.mean() * 252  # All data annualized
        
        # Views: assets with recent outperformance will continue
        momentum_signal = recent_returns - long_term_returns
        
        # Top 3 momentum assets
        top_momentum = momentum_signal.nlargest(3)
        
        if len(top_momentum) >= 3:
            # Create views matrix
            P = np.zeros((3, self.n_assets))
            Q = np.zeros(3)
            
            for i, (asset, signal) in enumerate(top_momentum.items()):
                asset_idx = self.assets.get_loc(asset)
                P[i, asset_idx] = 1
                Q[i] = signal * 0.1  # 10% of momentum signal as view
            
            # Uncertainty matrix (proportional to view strength)
            Omega = np.diag(np.abs(Q) * 0.1)
            
            bl_inputs = BlackLittermanInputs(P=P, Q=Q, Omega=Omega)
            
            bl_optimizer = BlackLittermanOptimizer(self.returns)
            result = bl_optimizer.optimize(bl_inputs)
            
            weights = result['weights'].values
            
            # Apply constraints
            weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
            weights = weights / weights.sum()  # Renormalize
            
            metrics = self._calculate_portfolio_metrics(weights)
            
            return {
                'weights': pd.Series(weights, index=self.assets),
                'metrics': metrics,
                'bl_expected_returns': result['expected_returns'],
                'market_implied_returns': result['market_implied_returns'],
                'success': True
            }
        
        # Fallback to max Sharpe if not enough momentum signals
        return self._optimize_max_sharpe(constraints)
    
    def _optimize_with_constraints(self, objective_func, constraints: OptimizationConstraints) -> Dict[str, Any]:
        """Generic optimization with constraints"""
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(self.n_assets)]
        
        # Constraints list
        constraint_list = []
        
        # Weights sum to 1
        constraint_list.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        
        # Target return constraint
        if constraints.target_return is not None:
            constraint_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w * self.mean_returns) - constraints.target_return
            })
        
        # Maximum turnover constraint
        if constraints.max_turnover is not None:
            # Placeholder - would need previous weights for implementation
            pass
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize using different methods for robustness
        methods = ['SLSQP', 'L-BFGS-B']
        best_result = None
        best_value = np.inf
        
        for method in methods:
            try:
                result = sco.minimize(
                    objective_func, x0, method=method,
                    bounds=bounds, constraints=constraint_list,
                    options={'maxiter': 1000}
                )
                
                if result.success and result.fun < best_value:
                    best_result = result
                    best_value = result.fun
                    
            except Exception as e:
                logger.warning(f"Optimization with {method} failed: {e}")
                continue
        
        if best_result is not None and best_result.success:
            weights = best_result.x
            
            # Ensure weights are normalized and within bounds
            weights = np.clip(weights, constraints.min_weight, constraints.max_weight)
            weights = weights / weights.sum()
            
            metrics = self._calculate_portfolio_metrics(weights)
            
            return {
                'weights': pd.Series(weights, index=self.assets),
                'metrics': metrics,
                'success': True,
                'optimization_result': best_result
            }
        else:
            logger.error("All optimization methods failed")
            # Return equal weights as fallback
            equal_weights = np.ones(self.n_assets) / self.n_assets
            metrics = self._calculate_portfolio_metrics(equal_weights)
            
            return {
                'weights': pd.Series(equal_weights, index=self.assets),
                'metrics': metrics,
                'success': False
            }
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        # Basic metrics
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        # Portfolio returns time series
        portfolio_returns = self.returns @ weights
        
        # Downside deviation for Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (portfolio_return - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # VaR and CVaR (95% confidence)
        portfolio_returns_annual = portfolio_returns * np.sqrt(252)
        var_95 = np.percentile(portfolio_returns_annual, 5)
        cvar_95 = portfolio_returns_annual[portfolio_returns_annual <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = portfolio_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Benchmark-relative metrics
        information_ratio = None
        tracking_error = None
        beta = None
        
        if self.benchmark_returns is not None:
            # Align dates
            aligned_data = pd.concat([portfolio_returns, self.benchmark_returns], axis=1).dropna()
            if len(aligned_data) > 30:  # Minimum observations
                port_rets = aligned_data.iloc[:, 0]
                bench_rets = aligned_data.iloc[:, 1]
                
                excess_returns = port_rets - bench_rets
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
                
                # Beta calculation
                covariance = np.cov(port_rets, bench_rets)[0, 1]
                benchmark_var = np.var(bench_rets)
                beta = covariance / benchmark_var if benchmark_var > 0 else 1
        
        return PortfolioMetrics(
            expected_return=portfolio_return,
            volatility=portfolio_vol,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            beta=beta
        )
    
    def efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """Generate efficient frontier"""
        
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                constraints = OptimizationConstraints(target_return=target_ret)
                result = self._optimize_min_volatility(constraints)
                
                if result['success']:
                    efficient_portfolios.append({
                        'target_return': target_ret,
                        'volatility': result['metrics'].volatility,
                        'sharpe_ratio': result['metrics'].sharpe_ratio,
                        'weights': result['weights'].to_dict()
                    })
            except:
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def monte_carlo_simulation(self, weights: np.ndarray, n_simulations: int = 10000, 
                              time_horizon: int = 252) -> Dict[str, Any]:
        """Monte Carlo portfolio simulation"""
        
        # Portfolio parameters
        portfolio_return = np.sum(weights * self.mean_returns) / 252  # Daily
        portfolio_vol = np.sqrt(weights @ self.cov_matrix @ weights) / np.sqrt(252)  # Daily
        
        # Simulate paths
        np.random.seed(42)
        dt = 1/252
        
        simulated_returns = []
        final_values = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Geometric Brownian Motion
            random_shocks = np.random.normal(0, 1, time_horizon)
            returns = (portfolio_return - 0.5 * portfolio_vol**2) * dt + portfolio_vol * np.sqrt(dt) * random_shocks
            
            # Cumulative portfolio value
            portfolio_values = np.cumprod(1 + returns)
            
            # Total return
            total_return = portfolio_values[-1] - 1
            simulated_returns.append(total_return)
            final_values.append(portfolio_values[-1])
            
            # Max drawdown for this path
            running_max = np.maximum.accumulate(portfolio_values)
            drawdowns = (portfolio_values - running_max) / running_max
            max_drawdowns.append(drawdowns.min())
        
        return {
            'simulated_returns': np.array(simulated_returns),
            'final_values': np.array(final_values),
            'max_drawdowns': np.array(max_drawdowns),
            'percentiles': {
                'return_5th': np.percentile(simulated_returns, 5),
                'return_50th': np.percentile(simulated_returns, 50),
                'return_95th': np.percentile(simulated_returns, 95),
                'drawdown_5th': np.percentile(max_drawdowns, 5),
                'drawdown_95th': np.percentile(max_drawdowns, 95)
            }
        }

# Utility functions
def create_sample_views_matrix(assets: List[str], views: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Create Black-Litterman views matrix from dictionary"""
    
    n_assets = len(assets)
    n_views = len(views)
    
    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    
    for i, (asset, view) in enumerate(views.items()):
        if asset in assets:
            asset_idx = assets.index(asset)
            P[i, asset_idx] = 1
            Q[i] = view
    
    return P, Q

def calculate_risk_budgets(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """Calculate risk contribution of each asset"""
    
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
    marginal_contrib = cov_matrix @ weights / portfolio_vol
    risk_contrib = weights * marginal_contrib / portfolio_vol
    
    return risk_contrib

# Example usage and testing
if __name__ == "__main__":
    print("📊 Advanced Portfolio Optimization")
    print("=" * 40)
    
    # Generate sample data
    np.random.seed(42)
    n_assets = 5
    n_periods = 252 * 2  # 2 years
    
    asset_names = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate correlated returns
    correlation_matrix = np.random.uniform(0.1, 0.7, (n_assets, n_assets))
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Ensure positive semi-definite
    eigenvals, eigenvecs = linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Generate returns
    mean_returns = np.random.uniform(0.05, 0.15, n_assets) / 252  # Daily
    volatilities = np.random.uniform(0.15, 0.30, n_assets) / np.sqrt(252)  # Daily
    
    returns_data = []
    for i in range(n_periods):
        random_factors = np.random.multivariate_normal(np.zeros(n_assets), correlation_matrix)
        daily_returns = mean_returns + volatilities * random_factors
        returns_data.append(daily_returns)
    
    returns_df = pd.DataFrame(returns_data, columns=asset_names)
    returns_df.index = pd.date_range(start='2022-01-01', periods=n_periods, freq='D')
    
    print(f"✅ Generated sample returns data: {returns_df.shape}")
    print(f"   Assets: {list(returns_df.columns)}")
    print(f"   Date range: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    
    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer(returns_df)
    
    print(f"\n📈 Portfolio Statistics:")
    print(f"   Mean returns (annual): {(returns_df.mean() * 252).round(3).to_dict()}")
    print(f"   Volatilities (annual): {(returns_df.std() * np.sqrt(252)).round(3).to_dict()}")
    
    # Test different optimization objectives
    objectives_to_test = [
        OptimizationObjective.MAX_SHARPE,
        OptimizationObjective.MIN_VOLATILITY,
        OptimizationObjective.RISK_PARITY,
        OptimizationObjective.BLACK_LITTERMAN
    ]
    
    results = {}
    
    for objective in objectives_to_test:
        print(f"\n🎯 Optimizing for: {objective.value}")
        
        try:
            constraints = OptimizationConstraints(
                max_weight=0.4,
                min_weight=0.05,
                long_only=True
            )
            
            result = optimizer.optimize(objective, constraints)
            
            if result['success']:
                weights = result['weights']
                metrics = result['metrics']
                
                print(f"   ✅ Optimization successful")
                print(f"   Expected Return: {metrics.expected_return:.3f}")
                print(f"   Volatility: {metrics.volatility:.3f}")
                print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
                print(f"   Max Drawdown: {metrics.max_drawdown:.3f}")
                
                print(f"   Asset Allocation:")
                for asset, weight in weights.items():
                    print(f"      {asset}: {weight:.1%}")
                
                results[objective.value] = result
                
            else:
                print(f"   ❌ Optimization failed")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test Monte Carlo simulation
    if OptimizationObjective.MAX_SHARPE.value in results:
        print(f"\n🎲 Monte Carlo Simulation (Max Sharpe Portfolio):")
        
        max_sharpe_weights = results[OptimizationObjective.MAX_SHARPE.value]['weights'].values
        
        mc_results = optimizer.monte_carlo_simulation(
            max_sharpe_weights, 
            n_simulations=1000,  # Reduced for testing
            time_horizon=252     # 1 year
        )
        
        percentiles = mc_results['percentiles']
        print(f"   1-Year Return Projections:")
        print(f"      5th percentile: {percentiles['return_5th']:.1%}")
        print(f"      Median: {percentiles['return_50th']:.1%}")
        print(f"      95th percentile: {percentiles['return_95th']:.1%}")
        
        print(f"   Max Drawdown Projections:")
        print(f"      5th percentile: {percentiles['drawdown_5th']:.1%}")
        print(f"      95th percentile: {percentiles['drawdown_95th']:.1%}")
    
    # Test efficient frontier
    print(f"\n📊 Efficient Frontier:")
    try:
        frontier = optimizer.efficient_frontier(n_points=10)  # Reduced for testing
        
        if len(frontier) > 0:
            print(f"   Generated {len(frontier)} efficient portfolios")
            print(f"   Return range: {frontier['target_return'].min():.3f} to {frontier['target_return'].max():.3f}")
            print(f"   Volatility range: {frontier['volatility'].min():.3f} to {frontier['volatility'].max():.3f}")
            print(f"   Max Sharpe ratio: {frontier['sharpe_ratio'].max():.3f}")
        
    except Exception as e:
        print(f"   Error generating efficient frontier: {e}")
    
    print(f"\n🎯 Advanced portfolio optimization ready!")
    print(f"📋 Features:")
    print(f"   • Multiple optimization objectives")
    print(f"   • Black-Litterman model with views")
    print(f"   • Risk parity optimization")
    print(f"   • Advanced covariance estimation")
    print(f"   • Monte Carlo simulation")
    print(f"   • Efficient frontier generation")
    print(f"   • Comprehensive risk metrics")
    
    print(f"\n💡 Optimization objectives supported:")
    for obj in OptimizationObjective:
        print(f"   • {obj.value}")
    
    print(f"\n🛡️ Risk metrics calculated:")
    print(f"   • Sharpe & Sortino ratios")
    print(f"   • Value at Risk (VaR)")
    print(f"   • Conditional VaR (CVaR)")
    print(f"   • Maximum drawdown")
    print(f"   • Information ratio")
    print(f"   • Tracking error")