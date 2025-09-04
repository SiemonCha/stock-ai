import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AssetInfo:
    """Asset information structure"""
    symbol: str
    name: str
    sector: str
    market_cap: float
    beta: float
    expected_return: float
    risk: float
    current_price: float
    
@dataclass
class PortfolioConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    max_sector_weight: float = 0.3
    min_positions: int = 3
    max_positions: int = 20
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    transaction_costs: float = 0.001
    
@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float
    diversification_ratio: float
    turnover: float
    transaction_costs: float
    optimization_method: str
    success: bool
    message: str

class MultiAssetOptimizer:
    """
    Advanced multi-asset portfolio optimization system
    """
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 lookback_period: int = 252,
                 rebalance_frequency: str = 'monthly'):
        self.risk_free_rate = risk_free_rate
        self.lookback_period = lookback_period
        self.rebalance_frequency = rebalance_frequency
        
        # Asset universe
        self.asset_universe = {}
        self.return_data = pd.DataFrame()
        self.covariance_matrix = np.array([])
        self.expected_returns = np.array([])
        
        # Market factors
        self.factor_models = {}
        self.regime_probabilities = {}
        
        # Optimization methods
        self.optimization_methods = {
            'markowitz': self._markowitz_optimization,
            'black_litterman': self._black_litterman_optimization,
            'risk_parity': self._risk_parity_optimization,
            'hierarchical_risk_parity': self._hierarchical_risk_parity,
            'kelly_criterion': self._kelly_criterion_optimization,
            'robust_optimization': self._robust_optimization,
            'regime_aware': self._regime_aware_optimization
        }
    
    def add_assets(self, symbols: List[str], fetch_data: bool = True) -> Dict[str, AssetInfo]:
        """Add assets to the optimization universe"""
        asset_info = {}
        
        if fetch_data:
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_symbol = {
                    executor.submit(self._fetch_asset_data, symbol): symbol 
                    for symbol in symbols
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        if data:
                            asset_info[symbol] = data
                            self.asset_universe[symbol] = data
                    except Exception as e:
                        logging.error(f"Failed to fetch data for {symbol}: {e}")
        
        # Fetch historical return data
        if asset_info:
            self._update_return_data(list(asset_info.keys()))
            self._calculate_expected_returns()
            self._calculate_covariance_matrix()
        
        return asset_info
    
    def _fetch_asset_data(self, symbol: str) -> Optional[AssetInfo]:
        """Fetch asset data from external sources"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            history = ticker.history(period=f'{self.lookback_period}d')
            
            if len(history) < 50:  # Need minimum data
                return None
            
            # Calculate metrics
            returns = history['Close'].pct_change().dropna()
            expected_return = returns.mean() * 252  # Annualized
            risk = returns.std() * np.sqrt(252)  # Annualized
            
            # Get market data for beta calculation
            try:
                market_data = yf.download('^GSPC', period=f'{self.lookback_period}d')['Close']
                market_returns = market_data.pct_change().dropna()
                
                # Align dates
                common_dates = returns.index.intersection(market_returns.index)
                if len(common_dates) > 50:
                    asset_rets = returns.loc[common_dates]
                    market_rets = market_returns.loc[common_dates]
                    beta = np.cov(asset_rets, market_rets)[0, 1] / np.var(market_rets)
                else:
                    beta = 1.0
            except:
                beta = 1.0
            
            return AssetInfo(
                symbol=symbol,
                name=info.get('longName', symbol),
                sector=info.get('sector', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                beta=beta,
                expected_return=expected_return,
                risk=risk,
                current_price=float(history['Close'].iloc[-1])
            )
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _update_return_data(self, symbols: List[str]):
        """Update historical return data for assets"""
        try:
            # Download data for all symbols
            data = yf.download(symbols, period=f'{self.lookback_period}d', group_by='ticker')
            
            if len(symbols) == 1:
                # Single symbol case
                returns = data['Close'].pct_change().dropna()
                self.return_data = pd.DataFrame({symbols[0]: returns})
            else:
                # Multiple symbols case
                return_data = {}
                for symbol in symbols:
                    if symbol in data.columns.levels[0]:
                        prices = data[symbol]['Close']
                        returns = prices.pct_change().dropna()
                        return_data[symbol] = returns
                
                self.return_data = pd.DataFrame(return_data).dropna()
            
        except Exception as e:
            logging.error(f"Error updating return data: {e}")
    
    def _calculate_expected_returns(self, method: str = 'historical'):
        """Calculate expected returns using various methods"""
        if self.return_data.empty:
            return
        
        if method == 'historical':
            self.expected_returns = self.return_data.mean() * 252
        
        elif method == 'capm':
            # CAPM-based expected returns
            market_return = 0.10  # Assumed market return
            expected_returns = {}
            
            for symbol in self.return_data.columns:
                if symbol in self.asset_universe:
                    beta = self.asset_universe[symbol].beta
                    expected_return = self.risk_free_rate + beta * (market_return - self.risk_free_rate)
                    expected_returns[symbol] = expected_return
            
            self.expected_returns = pd.Series(expected_returns)
        
        elif method == 'ewma':
            # Exponentially weighted moving average
            alpha = 0.94  # Decay factor
            weights = [(alpha ** i) for i in range(len(self.return_data))]
            weights.reverse()
            weights = np.array(weights) / np.sum(weights)
            
            self.expected_returns = np.average(self.return_data.values, axis=0, weights=weights) * 252
            self.expected_returns = pd.Series(self.expected_returns, index=self.return_data.columns)
    
    def _calculate_covariance_matrix(self, method: str = 'sample'):
        """Calculate covariance matrix using various methods"""
        if self.return_data.empty:
            return
        
        if method == 'sample':
            self.covariance_matrix = self.return_data.cov() * 252
        
        elif method == 'ledoit_wolf':
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_lw = lw.fit(self.return_data.values).covariance_ * 252
            self.covariance_matrix = pd.DataFrame(cov_lw, 
                                                 index=self.return_data.columns, 
                                                 columns=self.return_data.columns)
        
        elif method == 'ewma':
            # Exponentially weighted moving average covariance
            self.covariance_matrix = self.return_data.ewm(alpha=0.06).cov().iloc[-len(self.return_data.columns):] * 252
    
    def optimize_portfolio(self, 
                          method: str = 'markowitz',
                          constraints: Optional[PortfolioConstraints] = None,
                          current_weights: Optional[Dict[str, float]] = None,
                          objective: str = 'sharpe') -> OptimizationResult:
        """
        Optimize portfolio using specified method
        
        Args:
            method: Optimization method ('markowitz', 'black_litterman', etc.)
            constraints: Portfolio constraints
            current_weights: Current portfolio weights for turnover calculation
            objective: Optimization objective ('sharpe', 'return', 'risk', 'utility')
        """
        
        if constraints is None:
            constraints = PortfolioConstraints()
        
        if method not in self.optimization_methods:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Prepare optimization data
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        if n_assets == 0:
            return OptimizationResult(
                weights={}, expected_return=0, expected_risk=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
                transaction_costs=0, optimization_method=method, 
                success=False, message="No assets available"
            )
        
        try:
            # Run optimization
            result = self.optimization_methods[method](constraints, objective, current_weights)
            
            # Calculate portfolio metrics
            weights_array = np.array([result.weights.get(symbol, 0) for symbol in symbols])
            
            portfolio_return = float(np.dot(weights_array, self.expected_returns))
            portfolio_risk = float(np.sqrt(np.dot(weights_array, np.dot(self.covariance_matrix.values, weights_array))))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            # Calculate additional metrics
            max_drawdown = self._calculate_max_drawdown(weights_array)
            var_95 = self._calculate_var(weights_array, confidence=0.95)
            diversification_ratio = self._calculate_diversification_ratio(weights_array)
            
            # Calculate turnover
            turnover = 0.0
            transaction_costs_total = 0.0
            if current_weights:
                current_array = np.array([current_weights.get(symbol, 0) for symbol in symbols])
                turnover = float(np.sum(np.abs(weights_array - current_array)))
                transaction_costs_total = turnover * constraints.transaction_costs
            
            result.expected_return = portfolio_return
            result.expected_risk = portfolio_risk
            result.sharpe_ratio = sharpe_ratio
            result.max_drawdown = max_drawdown
            result.var_95 = var_95
            result.diversification_ratio = diversification_ratio
            result.turnover = turnover
            result.transaction_costs = transaction_costs_total
            result.success = True
            
            return result
            
        except Exception as e:
            logging.error(f"Portfolio optimization failed: {e}")
            return OptimizationResult(
                weights={}, expected_return=0, expected_risk=0, sharpe_ratio=0,
                max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
                transaction_costs=0, optimization_method=method, 
                success=False, message=str(e)
            )
    
    def _markowitz_optimization(self, 
                               constraints: PortfolioConstraints, 
                               objective: str,
                               current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Classical Markowitz mean-variance optimization"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        # Decision variables
        w = cp.Variable(n_assets, nonneg=True)
        
        # Objective function
        portfolio_return = cp.sum(cp.multiply(self.expected_returns.values, w))
        portfolio_risk = cp.quad_form(w, self.covariance_matrix.values)
        
        if objective == 'sharpe':
            # Maximize Sharpe ratio (approximate)
            obj = cp.Maximize(portfolio_return - 0.5 * portfolio_risk)
        elif objective == 'return':
            obj = cp.Maximize(portfolio_return)
        elif objective == 'risk':
            obj = cp.Minimize(portfolio_risk)
        elif objective == 'utility':
            # Utility maximization with risk aversion
            risk_aversion = 3.0
            obj = cp.Maximize(portfolio_return - 0.5 * risk_aversion * portfolio_risk)
        
        # Constraints
        constraint_list = [cp.sum(w) == 1]  # Budget constraint
        
        # Weight bounds
        if constraints.min_weight > 0:
            constraint_list.append(w >= constraints.min_weight)
        if constraints.max_weight < 1:
            constraint_list.append(w <= constraints.max_weight)
        
        # Target return constraint
        if constraints.target_return is not None:
            constraint_list.append(portfolio_return >= constraints.target_return)
        
        # Maximum risk constraint
        if constraints.max_risk is not None:
            constraint_list.append(portfolio_risk <= constraints.max_risk ** 2)
        
        # Sector constraints
        if constraints.max_sector_weight < 1.0:
            sectors = {}
            for i, symbol in enumerate(symbols):
                if symbol in self.asset_universe:
                    sector = self.asset_universe[symbol].sector
                    if sector not in sectors:
                        sectors[sector] = []
                    sectors[sector].append(i)
            
            for sector, indices in sectors.items():
                constraint_list.append(cp.sum(w[indices]) <= constraints.max_sector_weight)
        
        # Position count constraints (approximated)
        if constraints.min_positions > 1:
            # Encourage diversification by penalizing concentration
            constraint_list.append(cp.sum(cp.square(w)) <= 1.0 / constraints.min_positions)
        
        # Transaction costs (if current weights provided)
        if current_weights is not None:
            current_w = np.array([current_weights.get(symbol, 0) for symbol in symbols])
            turnover = cp.sum(cp.abs(w - current_w))
            obj = obj - constraints.transaction_costs * turnover
        
        # Solve optimization problem
        problem = cp.Problem(obj, constraint_list)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Optimization failed with status: {problem.status}")
        
        # Extract results
        optimal_weights = w.value
        weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, optimal_weights)}
        
        # Clean up small weights
        weights_dict = {k: v for k, v in weights_dict.items() if v > 1e-4}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0, expected_risk=0, sharpe_ratio=0,  # Will be calculated later
            max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
            transaction_costs=0, optimization_method='markowitz',
            success=True, message="Optimization successful"
        )
    
    def _black_litterman_optimization(self, 
                                     constraints: PortfolioConstraints,
                                     objective: str,
                                     current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Black-Litterman optimization with investor views"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        # Market capitalization weights (equilibrium portfolio)
        market_caps = np.array([
            self.asset_universe.get(symbol, AssetInfo('', '', '', 1e9, 1, 0, 0, 0)).market_cap 
            for symbol in symbols
        ])
        
        if np.sum(market_caps) > 0:
            w_market = market_caps / np.sum(market_caps)
        else:
            w_market = np.ones(n_assets) / n_assets
        
        # Risk aversion parameter
        risk_aversion = 3.0
        
        # Implied equilibrium returns
        pi = risk_aversion * np.dot(self.covariance_matrix.values, w_market)
        
        # Uncertainty in prior (tau parameter)
        tau = 1.0 / len(self.return_data)
        
        # Investor views (simplified example)
        # In practice, these would come from research/analysis
        P = np.eye(n_assets)  # Each view is about individual assets
        Q = self.expected_returns.values  # Views are historical returns
        omega = np.eye(n_assets) * 0.01  # Confidence in views
        
        # Black-Litterman formula
        sigma_inv = np.linalg.inv(self.covariance_matrix.values)
        
        try:
            # Combined covariance matrix
            M1 = np.linalg.inv(tau * self.covariance_matrix.values)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(omega), P))
            M3 = np.linalg.inv(M1 + M2)
            
            # Combined expected returns
            mu_bl = np.dot(M3, np.dot(M1, pi) + np.dot(P.T, np.dot(np.linalg.inv(omega), Q)))
            
            # Update expected returns
            expected_returns_bl = pd.Series(mu_bl, index=symbols)
            
            # Now use Markowitz optimization with BL returns
            original_returns = self.expected_returns.copy()
            self.expected_returns = expected_returns_bl
            
            result = self._markowitz_optimization(constraints, objective, current_weights)
            result.optimization_method = 'black_litterman'
            
            # Restore original returns
            self.expected_returns = original_returns
            
            return result
            
        except np.linalg.LinAlgError:
            # Fall back to regular Markowitz if BL fails
            result = self._markowitz_optimization(constraints, objective, current_weights)
            result.optimization_method = 'black_litterman_fallback'
            return result
    
    def _risk_parity_optimization(self, 
                                 constraints: PortfolioConstraints,
                                 objective: str,
                                 current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Risk parity optimization (equal risk contribution)"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        def risk_parity_objective(weights):
            """Objective function for risk parity"""
            weights = np.array(weights)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            
            # Risk contributions
            marginal_contrib = np.dot(self.covariance_matrix.values, weights) / portfolio_risk
            risk_contrib = weights * marginal_contrib
            
            # Target risk contribution (equal for all assets)
            target_contrib = portfolio_risk / n_assets
            
            # Minimize deviation from equal risk contribution
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # Constraints for scipy optimization
        constraints_scipy = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Budget constraint
        ]
        
        # Bounds
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result_opt = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_scipy,
            options={'maxiter': 1000}
        )
        
        if not result_opt.success:
            raise ValueError(f"Risk parity optimization failed: {result_opt.message}")
        
        optimal_weights = result_opt.x
        weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, optimal_weights)}
        
        # Clean up small weights
        weights_dict = {k: v for k, v in weights_dict.items() if v > 1e-4}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0, expected_risk=0, sharpe_ratio=0,
            max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
            transaction_costs=0, optimization_method='risk_parity',
            success=True, message="Risk parity optimization successful"
        )
    
    def _hierarchical_risk_parity(self, 
                                 constraints: PortfolioConstraints,
                                 objective: str,
                                 current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Hierarchical Risk Parity (HRP) optimization"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        # Calculate correlation matrix
        corr_matrix = self.return_data.corr().values
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        
        # Distance matrix from correlation
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Hierarchical clustering
        linkage_matrix = linkage(squareform(distance_matrix), method='single')
        
        # Get hierarchical tree
        tree = to_tree(linkage_matrix, rd_return_distance=False)
        
        def get_cluster_var(cluster_items):
            """Calculate cluster variance"""
            if len(cluster_items) == 1:
                return self.covariance_matrix.iloc[cluster_items[0], cluster_items[0]]
            
            cluster_cov = self.covariance_matrix.iloc[cluster_items, cluster_items]
            cluster_weights = np.ones(len(cluster_items)) / len(cluster_items)
            
            return np.dot(cluster_weights, np.dot(cluster_cov.values, cluster_weights))
        
        def get_rec_bipartition(tree_node, weights):
            """Recursively assign weights using inverse variance"""
            if tree_node.is_leaf():
                return
            
            # Get left and right clusters
            left_items = tree_node.left.pre_order()
            right_items = tree_node.right.pre_order()
            
            # Calculate cluster variances
            left_var = get_cluster_var(left_items)
            right_var = get_cluster_var(right_items)
            
            # Inverse variance weighting
            total_inv_var = 1/left_var + 1/right_var
            left_weight = (1/left_var) / total_inv_var
            right_weight = (1/right_var) / total_inv_var
            
            # Assign weights
            current_weight = np.sum([weights[i] for i in left_items + right_items])
            
            for i in left_items:
                weights[i] *= left_weight
            for i in right_items:
                weights[i] *= right_weight
            
            # Recurse
            get_rec_bipartition(tree_node.left, weights)
            get_rec_bipartition(tree_node.right, weights)
        
        # Initialize equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Apply recursive bipartition
        get_rec_bipartition(tree, weights)
        
        weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, weights)}
        
        # Clean up small weights
        weights_dict = {k: v for k, v in weights_dict.items() if v > 1e-4}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0, expected_risk=0, sharpe_ratio=0,
            max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
            transaction_costs=0, optimization_method='hierarchical_risk_parity',
            success=True, message="HRP optimization successful"
        )
    
    def _kelly_criterion_optimization(self, 
                                     constraints: PortfolioConstraints,
                                     objective: str,
                                     current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Kelly Criterion optimization for growth-optimal portfolio"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        def kelly_objective(weights):
            """Kelly criterion objective function"""
            weights = np.array(weights)
            
            # Expected log return
            expected_returns = self.expected_returns.values - self.risk_free_rate
            portfolio_variance = np.dot(weights, np.dot(self.covariance_matrix.values, weights))
            
            # Approximate Kelly formula for continuous time
            kelly_growth = np.dot(weights, expected_returns) - 0.5 * portfolio_variance
            
            return -kelly_growth  # Minimize negative growth
        
        # Constraints
        constraints_scipy = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        
        bounds = [(constraints.min_weight, constraints.max_weight) for _ in range(n_assets)]
        x0 = np.ones(n_assets) / n_assets
        
        result_opt = minimize(
            kelly_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_scipy,
            options={'maxiter': 1000}
        )
        
        if not result_opt.success:
            raise ValueError(f"Kelly optimization failed: {result_opt.message}")
        
        optimal_weights = result_opt.x
        weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, optimal_weights)}
        
        # Clean up small weights
        weights_dict = {k: v for k, v in weights_dict.items() if v > 1e-4}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0, expected_risk=0, sharpe_ratio=0,
            max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
            transaction_costs=0, optimization_method='kelly_criterion',
            success=True, message="Kelly optimization successful"
        )
    
    def _robust_optimization(self, 
                           constraints: PortfolioConstraints,
                           objective: str,
                           current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Robust optimization accounting for parameter uncertainty"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        # Create uncertainty sets for expected returns
        return_uncertainty = 0.02  # 2% uncertainty in expected returns
        
        w = cp.Variable(n_assets, nonneg=True)
        
        # Robust objective using worst-case expected return
        expected_returns_nominal = self.expected_returns.values
        
        # Worst-case portfolio return (conservative approach)
        portfolio_return_worst = cp.sum(cp.multiply(expected_returns_nominal - return_uncertainty, w))
        portfolio_risk = cp.quad_form(w, self.covariance_matrix.values)
        
        if objective == 'sharpe':
            obj = cp.Maximize(portfolio_return_worst - 0.5 * portfolio_risk)
        else:
            obj = cp.Maximize(portfolio_return_worst)
        
        # Standard constraints
        constraint_list = [
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]
        
        problem = cp.Problem(obj, constraint_list)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Robust optimization failed with status: {problem.status}")
        
        optimal_weights = w.value
        weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, optimal_weights)}
        
        # Clean up small weights
        weights_dict = {k: v for k, v in weights_dict.items() if v > 1e-4}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0, expected_risk=0, sharpe_ratio=0,
            max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
            transaction_costs=0, optimization_method='robust_optimization',
            success=True, message="Robust optimization successful"
        )
    
    def _regime_aware_optimization(self, 
                                  constraints: PortfolioConstraints,
                                  objective: str,
                                  current_weights: Optional[Dict[str, float]]) -> OptimizationResult:
        """Regime-aware optimization using multiple market scenarios"""
        symbols = list(self.return_data.columns)
        n_assets = len(symbols)
        
        # Define market regimes
        regimes = ['bull', 'bear', 'normal', 'volatile']
        regime_probs = [0.25, 0.25, 0.30, 0.20]  # Probabilities
        
        # Generate regime-specific parameters
        regime_returns = {}
        regime_covariances = {}
        
        for regime in regimes:
            if regime == 'bull':
                regime_returns[regime] = self.expected_returns * 1.5
                regime_covariances[regime] = self.covariance_matrix * 0.8
            elif regime == 'bear':
                regime_returns[regime] = self.expected_returns * -0.8
                regime_covariances[regime] = self.covariance_matrix * 1.5
            elif regime == 'volatile':
                regime_returns[regime] = self.expected_returns * 0.8
                regime_covariances[regime] = self.covariance_matrix * 2.0
            else:  # normal
                regime_returns[regime] = self.expected_returns
                regime_covariances[regime] = self.covariance_matrix
        
        w = cp.Variable(n_assets, nonneg=True)
        
        # Expected portfolio performance across regimes
        regime_utilities = []
        for i, regime in enumerate(regimes):
            portfolio_return = cp.sum(cp.multiply(regime_returns[regime].values, w))
            portfolio_risk = cp.quad_form(w, regime_covariances[regime].values)
            
            # Utility for this regime
            utility = portfolio_return - 0.5 * portfolio_risk
            regime_utilities.append(regime_probs[i] * utility)
        
        # Maximize expected utility across regimes
        obj = cp.Maximize(cp.sum(regime_utilities))
        
        constraint_list = [
            cp.sum(w) == 1,
            w >= constraints.min_weight,
            w <= constraints.max_weight
        ]
        
        problem = cp.Problem(obj, constraint_list)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            raise ValueError(f"Regime-aware optimization failed with status: {problem.status}")
        
        optimal_weights = w.value
        weights_dict = {symbol: float(weight) for symbol, weight in zip(symbols, optimal_weights)}
        
        # Clean up small weights
        weights_dict = {k: v for k, v in weights_dict.items() if v > 1e-4}
        
        return OptimizationResult(
            weights=weights_dict,
            expected_return=0, expected_risk=0, sharpe_ratio=0,
            max_drawdown=0, var_95=0, diversification_ratio=0, turnover=0,
            transaction_costs=0, optimization_method='regime_aware',
            success=True, message="Regime-aware optimization successful"
        )
    
    def _calculate_max_drawdown(self, weights: np.ndarray) -> float:
        """Calculate maximum drawdown for portfolio"""
        try:
            # Calculate portfolio returns
            portfolio_returns = np.dot(self.return_data.values, weights)
            
            # Calculate cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # Calculate drawdowns
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            
            return float(np.min(drawdowns))
            
        except:
            return 0.0
    
    def _calculate_var(self, weights: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR)"""
        try:
            portfolio_returns = np.dot(self.return_data.values, weights)
            return float(np.percentile(portfolio_returns, (1 - confidence) * 100))
        except:
            return 0.0
    
    def _calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            # Weighted average volatility
            individual_vols = np.sqrt(np.diag(self.covariance_matrix.values))
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.covariance_matrix.values, weights)))
            
            return float(weighted_avg_vol / portfolio_vol) if portfolio_vol > 0 else 1.0
            
        except:
            return 1.0
    
    def backtest_portfolio(self, 
                          weights: Dict[str, float],
                          start_date: str,
                          end_date: str,
                          rebalance_freq: str = 'monthly') -> Dict:
        """Backtest portfolio performance"""
        try:
            symbols = list(weights.keys())
            
            # Get historical data
            data = yf.download(symbols, start=start_date, end=end_date)['Close']
            
            if len(symbols) == 1:
                data = pd.DataFrame({symbols[0]: data})
            
            # Calculate returns
            returns = data.pct_change().dropna()
            
            # Calculate portfolio returns
            weight_array = np.array([weights.get(symbol, 0) for symbol in returns.columns])
            portfolio_returns = np.dot(returns.values, weight_array)
            
            # Performance metrics
            total_return = float(np.prod(1 + portfolio_returns) - 1)
            annualized_return = float((1 + total_return) ** (252 / len(portfolio_returns)) - 1)
            volatility = float(np.std(portfolio_returns) * np.sqrt(252))
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = float(np.min(drawdowns))
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                'portfolio_returns': portfolio_returns,
                'cumulative_returns': cumulative_returns
            }
            
        except Exception as e:
            logging.error(f"Backtesting failed: {e}")
            return {}
    
    def get_optimization_summary(self) -> Dict:
        """Get summary of optimization capabilities"""
        return {
            'available_methods': list(self.optimization_methods.keys()),
            'assets_in_universe': len(self.asset_universe),
            'data_period': len(self.return_data) if not self.return_data.empty else 0,
            'risk_free_rate': self.risk_free_rate,
            'last_updated': datetime.now()
        }