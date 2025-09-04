"""
Professional Risk Management System
Advanced position sizing, risk control, and portfolio protection for professional traders
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Risk management libraries
try:
    from scipy import stats
    from scipy.optimize import minimize
    import cvxpy as cp
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("Optimization libraries not available. Install: pip install scipy cvxpy")

# Portfolio management
try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    from pypfopt.discrete_allocation import DiscreteAllocation
    PORTFOLIO_OPT_AVAILABLE = True
except ImportError:
    PORTFOLIO_OPT_AVAILABLE = False
    print("Portfolio optimization not available. Install: pip install PyPortfolioOpt")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfessionalRiskManager:
    """
    Institutional-grade risk management system with:
    - Dynamic position sizing
    - Real-time risk monitoring
    - Portfolio-level risk controls
    - Advanced stop-loss strategies
    - Volatility-based position sizing
    - Correlation-based exposure limits
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        
        # Risk limits
        self.max_position_size = self.config.get('max_position_size', 0.05)  # 5% max per position
        self.max_sector_exposure = self.config.get('max_sector_exposure', 0.25)  # 25% max per sector
        self.max_portfolio_beta = self.config.get('max_portfolio_beta', 1.5)
        self.max_daily_var = self.config.get('max_daily_var', 0.02)  # 2% daily VaR
        
        # Risk metrics cache
        self.risk_metrics = {}
        self.correlation_matrix = None
        self.volatility_estimates = {}
        
        # Position tracking
        self.positions = {}
        self.trade_history = []
        
        logger.info("🛡️ Professional Risk Manager initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default risk management configuration"""
        return {
            'max_position_size': 0.05,
            'max_sector_exposure': 0.25,
            'max_portfolio_beta': 1.5,
            'max_daily_var': 0.02,
            'max_drawdown': 0.10,
            'volatility_lookback': 252,
            'correlation_lookback': 252,
            'confidence_level': 0.95,
            'rebalance_threshold': 0.05,
            'stop_loss_method': 'volatility_based',
            'position_sizing_method': 'kelly_fraction',
            'risk_free_rate': 0.05
        }
    
    def calculate_position_size(self, symbol: str, prediction_data: Dict[str, Any], 
                              portfolio_value: float, current_price: float) -> Dict[str, Any]:
        """
        Calculate optimal position size using multiple methods
        
        Methods:
        - Kelly Criterion (maximize log wealth)
        - Volatility-based sizing
        - Risk parity approach
        - Maximum Sharpe ratio
        - VAR-based sizing
        """
        
        logger.info(f"📊 Calculating position size for {symbol}")
        
        # Extract prediction metrics
        expected_return = prediction_data.get('expected_return', 0)
        confidence = prediction_data.get('confidence_score', 0.5)
        volatility = prediction_data.get('volatility', 0.20)
        directional_accuracy = prediction_data.get('directional_accuracy', 0.6)
        
        # Get historical data for advanced calculations
        price_history = prediction_data.get('price_history', [])
        if len(price_history) < 30:
            logger.warning(f"Limited price history for {symbol}, using conservative sizing")
            return self._conservative_position_size(portfolio_value, current_price)
        
        # Calculate returns
        returns = pd.Series(price_history).pct_change().dropna()
        
        # Method 1: Kelly Criterion
        kelly_fraction = self._calculate_kelly_fraction(returns, expected_return, directional_accuracy)
        
        # Method 2: Volatility-based sizing
        vol_based_size = self._calculate_volatility_based_size(volatility, portfolio_value)
        
        # Method 3: Risk parity
        risk_parity_size = self._calculate_risk_parity_size(symbol, volatility, portfolio_value)
        
        # Method 4: VAR-based sizing
        var_based_size = self._calculate_var_based_size(returns, portfolio_value, current_price)
        
        # Method 5: Confidence-weighted sizing
        confidence_weighted_size = self._calculate_confidence_weighted_size(
            expected_return, confidence, volatility, portfolio_value)
        
        # Combine methods with weights
        method_weights = {
            'kelly': 0.25,
            'volatility': 0.20,
            'risk_parity': 0.20,
            'var_based': 0.20,
            'confidence': 0.15
        }
        
        sizes = {
            'kelly': kelly_fraction * portfolio_value / current_price,
            'volatility': vol_based_size,
            'risk_parity': risk_parity_size,
            'var_based': var_based_size,
            'confidence': confidence_weighted_size
        }
        
        # Weighted average position size
        optimal_shares = sum(sizes[method] * weight for method, weight in method_weights.items())
        
        # Apply risk limits
        optimal_shares = self._apply_risk_limits(symbol, optimal_shares, portfolio_value, current_price)
        
        # Calculate position metrics
        position_value = optimal_shares * current_price
        position_weight = position_value / portfolio_value
        
        # Stop loss and take profit levels
        stop_loss = self._calculate_stop_loss(current_price, volatility, confidence)
        take_profit = self._calculate_take_profit(current_price, expected_return, volatility)
        
        result = {
            'symbol': symbol,
            'optimal_shares': int(optimal_shares),
            'position_value': position_value,
            'position_weight': position_weight,
            'stop_loss_price': stop_loss,
            'take_profit_price': take_profit,
            'expected_return': expected_return,
            'volatility': volatility,
            'confidence': confidence,
            'kelly_fraction': kelly_fraction,
            'risk_adjusted_sizing': True,
            'method_breakdown': sizes,
            'risk_metrics': {
                'max_loss': position_value * (1 - stop_loss / current_price),
                'risk_reward_ratio': (take_profit - current_price) / (current_price - stop_loss) if stop_loss < current_price else 0,
                'position_var': self._calculate_position_var(optimal_shares, current_price, volatility),
                'contribution_to_portfolio_risk': self._estimate_portfolio_risk_contribution(symbol, optimal_shares, current_price)
            }
        }
        
        logger.info(f"✅ Position size for {symbol}: {optimal_shares:,.0f} shares (${position_value:,.0f}, {position_weight:.1%})")
        
        return result
    
    def _calculate_kelly_fraction(self, returns: pd.Series, expected_return: float, 
                                 accuracy: float) -> float:
        """Calculate Kelly Criterion position size"""
        
        if len(returns) < 20:
            return 0.02  # Conservative fallback
        
        # Kelly formula: f* = (bp - q) / b
        # Where b = odds received on the wager, p = probability of winning, q = probability of losing
        
        # Estimate win probability from directional accuracy
        win_prob = accuracy
        lose_prob = 1 - accuracy
        
        # Average win and loss amounts
        winning_returns = returns[returns > 0]
        losing_returns = returns[returns < 0]
        
        if len(winning_returns) == 0 or len(losing_returns) == 0:
            return 0.02
        
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())
        
        if avg_loss == 0:
            return 0.02
        
        # Kelly fraction
        b = avg_win / avg_loss  # Ratio of win to loss
        kelly_f = (b * win_prob - lose_prob) / b
        
        # Cap Kelly fraction to prevent over-leverage
        kelly_f = max(0, min(kelly_f, 0.25))  # Max 25% Kelly
        
        return kelly_f
    
    def _calculate_volatility_based_size(self, volatility: float, portfolio_value: float) -> float:
        """Position size inversely proportional to volatility"""
        
        target_vol = self.config.get('target_position_volatility', 0.15)
        base_allocation = self.config.get('base_allocation', 0.05)
        
        if volatility <= 0:
            return 0
        
        # Inverse volatility weighting
        vol_adjusted_allocation = base_allocation * (target_vol / volatility)
        
        # Convert to shares (placeholder - would need current price)
        shares = vol_adjusted_allocation * portfolio_value / 100  # Assuming $100 price
        
        return max(0, shares)
    
    def _calculate_risk_parity_size(self, symbol: str, volatility: float, 
                                  portfolio_value: float) -> float:
        """Risk parity position sizing"""
        
        # In risk parity, each position contributes equally to portfolio risk
        # This is a simplified version
        
        target_risk_contribution = self.config.get('target_risk_contribution', 0.02)
        
        if volatility <= 0:
            return 0
        
        # Position size to achieve target risk contribution
        position_value = target_risk_contribution * portfolio_value / volatility
        
        # Convert to shares (placeholder)
        shares = position_value / 100  # Assuming $100 price
        
        return max(0, shares)
    
    def _calculate_var_based_size(self, returns: pd.Series, portfolio_value: float, 
                                 current_price: float) -> float:
        """VAR-based position sizing"""
        
        if len(returns) < 20:
            return portfolio_value * 0.01 / current_price
        
        # Calculate Value at Risk
        confidence_level = self.config.get('confidence_level', 0.95)
        var_percentile = (1 - confidence_level) * 100
        
        daily_var = np.percentile(returns, var_percentile)
        
        # Maximum loss we're willing to accept
        max_portfolio_loss = self.max_daily_var * portfolio_value
        
        # Position size that limits loss to max_portfolio_loss
        if daily_var >= 0:
            return 0  # No position if VAR is positive (shouldn't happen)
        
        max_position_value = max_portfolio_loss / abs(daily_var)
        shares = max_position_value / current_price
        
        return max(0, shares)
    
    def _calculate_confidence_weighted_size(self, expected_return: float, confidence: float,
                                          volatility: float, portfolio_value: float) -> float:
        """Position size weighted by prediction confidence"""
        
        base_size = portfolio_value * 0.03  # 3% base allocation
        
        # Adjust for confidence
        confidence_multiplier = confidence ** 2  # Square to emphasize high confidence
        
        # Adjust for expected return
        return_multiplier = max(0, min(2, 1 + expected_return * 10))  # Cap multiplier
        
        # Adjust for volatility
        vol_multiplier = max(0.1, 0.20 / max(volatility, 0.05))
        
        adjusted_size = base_size * confidence_multiplier * return_multiplier * vol_multiplier
        
        # Convert to shares (placeholder)
        shares = adjusted_size / 100  # Assuming $100 price
        
        return shares
    
    def _apply_risk_limits(self, symbol: str, shares: float, portfolio_value: float, 
                          current_price: float) -> float:
        """Apply portfolio-level risk limits"""
        
        position_value = shares * current_price
        
        # Maximum position size limit
        max_position_value = self.max_position_size * portfolio_value
        if position_value > max_position_value:
            shares = max_position_value / current_price
            logger.info(f"⚠️ Position size capped at {self.max_position_size:.1%} for {symbol}")
        
        # Sector exposure limits (would need sector mapping)
        # This is simplified - in practice you'd track sector exposures
        
        # Liquidity limits (ensure we can exit)
        # Would need average daily volume data
        
        # Correlation limits (avoid concentrated risk)
        # Would need correlation analysis
        
        return max(0, shares)
    
    def _calculate_stop_loss(self, current_price: float, volatility: float, 
                           confidence: float) -> float:
        """Calculate dynamic stop loss level"""
        
        method = self.config.get('stop_loss_method', 'volatility_based')
        
        if method == 'volatility_based':
            # Stop loss based on volatility (wider stops for volatile stocks)
            vol_multiplier = 2.0 * (1 - confidence * 0.5)  # Higher confidence = tighter stops
            stop_distance = volatility * vol_multiplier
            return current_price * (1 - stop_distance)
        
        elif method == 'percentage':
            # Fixed percentage stop loss
            stop_percentage = self.config.get('stop_loss_percentage', 0.08)
            return current_price * (1 - stop_percentage)
        
        elif method == 'atr_based':
            # ATR-based stop loss (would need ATR data)
            atr_multiplier = 2.0
            # Placeholder: use volatility as ATR proxy
            atr = current_price * volatility * 0.5
            return current_price - (atr * atr_multiplier)
        
        else:
            # Default to 8% stop loss
            return current_price * 0.92
    
    def _calculate_take_profit(self, current_price: float, expected_return: float, 
                             volatility: float) -> float:
        """Calculate take profit level"""
        
        # Risk-reward based take profit
        risk_reward_ratio = self.config.get('target_risk_reward', 2.0)
        
        # Calculate stop loss distance
        stop_loss = self._calculate_stop_loss(current_price, volatility, 0.5)
        stop_distance = current_price - stop_loss
        
        # Take profit at risk-reward multiple
        take_profit = current_price + (stop_distance * risk_reward_ratio)
        
        # Don't exceed reasonable expected return
        max_gain = current_price * (1 + abs(expected_return) * 2)
        take_profit = min(take_profit, max_gain)
        
        return take_profit
    
    def _calculate_position_var(self, shares: float, price: float, volatility: float) -> float:
        """Calculate position-level Value at Risk"""
        
        position_value = shares * price
        confidence_level = self.config.get('confidence_level', 0.95)
        
        # Parametric VAR
        z_score = stats.norm.ppf(1 - confidence_level)
        daily_var = position_value * volatility * z_score / np.sqrt(252)
        
        return abs(daily_var)
    
    def _estimate_portfolio_risk_contribution(self, symbol: str, shares: float, 
                                            price: float) -> float:
        """Estimate how much this position contributes to portfolio risk"""
        
        position_value = shares * price
        
        # Simplified risk contribution
        # In practice, would use marginal contribution to portfolio VAR
        
        return position_value * 0.001  # Placeholder
    
    def _conservative_position_size(self, portfolio_value: float, current_price: float) -> Dict[str, Any]:
        """Conservative position sizing when data is limited"""
        
        conservative_allocation = 0.02  # 2% of portfolio
        position_value = portfolio_value * conservative_allocation
        shares = position_value / current_price
        
        return {
            'optimal_shares': int(shares),
            'position_value': position_value,
            'position_weight': conservative_allocation,
            'stop_loss_price': current_price * 0.92,  # 8% stop loss
            'take_profit_price': current_price * 1.16,  # 16% take profit (2:1 ratio)
            'risk_adjusted_sizing': False,
            'note': 'Conservative sizing due to limited data'
        }
    
    def monitor_portfolio_risk(self, positions: Dict[str, Dict], market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Real-time portfolio risk monitoring"""
        
        logger.info("📊 Monitoring portfolio risk...")
        
        if not positions:
            return {'status': 'No positions to monitor'}
        
        portfolio_metrics = {}
        
        # Calculate portfolio-level metrics
        total_value = sum(pos['position_value'] for pos in positions.values())
        
        # Portfolio beta
        portfolio_beta = self._calculate_portfolio_beta(positions, market_data)
        
        # Portfolio VAR
        portfolio_var = self._calculate_portfolio_var(positions, market_data)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(positions)
        
        # Sector exposure
        sector_exposure = self._calculate_sector_exposure(positions)
        
        # Correlation risk
        correlation_risk = self._calculate_correlation_risk(positions, market_data)
        
        # Risk alerts
        alerts = []
        
        if portfolio_beta > self.max_portfolio_beta:
            alerts.append(f"Portfolio beta ({portfolio_beta:.2f}) exceeds limit ({self.max_portfolio_beta})")
        
        if portfolio_var > self.max_daily_var * total_value:
            alerts.append(f"Portfolio VAR (${portfolio_var:,.0f}) exceeds limit")
        
        if concentration_risk['max_position_weight'] > self.max_position_size:
            alerts.append(f"Position concentration ({concentration_risk['max_position_weight']:.1%}) exceeds limit")
        
        portfolio_metrics = {
            'total_portfolio_value': total_value,
            'portfolio_beta': portfolio_beta,
            'portfolio_var': portfolio_var,
            'concentration_risk': concentration_risk,
            'sector_exposure': sector_exposure,
            'correlation_risk': correlation_risk,
            'risk_alerts': alerts,
            'risk_score': self._calculate_overall_risk_score(portfolio_beta, portfolio_var, 
                                                           concentration_risk, len(alerts)),
            'last_updated': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Portfolio risk monitored - Risk Score: {portfolio_metrics['risk_score']:.1f}/10")
        
        return portfolio_metrics
    
    def _calculate_portfolio_beta(self, positions: Dict[str, Dict], 
                                market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio beta vs market"""
        
        # Simplified beta calculation
        # In practice, would use regression vs market index
        
        weighted_beta = 0
        total_weight = 0
        
        for symbol, position in positions.items():
            # Placeholder beta (would need actual calculation)
            stock_beta = 1.0  # Default to market beta
            weight = position['position_weight']
            
            weighted_beta += stock_beta * weight
            total_weight += weight
        
        return weighted_beta / total_weight if total_weight > 0 else 1.0
    
    def _calculate_portfolio_var(self, positions: Dict[str, Dict], 
                               market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate portfolio Value at Risk"""
        
        # Simplified VAR calculation
        # In practice, would use Monte Carlo or historical simulation
        
        portfolio_var = 0
        
        for symbol, position in positions.items():
            position_var = position.get('position_var', position['position_value'] * 0.02)
            portfolio_var += position_var ** 2  # Assuming independence (simplified)
        
        # Portfolio VAR (square root of sum of squares - independence assumption)
        return np.sqrt(portfolio_var)
    
    def _calculate_concentration_risk(self, positions: Dict[str, Dict]) -> Dict[str, Any]:
        """Calculate position concentration metrics"""
        
        if not positions:
            return {'max_position_weight': 0, 'herfindahl_index': 0}
        
        weights = [pos['position_weight'] for pos in positions.values()]
        
        max_weight = max(weights)
        herfindahl_index = sum(w ** 2 for w in weights)
        
        return {
            'max_position_weight': max_weight,
            'herfindahl_index': herfindahl_index,
            'effective_number_of_positions': 1 / herfindahl_index if herfindahl_index > 0 else 0,
            'concentration_score': min(10, max_weight * 20)  # 0-10 scale
        }
    
    def _calculate_sector_exposure(self, positions: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate sector exposure (placeholder)"""
        
        # In practice, would map symbols to sectors
        return {
            'Technology': 0.30,
            'Healthcare': 0.20,
            'Financials': 0.15,
            'Consumer': 0.35
        }
    
    def _calculate_correlation_risk(self, positions: Dict[str, Dict], 
                                  market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate correlation-based risk measures"""
        
        # Simplified correlation risk
        # In practice, would calculate correlation matrix and eigenvalues
        
        num_positions = len(positions)
        
        # Placeholder correlation risk
        avg_correlation = 0.3 if num_positions > 5 else 0.5
        
        return {
            'average_correlation': avg_correlation,
            'correlation_risk_score': avg_correlation * 10,  # 0-10 scale
            'diversification_benefit': 1 - avg_correlation
        }
    
    def _calculate_overall_risk_score(self, beta: float, var: float, concentration: Dict, 
                                    num_alerts: int) -> float:
        """Calculate overall portfolio risk score (0-10 scale)"""
        
        # Risk score components
        beta_score = min(10, abs(beta - 1) * 10)
        concentration_score = concentration.get('concentration_score', 5)
        alert_score = min(10, num_alerts * 2)
        
        # Weighted average
        overall_score = (beta_score * 0.3 + concentration_score * 0.4 + alert_score * 0.3)
        
        return max(0, min(10, overall_score))
    
    def generate_risk_report(self, portfolio_analysis: Dict[str, Any]) -> str:
        """Generate human-readable risk report"""
        
        report = f"""
🛡️ PORTFOLIO RISK REPORT
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 PORTFOLIO OVERVIEW
Portfolio Value: ${portfolio_analysis.get('total_portfolio_value', 0):,.0f}
Risk Score: {portfolio_analysis.get('risk_score', 0):.1f}/10
Portfolio Beta: {portfolio_analysis.get('portfolio_beta', 1.0):.2f}

⚠️ RISK ALERTS ({len(portfolio_analysis.get('risk_alerts', []))})
"""
        
        for alert in portfolio_analysis.get('risk_alerts', []):
            report += f"• {alert}\n"
        
        if not portfolio_analysis.get('risk_alerts'):
            report += "• No risk alerts - portfolio within limits\n"
        
        concentration = portfolio_analysis.get('concentration_risk', {})
        report += f"""
📈 CONCENTRATION ANALYSIS
Max Position Weight: {concentration.get('max_position_weight', 0):.1%}
Effective Positions: {concentration.get('effective_number_of_positions', 0):.1f}
Concentration Score: {concentration.get('concentration_score', 0):.1f}/10

💼 SECTOR EXPOSURE
"""
        
        sector_exposure = portfolio_analysis.get('sector_exposure', {})
        for sector, exposure in sector_exposure.items():
            report += f"{sector}: {exposure:.1%}\n"
        
        report += f"""
🔗 CORRELATION RISK
Average Correlation: {portfolio_analysis.get('correlation_risk', {}).get('average_correlation', 0):.2f}
Diversification Benefit: {portfolio_analysis.get('correlation_risk', {}).get('diversification_benefit', 0):.2f}

📋 RECOMMENDATIONS
• Maintain position sizes below {self.max_position_size:.1%} limit
• Monitor sector concentration limits
• Review correlations during market stress
• Update stop losses based on volatility changes
        """
        
        return report

# Convenience functions
def calculate_optimal_position(symbol: str, prediction_data: Dict, portfolio_value: float, 
                             current_price: float) -> Dict[str, Any]:
    """
    One-function interface for position sizing
    """
    risk_manager = ProfessionalRiskManager()
    return risk_manager.calculate_position_size(symbol, prediction_data, portfolio_value, current_price)

def monitor_portfolio(positions: Dict, market_data: Dict = None) -> Dict[str, Any]:
    """
    One-function interface for portfolio risk monitoring
    """
    risk_manager = ProfessionalRiskManager()
    return risk_manager.monitor_portfolio_risk(positions, market_data or {})