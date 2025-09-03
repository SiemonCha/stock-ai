import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import requests
import warnings
warnings.filterwarnings('ignore')

class FundamentalAnalyzer:
    def __init__(self):
        self.fundamental_data = {}
        
    def get_financial_metrics(self, symbol: str) -> Dict:
        """
        Get comprehensive fundamental analysis metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Financial ratios and metrics
            fundamentals = {
                # Valuation Metrics
                'pe_ratio': info.get('trailingPE', np.nan),
                'forward_pe': info.get('forwardPE', np.nan),
                'peg_ratio': info.get('pegRatio', np.nan),
                'price_to_book': info.get('priceToBook', np.nan),
                'price_to_sales': info.get('priceToSalesTrailing12Months', np.nan),
                'ev_to_revenue': info.get('enterpriseToRevenue', np.nan),
                'ev_to_ebitda': info.get('enterpriseToEbitda', np.nan),
                
                # Profitability Metrics
                'profit_margins': info.get('profitMargins', np.nan),
                'operating_margins': info.get('operatingMargins', np.nan),
                'return_on_assets': info.get('returnOnAssets', np.nan),
                'return_on_equity': info.get('returnOnEquity', np.nan),
                'revenue_growth': info.get('revenueGrowth', np.nan),
                'earnings_growth': info.get('earningsGrowth', np.nan),
                
                # Financial Health
                'current_ratio': info.get('currentRatio', np.nan),
                'quick_ratio': info.get('quickRatio', np.nan),
                'debt_to_equity': info.get('debtToEquity', np.nan),
                'total_debt': info.get('totalDebt', np.nan),
                'total_cash': info.get('totalCash', np.nan),
                'free_cash_flow': info.get('freeCashflow', np.nan),
                'operating_cash_flow': info.get('operatingCashflow', np.nan),
                
                # Market Metrics
                'market_cap': info.get('marketCap', np.nan),
                'enterprise_value': info.get('enterpriseValue', np.nan),
                'shares_outstanding': info.get('sharesOutstanding', np.nan),
                'float_shares': info.get('floatShares', np.nan),
                'shares_short': info.get('sharesShort', np.nan),
                'short_ratio': info.get('shortRatio', np.nan),
                'short_percent_float': info.get('shortPercentOfFloat', np.nan),
                
                # Dividend Information
                'dividend_yield': info.get('dividendYield', np.nan),
                'dividend_rate': info.get('dividendRate', np.nan),
                'payout_ratio': info.get('payoutRatio', np.nan),
                'five_year_avg_dividend_yield': info.get('fiveYearAvgDividendYield', np.nan),
                
                # Growth Metrics
                'quarterly_revenue_growth': info.get('quarterlyRevenueGrowth', np.nan),
                'quarterly_earnings_growth': info.get('quarterlyEarningsGrowth', np.nan),
                
                # Other Important Metrics
                'beta': info.get('beta', np.nan),
                'trailing_eps': info.get('trailingEps', np.nan),
                'forward_eps': info.get('forwardEps', np.nan),
                'book_value': info.get('bookValue', np.nan),
                'price_to_book': info.get('priceToBook', np.nan),
                
                # Business metrics
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'full_time_employees': info.get('fullTimeEmployees', np.nan),
                
                # Analyst recommendations
                'recommendation_mean': info.get('recommendationMean', np.nan),
                'recommendation_key': info.get('recommendationKey', 'none'),
                'number_of_analyst_opinions': info.get('numberOfAnalystOpinions', np.nan),
                'target_high_price': info.get('targetHighPrice', np.nan),
                'target_low_price': info.get('targetLowPrice', np.nan),
                'target_mean_price': info.get('targetMeanPrice', np.nan),
            }
            
            self.fundamental_data[symbol] = fundamentals
            return fundamentals
            
        except Exception as e:
            print(f"Error getting fundamentals for {symbol}: {str(e)}")
            return {}
    
    def calculate_financial_health_score(self, symbol: str) -> Dict:
        """
        Calculate a comprehensive financial health score
        """
        if symbol not in self.fundamental_data:
            self.get_financial_metrics(symbol)
        
        data = self.fundamental_data.get(symbol, {})
        if not data:
            return {'financial_health_score': 0, 'components': {}}
        
        scores = {}
        
        # Profitability Score (0-25 points)
        profitability_score = 0
        if data.get('return_on_equity', 0) > 0.15:
            profitability_score += 5
        elif data.get('return_on_equity', 0) > 0.10:
            profitability_score += 3
        elif data.get('return_on_equity', 0) > 0.05:
            profitability_score += 1
            
        if data.get('return_on_assets', 0) > 0.05:
            profitability_score += 5
        elif data.get('return_on_assets', 0) > 0.02:
            profitability_score += 3
        elif data.get('return_on_assets', 0) > 0:
            profitability_score += 1
            
        if data.get('profit_margins', 0) > 0.20:
            profitability_score += 5
        elif data.get('profit_margins', 0) > 0.10:
            profitability_score += 3
        elif data.get('profit_margins', 0) > 0.05:
            profitability_score += 1
            
        if data.get('operating_margins', 0) > 0.15:
            profitability_score += 5
        elif data.get('operating_margins', 0) > 0.10:
            profitability_score += 3
        elif data.get('operating_margins', 0) > 0.05:
            profitability_score += 1
            
        if data.get('revenue_growth', 0) > 0.10:
            profitability_score += 5
        elif data.get('revenue_growth', 0) > 0.05:
            profitability_score += 3
        elif data.get('revenue_growth', 0) > 0:
            profitability_score += 1
        
        scores['profitability'] = profitability_score
        
        # Financial Stability Score (0-25 points)
        stability_score = 0
        if data.get('current_ratio', 0) > 2:
            stability_score += 5
        elif data.get('current_ratio', 0) > 1.5:
            stability_score += 3
        elif data.get('current_ratio', 0) > 1:
            stability_score += 1
            
        debt_to_equity = data.get('debt_to_equity', float('inf'))
        if debt_to_equity < 0.5:
            stability_score += 10
        elif debt_to_equity < 1:
            stability_score += 5
        elif debt_to_equity < 2:
            stability_score += 2
            
        if data.get('free_cash_flow', 0) > 0:
            stability_score += 10
        
        scores['stability'] = stability_score
        
        # Valuation Score (0-25 points)
        valuation_score = 0
        pe_ratio = data.get('pe_ratio', float('inf'))
        if 10 < pe_ratio < 20:
            valuation_score += 10
        elif 5 < pe_ratio < 30:
            valuation_score += 5
        elif pe_ratio > 0:
            valuation_score += 1
            
        pb_ratio = data.get('price_to_book', float('inf'))
        if pb_ratio < 1:
            valuation_score += 5
        elif pb_ratio < 2:
            valuation_score += 3
        elif pb_ratio < 3:
            valuation_score += 1
            
        peg_ratio = data.get('peg_ratio', float('inf'))
        if 0 < peg_ratio < 1:
            valuation_score += 10
        elif 1 <= peg_ratio < 2:
            valuation_score += 5
        
        scores['valuation'] = valuation_score
        
        # Market Sentiment Score (0-25 points)
        sentiment_score = 0
        rec_mean = data.get('recommendation_mean', 3)
        if rec_mean <= 2:
            sentiment_score += 15
        elif rec_mean <= 2.5:
            sentiment_score += 10
        elif rec_mean <= 3:
            sentiment_score += 5
        
        short_ratio = data.get('short_ratio', 0)
        if short_ratio < 2:
            sentiment_score += 5
        elif short_ratio < 5:
            sentiment_score += 3
        elif short_ratio < 10:
            sentiment_score += 1
            
        beta = data.get('beta', 1)
        if 0.5 <= beta <= 1.5:
            sentiment_score += 5
        elif beta > 0:
            sentiment_score += 2
        
        scores['sentiment'] = sentiment_score
        
        # Calculate total score
        total_score = sum(scores.values())
        
        return {
            'financial_health_score': total_score,
            'max_score': 100,
            'percentage': (total_score / 100) * 100,
            'components': scores,
            'rating': self._get_rating(total_score)
        }
    
    def _get_rating(self, score: int) -> str:
        """Convert numeric score to rating"""
        if score >= 80:
            return 'Excellent'
        elif score >= 65:
            return 'Good'
        elif score >= 50:
            return 'Fair'
        elif score >= 35:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def compare_to_sector(self, symbol: str, sector_symbols: List[str]) -> Dict:
        """
        Compare company fundamentals to sector averages
        """
        if symbol not in self.fundamental_data:
            self.get_financial_metrics(symbol)
        
        company_data = self.fundamental_data.get(symbol, {})
        
        # Get sector data
        sector_data = []
        for s in sector_symbols:
            if s != symbol:
                if s not in self.fundamental_data:
                    self.get_financial_metrics(s)
                sector_data.append(self.fundamental_data.get(s, {}))
        
        # Calculate sector averages
        metrics_to_compare = [
            'pe_ratio', 'price_to_book', 'return_on_equity', 'return_on_assets',
            'debt_to_equity', 'profit_margins', 'revenue_growth', 'current_ratio'
        ]
        
        comparisons = {}
        for metric in metrics_to_compare:
            company_value = company_data.get(metric, np.nan)
            sector_values = [d.get(metric, np.nan) for d in sector_data if not np.isnan(d.get(metric, np.nan))]
            
            if sector_values:
                sector_avg = np.mean(sector_values)
                sector_median = np.median(sector_values)
                
                comparisons[metric] = {
                    'company_value': company_value,
                    'sector_average': sector_avg,
                    'sector_median': sector_median,
                    'percentile': self._calculate_percentile(company_value, sector_values),
                    'vs_sector': 'Above Average' if company_value > sector_avg else 'Below Average'
                }
        
        return comparisons
    
    def _calculate_percentile(self, value: float, values: List[float]) -> float:
        """Calculate percentile of value in list of values"""
        if np.isnan(value) or not values:
            return np.nan
        
        return (sum(1 for v in values if v <= value) / len(values)) * 100
    
    def get_growth_analysis(self, symbol: str) -> Dict:
        """
        Analyze growth trends and sustainability
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical financials
            financials = ticker.financials
            quarterly_financials = ticker.quarterly_financials
            
            if financials.empty:
                return {'error': 'No financial data available'}
            
            # Revenue growth analysis
            if 'Total Revenue' in financials.index:
                revenue_data = financials.loc['Total Revenue'].sort_index()
                revenue_growth_rates = revenue_data.pct_change().dropna()
                
                growth_analysis = {
                    'revenue_growth_trend': 'increasing' if revenue_growth_rates.iloc[-1] > revenue_growth_rates.iloc[0] else 'decreasing',
                    'avg_annual_revenue_growth': revenue_growth_rates.mean(),
                    'revenue_growth_volatility': revenue_growth_rates.std(),
                    'consistent_growth': len([x for x in revenue_growth_rates if x > 0]) / len(revenue_growth_rates)
                }
                
                return growth_analysis
            
        except Exception as e:
            print(f"Error in growth analysis for {symbol}: {str(e)}")
            return {'error': str(e)}
        
        return {'error': 'Insufficient data for analysis'}
    
    def create_fundamental_features(self, symbol: str) -> Dict:
        """
        Create feature vector from fundamental data for ML model
        """
        fundamentals = self.get_financial_metrics(symbol)
        health_score = self.calculate_financial_health_score(symbol)
        
        # Create normalized features
        features = {}
        
        # Valuation features
        features['pe_ratio_norm'] = min(fundamentals.get('pe_ratio', 50), 100) / 100
        features['pb_ratio_norm'] = min(fundamentals.get('price_to_book', 10), 20) / 20
        features['peg_ratio_norm'] = min(fundamentals.get('peg_ratio', 5), 10) / 10
        
        # Profitability features
        features['roe_norm'] = min(max(fundamentals.get('return_on_equity', 0), -0.5), 1)
        features['roa_norm'] = min(max(fundamentals.get('return_on_assets', 0), -0.2), 0.5) / 0.5
        features['profit_margin_norm'] = min(max(fundamentals.get('profit_margins', 0), -0.5), 1)
        
        # Growth features
        features['revenue_growth_norm'] = min(max(fundamentals.get('revenue_growth', 0), -0.5), 2) / 2
        features['earnings_growth_norm'] = min(max(fundamentals.get('earnings_growth', 0), -1), 3) / 3
        
        # Financial health features
        features['current_ratio_norm'] = min(fundamentals.get('current_ratio', 0), 5) / 5
        features['debt_equity_norm'] = 1 - min(fundamentals.get('debt_to_equity', 0), 5) / 5
        
        # Market sentiment features
        features['recommendation_norm'] = (5 - fundamentals.get('recommendation_mean', 3)) / 5
        features['short_ratio_norm'] = 1 - min(fundamentals.get('short_ratio', 0), 20) / 20
        
        # Health score
        features['health_score_norm'] = health_score['percentage'] / 100
        
        # Fill NaN values
        for key, value in features.items():
            if np.isnan(value):
                features[key] = 0.5  # Neutral value
        
        return features