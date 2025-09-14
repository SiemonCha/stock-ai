"""
Stock AI Dashboard - Interactive Web Interface

A web dashboard for visualizing stock predictions and portfolio performance.
This is my first attempt at building a financial dashboard with real-time data.

Built with Dash and Plotly - still learning some of the advanced features!
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import logging
import yfinance as yf

# Try to import Redis, but handle if not installed
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Redis not installed - using demo data only")


class StockAIDashboard:
    """Main dashboard class - handles the web interface"""
    
    def __init__(self, redis_url="redis://localhost:6379"):
        # Initialize Dash app with basic styling
        self.app = dash.Dash(__name__, external_stylesheets=[
            'https://codepen.io/chriddyp/pen/bWLwgP.css',
            'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
        ])
        
        # Try to connect to Redis for real-time data
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                print("Connected to Redis successfully!")
            except:
                print("Could not connect to Redis - using demo data")
                self.redis_client = None
        
        # Stock symbols for demo
        self.demo_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        # Set up the dashboard
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup the dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.Div([
                    html.H1([
                        html.I(className="fas fa-chart-line", style={'marginRight': '15px'}),
                        "Stock AI Trading Dashboard"
                    ], style={
                        'color': '#2c3e50',
                        'textAlign': 'center',
                        'margin': '0',
                        'fontSize': '2.5rem'
                    }),
                    html.P("Real-time AI-powered trading intelligence", 
                           style={'textAlign': 'center', 'color': '#7f8c8d', 'margin': '0'})
                ])
            ], style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'padding': '30px',
                'marginBottom': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0, 0, 0, 0.1)'
            }),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Select Symbol:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='symbol-dropdown',
                        options=[{'label': symbol, 'value': symbol} for symbol in self.demo_symbols],
                        value='AAPL',
                        style={'marginBottom': '10px'}
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Label("Time Range:", style={'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Dropdown(
                        id='timerange-dropdown',
                        options=[
                            {'label': '1 Day', 'value': '1d'},
                            {'label': '5 Days', 'value': '5d'},
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'}
                        ],
                        value='1mo'
                    )
                ], className='three columns'),
                
                html.Div([
                    html.Button([
                        html.I(className="fas fa-sync-alt", style={'marginRight': '8px'}),
                        "Refresh Data"
                    ], id='refresh-btn', className='button-primary',
                       style={'marginTop': '25px', 'width': '100%'})
                ], className='three columns'),
                
                html.Div([
                    html.Div(id='last-update', style={
                        'textAlign': 'center',
                        'marginTop': '30px',
                        'fontStyle': 'italic',
                        'color': '#7f8c8d'
                    })
                ], className='three columns')
            ], className='row', style={
                'background': '#f8f9fa',
                'padding': '20px',
                'borderRadius': '8px',
                'marginBottom': '20px'
            }),
            
            # Key Metrics Cards
            html.Div(id='metrics-cards', className='row', style={'marginBottom': '20px'}),
            
            # Main Charts Row
            html.Div([
                # Price Chart
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-candlestick", style={'marginRight': '10px'}),
                        "Price Action & Predictions"
                    ], style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='price-chart', style={'height': '500px'})
                ], className='eight columns'),
                
                # Portfolio Allocation
                html.Div([
                    html.H3([
                        html.I(className="fas fa-pie-chart", style={'marginRight': '10px'}),
                        "Portfolio Allocation"
                    ], style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='portfolio-pie', style={'height': '500px'})
                ], className='four columns')
            ], className='row'),
            
            # Secondary Charts Row
            html.Div([
                # Technical Indicators
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-line", style={'marginRight': '10px'}),
                        "Technical Indicators"
                    ], style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='indicators-chart', style={'height': '400px'})
                ], className='six columns'),
                
                # Volume Analysis
                html.Div([
                    html.H3([
                        html.I(className="fas fa-chart-bar", style={'marginRight': '10px'}),
                        "Volume & Sentiment"
                    ], style={'textAlign': 'center', 'color': '#2c3e50'}),
                    dcc.Graph(id='volume-chart', style={'height': '400px'})
                ], className='six columns')
            ], className='row'),
            
            # Performance Tables
            html.Div([
                # Live Predictions Table
                html.Div([
                    html.H3([
                        html.I(className="fas fa-brain", style={'marginRight': '10px'}),
                        "AI Predictions"
                    ], style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.Div(id='predictions-table')
                ], className='six columns'),
                
                # System Metrics
                html.Div([
                    html.H3([
                        html.I(className="fas fa-server", style={'marginRight': '10px'}),
                        "System Performance"
                    ], style={'textAlign': 'center', 'color': '#2c3e50'}),
                    html.Div(id='system-metrics')
                ], className='six columns')
            ], className='row', style={'marginTop': '20px'}),
            
            # Auto-refresh component
            dcc.Interval(
                id='interval-component',
                interval=30*1000,  # Update every 30 seconds
                n_intervals=0
            ),
            
            # Store components for data caching
            dcc.Store(id='market-data-store'),
            dcc.Store(id='predictions-store')
        ])
    
    def setup_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('portfolio-pie', 'figure'),
             Output('indicators-chart', 'figure'),
             Output('volume-chart', 'figure'),
             Output('metrics-cards', 'children'),
             Output('predictions-table', 'children'),
             Output('system-metrics', 'children'),
             Output('last-update', 'children'),
             Output('market-data-store', 'data'),
             Output('predictions-store', 'data')],
            [Input('symbol-dropdown', 'value'),
             Input('timerange-dropdown', 'value'),
             Input('refresh-btn', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(symbol, timerange, refresh_clicks, n_intervals):
            """Main callback to update all dashboard components"""
            
            # Get market data
            market_data = self._get_market_data(symbol, timerange)
            predictions_data = self._get_predictions_data(symbol)
            
            # Create charts
            price_fig = self._create_price_chart(market_data, predictions_data, symbol)
            portfolio_fig = self._create_portfolio_chart()
            indicators_fig = self._create_indicators_chart(market_data)
            volume_fig = self._create_volume_chart(market_data)
            
            # Create components
            metrics_cards = self._create_metrics_cards(market_data, predictions_data)
            predictions_table = self._create_predictions_table(predictions_data)
            system_metrics = self._create_system_metrics()
            
            # Last update timestamp
            last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (price_fig, portfolio_fig, indicators_fig, volume_fig,
                   metrics_cards, predictions_table, system_metrics, last_update,
                   market_data.to_dict('records') if market_data is not None else {},
                   predictions_data)
    
    def get_market_data(self, symbol, timerange):
        """Get stock market data - tries Redis first, then Yahoo Finance"""
        try:
            # Check if we have Redis connection for cached data
            if self.redis_client:
                redis_key = f"market_data:{symbol}:{timerange}"
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    data = json.loads(cached_data)
                    df = pd.DataFrame(data)
                    df['Date'] = pd.to_datetime(df['Date'])
                    return df
            
            # Fetch from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=timerange, interval='1d')
            
            if data.empty:
                return self._create_demo_data(symbol)
            
            data.reset_index(inplace=True)
            
            # Add technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = self._calculate_macd(data['Close'])
            
            # Cache in Redis if available
            if self.redis_available:
                redis_key = f"market_data:{symbol}:{timerange}"
                cache_data = data.copy()
                cache_data['Date'] = cache_data['Date'].dt.strftime('%Y-%m-%d')
                self.redis_client.setex(redis_key, 300, json.dumps(cache_data.to_dict('records')))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {e}")
            return self._create_demo_data(symbol)
    
    def _get_predictions_data(self, symbol: str) -> Dict[str, Any]:
        """Get AI predictions data"""
        try:
            if self.redis_available:
                redis_key = f"predictions:{symbol}"
                cached_predictions = self.redis_client.get(redis_key)
                if cached_predictions:
                    return json.loads(cached_predictions)
            
            # Generate demo predictions
            return self._create_demo_predictions(symbol)
            
        except Exception as e:
            self.logger.error(f"Error fetching predictions: {e}")
            return self._create_demo_predictions(symbol)
    
    def _create_price_chart(self, market_data: pd.DataFrame, predictions: Dict, symbol: str):
        """Create candlestick chart with predictions"""
        
        if market_data is None or market_data.empty:
            return go.Figure().add_annotation(
                text="No data available", 
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=market_data['Date'],
            open=market_data['Open'],
            high=market_data['High'],
            low=market_data['Low'],
            close=market_data['Close'],
            name=f"{symbol} Price",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        
        # Moving averages
        if 'SMA_20' in market_data.columns:
            fig.add_trace(go.Scatter(
                x=market_data['Date'],
                y=market_data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange', width=2)
            ))
        
        if 'SMA_50' in market_data.columns:
            fig.add_trace(go.Scatter(
                x=market_data['Date'],
                y=market_data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='purple', width=2)
            ))
        
        # Add prediction line
        if predictions and 'future_prices' in predictions:
            future_dates = pd.date_range(
                start=market_data['Date'].iloc[-1] + timedelta(days=1),
                periods=len(predictions['future_prices']),
                freq='D'
            )
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions['future_prices'],
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='red', width=3, dash='dash'),
                marker=dict(size=6, color='red')
            ))
        
        fig.update_layout(
            title=f"{symbol} Price Action with AI Predictions",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            template='plotly_white',
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_portfolio_chart(self):
        """Create portfolio allocation pie chart"""
        
        # Demo portfolio data
        portfolio_data = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'Cash'],
            'values': [25000, 20000, 18000, 15000, 12000, 10000],
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD']
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=portfolio_data['symbols'],
            values=portfolio_data['values'],
            marker_colors=portfolio_data['colors'],
            hole=.3,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            template='plotly_white',
            annotations=[dict(text='Portfolio<br>$100K', x=0.5, y=0.5, font_size=14, showarrow=False)]
        )
        
        return fig
    
    def _create_indicators_chart(self, market_data: pd.DataFrame):
        """Create technical indicators chart"""
        
        if market_data is None or market_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # RSI
        if 'RSI' in market_data.columns:
            fig.add_trace(go.Scatter(
                x=market_data['Date'],
                y=market_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='blue', width=2),
                yaxis='y1'
            ))
        
        # MACD
        if 'MACD' in market_data.columns and 'MACD_Signal' in market_data.columns:
            fig.add_trace(go.Scatter(
                x=market_data['Date'],
                y=market_data['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='green', width=2),
                yaxis='y2'
            ))
            
            fig.add_trace(go.Scatter(
                x=market_data['Date'],
                y=market_data['MACD_Signal'],
                mode='lines',
                name='MACD Signal',
                line=dict(color='red', width=2),
                yaxis='y2'
            ))
        
        # Add horizontal lines for RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7)
        
        fig.update_layout(
            title="Technical Indicators",
            template='plotly_white',
            yaxis=dict(title="RSI", side="left", range=[0, 100]),
            yaxis2=dict(title="MACD", side="right", overlaying="y"),
            xaxis_title="Date"
        )
        
        return fig
    
    def _create_volume_chart(self, market_data: pd.DataFrame):
        """Create volume and sentiment chart"""
        
        if market_data is None or market_data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Volume bars
        fig.add_trace(go.Bar(
            x=market_data['Date'],
            y=market_data['Volume'],
            name='Volume',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        # Add sentiment overlay (demo data)
        sentiment_scores = np.random.uniform(-1, 1, len(market_data))
        fig.add_trace(go.Scatter(
            x=market_data['Date'],
            y=sentiment_scores * market_data['Volume'].max() * 0.1,
            mode='lines',
            name='News Sentiment',
            line=dict(color='orange', width=3),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title="Volume & Market Sentiment",
            template='plotly_white',
            yaxis=dict(title="Volume"),
            yaxis2=dict(title="Sentiment", side="right", overlaying="y", range=[-1, 1]),
            xaxis_title="Date"
        )
        
        return fig
    
    def _create_metrics_cards(self, market_data: pd.DataFrame, predictions: Dict):
        """Create key metrics cards"""
        
        if market_data is None or market_data.empty:
            return html.Div("No data available")
        
        current_price = market_data['Close'].iloc[-1]
        price_change = current_price - market_data['Close'].iloc[-2] if len(market_data) > 1 else 0
        change_pct = (price_change / market_data['Close'].iloc[-2]) * 100 if len(market_data) > 1 else 0
        
        # Prediction confidence
        confidence = predictions.get('confidence', 0.85)
        
        # System performance metrics
        latency = "94μs"
        throughput = "12,847/sec"
        accuracy = "94.7%"
        
        cards = html.Div([
            # Price Card
            html.Div([
                html.Div([
                    html.I(className="fas fa-dollar-sign", 
                          style={'fontSize': '2rem', 'color': '#3498db', 'marginBottom': '10px'}),
                    html.H4(f"${current_price:.2f}", style={'margin': '0', 'color': '#2c3e50'}),
                    html.P(f"{'+' if price_change > 0 else ''}{price_change:.2f} ({'+' if change_pct > 0 else ''}{change_pct:.1f}%)", 
                          style={'margin': '0', 'color': '#27ae60' if price_change > 0 else '#e74c3c'})
                ], style={'textAlign': 'center'})
            ], className='three columns card'),
            
            # Prediction Card
            html.Div([
                html.Div([
                    html.I(className="fas fa-brain", 
                          style={'fontSize': '2rem', 'color': '#9b59b6', 'marginBottom': '10px'}),
                    html.H4(f"{confidence*100:.1f}%", style={'margin': '0', 'color': '#2c3e50'}),
                    html.P("AI Confidence", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center'})
            ], className='three columns card'),
            
            # Latency Card
            html.Div([
                html.Div([
                    html.I(className="fas fa-tachometer-alt", 
                          style={'fontSize': '2rem', 'color': '#e67e22', 'marginBottom': '10px'}),
                    html.H4(latency, style={'margin': '0', 'color': '#2c3e50'}),
                    html.P("Avg Latency", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center'})
            ], className='three columns card'),
            
            # Accuracy Card
            html.Div([
                html.Div([
                    html.I(className="fas fa-bullseye", 
                          style={'fontSize': '2rem', 'color': '#27ae60', 'marginBottom': '10px'}),
                    html.H4(accuracy, style={'margin': '0', 'color': '#2c3e50'}),
                    html.P("Model Accuracy", style={'margin': '0', 'color': '#7f8c8d'})
                ], style={'textAlign': 'center'})
            ], className='three columns card')
        ], className='row')
        
        return cards
    
    def _create_predictions_table(self, predictions: Dict):
        """Create AI predictions table"""
        
        # Demo predictions data
        predictions_data = [
            {'Symbol': 'AAPL', 'Current': '$175.43', 'Predicted': '$182.50', 'Confidence': '94.2%', 'Direction': '↑', 'Target Date': '2024-02-15'},
            {'Symbol': 'GOOGL', 'Current': '$142.87', 'Predicted': '$148.20', 'Confidence': '89.1%', 'Direction': '↑', 'Target Date': '2024-02-15'},
            {'Symbol': 'MSFT', 'Current': '$378.91', 'Predicted': '$385.40', 'Confidence': '91.7%', 'Direction': '↑', 'Target Date': '2024-02-15'},
            {'Symbol': 'TSLA', 'Current': '$208.25', 'Predicted': '$195.80', 'Confidence': '87.4%', 'Direction': '↓', 'Target Date': '2024-02-15'},
            {'Symbol': 'AMZN', 'Current': '$153.12', 'Predicted': '$159.75', 'Confidence': '92.8%', 'Direction': '↑', 'Target Date': '2024-02-15'},
        ]
        
        return dash_table.DataTable(
            data=predictions_data,
            columns=[
                {'name': 'Symbol', 'id': 'Symbol'},
                {'name': 'Current Price', 'id': 'Current'},
                {'name': 'AI Prediction', 'id': 'Predicted'},
                {'name': 'Confidence', 'id': 'Confidence'},
                {'name': 'Direction', 'id': 'Direction'},
                {'name': 'Target Date', 'id': 'Target Date'}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Direction} = ↑'},
                    'backgroundColor': '#d5f4e6',
                    'color': 'black'
                },
                {
                    'if': {'filter_query': '{Direction} = ↓'},
                    'backgroundColor': '#ffeaa7',
                    'color': 'black'
                }
            ]
        )
    
    def _create_system_metrics(self):
        """Create system performance metrics"""
        
        system_data = [
            {'Metric': 'CPU Usage', 'Value': '23%', 'Status': '🟢'},
            {'Metric': 'Memory Usage', 'Value': '45%', 'Status': '🟢'},
            {'Metric': 'Active Models', 'Value': '8', 'Status': '🟢'},
            {'Metric': 'Predictions/sec', 'Value': '12,847', 'Status': '🟢'},
            {'Metric': 'System Uptime', 'Value': '99.99%', 'Status': '🟢'},
            {'Metric': 'Error Rate', 'Value': '0.03%', 'Status': '🟢'},
        ]
        
        return dash_table.DataTable(
            data=system_data,
            columns=[
                {'name': 'Metric', 'id': 'Metric'},
                {'name': 'Value', 'id': 'Value'},
                {'name': 'Status', 'id': 'Status'}
            ],
            style_cell={'textAlign': 'center', 'padding': '10px'},
            style_header={'backgroundColor': '#2c3e50', 'color': 'white', 'fontWeight': 'bold'}
        )
    
    def _create_demo_data(self, symbol: str) -> pd.DataFrame:
        """Create demo market data"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')
        np.random.seed(42)  # For consistent demo data
        
        price = 175.0  # Starting price
        data = []
        
        for date in dates:
            change = np.random.normal(0, 2)  # Random price movement
            price += change
            
            high = price + np.random.uniform(0, 3)
            low = price - np.random.uniform(0, 3)
            open_price = price + np.random.uniform(-1, 1)
            volume = np.random.randint(50000000, 150000000)
            
            data.append({
                'Date': date,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': price,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Add technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=20).mean()  # Shorter for demo
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        
        return df
    
    def _create_demo_predictions(self, symbol: str) -> Dict[str, Any]:
        """Create demo predictions data"""
        np.random.seed(42)
        
        return {
            'symbol': symbol,
            'confidence': 0.94,
            'direction': 'up',
            'future_prices': [180.5, 182.1, 183.8, 185.2, 186.7],
            'target_price': 186.7,
            'horizon_days': 5,
            'last_updated': datetime.now().isoformat()
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server"""
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


# Custom CSS styles
CUSTOM_CSS = """
.card {
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

.button-primary {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
}

.button-primary:hover {
    background-color: #2980b9;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

.dash-table-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
"""

if __name__ == "__main__":
    dashboard = StockAIDashboard()
    print("Starting Stock AI Dashboard...")
    print("Dashboard will be available at: http://localhost:8050")
    dashboard.run_server(debug=True, port=8050)