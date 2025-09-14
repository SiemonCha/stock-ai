"""
Stock AI Dashboard - My First Financial Web App

This is my attempt at building a stock trading dashboard for my CS project.
Still learning Dash and Plotly, so some parts might be a bit messy!

Features I managed to implement:
- Real-time stock charts
- Basic portfolio view
- Simple predictions display

TODO: 
- Fix the refresh bug
- Add more chart types
- Improve the styling
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import yfinance as yf

# Try Redis connection but don't break if it's not available
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()
    REDIS_WORKS = True
    print("✅ Redis connected!")
except:
    REDIS_WORKS = False
    print("⚠️ Redis not available - using demo data")


class StockDashboard:
    """
    My stock dashboard class
    Probably not the most efficient code but it works!
    """
    
    def __init__(self):
        # Create the Dash app
        self.app = dash.Dash(__name__)
        
        # Some basic styling (found this CSS online)
        self.app.layout = self.create_layout()
        
        # Set up the callbacks (still figuring out how these work)
        self.setup_callbacks()
        
        # Demo stock symbols
        self.stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
        
    def create_layout(self):
        """Create the dashboard layout"""
        return html.Div([
            # Main title
            html.H1("Stock AI Dashboard", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
            
            html.P("My first attempt at a financial dashboard 📈", 
                   style={'textAlign': 'center', 'color': '#7f8c8d'}),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Select Stock:"),
                    dcc.Dropdown(
                        id='stock-dropdown',
                        options=[{'label': stock, 'value': stock} for stock in self.stocks],
                        value='AAPL'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Time Period:"),
                    dcc.Dropdown(
                        id='period-dropdown',
                        options=[
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '1 Year', 'value': '1y'}
                        ],
                        value='3mo'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),
                
                html.Div([
                    html.Button('Refresh Data', id='refresh-btn', 
                               style={'marginTop': '25px', 'padding': '10px 20px'})
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
            ], style={'marginBottom': '30px'}),
            
            # Charts section
            html.Div([
                # Stock price chart
                html.Div([
                    dcc.Graph(id='stock-chart')
                ], style={'width': '70%', 'display': 'inline-block'}),
                
                # Portfolio pie chart (demo data)
                html.Div([
                    dcc.Graph(id='portfolio-chart')
                ], style={'width': '30%', 'display': 'inline-block'})
            ]),
            
            # Predictions table
            html.Div([
                html.H3("AI Predictions (Demo)", style={'textAlign': 'center'}),
                html.Div(id='predictions-table')
            ], style={'marginTop': '30px'}),
            
            # Auto-refresh every minute (maybe too frequent?)
            dcc.Interval(
                id='interval-update',
                interval=60*1000,  # 60 seconds
                n_intervals=0
            )
        ], style={'margin': '20px'})
    
    def setup_callbacks(self):
        """Set up the callback functions"""
        
        @self.app.callback(
            [Output('stock-chart', 'figure'),
             Output('portfolio-chart', 'figure'),
             Output('predictions-table', 'children')],
            [Input('stock-dropdown', 'value'),
             Input('period-dropdown', 'value'),
             Input('refresh-btn', 'n_clicks'),
             Input('interval-update', 'n_intervals')]
        )
        def update_dashboard(stock, period, refresh_clicks, intervals):
            # Get stock data
            stock_data = self.get_stock_data(stock, period)
            
            # Create charts
            stock_fig = self.create_stock_chart(stock_data, stock)
            portfolio_fig = self.create_portfolio_chart()
            predictions_table = self.create_predictions_table()
            
            return stock_fig, portfolio_fig, predictions_table
    
    def get_stock_data(self, symbol, period):
        """
        Get stock data from Yahoo Finance
        Should probably add error handling here...
        """
        try:
            # Try to get from cache first (if Redis is working)
            if REDIS_WORKS:
                cache_key = f"stock_data:{symbol}:{period}"
                cached = redis_client.get(cache_key)
                if cached:
                    data = json.loads(cached)
                    return pd.DataFrame(data)
            
            # Get from Yahoo Finance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if not data.empty:
                data.reset_index(inplace=True)
                
                # Add some simple technical indicators
                data['SMA_20'] = data['Close'].rolling(window=20).mean()
                data['SMA_50'] = data['Close'].rolling(window=50).mean()
                
                # Cache the data (if Redis works)
                if REDIS_WORKS:
                    cache_data = data.copy()
                    cache_data['Date'] = cache_data['Date'].astype(str)
                    redis_client.setex(cache_key, 300, json.dumps(cache_data.to_dict('records')))
                
                return data
            else:
                return self.create_demo_data(symbol)
                
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            return self.create_demo_data(symbol)
    
    def create_stock_chart(self, data, symbol):
        """Create the main stock chart"""
        if data is None or data.empty:
            return go.Figure().add_annotation(
                text="No data available", 
                x=0.5, y=0.5, showarrow=False
            )
        
        fig = go.Figure()
        
        # Main price line
        fig.add_trace(go.Scatter(
            x=data['Date'],
            y=data['Close'],
            mode='lines',
            name=f'{symbol} Price',
            line=dict(color='blue', width=2)
        ))
        
        # Moving averages (if we have enough data)
        if len(data) > 20 and 'SMA_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['SMA_20'],
                mode='lines',
                name='20-day MA',
                line=dict(color='orange', width=1)
            ))
        
        if len(data) > 50 and 'SMA_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data['Date'],
                y=data['SMA_50'],
                mode='lines',
                name='50-day MA',
                line=dict(color='red', width=1)
            ))
        
        # Add some demo predictions (just fake data for now)
        if len(data) > 5:
            future_dates = [data['Date'].iloc[-1] + timedelta(days=i) for i in range(1, 8)]
            last_price = data['Close'].iloc[-1]
            # Simple trend prediction (very basic)
            trend = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / 5
            future_prices = [last_price + trend * i + np.random.normal(0, 1) for i in range(1, 8)]
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_prices,
                mode='lines+markers',
                name='AI Prediction',
                line=dict(color='red', dash='dash'),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"{symbol} Stock Price with Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        
        return fig
    
    def create_portfolio_chart(self):
        """Create a simple portfolio pie chart with demo data"""
        # Just demo data for now
        portfolio = {
            'stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'Cash'],
            'values': [30, 25, 20, 15, 10],
            'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=portfolio['stocks'],
            values=portfolio['values'],
            marker_colors=portfolio['colors'],
            hole=.3
        )])
        
        fig.update_layout(
            title="Portfolio Allocation (%)",
            annotations=[dict(text='Demo<br>Portfolio', x=0.5, y=0.5, font_size=12, showarrow=False)]
        )
        
        return fig
    
    def create_predictions_table(self):
        """Create predictions table with demo data"""
        # Demo predictions data (would be from ML models in real version)
        predictions = [
            {'Stock': 'AAPL', 'Current': '$175.43', 'Predicted': '$182.50', 'Confidence': '87%', 'Direction': '↑'},
            {'Stock': 'GOOGL', 'Current': '$142.87', 'Predicted': '$148.20', 'Confidence': '82%', 'Direction': '↑'},
            {'Stock': 'MSFT', 'Current': '$378.91', 'Predicted': '$385.40', 'Confidence': '79%', 'Direction': '↑'},
            {'Stock': 'TSLA', 'Current': '$208.25', 'Predicted': '$195.80', 'Confidence': '74%', 'Direction': '↓'},
        ]
        
        return dash_table.DataTable(
            data=predictions,
            columns=[
                {'name': 'Stock', 'id': 'Stock'},
                {'name': 'Current Price', 'id': 'Current'},
                {'name': 'Predicted Price', 'id': 'Predicted'},
                {'name': 'Confidence', 'id': 'Confidence'},
                {'name': 'Direction', 'id': 'Direction'}
            ],
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': '#3498db', 'color': 'white'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{Direction} = ↑'},
                    'backgroundColor': '#d5f4e6'
                },
                {
                    'if': {'filter_query': '{Direction} = ↓'},
                    'backgroundColor': '#ffeaa7'
                }
            ]
        )
    
    def create_demo_data(self, symbol):
        """Create some demo data when real data isn't available"""
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        np.random.seed(42)  # So the demo data is consistent
        
        # Start with a base price
        base_price = {'AAPL': 175, 'GOOGL': 140, 'MSFT': 380, 'TSLA': 200, 'NVDA': 700}.get(symbol, 100)
        
        prices = []
        current_price = base_price
        
        for _ in dates:
            # Random walk with slight upward bias
            change = np.random.normal(0.5, 2)  
            current_price = max(current_price + change, base_price * 0.8)  # Don't go too low
            prices.append(current_price)
        
        data = pd.DataFrame({
            'Date': dates,
            'Close': prices,
            'Open': [p + np.random.normal(0, 1) for p in prices],
            'High': [p + abs(np.random.normal(2, 1)) for p in prices],
            'Low': [p - abs(np.random.normal(2, 1)) for p in prices],
            'Volume': [np.random.randint(50000000, 150000000) for _ in prices]
        })
        
        return data
    
    def run(self, debug=True, port=8050):
        """Start the dashboard server"""
        print(f"🚀 Starting Stock AI Dashboard...")
        print(f"📊 Open your browser to: http://localhost:{port}")
        print("💡 Note: This is demo data for learning purposes!")
        
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


# Run the dashboard
if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.run(debug=True)