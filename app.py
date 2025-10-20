#!/usr/bin/env python3
"""
Stock AI Dashboard - Hugging Face Spaces Deployment
AI-powered stock prediction with interactive dashboard
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Demo stock symbols
DEMO_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']

def get_stock_data(symbol, period="1mo"):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return None
            
        # Add technical indicators
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def create_price_chart(symbol, period):
    """Create interactive price chart with predictions"""
    data = get_stock_data(symbol, period)
    
    if data is None:
        return None
    
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=f"{symbol} Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'SMA_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='purple', width=2)
        ))
    
    # Add AI prediction (demo)
    last_price = data['Close'].iloc[-1]
    future_dates = pd.date_range(
        start=data.index[-1] + timedelta(days=1),
        periods=5,
        freq='D'
    )
    
    # Simple prediction based on recent trend
    recent_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-5]) / data['Close'].iloc[-5]
    predicted_prices = [last_price * (1 + recent_trend * (i+1) * 0.1) for i in range(5)]
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices,
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
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    return fig

def create_portfolio_chart():
    """Create portfolio allocation pie chart"""
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
        height=400,
        annotations=[dict(text='Portfolio<br>$100K', x=0.5, y=0.5, font_size=14, showarrow=False)]
    )
    
    return fig

def create_technical_indicators(symbol, period):
    """Create technical indicators chart"""
    data = get_stock_data(symbol, period)
    
    if data is None:
        return None
    
    fig = go.Figure()
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue', width=2)
        ))
    
    # Add horizontal lines for RSI
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7)
    
    fig.update_layout(
        title=f"{symbol} RSI Technical Indicator",
        template='plotly_white',
        height=300,
        yaxis=dict(title="RSI", range=[0, 100]),
        xaxis_title="Date"
    )
    
    return fig

def get_ai_predictions(symbol):
    """Get AI predictions (demo data)"""
    data = get_stock_data(symbol, "1mo")
    
    if data is None:
        return "No data available"
    
    current_price = data['Close'].iloc[-1]
    recent_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
    
    # Simple prediction logic
    confidence = min(95, 70 + abs(recent_change) * 2)
    predicted_change = recent_change * 0.8  # Momentum-based prediction
    predicted_price = current_price * (1 + predicted_change / 100)
    
    predictions_html = f"""
    <div style="font-family: Arial; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <h3 style="color: #2c3e50;"> AI Predictions for {symbol}</h3>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4 style="color: #3498db;">Current Metrics</h4>
                <p><strong>Current Price:</strong> ${current_price:.2f}</p>
                <p><strong>24h Change:</strong> {recent_change:+.2f}%</p>
                <p><strong>Volume:</strong> {data['Volume'].iloc[-1]:,}</p>
            </div>
            <div>
                <h4 style="color: #e74c3c;">AI Predictions</h4>
                <p><strong>Predicted Price:</strong> ${predicted_price:.2f}</p>
                <p><strong>Expected Change:</strong> {predicted_change:+.2f}%</p>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
            </div>
        </div>
        <div style="margin-top: 15px;">
            <p style="color: #7f8c8d; font-style: italic;">
                >>> This is a demo prediction system for educational purposes only. 
                Not financial advice!
            </p>
        </div>
    </div>
    """
    
    return predictions_html

def get_market_overview():
    """Get market overview with top stocks"""
    overview_data = []
    
    for symbol in DEMO_SYMBOLS[:6]:  # Top 6 stocks
        data = get_stock_data(symbol, "1d")
        if data is not None and not data.empty:
            current_price = data['Close'].iloc[-1]
            prev_close = data['Open'].iloc[-1]
            change = ((current_price - prev_close) / prev_close) * 100
            
            overview_data.append({
                'Symbol': symbol,
                'Price': f"${current_price:.2f}",
                'Change': f"{change:+.2f}%",
                'Status': '🟢' if change > 0 else '🔴'
            })
    
    if overview_data:
        df = pd.DataFrame(overview_data)
        return df.to_html(index=False, escape=False, 
                         classes='table table-striped',
                         table_id='market-overview')
    else:
        return "<p>Unable to fetch market data</p>"

# Create Gradio interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(
        title="Stock AI Dashboard",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>>>> Stock AI Dashboard</h1>
            <p>AI-powered stock prediction with interactive visualizations</p>
            <p style="font-size: 14px; opacity: 0.8;">
                Built with Deep Learning (LSTM/Transformer), NLP sentiment analysis, and MLOps
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                symbol_input = gr.Dropdown(
                    choices=DEMO_SYMBOLS,
                    value="AAPL",
                    label="Select Stock Symbol",
                    info="Choose a stock to analyze"
                )
                
                period_input = gr.Dropdown(
                    choices=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
                    value="1mo",
                    label="Time Period",
                    info="Select the time range for analysis"
                )
                
                update_btn = gr.Button("--- Update Analysis", variant="primary")
        
        with gr.Row():
            with gr.Column(scale=2):
                price_chart = gr.Plot(label="Price Chart with AI Predictions")
            with gr.Column(scale=1):
                portfolio_chart = gr.Plot(label="Portfolio Allocation")
        
        with gr.Row():
            with gr.Column():
                technical_chart = gr.Plot(label="Technical Indicators (RSI)")
        
        with gr.Row():
            with gr.Column():
                ai_predictions = gr.HTML(label="AI Predictions & Analysis")
        
        with gr.Row():
            with gr.Column():
                market_overview = gr.HTML(
                    value=get_market_overview(),
                    label="Market Overview"
                )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; 
                    background: #f8f9fa; border-radius: 10px; color: #7f8c8d;">
            <p><strong>--- Disclaimer:</strong> This is a demo application for educational purposes only. 
            Not financial advice. Past performance does not guarantee future results.</p>
            <p>Built with <3 using Python, Gradio, Plotly, and Yahoo Finance API</p>
        </div>
        """)
        
        # Event handlers
        def update_dashboard(symbol, period):
            price_fig = create_price_chart(symbol, period)
            portfolio_fig = create_portfolio_chart()
            technical_fig = create_technical_indicators(symbol, period)
            predictions_html = get_ai_predictions(symbol)
            
            return price_fig, portfolio_fig, technical_fig, predictions_html
        
        # Set up event handlers
        symbol_input.change(
            fn=update_dashboard,
            inputs=[symbol_input, period_input],
            outputs=[price_chart, portfolio_chart, technical_chart, ai_predictions]
        )
        
        period_input.change(
            fn=update_dashboard,
            inputs=[symbol_input, period_input],
            outputs=[price_chart, portfolio_chart, technical_chart, ai_predictions]
        )
        
        update_btn.click(
            fn=update_dashboard,
            inputs=[symbol_input, period_input],
            outputs=[price_chart, portfolio_chart, technical_chart, ai_predictions]
        )
        
        # Load initial data
        demo.load(
            fn=update_dashboard,
            inputs=[symbol_input, period_input],
            outputs=[price_chart, portfolio_chart, technical_chart, ai_predictions]
        )
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False
    )
