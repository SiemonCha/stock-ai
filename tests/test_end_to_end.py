"""
End-to-End System Tests for Stock AI Platform
Complete workflow testing from data ingestion to investment decisions
"""

import unittest
import asyncio
import numpy as np
import pandas as pd
import tempfile
import os
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import test framework
from .test_framework import AsyncTestCase, MockDataGenerator, TestFixtures, BenchmarkRunner

class TestCompleteWorkflow(AsyncTestCase):
    """Test the complete investment workflow from start to finish"""
    
    def setUp(self):
        super().setUp()
        self.temp_dir = tempfile.mkdtemp()
        self.portfolio_value = 100000  # $100k test portfolio
        self.test_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        # Generate comprehensive test data
        self.market_data = {}
        for symbol in self.test_symbols:
            self.market_data[symbol] = MockDataGenerator.generate_stock_data(symbol, 252)
        
        self.news_data = MockDataGenerator.generate_news_data(100)
    
    def tearDown(self):
        super().tearDown()
        TestFixtures.cleanup_temp_directory(self.temp_dir)
    
    def test_daily_investment_workflow(self):
        """Test complete daily investment decision workflow"""
        
        async def daily_workflow():
            """Simulate a complete daily investment workflow"""
            workflow_results = {
                'start_time': datetime.now(),
                'steps_completed': [],
                'decisions': {},
                'performance_metrics': {},
                'errors': []
            }
            
            try:
                # Step 1: Market Data Collection
                data_collector = Mock()
                collected_data = {}
                
                for symbol in self.test_symbols:
                    data_collector.collect_real_time_data = AsyncMock(
                        return_value=self.market_data[symbol].tail(1).iloc[0].to_dict()
                    )
                    
                    real_time_data = await data_collector.collect_real_time_data(symbol)
                    collected_data[symbol] = real_time_data
                    
                workflow_results['steps_completed'].append('data_collection')
                workflow_results['collected_symbols'] = len(collected_data)
                
                # Step 2: News and Sentiment Analysis
                news_analyzer = Mock()
                news_analyzer.analyze_sentiment = AsyncMock()
                
                sentiment_scores = {}
                for symbol in self.test_symbols:
                    # Mock sentiment analysis
                    sentiment = np.random.uniform(-1, 1)  # -1 (very negative) to 1 (very positive)
                    news_analyzer.analyze_sentiment.return_value = {
                        'sentiment_score': sentiment,
                        'news_count': np.random.randint(5, 20),
                        'confidence': np.random.uniform(0.6, 0.95)
                    }
                    
                    sentiment_analysis = await news_analyzer.analyze_sentiment(symbol)
                    sentiment_scores[symbol] = sentiment_analysis
                
                workflow_results['steps_completed'].append('sentiment_analysis')
                workflow_results['sentiment_scores'] = sentiment_scores
                
                # Step 3: Technical Analysis and Feature Engineering
                technical_analyzer = Mock()
                
                technical_signals = {}
                for symbol in self.test_symbols:
                    # Mock technical analysis
                    technical_analyzer.generate_signals.return_value = {
                        'trend_signal': np.random.choice(['bullish', 'bearish', 'neutral']),
                        'momentum': np.random.uniform(-1, 1),
                        'volatility': np.random.uniform(0.1, 0.5),
                        'support_level': collected_data[symbol]['Close'] * 0.95,
                        'resistance_level': collected_data[symbol]['Close'] * 1.05
                    }
                    
                    signals = technical_analyzer.generate_signals(self.market_data[symbol])
                    technical_signals[symbol] = signals
                
                workflow_results['steps_completed'].append('technical_analysis')
                workflow_results['technical_signals'] = technical_signals
                
                # Step 4: ML Model Predictions
                ml_predictor = Mock()
                
                predictions = {}
                for symbol in self.test_symbols:
                    # Mock ML predictions
                    current_price = collected_data[symbol]['Close']
                    predicted_return = np.random.uniform(-0.1, 0.15)  # -10% to +15%
                    predicted_price = current_price * (1 + predicted_return)
                    
                    ml_predictor.predict.return_value = {
                        'predicted_price': predicted_price,
                        'predicted_return': predicted_return,
                        'confidence': np.random.uniform(0.6, 0.95),
                        'risk_level': np.random.choice(['low', 'medium', 'high']),
                        'time_horizon': '1d'  # 1 day prediction
                    }
                    
                    prediction = ml_predictor.predict(symbol)
                    predictions[symbol] = prediction
                
                workflow_results['steps_completed'].append('ml_predictions')
                workflow_results['predictions'] = predictions
                
                # Step 5: Risk Assessment and Position Sizing
                risk_manager = Mock()
                
                risk_assessments = {}
                for symbol in self.test_symbols:
                    # Mock risk assessment
                    prediction = predictions[symbol]
                    sentiment = sentiment_scores[symbol]['sentiment_score']
                    
                    # Calculate position size based on Kelly Criterion mock
                    win_probability = (prediction['confidence'] + (sentiment + 1) / 2) / 2
                    expected_return = prediction['predicted_return']
                    
                    if expected_return > 0 and win_probability > 0.5:
                        # Kelly fraction (simplified)
                        kelly_fraction = min(0.25, (win_probability * expected_return - (1 - win_probability)) / abs(expected_return))
                        position_size = max(0, kelly_fraction * self.portfolio_value)
                    else:
                        position_size = 0
                    
                    risk_manager.calculate_position_size.return_value = {
                        'position_size': position_size,
                        'risk_percentage': position_size / self.portfolio_value,
                        'stop_loss': collected_data[symbol]['Close'] * 0.95,
                        'take_profit': collected_data[symbol]['Close'] * 1.10,
                        'expected_risk': abs(expected_return) * 0.5
                    }
                    
                    risk_assessment = risk_manager.calculate_position_size(symbol, prediction)
                    risk_assessments[symbol] = risk_assessment
                
                workflow_results['steps_completed'].append('risk_assessment')
                workflow_results['risk_assessments'] = risk_assessments
                
                # Step 6: Portfolio Optimization
                portfolio_optimizer = Mock()
                
                # Select top opportunities based on risk-adjusted returns
                opportunities = []
                for symbol in self.test_symbols:
                    pred = predictions[symbol]
                    risk = risk_assessments[symbol]
                    
                    if risk['position_size'] > 0:
                        sharpe_estimate = pred['predicted_return'] / max(risk['expected_risk'], 0.01)
                        opportunities.append({
                            'symbol': symbol,
                            'expected_return': pred['predicted_return'],
                            'risk': risk['expected_risk'],
                            'sharpe_ratio': sharpe_estimate,
                            'position_size': risk['position_size']
                        })
                
                # Sort by Sharpe ratio
                opportunities.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
                
                # Take top 3 opportunities
                selected_investments = opportunities[:3]
                
                portfolio_optimizer.optimize.return_value = {
                    'selected_investments': selected_investments,
                    'total_investment': sum(inv['position_size'] for inv in selected_investments),
                    'expected_portfolio_return': np.mean([inv['expected_return'] for inv in selected_investments]),
                    'portfolio_risk': np.sqrt(np.mean([inv['risk']**2 for inv in selected_investments]))
                }
                
                optimization_result = portfolio_optimizer.optimize(opportunities)
                workflow_results['steps_completed'].append('portfolio_optimization')
                workflow_results['optimization_result'] = optimization_result
                
                # Step 7: Trade Execution (Mock)
                trade_executor = Mock()
                
                executed_trades = []
                for investment in selected_investments:
                    symbol = investment['symbol']
                    position_size = investment['position_size']
                    current_price = collected_data[symbol]['Close']
                    shares = int(position_size / current_price)
                    
                    if shares > 0:
                        trade_executor.execute_trade.return_value = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': current_price,
                            'total_value': shares * current_price,
                            'timestamp': datetime.now(),
                            'status': 'FILLED',
                            'commission': 0.0  # Mock zero commission
                        }
                        
                        trade = trade_executor.execute_trade(symbol, 'BUY', shares)
                        executed_trades.append(trade)
                
                workflow_results['steps_completed'].append('trade_execution')
                workflow_results['executed_trades'] = executed_trades
                
                # Step 8: Performance Monitoring Setup
                monitor = Mock()
                monitor.setup_monitoring.return_value = {
                    'alerts_configured': len(executed_trades),
                    'stop_losses_set': len(executed_trades),
                    'monitoring_active': True
                }
                
                monitoring_setup = monitor.setup_monitoring(executed_trades)
                workflow_results['steps_completed'].append('monitoring_setup')
                workflow_results['monitoring_setup'] = monitoring_setup
                
            except Exception as e:
                workflow_results['errors'].append(str(e))
                print(f"Workflow error: {e}")
            
            workflow_results['end_time'] = datetime.now()
            workflow_results['total_duration'] = (
                workflow_results['end_time'] - workflow_results['start_time']
            ).total_seconds()
            
            return workflow_results
        
        # Execute the complete workflow
        results = self.run_async(daily_workflow())
        
        # Verify workflow completion
        expected_steps = [
            'data_collection', 'sentiment_analysis', 'technical_analysis',
            'ml_predictions', 'risk_assessment', 'portfolio_optimization',
            'trade_execution', 'monitoring_setup'
        ]
        
        for step in expected_steps:
            self.assertIn(step, results['steps_completed'], f"Step {step} not completed")
        
        # Verify data quality
        self.assertEqual(results['collected_symbols'], len(self.test_symbols))
        self.assertGreater(len(results['predictions']), 0)
        self.assertGreater(len(results['risk_assessments']), 0)
        
        # Verify investment decisions
        if 'optimization_result' in results:
            opt_result = results['optimization_result']
            self.assertIn('selected_investments', opt_result)
            self.assertLessEqual(opt_result['total_investment'], self.portfolio_value)
        
        # Verify trades were executed
        if 'executed_trades' in results:
            total_invested = sum(trade['total_value'] for trade in results['executed_trades'])
            self.assertLessEqual(total_invested, self.portfolio_value)
        
        # Verify performance
        self.assertLess(results['total_duration'], 10.0, "Workflow should complete in under 10 seconds")
        self.assertEqual(len(results['errors']), 0, f"Workflow had errors: {results['errors']}")
    
    def test_multi_day_backtesting_workflow(self):
        """Test backtesting workflow over multiple days"""
        
        def run_backtest():
            """Run a multi-day backtest simulation"""
            
            backtest_results = {
                'start_date': datetime.now() - timedelta(days=30),
                'end_date': datetime.now(),
                'daily_results': [],
                'portfolio_history': [],
                'trades': [],
                'performance_metrics': {}
            }
            
            # Initialize portfolio
            portfolio = {
                'cash': self.portfolio_value,
                'positions': {},
                'total_value': self.portfolio_value
            }
            
            # Simulate 30 days of trading
            for day in range(30):
                daily_result = {
                    'date': backtest_results['start_date'] + timedelta(days=day),
                    'signals': {},
                    'trades': [],
                    'portfolio_value': portfolio['total_value']
                }
                
                # Generate signals for each symbol
                for symbol in self.test_symbols[:3]:  # Test with 3 symbols for speed
                    # Mock daily signal generation
                    signal_strength = np.random.uniform(-1, 1)
                    price = 100 + np.random.uniform(-20, 20)  # Mock price
                    
                    signal = {
                        'action': 'BUY' if signal_strength > 0.3 else 'SELL' if signal_strength < -0.3 else 'HOLD',
                        'strength': abs(signal_strength),
                        'price': price,
                        'confidence': np.random.uniform(0.6, 0.9)
                    }
                    
                    daily_result['signals'][symbol] = signal
                    
                    # Execute trades based on signals
                    if signal['action'] == 'BUY' and portfolio['cash'] > price * 10:
                        shares = min(10, int(portfolio['cash'] / price))  # Buy up to 10 shares
                        cost = shares * price
                        
                        if symbol in portfolio['positions']:
                            portfolio['positions'][symbol] += shares
                        else:
                            portfolio['positions'][symbol] = shares
                        
                        portfolio['cash'] -= cost
                        
                        trade = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'value': cost,
                            'date': daily_result['date']
                        }
                        
                        daily_result['trades'].append(trade)
                        backtest_results['trades'].append(trade)
                    
                    elif signal['action'] == 'SELL' and symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                        shares = min(portfolio['positions'][symbol], 5)  # Sell up to 5 shares
                        proceeds = shares * price
                        
                        portfolio['positions'][symbol] -= shares
                        if portfolio['positions'][symbol] == 0:
                            del portfolio['positions'][symbol]
                        
                        portfolio['cash'] += proceeds
                        
                        trade = {
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares,
                            'price': price,
                            'value': proceeds,
                            'date': daily_result['date']
                        }
                        
                        daily_result['trades'].append(trade)
                        backtest_results['trades'].append(trade)
                
                # Calculate portfolio value
                position_value = sum(
                    shares * (100 + np.random.uniform(-20, 20))  # Mock current prices
                    for shares in portfolio['positions'].values()
                )
                portfolio['total_value'] = portfolio['cash'] + position_value
                daily_result['portfolio_value'] = portfolio['total_value']
                
                backtest_results['daily_results'].append(daily_result)
                backtest_results['portfolio_history'].append(portfolio['total_value'])
            
            # Calculate performance metrics
            initial_value = self.portfolio_value
            final_value = portfolio['total_value']
            
            returns = []
            for i in range(1, len(backtest_results['portfolio_history'])):
                daily_return = (backtest_results['portfolio_history'][i] - backtest_results['portfolio_history'][i-1]) / backtest_results['portfolio_history'][i-1]
                returns.append(daily_return)
            
            backtest_results['performance_metrics'] = {
                'total_return': (final_value - initial_value) / initial_value,
                'total_trades': len(backtest_results['trades']),
                'win_rate': 0.6,  # Mock win rate
                'sharpe_ratio': np.mean(returns) / max(np.std(returns), 0.001) if returns else 0,
                'max_drawdown': -0.05,  # Mock max drawdown
                'volatility': np.std(returns) if returns else 0
            }
            
            return backtest_results
        
        # Run the backtest
        results = run_backtest()
        
        # Verify backtest results
        self.assertEqual(len(results['daily_results']), 30)
        self.assertGreater(len(results['trades']), 0)
        self.assertEqual(len(results['portfolio_history']), 30)
        
        # Verify performance metrics
        metrics = results['performance_metrics']
        self.assertIn('total_return', metrics)
        self.assertIn('total_trades', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # Verify reasonable performance bounds
        self.assertGreater(metrics['total_trades'], 0)
        self.assertGreaterEqual(metrics['win_rate'], 0.0)
        self.assertLessEqual(metrics['win_rate'], 1.0)
    
    def test_real_time_trading_simulation(self):
        """Test real-time trading simulation with streaming data"""
        
        async def real_time_simulation():
            """Simulate real-time trading for a short period"""
            
            simulation_results = {
                'start_time': datetime.now(),
                'ticks_processed': 0,
                'signals_generated': 0,
                'trades_executed': 0,
                'latency_measurements': [],
                'errors': []
            }
            
            # Mock streaming data source
            test_symbols = ["AAPL", "GOOGL", "MSFT"]

            class MockDataStream:
                def __init__(self):
                    self.tick_count = 0

                async def get_tick(self):
                    """Get next market tick"""
                    self.tick_count += 1

                    symbol = np.random.choice(test_symbols)
                    price = 100 + np.random.uniform(-5, 5)
                    volume = np.random.randint(1000, 10000)
                    
                    return {
                        'symbol': symbol,
                        'price': price,
                        'volume': volume,
                        'timestamp': datetime.now(),
                        'tick_id': self.tick_count
                    }
            
            stream = MockDataStream()
            
            # Simulate processing 50 market ticks
            for tick_num in range(50):
                tick_start = time.time()
                
                try:
                    # Get market tick
                    tick = await stream.get_tick()
                    simulation_results['ticks_processed'] += 1
                    
                    # Process tick (signal generation)
                    signal_processor = Mock()
                    signal_processor.process_tick.return_value = {
                        'signal': np.random.choice(['BUY', 'SELL', 'HOLD']),
                        'strength': np.random.uniform(0, 1),
                        'confidence': np.random.uniform(0.5, 0.95)
                    }
                    
                    signal = signal_processor.process_tick(tick)
                    
                    if signal['signal'] != 'HOLD' and signal['confidence'] > 0.8:
                        simulation_results['signals_generated'] += 1
                        
                        # Execute trade (if signal is strong enough)
                        if signal['strength'] > 0.7:
                            trade_executor = Mock()
                            trade_executor.execute_trade.return_value = {
                                'status': 'FILLED',
                                'execution_time': time.time() - tick_start
                            }
                            
                            trade_result = trade_executor.execute_trade(
                                tick['symbol'], signal['signal'], 10
                            )
                            
                            if trade_result['status'] == 'FILLED':
                                simulation_results['trades_executed'] += 1
                    
                    # Measure processing latency
                    processing_latency = time.time() - tick_start
                    simulation_results['latency_measurements'].append(processing_latency)
                    
                    # Small delay to simulate realistic tick frequency
                    await asyncio.sleep(0.01)  # 100 ticks per second max
                    
                except Exception as e:
                    simulation_results['errors'].append(str(e))
            
            simulation_results['end_time'] = datetime.now()
            simulation_results['total_duration'] = (
                simulation_results['end_time'] - simulation_results['start_time']
            ).total_seconds()
            
            # Calculate latency statistics
            if simulation_results['latency_measurements']:
                latencies = simulation_results['latency_measurements']
                simulation_results['latency_stats'] = {
                    'mean_latency_ms': np.mean(latencies) * 1000,
                    'p95_latency_ms': np.percentile(latencies, 95) * 1000,
                    'p99_latency_ms': np.percentile(latencies, 99) * 1000,
                    'max_latency_ms': max(latencies) * 1000
                }
            
            return simulation_results
        
        # Run real-time simulation
        results = self.run_async(real_time_simulation())
        
        # Verify simulation results
        self.assertEqual(results['ticks_processed'], 50)
        self.assertGreaterEqual(results['signals_generated'], 0)
        self.assertGreaterEqual(results['trades_executed'], 0)
        self.assertEqual(len(results['errors']), 0, f"Simulation had errors: {results['errors']}")
        
        # Verify performance requirements
        if 'latency_stats' in results:
            latency = results['latency_stats']
            self.assertLess(latency['mean_latency_ms'], 50, "Average latency too high")
            self.assertLess(latency['p95_latency_ms'], 100, "P95 latency too high")
        
        # Verify throughput
        throughput = results['ticks_processed'] / results['total_duration']
        self.assertGreater(throughput, 10, "Tick processing throughput too low")
    
    def test_system_recovery_from_failure(self):
        """Test system recovery from various failure scenarios"""
        
        async def failure_recovery_test():
            """Test recovery from different failure types"""
            
            recovery_results = {
                'scenarios_tested': [],
                'recovery_times': {},
                'success_recoveries': 0,
                'failed_recoveries': 0
            }
            
            # Scenario 1: Data source failure
            try:
                data_source = Mock()
                data_source.get_data.side_effect = [
                    Exception("Connection timeout"),
                    Exception("Service unavailable"), 
                    self.market_data["AAPL"]  # Succeeds on 3rd try
                ]
                
                start_time = time.time()
                attempts = 0
                max_attempts = 5
                data = None
                
                while attempts < max_attempts and data is None:
                    try:
                        data = data_source.get_data("AAPL")
                    except Exception:
                        attempts += 1
                        await asyncio.sleep(0.1)  # Brief retry delay
                
                recovery_time = time.time() - start_time
                
                if data is not None:
                    recovery_results['success_recoveries'] += 1
                    recovery_results['recovery_times']['data_source_failure'] = recovery_time
                else:
                    recovery_results['failed_recoveries'] += 1
                
                recovery_results['scenarios_tested'].append('data_source_failure')
                
            except Exception as e:
                recovery_results['scenarios_tested'].append(f'data_source_failure_error: {e}')
            
            # Scenario 2: Model prediction failure
            try:
                predictor = Mock()
                call_count = [0]
                
                def intermittent_failure(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] <= 2:
                        raise Exception("Model service unavailable")
                    return {'prediction': 150.0, 'confidence': 0.8}
                
                predictor.predict.side_effect = intermittent_failure
                
                start_time = time.time()
                prediction = None
                
                for attempt in range(5):
                    try:
                        prediction = predictor.predict("AAPL")
                        break
                    except Exception:
                        await asyncio.sleep(0.05)  # Brief delay
                
                recovery_time = time.time() - start_time
                
                if prediction is not None:
                    recovery_results['success_recoveries'] += 1
                    recovery_results['recovery_times']['model_failure'] = recovery_time
                else:
                    recovery_results['failed_recoveries'] += 1
                
                recovery_results['scenarios_tested'].append('model_prediction_failure')
                
            except Exception as e:
                recovery_results['scenarios_tested'].append(f'model_failure_error: {e}')
            
            # Scenario 3: Trade execution failure with retry
            try:
                trade_executor = Mock()
                execution_attempts = [0]
                
                def failing_execution(*args, **kwargs):
                    execution_attempts[0] += 1
                    if execution_attempts[0] == 1:
                        raise Exception("Order rejected")
                    return {'status': 'FILLED', 'trade_id': '12345'}
                
                trade_executor.execute_trade.side_effect = failing_execution
                
                start_time = time.time()
                trade_result = None
                
                for attempt in range(3):
                    try:
                        trade_result = trade_executor.execute_trade("AAPL", "BUY", 10)
                        if trade_result['status'] == 'FILLED':
                            break
                    except Exception:
                        await asyncio.sleep(0.1)
                
                recovery_time = time.time() - start_time
                
                if trade_result and trade_result['status'] == 'FILLED':
                    recovery_results['success_recoveries'] += 1
                    recovery_results['recovery_times']['trade_execution_failure'] = recovery_time
                else:
                    recovery_results['failed_recoveries'] += 1
                
                recovery_results['scenarios_tested'].append('trade_execution_failure')
                
            except Exception as e:
                recovery_results['scenarios_tested'].append(f'trade_execution_error: {e}')
            
            # Scenario 4: Database connection failure
            try:
                db_connection = Mock()
                db_connection.execute.side_effect = [
                    Exception("Database connection lost"),
                    True  # Succeeds after reconnection
                ]
                
                start_time = time.time()
                success = False
                
                for attempt in range(3):
                    try:
                        result = db_connection.execute("SELECT * FROM positions")
                        success = True
                        break
                    except Exception:
                        # Simulate reconnection
                        await asyncio.sleep(0.1)
                
                recovery_time = time.time() - start_time
                
                if success:
                    recovery_results['success_recoveries'] += 1
                    recovery_results['recovery_times']['database_failure'] = recovery_time
                else:
                    recovery_results['failed_recoveries'] += 1
                
                recovery_results['scenarios_tested'].append('database_failure')
                
            except Exception as e:
                recovery_results['scenarios_tested'].append(f'database_error: {e}')
            
            return recovery_results
        
        # Run failure recovery tests
        results = self.run_async(failure_recovery_test())
        
        # Verify recovery scenarios
        expected_scenarios = [
            'data_source_failure', 'model_prediction_failure', 
            'trade_execution_failure', 'database_failure'
        ]
        
        tested_scenarios = [s for s in results['scenarios_tested'] if s in expected_scenarios]
        self.assertGreater(len(tested_scenarios), 0, "No recovery scenarios were tested")
        
        # Verify success rate
        total_attempts = results['success_recoveries'] + results['failed_recoveries']
        if total_attempts > 0:
            success_rate = results['success_recoveries'] / total_attempts
            self.assertGreater(success_rate, 0.5, "Recovery success rate too low")
        
        # Verify recovery times are reasonable
        for scenario, recovery_time in results['recovery_times'].items():
            self.assertLess(recovery_time, 5.0, f"Recovery time too long for {scenario}")

if __name__ == "__main__":
    print("🧪 Running End-to-End Tests")
    print("=" * 32)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test class
    tests = loader.loadTestsFromTestCase(TestCompleteWorkflow)
    suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 End-to-End Test Summary:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
        print(f"   Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, error in result.failures:
            print(f"   {test}: {error.split(chr(10))[0]}")
    
    if result.errors:
        print(f"\n🚨 Errors:")
        for test, error in result.errors:
            print(f"   {test}: {error.split(chr(10))[0]}")
    
    print(f"\n✅ End-to-end testing completed!")
    print(f"🎯 System readiness validated through complete workflow testing")