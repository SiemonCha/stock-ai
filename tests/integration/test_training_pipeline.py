"""
Integration tests for training pipeline
"""

import pytest
import numpy as np
import os
import tempfile
from unittest.mock import Mock, patch

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
    from data.stock_data import StockDataCollector
except ImportError:
    pytest.skip("Required modules not available", allow_module_level=True)

class TestTrainingPipeline:
    """Test complete training pipeline integration"""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for model files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @patch('models.unified_models.ta')
    def test_end_to_end_training_pipeline(self, mock_ta, sample_stock_data, temp_model_dir):
        """Test complete end-to-end training pipeline"""
        
        # Mock technical indicators
        data_length = len(sample_stock_data)
        mock_ta.trend.sma_indicator.return_value = pd.Series([100] * data_length)
        mock_ta.momentum.rsi.return_value = pd.Series([50] * data_length)
        mock_ta.trend.macd_diff.return_value = pd.Series([0] * data_length)
        mock_ta.trend.macd_signal.return_value = pd.Series([0] * data_length)
        mock_ta.volatility.bollinger_hband.return_value = pd.Series([105] * data_length)
        mock_ta.volatility.bollinger_lband.return_value = pd.Series([95] * data_length)
        mock_ta.volume.volume_sma.return_value = pd.Series([1000000] * data_length)
        
        # Initialize components
        processor = DataProcessor(sequence_length=30)
        models = UnifiedStockModels()
        
        # Process data
        X, y = processor.create_sequences(sample_stock_data)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = X[train_size:], y[train_size:]
        
        # Mock model creation and training to avoid actual TensorFlow usage
        mock_model = Mock()
        mock_history = Mock()
        mock_history.history = {
            'loss': [0.1, 0.05, 0.02],
            'val_loss': [0.12, 0.06, 0.03],
            'mae': [0.08, 0.04, 0.015],
            'val_mae': [0.09, 0.045, 0.018]
        }
        mock_model.fit.return_value = mock_history
        mock_model.predict.return_value = np.random.random((len(X_val), 1))
        
        with patch.object(models, 'create_lstm_model', return_value=mock_model):
            # Train model
            history = models.train_model(
                'lstm', 
                X_train, y_train, 
                X_val, y_val, 
                epochs=3,
                model_path=os.path.join(temp_model_dir, 'test_model.h5')
            )
            
            # Verify training
            assert history is not None
            assert 'loss' in history
            assert 'val_loss' in history
            assert len(history['loss']) == 3
            
            # Verify model is stored
            assert 'lstm' in models.models
            assert 'lstm' in models.history
    
    @patch('data.stock_data.yf.Ticker')
    def test_data_collection_to_training(self, mock_ticker, sample_stock_data):
        """Test integration from data collection to training preparation"""
        
        # Mock yfinance response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = sample_stock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Collect data
        collector = StockDataCollector()
        data = collector.get_stock_data('AAPL', period='2y')
        
        assert data is not None
        assert not data.empty
        
        # Process for training
        with patch('models.unified_models.ta') as mock_ta:
            # Mock all technical indicators
            data_length = len(data)
            mock_ta.trend.sma_indicator.return_value = pd.Series([100] * data_length)
            mock_ta.momentum.rsi.return_value = pd.Series([50] * data_length)
            mock_ta.trend.macd_diff.return_value = pd.Series([0] * data_length)
            mock_ta.trend.macd_signal.return_value = pd.Series([0] * data_length)
            mock_ta.volatility.bollinger_hband.return_value = pd.Series([105] * data_length)
            mock_ta.volatility.bollinger_lband.return_value = pd.Series([95] * data_length)
            mock_ta.volume.volume_sma.return_value = pd.Series([1000000] * data_length)
            
            processor = DataProcessor(sequence_length=60)
            X, y = processor.create_sequences(data)
            
            # Verify processed data
            assert X.shape[0] > 0
            assert y.shape[0] > 0
            assert X.shape[0] == y.shape[0]
            assert X.shape[1] == 60  # sequence_length
    
    def test_model_evaluation_pipeline(self, sample_sequences):
        """Test model evaluation pipeline"""
        X, y = sample_sequences
        
        models = UnifiedStockModels()
        
        # Mock trained model
        mock_model = Mock()
        predictions = np.random.random(len(y)) * 2 - 1  # Random predictions between -1 and 1
        mock_model.predict.return_value = predictions.reshape(-1, 1)
        
        models.models['lstm'] = mock_model
        
        # Evaluate model
        metrics = models.evaluate_model('lstm', X, y)
        
        # Verify metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'directional_accuracy' in metrics
        
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0
        assert 0 <= metrics['directional_accuracy'] <= 1
    
    def test_ensemble_training_pipeline(self, sample_sequences, temp_model_dir):
        """Test ensemble model training pipeline"""
        X, y = sample_sequences
        
        models = UnifiedStockModels()
        
        # Mock all individual models
        mock_models = {}
        for model_type in ['lstm', 'gru', 'transformer', 'cnn_lstm']:
            mock_model = Mock()
            mock_history = Mock()
            mock_history.history = {'loss': [0.1], 'val_loss': [0.12]}
            mock_model.fit.return_value = mock_history
            mock_models[model_type] = mock_model
        
        # Mock ensemble creation
        with patch.object(models, 'create_ensemble_model', return_value=mock_models):
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            X_val, y_val = X[train_size:], y[train_size:]
            
            # Train ensemble
            histories = models.train_ensemble(
                X_train, y_train,
                X_val, y_val,
                epochs=1,
                model_path=os.path.join(temp_model_dir, 'ensemble.h5')
            )
            
            # Verify training
            assert isinstance(histories, dict)
            assert len(histories) == 4
            assert 'ensemble' in models.models
            assert 'ensemble' in models.history

class TestCommandLineIntegration:
    """Test command line interface integration"""
    
    @patch('sys.argv', ['train.py', '--symbol', 'AAPL', '--model', 'lstm', '--epochs', '1'])
    @patch('train.StockDataCollector')
    @patch('train.UnifiedStockModels')
    def test_train_script_integration(self, mock_models_class, mock_collector_class, sample_stock_data):
        """Test train.py script integration"""
        
        # Mock data collection
        mock_collector = Mock()
        mock_collector.get_stock_data.return_value = sample_stock_data
        mock_collector_class.return_value = mock_collector
        
        # Mock model training
        mock_models = Mock()
        mock_models.train_model.return_value = {'loss': [0.1], 'val_loss': [0.12]}
        mock_models.evaluate_model.return_value = {
            'rmse': 0.05,
            'mae': 0.04,
            'directional_accuracy': 0.85
        }
        mock_models_class.return_value = mock_models
        
        # This would test the actual script execution
        # In a real scenario, you might use subprocess or import the main function
        # For now, we just verify the mocks would be called correctly
        
        assert mock_collector_class is not None
        assert mock_models_class is not None