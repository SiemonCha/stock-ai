"""
Unit tests for model architectures
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

try:
    from models.unified_models import UnifiedStockModels, DataProcessor
except ImportError:
    pytest.skip("Model modules not available", allow_module_level=True)

class TestUnifiedStockModels:
    """Test UnifiedStockModels class"""
    
    def test_init(self):
        """Test model initialization"""
        models = UnifiedStockModels()
        assert models.models == {}
        assert models.scalers == {}
        assert models.history == {}
    
    def test_create_lstm_model(self):
        """Test LSTM model creation"""
        models = UnifiedStockModels()
        
        with patch('models.unified_models.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            
            result = models.create_lstm_model(60, 5)
            
            assert result == mock_model
            mock_model.compile.assert_called_once()
    
    def test_create_gru_model(self):
        """Test GRU model creation"""
        models = UnifiedStockModels()
        
        with patch('models.unified_models.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            
            result = models.create_gru_model(60, 5)
            
            assert result == mock_model
            mock_model.compile.assert_called_once()
    
    def test_create_transformer_model(self):
        """Test Transformer model creation"""
        models = UnifiedStockModels()
        
        with patch('models.unified_models.Model') as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model
            
            result = models.create_transformer_model(60, 5)
            
            assert result == mock_model
            mock_model.compile.assert_called_once()
    
    def test_create_cnn_lstm_model(self):
        """Test CNN-LSTM model creation"""
        models = UnifiedStockModels()
        
        with patch('models.unified_models.Sequential') as mock_sequential:
            mock_model = Mock()
            mock_sequential.return_value = mock_model
            
            result = models.create_cnn_lstm_model(60, 5)
            
            assert result == mock_model
            mock_model.compile.assert_called_once()
    
    def test_create_ensemble_model(self):
        """Test ensemble model creation"""
        models = UnifiedStockModels()
        
        # Mock individual model creation methods
        models.create_lstm_model = Mock(return_value=Mock())
        models.create_gru_model = Mock(return_value=Mock())
        models.create_transformer_model = Mock(return_value=Mock())
        models.create_cnn_lstm_model = Mock(return_value=Mock())
        
        result = models.create_ensemble_model(60, 5)
        
        assert isinstance(result, dict)
        assert 'lstm' in result
        assert 'gru' in result
        assert 'transformer' in result
        assert 'cnn_lstm' in result
    
    def test_train_model_single(self, sample_sequences, mock_model_config):
        """Test single model training"""
        models = UnifiedStockModels()
        X, y = sample_sequences
        
        # Mock model creation and training
        mock_model = Mock()
        mock_history = Mock()
        mock_history.history = {'loss': [0.1, 0.05], 'val_loss': [0.12, 0.06]}
        mock_model.fit.return_value = mock_history
        
        models.create_lstm_model = Mock(return_value=mock_model)
        
        with patch('models.unified_models.os.makedirs'), \
             patch('models.unified_models.EarlyStopping'), \
             patch('models.unified_models.ReduceLROnPlateau'):
            
            result = models.train_model('lstm', X[:80], y[:80], X[80:], y[80:], epochs=1)
            
            assert result == mock_history.history
            assert 'lstm' in models.models
            assert 'lstm' in models.history
    
    def test_predict_single_model(self, sample_sequences):
        """Test single model prediction"""
        models = UnifiedStockModels()
        X, y = sample_sequences
        
        # Setup mock model
        mock_model = Mock()
        mock_predictions = np.array([[0.5], [0.6], [0.7]])
        mock_model.predict.return_value = mock_predictions
        
        models.models['lstm'] = mock_model
        
        result = models.predict('lstm', X[:3])
        
        np.testing.assert_array_equal(result, [0.5, 0.6, 0.7])
    
    def test_predict_ensemble(self, sample_sequences):
        """Test ensemble prediction"""
        models = UnifiedStockModels()
        X, y = sample_sequences
        
        # Setup mock ensemble models
        mock_models = {
            'lstm': Mock(),
            'gru': Mock()
        }
        
        mock_models['lstm'].predict.return_value = np.array([[0.5], [0.6]])
        mock_models['gru'].predict.return_value = np.array([[0.7], [0.8]])
        
        models.models['ensemble'] = mock_models
        
        result = models.predict('ensemble', X[:2])
        
        # Should return average of predictions
        expected = np.array([0.6, 0.7])  # (0.5+0.7)/2, (0.6+0.8)/2
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_evaluate_model(self, sample_sequences):
        """Test model evaluation"""
        models = UnifiedStockModels()
        X, y = sample_sequences
        
        # Mock predict method
        models.predict = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        actual = np.array([0.12, 0.18, 0.35])
        result = models.evaluate_model('lstm', X[:3], actual)
        
        assert 'mse' in result
        assert 'mae' in result
        assert 'rmse' in result
        assert 'directional_accuracy' in result
        
        assert result['mse'] > 0
        assert result['mae'] > 0
        assert result['rmse'] > 0

class TestDataProcessor:
    """Test DataProcessor class"""
    
    def test_init(self):
        """Test processor initialization"""
        processor = DataProcessor(sequence_length=30)
        assert processor.sequence_length == 30
        assert processor.feature_columns is None
    
    @patch('models.unified_models.ta')
    def test_prepare_data(self, mock_ta, sample_stock_data):
        """Test data preparation with technical indicators"""
        processor = DataProcessor()
        
        # Mock technical indicator functions
        mock_ta.trend.sma_indicator.return_value = pd.Series([100] * len(sample_stock_data))
        mock_ta.momentum.rsi.return_value = pd.Series([50] * len(sample_stock_data))
        mock_ta.trend.macd_diff.return_value = pd.Series([0] * len(sample_stock_data))
        mock_ta.trend.macd_signal.return_value = pd.Series([0] * len(sample_stock_data))
        mock_ta.volatility.bollinger_hband.return_value = pd.Series([105] * len(sample_stock_data))
        mock_ta.volatility.bollinger_lband.return_value = pd.Series([95] * len(sample_stock_data))
        mock_ta.volume.volume_sma.return_value = pd.Series([1000000] * len(sample_stock_data))
        
        result = processor.prepare_data(sample_stock_data)
        
        assert 'sma_20' in result.columns
        assert 'rsi' in result.columns
        assert 'macd' in result.columns
        assert 'price_change' in result.columns
        assert len(result) == len(sample_stock_data)
    
    @patch('models.unified_models.ta')
    def test_create_sequences(self, mock_ta, sample_stock_data):
        """Test sequence creation"""
        processor = DataProcessor(sequence_length=10)
        
        # Mock technical indicators
        mock_ta.trend.sma_indicator.return_value = pd.Series([100] * len(sample_stock_data))
        mock_ta.momentum.rsi.return_value = pd.Series([50] * len(sample_stock_data))
        mock_ta.trend.macd_diff.return_value = pd.Series([0] * len(sample_stock_data))
        mock_ta.trend.macd_signal.return_value = pd.Series([0] * len(sample_stock_data))
        mock_ta.volatility.bollinger_hband.return_value = pd.Series([105] * len(sample_stock_data))
        mock_ta.volatility.bollinger_lband.return_value = pd.Series([95] * len(sample_stock_data))
        mock_ta.volume.volume_sma.return_value = pd.Series([1000000] * len(sample_stock_data))
        
        with patch.object(processor, 'scaler') as mock_scaler:
            mock_scaler.fit_transform.return_value = np.random.random((len(sample_stock_data), 5))
            
            X, y = processor.create_sequences(sample_stock_data)
            
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert X.shape[1] == 10  # sequence_length
            assert len(X) == len(y)
            assert len(X) == len(sample_stock_data) - 10
    
    def test_inverse_transform_target(self):
        """Test inverse transformation of target values"""
        processor = DataProcessor()
        processor.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Mock scaler
        processor.scaler = Mock()
        processor.scaler.inverse_transform.return_value = np.array([[100, 105, 95, 102, 1000000]])
        
        scaled_target = np.array([0.5])
        result = processor.inverse_transform_target(scaled_target, 'Close')
        
        assert isinstance(result, np.ndarray)
        processor.scaler.inverse_transform.assert_called_once()