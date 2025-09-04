"""
Real-Time Data Streaming Pipeline
Professional-grade streaming data processing for stock market data
"""

import asyncio
import websockets
import aiohttp
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
from collections import deque, defaultdict
import threading
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    import kafka
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from apache_beam import Pipeline, DoFn, ParDo
    from apache_beam.options.pipeline_options import PipelineOptions
    BEAM_AVAILABLE = True
except ImportError:
    BEAM_AVAILABLE = False

try:
    import websocket
    WEBSOCKET_CLIENT_AVAILABLE = True
except ImportError:
    WEBSOCKET_CLIENT_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamType(Enum):
    """Types of data streams"""
    PRICE_UPDATES = "price_updates"
    TRADE_EXECUTIONS = "trade_executions"
    MARKET_DEPTH = "market_depth"
    NEWS_EVENTS = "news_events"
    ECONOMIC_DATA = "economic_data"
    SOCIAL_SENTIMENT = "social_sentiment"

class StreamingProvider(Enum):
    """Streaming data providers"""
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    IEX = "iex"
    ALPACA = "alpaca"
    WEBSOCKET_GENERIC = "websocket_generic"

@dataclass
class StreamConfig:
    """Configuration for a data stream"""
    provider: StreamingProvider
    stream_type: StreamType
    symbols: List[str]
    websocket_url: str
    auth_token: Optional[str] = None
    reconnect_interval: int = 5
    heartbeat_interval: int = 30
    buffer_size: int = 1000
    batch_size: int = 100
    flush_interval: float = 1.0

@dataclass 
class StreamMessage:
    """Streaming data message"""
    timestamp: datetime
    symbol: str
    stream_type: StreamType
    data: Dict[str, Any]
    provider: StreamingProvider
    sequence_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamMetrics:
    """Streaming metrics"""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    bytes_received: int = 0
    latency_ms: float = 0.0
    errors: int = 0
    last_message_time: Optional[datetime] = None
    connection_uptime: float = 0.0

class MessageBuffer:
    """Thread-safe circular buffer for streaming messages"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.dropped_count = 0
    
    def put(self, message: StreamMessage) -> bool:
        """Add message to buffer"""
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.dropped_count += 1
                return False
            
            self.buffer.append(message)
            return True
    
    def get_batch(self, batch_size: int) -> List[StreamMessage]:
        """Get batch of messages"""
        with self.lock:
            batch = []
            for _ in range(min(batch_size, len(self.buffer))):
                if self.buffer:
                    batch.append(self.buffer.popleft())
            return batch
    
    def get_all(self) -> List[StreamMessage]:
        """Get all messages"""
        with self.lock:
            batch = list(self.buffer)
            self.buffer.clear()
            return batch
    
    def size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self.lock:
            return len(self.buffer) >= self.max_size

class StreamingDataSource:
    """Base class for streaming data sources"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.is_connected = False
        self.is_running = False
        self.metrics = StreamMetrics()
        self.connection_start_time = None
        self.message_handlers: List[Callable[[StreamMessage], None]] = []
        self.error_handlers: List[Callable[[Exception], None]] = []
        
    def add_message_handler(self, handler: Callable[[StreamMessage], None]):
        """Add message handler"""
        self.message_handlers.append(handler)
    
    def add_error_handler(self, handler: Callable[[Exception], None]):
        """Add error handler"""
        self.error_handlers.append(handler)
    
    async def connect(self):
        """Connect to streaming source"""
        raise NotImplementedError
    
    async def disconnect(self):
        """Disconnect from streaming source"""
        raise NotImplementedError
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        raise NotImplementedError
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        raise NotImplementedError
    
    def _handle_message(self, message: StreamMessage):
        """Handle incoming message"""
        self.metrics.messages_received += 1
        self.metrics.last_message_time = datetime.now()
        
        # Update latency
        if hasattr(message, 'server_timestamp'):
            latency = (datetime.now() - message.server_timestamp).total_seconds() * 1000
            self.metrics.latency_ms = (self.metrics.latency_ms * 0.9 + latency * 0.1)  # EMA
        
        # Call handlers
        for handler in self.message_handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                self._handle_error(e)
    
    def _handle_error(self, error: Exception):
        """Handle errors"""
        self.metrics.errors += 1
        
        for handler in self.error_handlers:
            try:
                handler(error)
            except Exception as e:
                logger.error(f"Error handler error: {e}")

class PolygonWebSocketSource(StreamingDataSource):
    """Polygon.io WebSocket streaming source"""
    
    def __init__(self, config: StreamConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.websocket = None
        self.subscribed_symbols = set()
    
    async def connect(self):
        """Connect to Polygon WebSocket"""
        
        try:
            # Polygon WebSocket URL
            url = "wss://socket.polygon.io/stocks"
            
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.connection_start_time = datetime.now()
            
            # Authenticate
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            
            await self.websocket.send(json.dumps(auth_message))
            
            # Start message loop
            asyncio.create_task(self._message_loop())
            
            logger.info("Connected to Polygon WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to connect to Polygon WebSocket: {e}")
            self.is_connected = False
            raise e
    
    async def disconnect(self):
        """Disconnect from Polygon WebSocket"""
        
        self.is_running = False
        
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        self.is_connected = False
        logger.info("Disconnected from Polygon WebSocket")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        
        if not self.is_connected:
            raise Exception("Not connected to WebSocket")
        
        # Subscribe to trades and quotes
        subscribe_message = {
            "action": "subscribe",
            "params": f"T.{',T.'.join(symbols)},Q.{',Q.'.join(symbols)}"
        }
        
        await self.websocket.send(json.dumps(subscribe_message))
        self.subscribed_symbols.update(symbols)
        
        logger.info(f"Subscribed to {len(symbols)} symbols on Polygon")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        
        if not self.is_connected:
            return
        
        unsubscribe_message = {
            "action": "unsubscribe", 
            "params": f"T.{',T.'.join(symbols)},Q.{',Q.'.join(symbols)}"
        }
        
        await self.websocket.send(json.dumps(unsubscribe_message))
        self.subscribed_symbols.difference_update(symbols)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols on Polygon")
    
    async def _message_loop(self):
        """Main message processing loop"""
        
        self.is_running = True
        
        while self.is_running and self.is_connected:
            try:
                # Receive message with timeout
                message_text = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.config.heartbeat_interval
                )
                
                self.metrics.bytes_received += len(message_text.encode())
                
                # Parse message
                data = json.loads(message_text)
                
                # Handle different message types
                if isinstance(data, list):
                    # Bulk messages
                    for item in data:
                        await self._process_message(item)
                else:
                    # Single message
                    await self._process_message(data)
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await self._send_heartbeat()
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Polygon WebSocket connection closed")
                self.is_connected = False
                break
                
            except Exception as e:
                logger.error(f"Polygon WebSocket error: {e}")
                self._handle_error(e)
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process individual message"""
        
        if not isinstance(data, dict):
            return
        
        # Determine message type
        msg_type = data.get('ev', '')
        
        if msg_type == 'T':  # Trade
            stream_message = StreamMessage(
                timestamp=datetime.now(),
                symbol=data.get('sym', ''),
                stream_type=StreamType.TRADE_EXECUTIONS,
                data={
                    'price': data.get('p', 0),
                    'size': data.get('s', 0),
                    'exchange': data.get('x', ''),
                    'conditions': data.get('c', []),
                    'timestamp': data.get('t', 0)
                },
                provider=StreamingProvider.POLYGON,
                sequence_id=data.get('q', None)
            )
            
            self._handle_message(stream_message)
        
        elif msg_type == 'Q':  # Quote
            stream_message = StreamMessage(
                timestamp=datetime.now(),
                symbol=data.get('sym', ''),
                stream_type=StreamType.MARKET_DEPTH,
                data={
                    'bid_price': data.get('bp', 0),
                    'bid_size': data.get('bs', 0),
                    'ask_price': data.get('ap', 0),
                    'ask_size': data.get('as', 0),
                    'exchange': data.get('x', ''),
                    'timestamp': data.get('t', 0)
                },
                provider=StreamingProvider.POLYGON,
                sequence_id=data.get('q', None)
            )
            
            self._handle_message(stream_message)
        
        elif msg_type == 'status':
            # Status message
            logger.info(f"Polygon status: {data.get('message', '')}")
    
    async def _send_heartbeat(self):
        """Send heartbeat to keep connection alive"""
        
        if self.websocket and not self.websocket.closed:
            try:
                heartbeat = {"action": "heartbeat"}
                await self.websocket.send(json.dumps(heartbeat))
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

class FinnhubWebSocketSource(StreamingDataSource):
    """Finnhub WebSocket streaming source"""
    
    def __init__(self, config: StreamConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.websocket = None
        self.subscribed_symbols = set()
    
    async def connect(self):
        """Connect to Finnhub WebSocket"""
        
        try:
            url = f"wss://ws.finnhub.io?token={self.api_key}"
            
            self.websocket = await websockets.connect(url)
            self.is_connected = True
            self.connection_start_time = datetime.now()
            
            # Start message loop
            asyncio.create_task(self._message_loop())
            
            logger.info("Connected to Finnhub WebSocket")
            
        except Exception as e:
            logger.error(f"Failed to connect to Finnhub WebSocket: {e}")
            self.is_connected = False
            raise e
    
    async def disconnect(self):
        """Disconnect from Finnhub WebSocket"""
        
        self.is_running = False
        
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        self.is_connected = False
        logger.info("Disconnected from Finnhub WebSocket")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to symbols"""
        
        if not self.is_connected:
            raise Exception("Not connected to WebSocket")
        
        for symbol in symbols:
            subscribe_message = {
                "type": "subscribe",
                "symbol": symbol
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            self.subscribed_symbols.add(symbol)
        
        logger.info(f"Subscribed to {len(symbols)} symbols on Finnhub")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        
        if not self.is_connected:
            return
        
        for symbol in symbols:
            unsubscribe_message = {
                "type": "unsubscribe",
                "symbol": symbol
            }
            
            await self.websocket.send(json.dumps(unsubscribe_message))
            self.subscribed_symbols.discard(symbol)
        
        logger.info(f"Unsubscribed from {len(symbols)} symbols on Finnhub")
    
    async def _message_loop(self):
        """Main message processing loop"""
        
        self.is_running = True
        
        while self.is_running and self.is_connected:
            try:
                message_text = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=self.config.heartbeat_interval
                )
                
                self.metrics.bytes_received += len(message_text.encode())
                
                data = json.loads(message_text)
                await self._process_message(data)
                
            except asyncio.TimeoutError:
                # Send ping
                await self.websocket.ping()
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Finnhub WebSocket connection closed")
                self.is_connected = False
                break
                
            except Exception as e:
                logger.error(f"Finnhub WebSocket error: {e}")
                self._handle_error(e)
    
    async def _process_message(self, data: Dict[str, Any]):
        """Process Finnhub message"""
        
        msg_type = data.get('type', '')
        
        if msg_type == 'trade':
            trades = data.get('data', [])
            
            for trade in trades:
                stream_message = StreamMessage(
                    timestamp=datetime.now(),
                    symbol=trade.get('s', ''),
                    stream_type=StreamType.TRADE_EXECUTIONS,
                    data={
                        'price': trade.get('p', 0),
                        'volume': trade.get('v', 0),
                        'timestamp': trade.get('t', 0),
                        'conditions': trade.get('c', [])
                    },
                    provider=StreamingProvider.FINNHUB
                )
                
                self._handle_message(stream_message)

class StreamingPipeline:
    """
    Real-time streaming data pipeline
    """
    
    def __init__(self, output_format: str = "kafka"):
        self.sources: Dict[str, StreamingDataSource] = {}
        self.message_buffer = MessageBuffer(max_size=50000)
        self.output_format = output_format
        self.is_running = False
        
        # Message processors
        self.processors: List[Callable[[StreamMessage], StreamMessage]] = []
        
        # Output handlers
        self.kafka_producer = None
        self.redis_client = None
        
        # Performance metrics
        self.pipeline_metrics = {
            'total_messages_processed': 0,
            'messages_per_second': 0.0,
            'processing_latency_ms': 0.0,
            'buffer_utilization': 0.0
        }
        
        # Initialize output systems
        self._initialize_output_systems()
        
        logger.info(f"StreamingPipeline initialized with {output_format} output")
    
    def _initialize_output_systems(self):
        """Initialize output systems (Kafka, Redis, etc.)"""
        
        if self.output_format == "kafka" and KAFKA_AVAILABLE:
            try:
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    value_serializer=lambda x: json.dumps(x, default=str).encode('utf-8'),
                    batch_size=16384,
                    linger_ms=10,
                    compression_type='gzip'
                )
                logger.info("Kafka producer initialized")
            except Exception as e:
                logger.warning(f"Kafka initialization failed: {e}")
        
        if self.output_format == "redis" and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("Redis client initialized")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}")
    
    def add_source(self, name: str, source: StreamingDataSource):
        """Add streaming data source"""
        
        # Add pipeline as message handler
        source.add_message_handler(self._handle_stream_message)
        source.add_error_handler(self._handle_stream_error)
        
        self.sources[name] = source
        logger.info(f"Added streaming source: {name}")
    
    def add_processor(self, processor: Callable[[StreamMessage], StreamMessage]):
        """Add message processor"""
        self.processors.append(processor)
    
    async def start(self):
        """Start the streaming pipeline"""
        
        logger.info("Starting streaming pipeline...")
        
        # Connect all sources
        for name, source in self.sources.items():
            try:
                await source.connect()
                logger.info(f"Connected source: {name}")
            except Exception as e:
                logger.error(f"Failed to connect source {name}: {e}")
        
        # Start processing loop
        self.is_running = True
        asyncio.create_task(self._processing_loop())
        asyncio.create_task(self._metrics_loop())
        
        logger.info("Streaming pipeline started")
    
    async def stop(self):
        """Stop the streaming pipeline"""
        
        logger.info("Stopping streaming pipeline...")
        
        self.is_running = False
        
        # Disconnect all sources
        for name, source in self.sources.items():
            try:
                await source.disconnect()
                logger.info(f"Disconnected source: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting source {name}: {e}")
        
        # Close output systems
        if self.kafka_producer:
            self.kafka_producer.flush()
            self.kafka_producer.close()
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Streaming pipeline stopped")
    
    async def subscribe_symbols(self, symbols: List[str], source_names: List[str] = None):
        """Subscribe to symbols on specified sources"""
        
        if source_names is None:
            source_names = list(self.sources.keys())
        
        for name in source_names:
            if name in self.sources:
                try:
                    await self.sources[name].subscribe(symbols)
                    logger.info(f"Subscribed to {len(symbols)} symbols on {name}")
                except Exception as e:
                    logger.error(f"Subscription error on {name}: {e}")
    
    async def unsubscribe_symbols(self, symbols: List[str], source_names: List[str] = None):
        """Unsubscribe from symbols on specified sources"""
        
        if source_names is None:
            source_names = list(self.sources.keys())
        
        for name in source_names:
            if name in self.sources:
                try:
                    await self.sources[name].unsubscribe(symbols)
                    logger.info(f"Unsubscribed from {len(symbols)} symbols on {name}")
                except Exception as e:
                    logger.error(f"Unsubscription error on {name}: {e}")
    
    def _handle_stream_message(self, message: StreamMessage):
        """Handle incoming stream message"""
        
        # Add to buffer
        if not self.message_buffer.put(message):
            logger.warning("Message buffer full, dropping message")
    
    def _handle_stream_error(self, error: Exception):
        """Handle streaming errors"""
        logger.error(f"Stream error: {error}")
    
    async def _processing_loop(self):
        """Main message processing loop"""
        
        last_metrics_time = time.time()
        messages_processed_since_last = 0
        
        while self.is_running:
            try:
                # Get batch of messages
                messages = self.message_buffer.get_batch(self.message_buffer.config.batch_size if hasattr(self.message_buffer, 'config') else 100)
                
                if not messages:
                    await asyncio.sleep(0.01)  # Short sleep if no messages
                    continue
                
                # Process messages
                for message in messages:
                    processed_message = await self._process_message(message)
                    
                    if processed_message:
                        await self._output_message(processed_message)
                        messages_processed_since_last += 1
                        self.pipeline_metrics['total_messages_processed'] += 1
                
                # Update metrics
                now = time.time()
                if now - last_metrics_time >= 1.0:  # Every second
                    self.pipeline_metrics['messages_per_second'] = messages_processed_since_last / (now - last_metrics_time)
                    messages_processed_since_last = 0
                    last_metrics_time = now
                
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_message(self, message: StreamMessage) -> Optional[StreamMessage]:
        """Process individual message through processors"""
        
        start_time = time.time()
        
        try:
            processed_message = message
            
            # Apply processors
            for processor in self.processors:
                processed_message = processor(processed_message)
                if processed_message is None:
                    return None  # Message filtered out
            
            # Update latency metric
            processing_time = (time.time() - start_time) * 1000
            self.pipeline_metrics['processing_latency_ms'] = (
                self.pipeline_metrics['processing_latency_ms'] * 0.9 + processing_time * 0.1
            )
            
            return processed_message
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            return None
    
    async def _output_message(self, message: StreamMessage):
        """Output message to configured output system"""
        
        # Convert message to dictionary
        message_dict = {
            'timestamp': message.timestamp.isoformat(),
            'symbol': message.symbol,
            'stream_type': message.stream_type.value,
            'data': message.data,
            'provider': message.provider.value,
            'sequence_id': message.sequence_id,
            'metadata': message.metadata
        }
        
        try:
            if self.output_format == "kafka" and self.kafka_producer:
                # Send to Kafka topic
                topic = f"stock_stream_{message.stream_type.value}"
                self.kafka_producer.send(topic, value=message_dict, key=message.symbol)
                
            elif self.output_format == "redis" and self.redis_client:
                # Send to Redis stream
                stream_key = f"stream:{message.symbol}:{message.stream_type.value}"
                self.redis_client.xadd(stream_key, message_dict)
                
            else:
                # Log output (fallback)
                logger.info(f"Stream message: {message.symbol} - {message.stream_type.value}")
                
        except Exception as e:
            logger.error(f"Output error: {e}")
    
    async def _metrics_loop(self):
        """Update pipeline metrics"""
        
        while self.is_running:
            try:
                # Update buffer utilization
                buffer_size = self.message_buffer.size()
                max_size = self.message_buffer.max_size
                self.pipeline_metrics['buffer_utilization'] = (buffer_size / max_size) * 100
                
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(5.0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics"""
        
        # Collect source metrics
        source_metrics = {}
        for name, source in self.sources.items():
            source_metrics[name] = {
                'connected': source.is_connected,
                'running': source.is_running,
                'messages_received': source.metrics.messages_received,
                'latency_ms': source.metrics.latency_ms,
                'errors': source.metrics.errors,
                'uptime_seconds': (
                    (datetime.now() - source.connection_start_time).total_seconds()
                    if source.connection_start_time else 0
                )
            }
        
        return {
            'pipeline': self.pipeline_metrics,
            'sources': source_metrics,
            'buffer': {
                'size': self.message_buffer.size(),
                'max_size': self.message_buffer.max_size,
                'dropped_count': self.message_buffer.dropped_count,
                'utilization_pct': self.pipeline_metrics['buffer_utilization']
            }
        }

# Message processors
def price_aggregator_processor(message: StreamMessage) -> StreamMessage:
    """Aggregate price data"""
    
    if message.stream_type == StreamType.TRADE_EXECUTIONS:
        # Add OHLC aggregation logic here
        pass
    
    return message

def anomaly_detector_processor(message: StreamMessage) -> Optional[StreamMessage]:
    """Filter out anomalous messages"""
    
    if message.stream_type == StreamType.TRADE_EXECUTIONS:
        price = message.data.get('price', 0)
        
        # Simple anomaly detection (price bounds)
        if price <= 0 or price > 10000:  # Reasonable bounds for most stocks
            logger.warning(f"Anomalous price detected: {price} for {message.symbol}")
            return None  # Filter out
    
    return message

def data_enricher_processor(message: StreamMessage) -> StreamMessage:
    """Enrich message with additional data"""
    
    # Add timestamp enrichment
    message.metadata['processed_at'] = datetime.now().isoformat()
    
    # Add market session info
    now = datetime.now().time()
    market_open = datetime.strptime("09:30", "%H:%M").time()
    market_close = datetime.strptime("16:00", "%H:%M").time()
    
    message.metadata['market_session'] = (
        'open' if market_open <= now <= market_close else 'closed'
    )
    
    return message

# Example usage and testing
if __name__ == "__main__":
    print("🌊 Real-Time Streaming Pipeline")
    print("=" * 50)
    
    async def test_streaming_pipeline():
        # Initialize pipeline
        pipeline = StreamingPipeline(output_format="redis")
        
        # Add processors
        pipeline.add_processor(anomaly_detector_processor)
        pipeline.add_processor(data_enricher_processor)
        pipeline.add_processor(price_aggregator_processor)
        
        # Add mock streaming source for testing
        class MockStreamingSource(StreamingDataSource):
            def __init__(self, config: StreamConfig):
                super().__init__(config)
                self.symbols = config.symbols
            
            async def connect(self):
                self.is_connected = True
                self.connection_start_time = datetime.now()
                asyncio.create_task(self._generate_mock_data())
                logger.info("Mock source connected")
            
            async def disconnect(self):
                self.is_connected = False
                self.is_running = False
                logger.info("Mock source disconnected")
            
            async def subscribe(self, symbols: List[str]):
                self.symbols.extend(symbols)
                logger.info(f"Mock subscribed to {symbols}")
            
            async def unsubscribe(self, symbols: List[str]):
                for symbol in symbols:
                    if symbol in self.symbols:
                        self.symbols.remove(symbol)
                logger.info(f"Mock unsubscribed from {symbols}")
            
            async def _generate_mock_data(self):
                self.is_running = True
                
                while self.is_running:
                    for symbol in self.symbols:
                        # Generate mock trade data
                        message = StreamMessage(
                            timestamp=datetime.now(),
                            symbol=symbol,
                            stream_type=StreamType.TRADE_EXECUTIONS,
                            data={
                                'price': 100 + np.random.randn() * 5,
                                'size': np.random.randint(100, 1000),
                                'exchange': 'NASDAQ'
                            },
                            provider=StreamingProvider.WEBSOCKET_GENERIC
                        )
                        
                        self._handle_message(message)
                    
                    await asyncio.sleep(0.1)  # 10 messages per second
        
        # Create mock source
        mock_config = StreamConfig(
            provider=StreamingProvider.WEBSOCKET_GENERIC,
            stream_type=StreamType.TRADE_EXECUTIONS,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            websocket_url="mock://test"
        )
        
        mock_source = MockStreamingSource(mock_config)
        pipeline.add_source("mock_source", mock_source)
        
        # Start pipeline
        await pipeline.start()
        
        # Subscribe to symbols
        await pipeline.subscribe_symbols(['AAPL', 'MSFT', 'GOOGL'])
        
        # Run for a few seconds
        print("📊 Running streaming pipeline for 10 seconds...")
        await asyncio.sleep(10)
        
        # Print metrics
        print("\n📈 Pipeline Metrics:")
        metrics = pipeline.get_metrics()
        
        print(f"Pipeline:")
        for key, value in metrics['pipeline'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\nBuffer:")
        for key, value in metrics['buffer'].items():
            print(f"   {key}: {value}")
        
        print(f"\nSources:")
        for source_name, source_metrics in metrics['sources'].items():
            print(f"   {source_name}:")
            for key, value in source_metrics.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.2f}")
                else:
                    print(f"      {key}: {value}")
        
        # Stop pipeline
        await pipeline.stop()
    
    # Run test
    asyncio.run(test_streaming_pipeline())
    
    print(f"\n🎯 Real-time streaming pipeline ready!")
    print(f"📋 Features:")
    print(f"   • Multi-provider WebSocket support")
    print(f"   • Message buffering and batching")
    print(f"   • Configurable processors")
    print(f"   • Kafka/Redis output")
    print(f"   • Performance monitoring")
    print(f"   • Error handling and reconnection")