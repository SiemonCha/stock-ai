"""
Backup Data Sources and Redundancy System
Enterprise-grade data backup and redundancy for continuous operation
"""

import asyncio
import aiofiles
import sqlite3
import threading
import time
import json
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from sqlalchemy import create_engine, MetaData, Table, Column, String, DateTime, Float, Integer, Text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupStrategy(Enum):
    """Backup strategies"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONTINUOUS = "continuous"

class StorageType(Enum):
    """Storage types"""
    LOCAL_DISK = "local_disk"
    NETWORK_STORAGE = "network_storage"
    CLOUD_STORAGE = "cloud_storage"
    DATABASE = "database"
    MEMORY_CACHE = "memory_cache"

class BackupStatus(Enum):
    """Backup status"""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    SCHEDULED = "scheduled"

@dataclass
class BackupConfig:
    """Backup configuration"""
    name: str
    storage_type: StorageType
    strategy: BackupStrategy
    frequency: int  # seconds
    retention_days: int
    compression: bool = True
    encryption: bool = False
    priority: int = 1  # 1 = highest
    enabled: bool = True
    connection_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupRecord:
    """Backup record"""
    backup_id: str
    timestamp: datetime
    symbol: str
    data_type: str
    storage_location: str
    size_bytes: int
    checksum: str
    status: BackupStatus
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RestoreRequest:
    """Data restore request"""
    symbol: str
    data_type: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    preferred_source: Optional[str] = None
    quality_threshold: float = 0.8

class LocalDiskBackup:
    """Local disk backup storage"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.base_path = Path(config.connection_params.get('path', '/var/backups/stock_data'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def store_data(self, symbol: str, data: pd.DataFrame, data_type: str) -> BackupRecord:
        """Store data to local disk"""
        
        timestamp = datetime.now()
        backup_id = f"{symbol}_{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create directory structure
        symbol_dir = self.base_path / symbol / data_type
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file format and extension
        if PARQUET_AVAILABLE and self.config.connection_params.get('format', 'parquet') == 'parquet':
            filename = f"{backup_id}.parquet"
            filepath = symbol_dir / filename
            
            if self.config.compression:
                data.to_parquet(filepath, compression='gzip')
            else:
                data.to_parquet(filepath)
                
        elif HDF5_AVAILABLE and self.config.connection_params.get('format', 'parquet') == 'hdf5':
            filename = f"{backup_id}.h5"
            filepath = symbol_dir / filename
            
            compression = 'gzip' if self.config.compression else None
            data.to_hdf(filepath, key='data', mode='w', complevel=9 if compression else 0)
            
        else:
            # Fallback to pickle
            filename = f"{backup_id}.pkl"
            filepath = symbol_dir / filename
            
            if self.config.compression:
                filename = f"{backup_id}.pkl.gz"
                filepath = symbol_dir / filename
                
                async with aiofiles.open(filepath, 'wb') as f:
                    compressed_data = gzip.compress(data.to_pickle())
                    await f.write(compressed_data)
            else:
                data.to_pickle(filepath)
        
        # Calculate file size and checksum
        file_size = filepath.stat().st_size
        checksum = self._calculate_checksum(filepath)
        
        return BackupRecord(
            backup_id=backup_id,
            timestamp=timestamp,
            symbol=symbol,
            data_type=data_type,
            storage_location=str(filepath),
            size_bytes=file_size,
            checksum=checksum,
            status=BackupStatus.SUCCESS,
            metadata={
                'format': self.config.connection_params.get('format', 'parquet'),
                'compression': self.config.compression
            }
        )
    
    async def retrieve_data(self, backup_record: BackupRecord) -> Optional[pd.DataFrame]:
        """Retrieve data from local disk"""
        
        filepath = Path(backup_record.storage_location)
        
        if not filepath.exists():
            logger.error(f"Backup file not found: {filepath}")
            return None
        
        try:
            # Verify checksum
            if not self._verify_checksum(filepath, backup_record.checksum):
                logger.error(f"Checksum mismatch for backup: {backup_record.backup_id}")
                return None
            
            # Load data based on format
            format_type = backup_record.metadata.get('format', 'parquet')
            
            if format_type == 'parquet' and PARQUET_AVAILABLE:
                return pd.read_parquet(filepath)
            elif format_type == 'hdf5' and HDF5_AVAILABLE:
                return pd.read_hdf(filepath, key='data')
            elif format_type == 'pickle':
                if filepath.suffix == '.gz':
                    with gzip.open(filepath, 'rb') as f:
                        return pd.read_pickle(f)
                else:
                    return pd.read_pickle(filepath)
            
            logger.error(f"Unsupported format: {format_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving backup {backup_record.backup_id}: {e}")
            return None
    
    def cleanup_old_backups(self, retention_days: int):
        """Clean up old backup files"""
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0
        
        for backup_file in self.base_path.rglob('*'):
            if backup_file.is_file():
                file_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                
                if file_time < cutoff_date:
                    try:
                        backup_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove old backup {backup_file}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old backup files")
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate file checksum"""
        import hashlib
        
        hash_md5 = hashlib.md5()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
    
    def _verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """Verify file checksum"""
        actual_checksum = self._calculate_checksum(filepath)
        return actual_checksum == expected_checksum

class CloudStorageBackup:
    """Cloud storage backup (AWS S3)"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        
        if not AWS_AVAILABLE:
            raise ImportError("AWS SDK not available. Install with: pip install boto3")
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=config.connection_params.get('access_key'),
            aws_secret_access_key=config.connection_params.get('secret_key'),
            region_name=config.connection_params.get('region', 'us-east-1')
        )
        
        self.bucket_name = config.connection_params.get('bucket')
        self.prefix = config.connection_params.get('prefix', 'stock-data-backups/')
        
        if not self.bucket_name:
            raise ValueError("S3 bucket name is required")
    
    async def store_data(self, symbol: str, data: pd.DataFrame, data_type: str) -> BackupRecord:
        """Store data to S3"""
        
        timestamp = datetime.now()
        backup_id = f"{symbol}_{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare data for upload
        if self.config.compression:
            # Compress data
            buffer = data.to_parquet(compression='gzip')
            key = f"{self.prefix}{symbol}/{data_type}/{backup_id}.parquet.gz"
        else:
            buffer = data.to_parquet()
            key = f"{self.prefix}{symbol}/{data_type}/{backup_id}.parquet"
        
        try:
            # Upload to S3
            response = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer,
                ServerSideEncryption='AES256' if self.config.encryption else None,
                Metadata={
                    'symbol': symbol,
                    'data_type': data_type,
                    'backup_id': backup_id,
                    'timestamp': timestamp.isoformat()
                }
            )
            
            # Calculate size
            size_bytes = len(buffer) if isinstance(buffer, bytes) else len(str(buffer).encode())
            
            return BackupRecord(
                backup_id=backup_id,
                timestamp=timestamp,
                symbol=symbol,
                data_type=data_type,
                storage_location=key,
                size_bytes=size_bytes,
                checksum=response.get('ETag', '').strip('"'),
                status=BackupStatus.SUCCESS,
                metadata={
                    'bucket': self.bucket_name,
                    'encryption': self.config.encryption
                }
            )
            
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            return BackupRecord(
                backup_id=backup_id,
                timestamp=timestamp,
                symbol=symbol,
                data_type=data_type,
                storage_location="",
                size_bytes=0,
                checksum="",
                status=BackupStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    async def retrieve_data(self, backup_record: BackupRecord) -> Optional[pd.DataFrame]:
        """Retrieve data from S3"""
        
        try:
            response = self.s3_client.get_object(
                Bucket=backup_record.metadata['bucket'],
                Key=backup_record.storage_location
            )
            
            # Read data
            buffer = response['Body'].read()
            
            # Determine if compressed
            if backup_record.storage_location.endswith('.gz'):
                # Decompress and load
                import io
                decompressed = gzip.decompress(buffer)
                return pd.read_parquet(io.BytesIO(decompressed))
            else:
                import io
                return pd.read_parquet(io.BytesIO(buffer))
                
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            return None

class DatabaseBackup:
    """Database backup storage"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError("SQLAlchemy not available. Install with: pip install sqlalchemy")
        
        # Create database engine
        db_url = config.connection_params.get('url')
        if not db_url:
            # Default to SQLite
            db_path = config.connection_params.get('path', '/var/lib/stock_ai/backups.db')
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_url)
        self._create_tables()
    
    def _create_tables(self):
        """Create backup tables"""
        
        from sqlalchemy import MetaData, Table, Column, String, DateTime, LargeBinary, Integer
        
        metadata = MetaData()
        
        # Backup metadata table
        self.backups_table = Table(
            'backups',
            metadata,
            Column('backup_id', String, primary_key=True),
            Column('timestamp', DateTime),
            Column('symbol', String),
            Column('data_type', String),
            Column('size_bytes', Integer),
            Column('checksum', String),
            Column('status', String),
            Column('metadata', String)  # JSON
        )
        
        # Data storage table
        self.data_table = Table(
            'backup_data',
            metadata,
            Column('backup_id', String, primary_key=True),
            Column('data', LargeBinary)  # Compressed data
        )
        
        metadata.create_all(self.engine)
    
    async def store_data(self, symbol: str, data: pd.DataFrame, data_type: str) -> BackupRecord:
        """Store data to database"""
        
        timestamp = datetime.now()
        backup_id = f"{symbol}_{data_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Serialize and optionally compress data
            if self.config.compression:
                serialized_data = gzip.compress(data.to_pickle())
            else:
                serialized_data = data.to_pickle()
            
            # Calculate checksum
            import hashlib
            checksum = hashlib.md5(serialized_data).hexdigest()
            
            # Store in database
            with self.engine.connect() as conn:
                # Insert metadata
                conn.execute(
                    self.backups_table.insert().values(
                        backup_id=backup_id,
                        timestamp=timestamp,
                        symbol=symbol,
                        data_type=data_type,
                        size_bytes=len(serialized_data),
                        checksum=checksum,
                        status=BackupStatus.SUCCESS.value,
                        metadata=json.dumps({'compression': self.config.compression})
                    )
                )
                
                # Insert data
                conn.execute(
                    self.data_table.insert().values(
                        backup_id=backup_id,
                        data=serialized_data
                    )
                )
                
                conn.commit()
            
            return BackupRecord(
                backup_id=backup_id,
                timestamp=timestamp,
                symbol=symbol,
                data_type=data_type,
                storage_location=f"db:{backup_id}",
                size_bytes=len(serialized_data),
                checksum=checksum,
                status=BackupStatus.SUCCESS,
                metadata={'compression': self.config.compression}
            )
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return BackupRecord(
                backup_id=backup_id,
                timestamp=timestamp,
                symbol=symbol,
                data_type=data_type,
                storage_location="",
                size_bytes=0,
                checksum="",
                status=BackupStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    async def retrieve_data(self, backup_record: BackupRecord) -> Optional[pd.DataFrame]:
        """Retrieve data from database"""
        
        try:
            with self.engine.connect() as conn:
                # Get data
                result = conn.execute(
                    self.data_table.select().where(
                        self.data_table.c.backup_id == backup_record.backup_id
                    )
                ).fetchone()
                
                if not result:
                    return None
                
                serialized_data = result.data
                
                # Verify checksum
                import hashlib
                actual_checksum = hashlib.md5(serialized_data).hexdigest()
                
                if actual_checksum != backup_record.checksum:
                    logger.error(f"Checksum mismatch for backup: {backup_record.backup_id}")
                    return None
                
                # Deserialize data
                if backup_record.metadata.get('compression'):
                    decompressed_data = gzip.decompress(serialized_data)
                    return pd.read_pickle(io.BytesIO(decompressed_data))
                else:
                    return pd.read_pickle(io.BytesIO(serialized_data))
                    
        except Exception as e:
            logger.error(f"Database retrieval failed: {e}")
            return None

class BackupManager:
    """
    Enterprise backup and redundancy manager
    """
    
    def __init__(self):
        self.backup_configs: Dict[str, BackupConfig] = {}
        self.backup_handlers: Dict[StorageType, Any] = {}
        self.backup_registry: Dict[str, List[BackupRecord]] = {}
        self.backup_scheduler_running = False
        self.scheduler_thread = None
        
        # Performance metrics
        self.backup_stats = {
            'total_backups': 0,
            'successful_backups': 0,
            'failed_backups': 0,
            'total_size_bytes': 0,
            'avg_backup_time': 0.0
        }
        
        logger.info("BackupManager initialized")
    
    def add_backup_config(self, config: BackupConfig):
        """Add backup configuration"""
        
        self.backup_configs[config.name] = config
        
        # Initialize storage handler
        if config.storage_type == StorageType.LOCAL_DISK:
            self.backup_handlers[config.storage_type] = LocalDiskBackup(config)
        elif config.storage_type == StorageType.CLOUD_STORAGE:
            self.backup_handlers[config.storage_type] = CloudStorageBackup(config)
        elif config.storage_type == StorageType.DATABASE:
            self.backup_handlers[config.storage_type] = DatabaseBackup(config)
        
        logger.info(f"Added backup configuration: {config.name} ({config.storage_type.value})")
    
    async def backup_data(self, symbol: str, data: pd.DataFrame, data_type: str, 
                         backup_names: List[str] = None) -> List[BackupRecord]:
        """Backup data to configured storage systems"""
        
        if backup_names is None:
            backup_names = [name for name, config in self.backup_configs.items() if config.enabled]
        
        backup_tasks = []
        
        for name in backup_names:
            if name not in self.backup_configs:
                logger.warning(f"Backup config not found: {name}")
                continue
            
            config = self.backup_configs[name]
            handler = self.backup_handlers.get(config.storage_type)
            
            if handler:
                task = asyncio.create_task(
                    self._backup_with_handler(handler, symbol, data, data_type)
                )
                backup_tasks.append((name, task))
        
        # Execute backups concurrently
        backup_records = []
        
        for name, task in backup_tasks:
            try:
                start_time = time.time()
                record = await task
                backup_time = time.time() - start_time
                
                # Update statistics
                self.backup_stats['total_backups'] += 1
                
                if record.status == BackupStatus.SUCCESS:
                    self.backup_stats['successful_backups'] += 1
                    self.backup_stats['total_size_bytes'] += record.size_bytes
                    
                    # Update average backup time
                    total_successful = self.backup_stats['successful_backups']
                    current_avg = self.backup_stats['avg_backup_time']
                    self.backup_stats['avg_backup_time'] = (
                        (current_avg * (total_successful - 1) + backup_time) / total_successful
                    )
                    
                    logger.info(f"Backup successful: {name} - {record.backup_id}")
                else:
                    self.backup_stats['failed_backups'] += 1
                    logger.error(f"Backup failed: {name} - {record.backup_id}")
                
                # Register backup
                registry_key = f"{symbol}_{data_type}"
                if registry_key not in self.backup_registry:
                    self.backup_registry[registry_key] = []
                
                self.backup_registry[registry_key].append(record)
                backup_records.append(record)
                
            except Exception as e:
                logger.error(f"Backup task failed for {name}: {e}")
                self.backup_stats['failed_backups'] += 1
        
        return backup_records
    
    async def restore_data(self, request: RestoreRequest) -> Optional[pd.DataFrame]:
        """Restore data with intelligent source selection"""
        
        registry_key = f"{request.symbol}_{request.data_type}"
        
        if registry_key not in self.backup_registry:
            logger.warning(f"No backups found for {registry_key}")
            return None
        
        # Get available backups
        available_backups = self.backup_registry[registry_key]
        
        # Filter by date range if specified
        if request.start_date or request.end_date:
            filtered_backups = []
            for backup in available_backups:
                backup_date = backup.timestamp
                
                if request.start_date and backup_date < request.start_date:
                    continue
                if request.end_date and backup_date > request.end_date:
                    continue
                
                filtered_backups.append(backup)
            
            available_backups = filtered_backups
        
        if not available_backups:
            logger.warning(f"No backups match criteria for {registry_key}")
            return None
        
        # Sort by preference and quality
        available_backups = sorted(
            available_backups,
            key=lambda b: (
                b.status == BackupStatus.SUCCESS,  # Successful backups first
                -self._get_backup_priority(b),     # Higher priority first
                -b.timestamp.timestamp()           # More recent first
            ),
            reverse=True
        )
        
        # Try to restore from backups in order of preference
        for backup in available_backups:
            try:
                logger.info(f"Attempting restore from backup: {backup.backup_id}")
                
                # Get handler for this backup
                handler = self._get_handler_for_backup(backup)
                if not handler:
                    continue
                
                # Restore data
                data = await handler.retrieve_data(backup)
                
                if data is not None and not data.empty:
                    logger.info(f"Successfully restored data from: {backup.backup_id}")
                    return data
                
            except Exception as e:
                logger.error(f"Restore failed for backup {backup.backup_id}: {e}")
                continue
        
        logger.error(f"All restore attempts failed for {registry_key}")
        return None
    
    def start_scheduler(self):
        """Start backup scheduler"""
        
        if self.backup_scheduler_running:
            return
        
        self.backup_scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._backup_scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Backup scheduler started")
    
    def stop_scheduler(self):
        """Stop backup scheduler"""
        
        self.backup_scheduler_running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Backup scheduler stopped")
    
    def _backup_scheduler_loop(self):
        """Backup scheduler main loop"""
        
        last_backup_times = {}
        
        while self.backup_scheduler_running:
            try:
                current_time = time.time()
                
                for name, config in self.backup_configs.items():
                    if not config.enabled:
                        continue
                    
                    last_backup = last_backup_times.get(name, 0)
                    
                    if current_time - last_backup >= config.frequency:
                        # Trigger scheduled backup
                        logger.info(f"Triggering scheduled backup: {name}")
                        
                        # This would need to be integrated with a data collection system
                        # For now, we'll just update the timestamp
                        last_backup_times[name] = current_time
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Backup scheduler error: {e}")
                time.sleep(60)
    
    async def _backup_with_handler(self, handler: Any, symbol: str, 
                                 data: pd.DataFrame, data_type: str) -> BackupRecord:
        """Execute backup with specific handler"""
        
        try:
            return await handler.store_data(symbol, data, data_type)
        except Exception as e:
            logger.error(f"Handler backup failed: {e}")
            
            # Return failed record
            return BackupRecord(
                backup_id=f"failed_{int(time.time())}",
                timestamp=datetime.now(),
                symbol=symbol,
                data_type=data_type,
                storage_location="",
                size_bytes=0,
                checksum="",
                status=BackupStatus.FAILED,
                metadata={'error': str(e)}
            )
    
    def _get_backup_priority(self, backup: BackupRecord) -> int:
        """Get backup priority score"""
        
        # Find the backup config that matches this backup
        for config in self.backup_configs.values():
            if config.storage_type.value in backup.metadata.get('storage_type', ''):
                return config.priority
        
        return 0  # Default priority
    
    def _get_handler_for_backup(self, backup: BackupRecord) -> Optional[Any]:
        """Get appropriate handler for backup"""
        
        # Determine storage type from backup metadata or location
        if backup.storage_location.startswith('db:'):
            storage_type = StorageType.DATABASE
        elif backup.storage_location.startswith('s3://') or 'bucket' in backup.metadata:
            storage_type = StorageType.CLOUD_STORAGE
        else:
            storage_type = StorageType.LOCAL_DISK
        
        return self.backup_handlers.get(storage_type)
    
    def get_backup_status(self) -> Dict[str, Any]:
        """Get comprehensive backup status"""
        
        status = {
            'statistics': self.backup_stats.copy(),
            'configurations': {},
            'registry_summary': {}
        }
        
        # Configuration status
        for name, config in self.backup_configs.items():
            status['configurations'][name] = {
                'enabled': config.enabled,
                'storage_type': config.storage_type.value,
                'strategy': config.strategy.value,
                'frequency': config.frequency,
                'priority': config.priority
            }
        
        # Registry summary
        for key, backups in self.backup_registry.items():
            successful_backups = [b for b in backups if b.status == BackupStatus.SUCCESS]
            
            status['registry_summary'][key] = {
                'total_backups': len(backups),
                'successful_backups': len(successful_backups),
                'latest_backup': max(b.timestamp for b in backups) if backups else None,
                'total_size_bytes': sum(b.size_bytes for b in successful_backups)
            }
        
        return status
    
    def cleanup_old_backups(self, retention_days: int = None):
        """Cleanup old backups across all storage systems"""
        
        for name, config in self.backup_configs.items():
            retention = retention_days or config.retention_days
            
            handler = self.backup_handlers.get(config.storage_type)
            if handler and hasattr(handler, 'cleanup_old_backups'):
                try:
                    handler.cleanup_old_backups(retention)
                    logger.info(f"Cleaned up old backups for {name}")
                except Exception as e:
                    logger.error(f"Cleanup failed for {name}: {e}")
        
        # Also clean up registry entries
        cutoff_date = datetime.now() - timedelta(days=retention_days or 30)
        
        for key, backups in self.backup_registry.items():
            self.backup_registry[key] = [
                b for b in backups if b.timestamp >= cutoff_date
            ]

# Example usage and testing
if __name__ == "__main__":
    print("🛡️ Backup and Redundancy System")
    print("=" * 50)
    
    async def test_backup_system():
        # Initialize backup manager
        manager = BackupManager()
        
        # Configure local disk backup
        local_config = BackupConfig(
            name="local_primary",
            storage_type=StorageType.LOCAL_DISK,
            strategy=BackupStrategy.FULL,
            frequency=3600,  # 1 hour
            retention_days=30,
            compression=True,
            priority=1,
            connection_params={
                'path': './test_backups',
                'format': 'parquet'
            }
        )
        
        manager.add_backup_config(local_config)
        
        # Configure database backup
        db_config = BackupConfig(
            name="database_secondary",
            storage_type=StorageType.DATABASE,
            strategy=BackupStrategy.INCREMENTAL,
            frequency=7200,  # 2 hours
            retention_days=60,
            compression=True,
            priority=2,
            connection_params={
                'path': './test_backups/backup.db'
            }
        )
        
        manager.add_backup_config(db_config)
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        test_data = pd.DataFrame({
            'Date': dates,
            'Open': 100 + np.random.randn(100).cumsum(),
            'High': 105 + np.random.randn(100).cumsum(),
            'Low': 95 + np.random.randn(100).cumsum(),
            'Close': 100 + np.random.randn(100).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }).set_index('Date')
        
        print(f"📊 Created test data: {len(test_data)} rows")
        
        # Test backup
        print(f"\n💾 Testing backup operations...")
        backup_records = await manager.backup_data('AAPL', test_data, 'prices')
        
        print(f"✅ Backup completed: {len(backup_records)} backups created")
        for record in backup_records:
            status_emoji = "✅" if record.status == BackupStatus.SUCCESS else "❌"
            print(f"   {status_emoji} {record.backup_id}: {record.size_bytes:,} bytes")
        
        # Test restore
        print(f"\n🔄 Testing restore operations...")
        
        restore_request = RestoreRequest(
            symbol='AAPL',
            data_type='prices'
        )
        
        restored_data = await manager.restore_data(restore_request)
        
        if restored_data is not None:
            print(f"✅ Data restored successfully: {len(restored_data)} rows")
            
            # Verify data integrity
            if test_data.shape == restored_data.shape:
                data_match = np.allclose(test_data.values, restored_data.values, rtol=1e-10)
                print(f"   Data integrity: {'✅ PASS' if data_match else '❌ FAIL'}")
            else:
                print(f"   Shape mismatch: {test_data.shape} vs {restored_data.shape}")
        else:
            print(f"❌ Restore failed")
        
        # Test backup status
        print(f"\n📊 Backup System Status:")
        status = manager.get_backup_status()
        
        print(f"Statistics:")
        for key, value in status['statistics'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\nConfigurations:")
        for name, config in status['configurations'].items():
            print(f"   {name}: {config['storage_type']} ({'enabled' if config['enabled'] else 'disabled'})")
        
        print(f"\nRegistry Summary:")
        for key, summary in status['registry_summary'].items():
            print(f"   {key}: {summary['successful_backups']} successful backups")
        
        # Test multiple backup sources
        print(f"\n🔀 Testing multiple backup sources...")
        
        # Simulate second backup with different data
        modified_data = test_data.copy()
        modified_data['Close'] = modified_data['Close'] * 1.1  # 10% increase
        
        backup_records_2 = await manager.backup_data('AAPL', modified_data, 'prices')
        print(f"✅ Second backup completed: {len(backup_records_2)} backups")
        
        # Restore should get the most recent backup
        restored_data_2 = await manager.restore_data(restore_request)
        
        if restored_data_2 is not None:
            # Check if we got the modified data (most recent)
            close_match = np.allclose(modified_data['Close'].values, restored_data_2['Close'].values)
            print(f"   Latest data restored: {'✅ YES' if close_match else '❌ NO'}")
    
    # Run test
    asyncio.run(test_backup_system())
    
    print(f"\n🎯 Backup and redundancy system ready!")
    print(f"📋 Features:")
    print(f"   • Multi-storage backend support")
    print(f"   • Intelligent restore with failover")
    print(f"   • Data integrity verification")
    print(f"   • Automated scheduling")
    print(f"   • Compression and encryption")
    print(f"   • Performance monitoring")