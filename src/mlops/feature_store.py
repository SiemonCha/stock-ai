"""
Feature Store Module

Redis-based feature storage with versioning and metadata tracking.
Falls back to in-memory cache if Redis is unavailable.
"""

import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from io import StringIO
import pandas as pd
import numpy as np

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class FeatureStore:
    """
    Simple feature store for caching computed features.

    Uses Redis for distributed caching with automatic fallback to
    in-memory storage when Redis is unavailable.
    """

    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, ttl_hours: int = 24):
        """
        Initialize feature store.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            ttl_hours: Time-to-live for cached features in hours
        """
        self.ttl_seconds = ttl_hours * 3600
        self.use_redis = False
        self.cache = {}  # In-memory fallback

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=False,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                self.use_redis = True
            except Exception:
                pass  # Fall back to in-memory

    def _make_key(self, symbol: str, feature_group: str, version: str = "v1") -> str:
        """Generate storage key for features."""
        return f"features:{symbol}:{feature_group}:{version}"

    def _make_metadata_key(self, feature_key: str) -> str:
        """Generate storage key for feature metadata."""
        return f"{feature_key}:metadata"

    def store_features(
        self,
        symbol: str,
        feature_group: str,
        features: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
        version: str = "v1"
    ) -> bool:
        """
        Store computed features with metadata.

        Args:
            symbol: Asset symbol (e.g., "AAPL")
            feature_group: Feature group name (e.g., "technical_indicators")
            features: DataFrame containing computed features
            metadata: Optional metadata about feature computation
            version: Feature version string

        Returns:
            True if stored successfully, False otherwise
        """
        feature_key = self._make_key(symbol, feature_group, version)

        feature_metadata = {
            "symbol": symbol,
            "feature_group": feature_group,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "shape": features.shape,
            "columns": list(features.columns),
            "metadata": metadata or {}
        }

        try:
            features_json = features.to_json(orient='split', date_format='iso')

            if self.use_redis:
                self.redis_client.setex(feature_key, self.ttl_seconds, features_json)
                self.redis_client.setex(
                    self._make_metadata_key(feature_key),
                    self.ttl_seconds,
                    json.dumps(feature_metadata)
                )
            else:
                # In-memory fallback
                self.cache[feature_key] = {
                    "features": features_json,
                    "metadata": feature_metadata,
                    "expires_at": datetime.now() + timedelta(seconds=self.ttl_seconds)
                }

            return True

        except Exception:
            return False

    def get_features(
        self,
        symbol: str,
        feature_group: str,
        version: str = "v1"
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve stored features.

        Args:
            symbol: Asset symbol
            feature_group: Feature group name
            version: Feature version string

        Returns:
            DataFrame with features, or None if not found or expired
        """
        feature_key = self._make_key(symbol, feature_group, version)

        try:
            if self.use_redis:
                features_json = self.redis_client.get(feature_key)
                if features_json:
                    return pd.read_json(StringIO(features_json.decode()), orient='split')
            else:
                if feature_key in self.cache:
                    cached = self.cache[feature_key]
                    # Check expiration
                    if datetime.now() < cached["expires_at"]:
                        return pd.read_json(StringIO(cached["features"]), orient='split')
                    else:
                        del self.cache[feature_key]

            return None

        except Exception:
            return None

    def get_metadata(self, symbol: str, feature_group: str, version: str = "v1") -> Optional[Dict]:
        """
        Retrieve feature metadata.

        Args:
            symbol: Asset symbol
            feature_group: Feature group name
            version: Feature version string

        Returns:
            Metadata dictionary, or None if not found
        """
        feature_key = self._make_key(symbol, feature_group, version)
        metadata_key = self._make_metadata_key(feature_key)

        try:
            if self.use_redis:
                metadata_json = self.redis_client.get(metadata_key)
                if metadata_json:
                    return json.loads(metadata_json)
            else:
                if feature_key in self.cache:
                    return self.cache[feature_key]["metadata"]

            return None

        except Exception:
            return None

    def list_feature_groups(self, symbol: str) -> List[str]:
        """
        List all available feature groups for a symbol.

        Args:
            symbol: Asset symbol

        Returns:
            List of feature group names
        """
        if self.use_redis:
            pattern = f"features:{symbol}:*"
            keys = self.redis_client.keys(pattern)
            groups = set()
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                if ":metadata" not in key_str:
                    parts = key_str.split(":")
                    if len(parts) >= 3:
                        groups.add(parts[2])
            return list(groups)
        else:
            groups = set()
            for key in self.cache.keys():
                if key.startswith(f"features:{symbol}:"):
                    parts = key.split(":")
                    if len(parts) >= 3:
                        groups.add(parts[2])
            return list(groups)

    def invalidate(self, symbol: str, feature_group: Optional[str] = None, version: str = "v1"):
        """
        Invalidate (delete) cached features.

        Args:
            symbol: Asset symbol
            feature_group: Specific group to invalidate, or None for all groups
            version: Feature version string
        """
        if feature_group:
            feature_key = self._make_key(symbol, feature_group, version)
            if self.use_redis:
                self.redis_client.delete(feature_key)
                self.redis_client.delete(self._make_metadata_key(feature_key))
            else:
                self.cache.pop(feature_key, None)
        else:
            # Invalidate all features for symbol
            if self.use_redis:
                pattern = f"features:{symbol}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            else:
                keys_to_delete = [k for k in self.cache.keys() if k.startswith(f"features:{symbol}:")]
                for key in keys_to_delete:
                    del self.cache[key]


def cache_features(symbol: str, feature_group: str, compute_fn, **kwargs):
    """
    Convenience function for caching features with a compute function.

    Args:
        symbol: Asset symbol
        feature_group: Feature group name
        compute_fn: Function that computes features
        **kwargs: Arguments to pass to compute function

    Returns:
        DataFrame with features (cached or freshly computed)

    Example:
        def compute_indicators(symbol):
            # ... compute features
            return features_df

        features = cache_features("AAPL", "indicators", compute_indicators)
    """
    store = FeatureStore()
    features = store.get_features(symbol, feature_group)

    if features is not None:
        return features

    # Compute if not cached
    features = compute_fn(symbol, **kwargs)
    store.store_features(symbol, feature_group, features)

    return features


if __name__ == "__main__":
    # Simple test
    store = FeatureStore()

    # Create dummy features
    features = pd.DataFrame({
        'sma_20': np.random.rand(100),
        'rsi': np.random.rand(100),
        'macd': np.random.rand(100)
    })

    # Store features
    store.store_features(
        symbol="AAPL",
        feature_group="technical_indicators",
        features=features,
        metadata={"source": "yfinance", "method": "ta-lib"}
    )

    # Retrieve features
    retrieved = store.get_features("AAPL", "technical_indicators")
    print(f"Retrieved shape: {retrieved.shape if retrieved is not None else 'None'}")

    # Get metadata
    metadata = store.get_metadata("AAPL", "technical_indicators")
    print(f"Metadata: {json.dumps(metadata, indent=2) if metadata else 'None'}")

    # List feature groups
    groups = store.list_feature_groups("AAPL")
    print(f"Feature groups: {groups}")