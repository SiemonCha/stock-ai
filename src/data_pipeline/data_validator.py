"""
Comprehensive Data Validation and Quality Control System
Professional-grade data validation for stock market data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

try:
    import great_expectations as ge
    from great_expectations.dataset import PandasDataset
    GREAT_EXPECTATIONS_AVAILABLE = True
except ImportError:
    GREAT_EXPECTATIONS_AVAILABLE = False

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    PANDERA_AVAILABLE = True
except ImportError:
    PANDERA_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation issue severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DataQualityDimension(Enum):
    """Data quality dimensions"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"

@dataclass
class ValidationIssue:
    """Data validation issue"""
    dimension: DataQualityDimension
    severity: ValidationSeverity
    description: str
    column: Optional[str] = None
    row_indices: Optional[List[int]] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    symbol: str
    data_type: str
    timestamp: datetime
    total_rows: int
    total_columns: int
    quality_score: float
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)

class DataValidator:
    """
    Comprehensive data validation and quality control system
    """
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        
        # Quality thresholds
        self.quality_thresholds = {
            'completeness_threshold': 0.95,
            'accuracy_threshold': 0.90,
            'consistency_threshold': 0.95,
            'outlier_threshold': 0.05,
            'duplicate_threshold': 0.01
        }
        
        logger.info("DataValidator initialized with comprehensive validation rules")
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize comprehensive validation rules for different data types"""
        
        rules = {
            'prices': {
                'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'optional_columns': ['Adj Close', 'Dividends', 'Stock Splits'],
                'numeric_columns': ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'],
                'positive_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'price_relationships': [
                    ('High', 'Low', 'greater_equal'),
                    ('High', 'Open', 'greater_equal'),
                    ('High', 'Close', 'greater_equal'),
                    ('Low', 'Open', 'less_equal'),
                    ('Low', 'Close', 'less_equal')
                ],
                'volume_checks': {
                    'min_volume': 0,
                    'max_daily_change': 50.0  # 50x volume spike threshold
                },
                'price_checks': {
                    'max_daily_change': 0.50,  # 50% daily price change threshold
                    'min_price': 0.01
                }
            },
            'fundamentals': {
                'required_columns': ['symbol'],
                'numeric_columns': ['marketCap', 'enterpriseValue', 'trailingPE', 'forwardPE'],
                'positive_columns': ['marketCap', 'enterpriseValue', 'totalRevenue', 'totalCash'],
                'ratio_checks': {
                    'pe_ratio_max': 1000,
                    'debt_to_equity_max': 10.0,
                    'current_ratio_min': 0.1
                }
            },
            'economic': {
                'required_columns': ['Value'],
                'numeric_columns': ['Value'],
                'time_series_checks': {
                    'max_gap_days': 90,
                    'min_observations': 10
                }
            }
        }
        
        return rules
    
    def validate_data(self, data: pd.DataFrame, symbol: str, data_type: str = 'prices') -> ValidationReport:
        """
        Perform comprehensive data validation
        """
        
        logger.info(f"Validating {data_type} data for {symbol}: {len(data)} rows")
        
        # Initialize report
        report = ValidationReport(
            symbol=symbol,
            data_type=data_type,
            timestamp=datetime.now(),
            total_rows=len(data),
            total_columns=len(data.columns)
        )
        
        if data.empty:
            report.issues.append(ValidationIssue(
                dimension=DataQualityDimension.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                description="Dataset is empty"
            ))
            report.quality_score = 0.0
            return report
        
        # Run validation checks
        self._validate_completeness(data, report, data_type)
        self._validate_accuracy(data, report, data_type)
        self._validate_consistency(data, report, data_type)
        self._validate_timeliness(data, report, data_type)
        self._validate_validity(data, report, data_type)
        self._validate_uniqueness(data, report, data_type)
        
        # Detect anomalies
        self._detect_anomalies(data, report, data_type)
        
        # Calculate overall quality score
        report.quality_score = self._calculate_quality_score(report)
        
        # Add summary metrics
        self._add_summary_metrics(data, report)
        
        logger.info(f"Validation complete. Quality score: {report.quality_score:.3f}")
        
        return report
    
    def _validate_completeness(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Validate data completeness"""
        
        rules = self.validation_rules.get(data_type, {})
        required_columns = rules.get('required_columns', [])
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            report.issues.append(ValidationIssue(
                dimension=DataQualityDimension.COMPLETENESS,
                severity=ValidationSeverity.CRITICAL,
                description=f"Missing required columns: {missing_columns}"
            ))
            report.failed_checks.append("required_columns")
        else:
            report.passed_checks.append("required_columns")
        
        # Check missing values
        for column in data.columns:
            missing_count = data[column].isna().sum()
            missing_pct = missing_count / len(data)
            
            if missing_pct > (1 - self.quality_thresholds['completeness_threshold']):
                severity = ValidationSeverity.CRITICAL if missing_pct > 0.5 else ValidationSeverity.ERROR
                report.issues.append(ValidationIssue(
                    dimension=DataQualityDimension.COMPLETENESS,
                    severity=severity,
                    description=f"High missing values in {column}: {missing_pct:.1%}",
                    column=column,
                    metadata={'missing_count': missing_count, 'missing_pct': missing_pct}
                ))
                report.failed_checks.append(f"completeness_{column}")
            else:
                report.passed_checks.append(f"completeness_{column}")
        
        # Check data density (consecutive missing values)
        if isinstance(data.index, pd.DatetimeIndex):
            for column in data.select_dtypes(include=[np.number]).columns:
                consecutive_missing = self._find_consecutive_missing(data[column])
                if consecutive_missing > 5:  # More than 5 consecutive missing values
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.COMPLETENESS,
                        severity=ValidationSeverity.WARNING,
                        description=f"Consecutive missing values in {column}: {consecutive_missing} days",
                        column=column,
                        metadata={'consecutive_missing': consecutive_missing}
                    ))
    
    def _validate_accuracy(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Validate data accuracy"""
        
        rules = self.validation_rules.get(data_type, {})
        
        if data_type == 'prices':
            # Validate price relationships
            price_relationships = rules.get('price_relationships', [])
            for col1, col2, relationship in price_relationships:
                if col1 in data.columns and col2 in data.columns:
                    violations = self._check_price_relationship(data, col1, col2, relationship)
                    if violations:
                        report.issues.append(ValidationIssue(
                            dimension=DataQualityDimension.ACCURACY,
                            severity=ValidationSeverity.ERROR,
                            description=f"Price relationship violation: {col1} should be {relationship} {col2}",
                            row_indices=violations,
                            metadata={'violation_count': len(violations)}
                        ))
                        report.failed_checks.append(f"price_relationship_{col1}_{col2}")
                    else:
                        report.passed_checks.append(f"price_relationship_{col1}_{col2}")
            
            # Validate volume consistency
            if 'Volume' in data.columns:
                volume_spikes = self._detect_volume_spikes(data['Volume'])
                if volume_spikes:
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.ACCURACY,
                        severity=ValidationSeverity.WARNING,
                        description=f"Unusual volume spikes detected: {len(volume_spikes)} instances",
                        column='Volume',
                        row_indices=volume_spikes,
                        metadata={'spike_count': len(volume_spikes)}
                    ))
            
            # Validate price movement reasonableness
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in data.columns:
                    extreme_changes = self._detect_extreme_price_changes(data[col])
                    if extreme_changes:
                        report.issues.append(ValidationIssue(
                            dimension=DataQualityDimension.ACCURACY,
                            severity=ValidationSeverity.WARNING,
                            description=f"Extreme price changes in {col}: {len(extreme_changes)} instances",
                            column=col,
                            row_indices=extreme_changes
                        ))
        
        # Generic numeric range validation
        numeric_columns = rules.get('numeric_columns', [])
        positive_columns = rules.get('positive_columns', [])
        
        for col in numeric_columns:
            if col in data.columns:
                # Check for infinite values
                inf_count = np.isinf(data[col]).sum()
                if inf_count > 0:
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.ACCURACY,
                        severity=ValidationSeverity.ERROR,
                        description=f"Infinite values in {col}: {inf_count} instances",
                        column=col,
                        metadata={'inf_count': inf_count}
                    ))
                
                # Check positive constraints
                if col in positive_columns:
                    negative_count = (data[col] < 0).sum()
                    if negative_count > 0:
                        negative_indices = data[data[col] < 0].index.tolist()
                        report.issues.append(ValidationIssue(
                            dimension=DataQualityDimension.ACCURACY,
                            severity=ValidationSeverity.ERROR,
                            description=f"Negative values in {col}: {negative_count} instances",
                            column=col,
                            row_indices=negative_indices,
                            metadata={'negative_count': negative_count}
                        ))
    
    def _validate_consistency(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Validate data consistency"""
        
        # Check data type consistency
        for column in data.columns:
            if column in data.select_dtypes(include=[np.number]).columns:
                # Check for mixed data types (should all be numeric)
                non_numeric = pd.to_numeric(data[column], errors='coerce').isna() & data[column].notna()
                if non_numeric.any():
                    non_numeric_indices = data[non_numeric].index.tolist()
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.CONSISTENCY,
                        severity=ValidationSeverity.ERROR,
                        description=f"Mixed data types in numeric column {column}",
                        column=column,
                        row_indices=non_numeric_indices
                    ))
        
        # Check index consistency (for time series)
        if isinstance(data.index, pd.DatetimeIndex):
            # Check for duplicate timestamps
            duplicate_dates = data.index.duplicated()
            if duplicate_dates.any():
                duplicate_indices = data[duplicate_dates].index.tolist()
                report.issues.append(ValidationIssue(
                    dimension=DataQualityDimension.CONSISTENCY,
                    severity=ValidationSeverity.ERROR,
                    description=f"Duplicate timestamps: {duplicate_dates.sum()} instances",
                    row_indices=list(range(len(duplicate_indices))),
                    metadata={'duplicate_count': len(duplicate_indices)}
                ))
            
            # Check chronological order
            if not data.index.is_monotonic_increasing:
                report.issues.append(ValidationIssue(
                    dimension=DataQualityDimension.CONSISTENCY,
                    severity=ValidationSeverity.WARNING,
                    description="Index is not in chronological order"
                ))
        
        # Cross-column consistency checks for prices
        if data_type == 'prices':
            # Check that Adj Close is reasonably related to Close
            if 'Close' in data.columns and 'Adj Close' in data.columns:
                adj_ratio = data['Adj Close'] / data['Close']
                extreme_adj = (adj_ratio < 0.1) | (adj_ratio > 10)  # 10x adjustment seems extreme
                if extreme_adj.any():
                    extreme_indices = data[extreme_adj].index.tolist()
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.CONSISTENCY,
                        severity=ValidationSeverity.WARNING,
                        description=f"Extreme price adjustments: {extreme_adj.sum()} instances",
                        row_indices=list(range(len(extreme_indices))),
                        metadata={'extreme_adj_count': extreme_adj.sum()}
                    ))
    
    def _validate_timeliness(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Validate data timeliness"""
        
        if not isinstance(data.index, pd.DatetimeIndex):
            return
        
        # Check data recency
        latest_date = data.index.max()
        days_old = (datetime.now() - latest_date).days
        
        if days_old > 7:  # More than a week old
            severity = ValidationSeverity.WARNING if days_old < 30 else ValidationSeverity.ERROR
            report.issues.append(ValidationIssue(
                dimension=DataQualityDimension.TIMELINESS,
                severity=severity,
                description=f"Data is {days_old} days old (latest: {latest_date.date()})",
                metadata={'days_old': days_old, 'latest_date': latest_date}
            ))
        
        # Check for large gaps in time series
        time_diffs = data.index.to_series().diff()
        median_gap = time_diffs.median()
        large_gaps = time_diffs > median_gap * 5  # More than 5x the median gap
        
        if large_gaps.any():
            gap_indices = data[large_gaps].index.tolist()
            report.issues.append(ValidationIssue(
                dimension=DataQualityDimension.TIMELINESS,
                severity=ValidationSeverity.WARNING,
                description=f"Large time gaps detected: {large_gaps.sum()} instances",
                row_indices=list(range(len(gap_indices))),
                metadata={'large_gap_count': large_gaps.sum()}
            ))
    
    def _validate_validity(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Validate data validity (business rules)"""
        
        rules = self.validation_rules.get(data_type, {})
        
        if data_type == 'prices':
            price_checks = rules.get('price_checks', {})
            
            # Check minimum price
            min_price = price_checks.get('min_price', 0.01)
            price_columns = ['Open', 'High', 'Low', 'Close']
            
            for col in price_columns:
                if col in data.columns:
                    below_min = data[col] < min_price
                    if below_min.any():
                        below_min_indices = data[below_min].index.tolist()
                        report.issues.append(ValidationIssue(
                            dimension=DataQualityDimension.VALIDITY,
                            severity=ValidationSeverity.WARNING,
                            description=f"Prices below minimum threshold in {col}: {below_min.sum()} instances",
                            column=col,
                            row_indices=list(range(len(below_min_indices))),
                            expected=min_price
                        ))
            
            # Check volume validity
            if 'Volume' in data.columns:
                zero_volume = data['Volume'] == 0
                if zero_volume.any():
                    zero_volume_indices = data[zero_volume].index.tolist()
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.VALIDITY,
                        severity=ValidationSeverity.INFO,
                        description=f"Zero volume days: {zero_volume.sum()} instances",
                        column='Volume',
                        row_indices=list(range(len(zero_volume_indices)))
                    ))
        
        elif data_type == 'fundamentals':
            ratio_checks = rules.get('ratio_checks', {})
            
            # Check PE ratio validity
            if 'trailingPE' in data.columns:
                pe_max = ratio_checks.get('pe_ratio_max', 1000)
                extreme_pe = (data['trailingPE'] < 0) | (data['trailingPE'] > pe_max)
                if extreme_pe.any():
                    report.issues.append(ValidationIssue(
                        dimension=DataQualityDimension.VALIDITY,
                        severity=ValidationSeverity.WARNING,
                        description=f"Extreme P/E ratios: {extreme_pe.sum()} instances",
                        column='trailingPE',
                        expected=f"0 < PE < {pe_max}"
                    ))
    
    def _validate_uniqueness(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Validate data uniqueness"""
        
        # Check for duplicate rows
        duplicate_rows = data.duplicated()
        if duplicate_rows.any():
            duplicate_indices = data[duplicate_rows].index.tolist()
            report.issues.append(ValidationIssue(
                dimension=DataQualityDimension.UNIQUENESS,
                severity=ValidationSeverity.WARNING,
                description=f"Duplicate rows: {duplicate_rows.sum()} instances",
                row_indices=list(range(len(duplicate_indices))),
                metadata={'duplicate_count': duplicate_rows.sum()}
            ))
        
        # Check for duplicate index values (already covered in consistency)
        # Additional uniqueness checks can be added here
    
    def _detect_anomalies(self, data: pd.DataFrame, report: ValidationReport, data_type: str):
        """Detect statistical anomalies in the data"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return
        
        # Prepare data for anomaly detection
        numeric_data_filled = numeric_data.fillna(numeric_data.median())
        
        if len(numeric_data_filled) < 10:  # Need enough data for anomaly detection
            return
        
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(numeric_data_filled)
            
            # Detect anomalies using Isolation Forest
            anomaly_labels = self.anomaly_detector.fit_predict(scaled_data)
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            if len(anomaly_indices) > 0:
                # Convert to original index
                original_anomaly_indices = numeric_data_filled.iloc[anomaly_indices].index.tolist()
                
                report.issues.append(ValidationIssue(
                    dimension=DataQualityDimension.ACCURACY,
                    severity=ValidationSeverity.INFO,
                    description=f"Statistical anomalies detected: {len(anomaly_indices)} instances",
                    row_indices=list(range(len(original_anomaly_indices))),
                    metadata={'anomaly_count': len(anomaly_indices)}
                ))
                
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
    
    def _check_price_relationship(self, data: pd.DataFrame, col1: str, col2: str, relationship: str) -> List[int]:
        """Check price relationships (e.g., High >= Low)"""
        
        if relationship == 'greater_equal':
            violations = data[data[col1] < data[col2]]
        elif relationship == 'less_equal':
            violations = data[data[col1] > data[col2]]
        elif relationship == 'equal':
            violations = data[data[col1] != data[col2]]
        else:
            return []
        
        return violations.index.tolist()
    
    def _detect_volume_spikes(self, volume_series: pd.Series, threshold_multiplier: float = 10) -> List[int]:
        """Detect unusual volume spikes"""
        
        # Calculate rolling median volume
        rolling_median = volume_series.rolling(window=20, min_periods=5).median()
        
        # Find spikes
        spikes = volume_series > (rolling_median * threshold_multiplier)
        
        return volume_series[spikes].index.tolist()
    
    def _detect_extreme_price_changes(self, price_series: pd.Series, threshold: float = 0.25) -> List[int]:
        """Detect extreme daily price changes"""
        
        # Calculate daily percentage change
        pct_change = price_series.pct_change().abs()
        
        # Find extreme changes
        extreme = pct_change > threshold
        
        return price_series[extreme].index.tolist()
    
    def _find_consecutive_missing(self, series: pd.Series) -> int:
        """Find maximum consecutive missing values"""
        
        is_missing = series.isna()
        
        if not is_missing.any():
            return 0
        
        # Find consecutive groups of missing values
        groups = (is_missing != is_missing.shift()).cumsum()
        consecutive_counts = is_missing.groupby(groups).sum()
        
        return consecutive_counts.max() if len(consecutive_counts) > 0 else 0
    
    def _calculate_quality_score(self, report: ValidationReport) -> float:
        """Calculate overall data quality score"""
        
        if not report.issues:
            return 1.0
        
        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.02,
            ValidationSeverity.WARNING: 0.05,
            ValidationSeverity.ERROR: 0.15,
            ValidationSeverity.CRITICAL: 0.50
        }
        
        total_penalty = 0.0
        for issue in report.issues:
            total_penalty += severity_weights.get(issue.severity, 0.1)
        
        # Calculate score (max penalty of 1.0)
        quality_score = max(0.0, 1.0 - min(1.0, total_penalty))
        
        return quality_score
    
    def _add_summary_metrics(self, data: pd.DataFrame, report: ValidationReport):
        """Add summary metrics to the validation report"""
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        report.metrics = {
            'data_shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
            'numeric_columns': len(numeric_data.columns),
            'categorical_columns': len(data.select_dtypes(include=['object']).columns),
            'missing_values_total': data.isna().sum().sum(),
            'missing_percentage': (data.isna().sum().sum() / (data.shape[0] * data.shape[1])) * 100,
            'duplicate_rows': data.duplicated().sum(),
            'issue_count_by_severity': {
                severity.value: sum(1 for issue in report.issues if issue.severity == severity)
                for severity in ValidationSeverity
            }
        }
        
        # Add statistical summary for numeric columns
        if not numeric_data.empty:
            report.metrics['numeric_summary'] = {
                'mean': numeric_data.mean().to_dict(),
                'std': numeric_data.std().to_dict(),
                'min': numeric_data.min().to_dict(),
                'max': numeric_data.max().to_dict(),
                'skewness': numeric_data.skew().to_dict(),
                'kurtosis': numeric_data.kurtosis().to_dict()
            }
    
    def generate_validation_summary(self, report: ValidationReport) -> str:
        """Generate human-readable validation summary"""
        
        summary_lines = [
            f"📊 DATA VALIDATION REPORT",
            f"=" * 50,
            f"Symbol: {report.symbol}",
            f"Data Type: {report.data_type}",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Data Shape: {report.total_rows} rows × {report.total_columns} columns",
            f"Quality Score: {report.quality_score:.3f}/1.000",
            f"",
            f"📈 SUMMARY STATISTICS",
            f"-" * 30
        ]
        
        # Add basic metrics
        if 'missing_percentage' in report.metrics:
            summary_lines.append(f"Missing Data: {report.metrics['missing_percentage']:.1f}%")
        
        if 'duplicate_rows' in report.metrics:
            summary_lines.append(f"Duplicate Rows: {report.metrics['duplicate_rows']}")
        
        # Add issue summary
        summary_lines.extend([
            f"",
            f"🚨 ISSUES FOUND",
            f"-" * 20
        ])
        
        issue_counts = report.metrics.get('issue_count_by_severity', {})
        for severity in ValidationSeverity:
            count = issue_counts.get(severity.value, 0)
            if count > 0:
                emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🔴"}
                summary_lines.append(f"{emoji.get(severity.value, '•')} {severity.value}: {count}")
        
        # Add top issues
        if report.issues:
            summary_lines.extend([
                f"",
                f"🔍 TOP ISSUES",
                f"-" * 15
            ])
            
            # Sort issues by severity
            sorted_issues = sorted(report.issues, 
                                 key=lambda x: ['INFO', 'WARNING', 'ERROR', 'CRITICAL'].index(x.severity.value))
            
            for i, issue in enumerate(sorted_issues[:5]):  # Top 5 issues
                summary_lines.append(f"{i+1}. {issue.severity.value}: {issue.description}")
        
        # Add recommendations
        summary_lines.extend([
            f"",
            f"💡 RECOMMENDATIONS",
            f"-" * 20
        ])
        
        if report.quality_score < 0.7:
            summary_lines.append("• Data quality is below acceptable threshold - review critical issues")
        elif report.quality_score < 0.9:
            summary_lines.append("• Data quality is acceptable but could be improved")
        else:
            summary_lines.append("• Data quality is excellent")
        
        if any(issue.severity == ValidationSeverity.CRITICAL for issue in report.issues):
            summary_lines.append("• Address critical issues immediately before using data")
        
        return "\n".join(summary_lines)

class DataQualityMonitor:
    """
    Monitor data quality over time and detect degradation
    """
    
    def __init__(self):
        self.quality_history = {}
        self.thresholds = {
            'quality_degradation_threshold': 0.1,  # 10% drop in quality
            'consecutive_failures_threshold': 3
        }
    
    def track_quality(self, symbol: str, data_type: str, quality_score: float, issues: List[ValidationIssue]):
        """Track data quality metrics over time"""
        
        key = f"{symbol}_{data_type}"
        
        if key not in self.quality_history:
            self.quality_history[key] = []
        
        quality_record = {
            'timestamp': datetime.now(),
            'quality_score': quality_score,
            'issue_count': len(issues),
            'critical_issues': sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL),
            'error_issues': sum(1 for issue in issues if issue.severity == ValidationSeverity.ERROR)
        }
        
        self.quality_history[key].append(quality_record)
        
        # Keep only last 100 records
        if len(self.quality_history[key]) > 100:
            self.quality_history[key] = self.quality_history[key][-100:]
    
    def detect_quality_degradation(self, symbol: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Detect if data quality has degraded significantly"""
        
        key = f"{symbol}_{data_type}"
        
        if key not in self.quality_history or len(self.quality_history[key]) < 5:
            return None
        
        history = self.quality_history[key]
        recent_scores = [record['quality_score'] for record in history[-5:]]
        older_scores = [record['quality_score'] for record in history[-10:-5]] if len(history) >= 10 else []
        
        if not older_scores:
            return None
        
        recent_avg = np.mean(recent_scores)
        older_avg = np.mean(older_scores)
        
        # Check for significant degradation
        if older_avg - recent_avg > self.thresholds['quality_degradation_threshold']:
            return {
                'type': 'quality_degradation',
                'recent_avg_quality': recent_avg,
                'previous_avg_quality': older_avg,
                'degradation': older_avg - recent_avg,
                'timestamp': datetime.now()
            }
        
        # Check for consecutive failures
        recent_critical_count = sum(record['critical_issues'] for record in history[-3:])
        if recent_critical_count >= self.thresholds['consecutive_failures_threshold']:
            return {
                'type': 'consecutive_failures',
                'critical_issues_count': recent_critical_count,
                'timestamp': datetime.now()
            }
        
        return None

# Example usage and testing
if __name__ == "__main__":
    print("🔍 Comprehensive Data Validation System")
    print("=" * 50)
    
    # Create test data with various quality issues
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
    
    # Introduce some quality issues for testing
    test_data.loc[test_data.index[10], 'High'] = test_data.loc[test_data.index[10], 'Low'] - 1  # Price relationship violation
    test_data.loc[test_data.index[20:25], 'Volume'] = np.nan  # Missing values
    test_data.loc[test_data.index[30], 'Volume'] = test_data.loc[test_data.index[30], 'Volume'] * 50  # Volume spike
    test_data.loc[test_data.index[40], 'Close'] = -5  # Negative price
    
    # Test validation
    validator = DataValidator()
    
    print("📊 Testing data validation...")
    report = validator.validate_data(test_data, 'AAPL', 'prices')
    
    # Print summary
    summary = validator.generate_validation_summary(report)
    print(summary)
    
    # Test quality monitoring
    print(f"\n📈 Testing quality monitoring...")
    monitor = DataQualityMonitor()
    
    # Track quality over time
    for i in range(5):
        # Simulate degrading quality
        quality_score = max(0.5, 0.9 - i * 0.1)
        issues = [ValidationIssue(
            dimension=DataQualityDimension.ACCURACY,
            severity=ValidationSeverity.ERROR if i > 2 else ValidationSeverity.WARNING,
            description=f"Test issue {i}"
        )]
        
        monitor.track_quality('AAPL', 'prices', quality_score, issues)
    
    # Check for degradation
    degradation = monitor.detect_quality_degradation('AAPL', 'prices')
    if degradation:
        print(f"🚨 Quality degradation detected: {degradation}")
    else:
        print("✅ No quality degradation detected")
    
    print(f"\n🎯 Data validation system ready!")
    print(f"📋 Features:")
    print(f"   • 6 quality dimensions")
    print(f"   • Statistical anomaly detection")
    print(f"   • Business rule validation")
    print(f"   • Quality score calculation")
    print(f"   • Trend monitoring")