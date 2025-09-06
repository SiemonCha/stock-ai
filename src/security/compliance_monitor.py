"""
Compliance Monitoring System for Financial Services

This module provides comprehensive compliance monitoring for financial regulations
including FINRA, SEC, GDPR, SOC2, and other applicable standards for stock trading systems.
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import redis


class ComplianceViolationType(Enum):
    TRADING_LIMIT_EXCEEDED = "trading_limit_exceeded"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    DATA_RETENTION_VIOLATION = "data_retention_violation"
    ACCESS_CONTROL_BREACH = "access_control_breach"
    AUDIT_LOG_MISSING = "audit_log_missing"
    ENCRYPTION_FAILURE = "encryption_failure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    REGULATORY_REPORTING_LATE = "regulatory_reporting_late"
    MARKET_MANIPULATION = "market_manipulation"
    INSIDER_TRADING_RISK = "insider_trading_risk"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ComplianceViolation:
    violation_type: ComplianceViolationType
    risk_level: RiskLevel
    timestamp: datetime
    user_id: Optional[str]
    resource: str
    description: str
    evidence: Dict[str, Any]
    regulatory_framework: List[str]
    remediation_required: bool = True
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class ComplianceRule:
    rule_id: str
    name: str
    description: str
    regulatory_framework: str
    violation_type: ComplianceViolationType
    risk_level: RiskLevel
    conditions: Dict[str, Any]
    threshold_values: Dict[str, float]
    enabled: bool = True
    auto_remediation: bool = False


class FINRAMonitor:
    """
    FINRA compliance monitoring for trading activities
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # FINRA trading limits and thresholds
        self.daily_trading_limit = 10000000  # $10M per day
        self.position_concentration_limit = 0.05  # 5% of portfolio
        self.max_order_size = 1000000  # $1M per order
        
        # Pattern detection parameters
        self.wash_trade_threshold = 0.02  # 2% price difference
        self.layering_threshold = 5  # Number of orders
        self.spoofing_cancel_ratio = 0.8  # 80% cancellation rate
    
    async def monitor_trading_activity(self) -> List[ComplianceViolation]:
        """Monitor trading activity for FINRA violations"""
        
        violations = []
        
        # Get recent trading data
        trading_data = await self._get_trading_data()
        
        if not trading_data:
            return violations
        
        # Check daily trading limits
        daily_volume = sum(trade['amount'] for trade in trading_data)
        if daily_volume > self.daily_trading_limit:
            violations.append(ComplianceViolation(
                violation_type=ComplianceViolationType.TRADING_LIMIT_EXCEEDED,
                risk_level=RiskLevel.HIGH,
                timestamp=datetime.now(),
                user_id=None,
                resource="trading_system",
                description=f"Daily trading volume ${daily_volume:,.2f} exceeds limit ${self.daily_trading_limit:,.2f}",
                evidence={"daily_volume": daily_volume, "limit": self.daily_trading_limit},
                regulatory_framework=["FINRA"]
            ))
        
        # Check for wash trading patterns
        wash_violations = await self._detect_wash_trading(trading_data)
        violations.extend(wash_violations)
        
        # Check for layering/spoofing
        layering_violations = await self._detect_layering(trading_data)
        violations.extend(layering_violations)
        
        # Check position concentration
        concentration_violations = await self._check_position_concentration()
        violations.extend(concentration_violations)
        
        return violations
    
    async def _get_trading_data(self) -> List[Dict[str, Any]]:
        """Get recent trading data from Redis"""
        try:
            # Get trades from last 24 hours
            trades_json = self.redis_client.lrange('recent_trades', 0, -1)
            trades = [json.loads(trade) for trade in trades_json]
            
            # Filter for today's trades
            today = datetime.now().date()
            today_trades = [
                trade for trade in trades 
                if datetime.fromisoformat(trade['timestamp']).date() == today
            ]
            
            return today_trades
            
        except Exception as e:
            self.logger.error(f"Error getting trading data: {e}")
            return []
    
    async def _detect_wash_trading(self, trading_data: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Detect potential wash trading patterns"""
        
        violations = []
        
        # Group trades by symbol and user
        user_trades = {}
        for trade in trading_data:
            user_id = trade.get('user_id')
            symbol = trade.get('symbol')
            key = f"{user_id}_{symbol}"
            
            if key not in user_trades:
                user_trades[key] = []
            user_trades[key].append(trade)
        
        # Look for buy/sell patterns within short time windows
        for key, trades in user_trades.items():
            user_id, symbol = key.split('_')
            
            # Sort by timestamp
            trades.sort(key=lambda x: x['timestamp'])
            
            for i in range(len(trades) - 1):
                current_trade = trades[i]
                next_trade = trades[i + 1]
                
                # Check if trades are opposite directions within 5 minutes
                time_diff = (
                    datetime.fromisoformat(next_trade['timestamp']) - 
                    datetime.fromisoformat(current_trade['timestamp'])
                ).total_seconds()
                
                if (time_diff < 300 and  # Within 5 minutes
                    current_trade['side'] != next_trade['side'] and  # Opposite directions
                    abs(current_trade['price'] - next_trade['price']) / current_trade['price'] < self.wash_trade_threshold):
                    
                    violations.append(ComplianceViolation(
                        violation_type=ComplianceViolationType.SUSPICIOUS_PATTERN,
                        risk_level=RiskLevel.HIGH,
                        timestamp=datetime.now(),
                        user_id=user_id,
                        resource=f"trading_{symbol}",
                        description=f"Potential wash trading detected in {symbol}",
                        evidence={
                            "symbol": symbol,
                            "trade_1": current_trade,
                            "trade_2": next_trade,
                            "time_diff_seconds": time_diff
                        },
                        regulatory_framework=["FINRA", "SEC"]
                    ))
        
        return violations
    
    async def _detect_layering(self, trading_data: List[Dict[str, Any]]) -> List[ComplianceViolation]:
        """Detect layering/spoofing patterns"""
        
        violations = []
        
        # Get order data (including cancellations)
        order_data = await self._get_order_data()
        
        if not order_data:
            return violations
        
        # Group orders by user and symbol
        user_orders = {}
        for order in order_data:
            user_id = order.get('user_id')
            symbol = order.get('symbol')
            key = f"{user_id}_{symbol}"
            
            if key not in user_orders:
                user_orders[key] = []
            user_orders[key].append(order)
        
        # Detect high cancellation rates (spoofing indicator)
        for key, orders in user_orders.items():
            user_id, symbol = key.split('_')
            
            if len(orders) < self.layering_threshold:
                continue
            
            cancelled_orders = [o for o in orders if o.get('status') == 'cancelled']
            cancellation_rate = len(cancelled_orders) / len(orders)
            
            if cancellation_rate > self.spoofing_cancel_ratio:
                violations.append(ComplianceViolation(
                    violation_type=ComplianceViolationType.MARKET_MANIPULATION,
                    risk_level=RiskLevel.CRITICAL,
                    timestamp=datetime.now(),
                    user_id=user_id,
                    resource=f"orders_{symbol}",
                    description=f"High order cancellation rate ({cancellation_rate:.1%}) indicates potential spoofing",
                    evidence={
                        "symbol": symbol,
                        "total_orders": len(orders),
                        "cancelled_orders": len(cancelled_orders),
                        "cancellation_rate": cancellation_rate
                    },
                    regulatory_framework=["FINRA", "SEC"]
                ))
        
        return violations
    
    async def _get_order_data(self) -> List[Dict[str, Any]]:
        """Get order data including cancellations"""
        try:
            orders_json = self.redis_client.lrange('recent_orders', 0, -1)
            return [json.loads(order) for order in orders_json]
        except Exception as e:
            self.logger.error(f"Error getting order data: {e}")
            return []
    
    async def _check_position_concentration(self) -> List[ComplianceViolation]:
        """Check for excessive position concentration"""
        
        violations = []
        
        try:
            # Get current positions
            positions_json = self.redis_client.get('current_positions')
            if not positions_json:
                return violations
            
            positions = json.loads(positions_json)
            total_portfolio_value = sum(pos['market_value'] for pos in positions.values())
            
            for symbol, position in positions.items():
                concentration = position['market_value'] / total_portfolio_value
                
                if concentration > self.position_concentration_limit:
                    violations.append(ComplianceViolation(
                        violation_type=ComplianceViolationType.TRADING_LIMIT_EXCEEDED,
                        risk_level=RiskLevel.MEDIUM,
                        timestamp=datetime.now(),
                        user_id=None,
                        resource=f"position_{symbol}",
                        description=f"Position concentration in {symbol} ({concentration:.1%}) exceeds limit ({self.position_concentration_limit:.1%})",
                        evidence={
                            "symbol": symbol,
                            "position_value": position['market_value'],
                            "total_portfolio": total_portfolio_value,
                            "concentration": concentration
                        },
                        regulatory_framework=["FINRA"]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error checking position concentration: {e}")
        
        return violations


class GDPRMonitor:
    """
    GDPR compliance monitoring for data protection
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # GDPR retention periods (in days)
        self.retention_periods = {
            'user_data': 2555,  # 7 years
            'trading_data': 2555,  # 7 years
            'audit_logs': 2555,  # 7 years
            'session_data': 365,  # 1 year
            'temporary_data': 30  # 30 days
        }
    
    async def monitor_data_retention(self) -> List[ComplianceViolation]:
        """Monitor data retention compliance"""
        
        violations = []
        
        # Check each data type for retention compliance
        for data_type, retention_days in self.retention_periods.items():
            expired_records = await self._find_expired_records(data_type, retention_days)
            
            if expired_records:
                violations.append(ComplianceViolation(
                    violation_type=ComplianceViolationType.DATA_RETENTION_VIOLATION,
                    risk_level=RiskLevel.HIGH,
                    timestamp=datetime.now(),
                    user_id=None,
                    resource=f"data_{data_type}",
                    description=f"Found {len(expired_records)} expired {data_type} records beyond retention period",
                    evidence={
                        "data_type": data_type,
                        "retention_days": retention_days,
                        "expired_count": len(expired_records),
                        "sample_records": expired_records[:5]  # First 5 for evidence
                    },
                    regulatory_framework=["GDPR"]
                ))
        
        return violations
    
    async def _find_expired_records(self, data_type: str, retention_days: int) -> List[Dict[str, Any]]:
        """Find records that exceed retention period"""
        
        expired_records = []
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        try:
            # This is a simplified check - in practice, you'd query your database
            keys = self.redis_client.keys(f"{data_type}:*")
            
            for key in keys:
                data_json = self.redis_client.get(key)
                if data_json:
                    data = json.loads(data_json)
                    created_at = datetime.fromisoformat(data.get('created_at', datetime.now().isoformat()))
                    
                    if created_at < cutoff_date:
                        expired_records.append({
                            'key': key.decode('utf-8'),
                            'created_at': created_at.isoformat(),
                            'age_days': (datetime.now() - created_at).days
                        })
            
        except Exception as e:
            self.logger.error(f"Error checking retention for {data_type}: {e}")
        
        return expired_records
    
    async def monitor_consent_compliance(self) -> List[ComplianceViolation]:
        """Monitor user consent compliance"""
        
        violations = []
        
        try:
            # Check for users without valid consent
            user_keys = self.redis_client.keys("user:*")
            
            for user_key in user_keys:
                user_data = self.redis_client.hgetall(user_key)
                
                if not user_data.get(b'gdpr_consent'):
                    user_id = user_key.decode('utf-8').replace('user:', '')
                    
                    violations.append(ComplianceViolation(
                        violation_type=ComplianceViolationType.DATA_RETENTION_VIOLATION,
                        risk_level=RiskLevel.HIGH,
                        timestamp=datetime.now(),
                        user_id=user_id,
                        resource=f"user_consent_{user_id}",
                        description=f"User {user_id} missing GDPR consent",
                        evidence={
                            "user_id": user_id,
                            "consent_status": "missing"
                        },
                        regulatory_framework=["GDPR"]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error checking consent compliance: {e}")
        
        return violations


class SOC2Monitor:
    """
    SOC2 compliance monitoring for security controls
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def monitor_access_controls(self) -> List[ComplianceViolation]:
        """Monitor access control compliance"""
        
        violations = []
        
        # Check for inactive sessions
        session_keys = self.redis_client.keys("session:*")
        current_time = datetime.now()
        
        for session_key in session_keys:
            session_data_json = self.redis_client.get(session_key)
            if session_data_json:
                session_data = json.loads(session_data_json)
                created_at = datetime.fromisoformat(session_data['created_at'])
                
                # Sessions older than 8 hours should be flagged
                if (current_time - created_at).total_seconds() > 28800:
                    violations.append(ComplianceViolation(
                        violation_type=ComplianceViolationType.ACCESS_CONTROL_BREACH,
                        risk_level=RiskLevel.MEDIUM,
                        timestamp=datetime.now(),
                        user_id=session_data.get('user_id'),
                        resource="session_management",
                        description=f"Long-running session detected (active for {(current_time - created_at).total_seconds() / 3600:.1f} hours)",
                        evidence={
                            "session_id": session_key.decode('utf-8'),
                            "created_at": created_at.isoformat(),
                            "duration_hours": (current_time - created_at).total_seconds() / 3600
                        },
                        regulatory_framework=["SOC2"]
                    ))
        
        return violations
    
    async def monitor_audit_logs(self) -> List[ComplianceViolation]:
        """Monitor audit log completeness"""
        
        violations = []
        
        try:
            # Check for gaps in audit logs
            security_events = self.redis_client.lrange('security_events', 0, -1)
            
            if len(security_events) == 0:
                violations.append(ComplianceViolation(
                    violation_type=ComplianceViolationType.AUDIT_LOG_MISSING,
                    risk_level=RiskLevel.HIGH,
                    timestamp=datetime.now(),
                    user_id=None,
                    resource="audit_logging",
                    description="No security events logged in recent period",
                    evidence={"event_count": 0},
                    regulatory_framework=["SOC2"]
                ))
            
            # Check for missing critical events
            events = [json.loads(event) for event in security_events]
            recent_events = [
                event for event in events 
                if datetime.fromisoformat(event['timestamp']) > datetime.now() - timedelta(hours=24)
            ]
            
            required_event_types = ['authentication', 'authorization', 'data_access']
            for event_type in required_event_types:
                if not any(event['event_type'] == event_type for event in recent_events):
                    violations.append(ComplianceViolation(
                        violation_type=ComplianceViolationType.AUDIT_LOG_MISSING,
                        risk_level=RiskLevel.MEDIUM,
                        timestamp=datetime.now(),
                        user_id=None,
                        resource="audit_logging",
                        description=f"Missing {event_type} events in recent audit logs",
                        evidence={
                            "missing_event_type": event_type,
                            "recent_events_count": len(recent_events)
                        },
                        regulatory_framework=["SOC2"]
                    ))
            
        except Exception as e:
            self.logger.error(f"Error monitoring audit logs: {e}")
        
        return violations


class ComplianceReporter:
    """
    Generate compliance reports for regulators
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
    
    async def generate_finra_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate FINRA compliance report"""
        
        report = {
            "report_type": "FINRA_COMPLIANCE",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "generated_at": datetime.now().isoformat(),
            "trading_summary": {},
            "violations": [],
            "risk_metrics": {}
        }
        
        try:
            # Get trading data for period
            trading_data = await self._get_trading_data_for_period(start_date, end_date)
            
            # Calculate trading summary
            report["trading_summary"] = {
                "total_trades": len(trading_data),
                "total_volume": sum(trade.get('amount', 0) for trade in trading_data),
                "unique_symbols": len(set(trade.get('symbol') for trade in trading_data)),
                "unique_users": len(set(trade.get('user_id') for trade in trading_data))
            }
            
            # Get violations for period
            violations_json = self.redis_client.lrange('compliance_violations', 0, -1)
            violations = [json.loads(v) for v in violations_json]
            
            period_violations = [
                v for v in violations 
                if start_date <= datetime.fromisoformat(v['timestamp']) <= end_date
                and 'FINRA' in v.get('regulatory_framework', [])
            ]
            
            report["violations"] = period_violations
            
            # Calculate risk metrics
            report["risk_metrics"] = {
                "total_violations": len(period_violations),
                "high_risk_violations": len([v for v in period_violations if v['risk_level'] == 'HIGH']),
                "critical_violations": len([v for v in period_violations if v['risk_level'] == 'CRITICAL']),
                "avg_daily_volume": report["trading_summary"]["total_volume"] / max(1, (end_date - start_date).days),
                "compliance_score": self._calculate_compliance_score(period_violations, trading_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating FINRA report: {e}")
            report["error"] = str(e)
        
        return report
    
    async def _get_trading_data_for_period(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get trading data for specific period"""
        
        trades_json = self.redis_client.lrange('all_trades', 0, -1)  # This would be a larger dataset
        trades = [json.loads(trade) for trade in trades_json]
        
        return [
            trade for trade in trades
            if start_date <= datetime.fromisoformat(trade['timestamp']) <= end_date
        ]
    
    def _calculate_compliance_score(self, violations: List[Dict[str, Any]], trading_data: List[Dict[str, Any]]) -> float:
        """Calculate overall compliance score (0.0 - 1.0)"""
        
        if not trading_data:
            return 1.0  # No trading, perfect compliance
        
        # Weight violations by severity
        violation_weights = {
            'LOW': 0.1,
            'MEDIUM': 0.3,
            'HIGH': 0.6,
            'CRITICAL': 1.0
        }
        
        total_violation_score = sum(
            violation_weights.get(v.get('risk_level', 'LOW'), 0.1)
            for v in violations
        )
        
        # Normalize by trading volume
        volume_factor = len(trading_data) / 1000  # Normalize per 1000 trades
        risk_score = total_violation_score / max(1, volume_factor)
        
        # Convert to compliance score (inverse of risk)
        compliance_score = max(0.0, 1.0 - min(1.0, risk_score))
        
        return compliance_score


class ComplianceMonitor:
    """
    Main compliance monitoring orchestrator
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize monitoring components
        self.finra_monitor = FINRAMonitor(self.redis_client)
        self.gdpr_monitor = GDPRMonitor(self.redis_client)
        self.soc2_monitor = SOC2Monitor(self.redis_client)
        self.reporter = ComplianceReporter(self.redis_client)
        
        # Violation storage
        self.violations: List[ComplianceViolation] = []
        
        # Notification settings
        self.notification_settings = {
            'email_alerts': True,
            'alert_email': 'compliance@company.com',
            'smtp_server': 'smtp.company.com',
            'critical_immediate': True
        }
        
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    async def start_monitoring(self, interval: int = 300):  # 5 minutes
        """Start continuous compliance monitoring"""
        
        self.running = True
        self.logger.info("Starting compliance monitoring...")
        
        while self.running:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in compliance monitoring cycle: {e}")
                await asyncio.sleep(60)  # Retry in 1 minute
    
    def stop_monitoring(self):
        """Stop compliance monitoring"""
        self.running = False
        self.logger.info("Compliance monitoring stopped")
    
    async def _monitoring_cycle(self):
        """Execute one complete monitoring cycle"""
        
        self.logger.info("Running compliance monitoring cycle...")
        
        # Collect violations from all monitors
        all_violations = []
        
        # FINRA monitoring
        try:
            finra_violations = await self.finra_monitor.monitor_trading_activity()
            all_violations.extend(finra_violations)
        except Exception as e:
            self.logger.error(f"Error in FINRA monitoring: {e}")
        
        # GDPR monitoring
        try:
            gdpr_violations = await self.gdpr_monitor.monitor_data_retention()
            gdpr_violations.extend(await self.gdpr_monitor.monitor_consent_compliance())
            all_violations.extend(gdpr_violations)
        except Exception as e:
            self.logger.error(f"Error in GDPR monitoring: {e}")
        
        # SOC2 monitoring
        try:
            soc2_violations = await self.soc2_monitor.monitor_access_controls()
            soc2_violations.extend(await self.soc2_monitor.monitor_audit_logs())
            all_violations.extend(soc2_violations)
        except Exception as e:
            self.logger.error(f"Error in SOC2 monitoring: {e}")
        
        # Process violations
        for violation in all_violations:
            await self._process_violation(violation)
        
        self.logger.info(f"Compliance cycle completed. Found {len(all_violations)} violations.")
    
    async def _process_violation(self, violation: ComplianceViolation):
        """Process a compliance violation"""
        
        # Store violation
        self.violations.append(violation)
        
        # Store in Redis for persistence
        violation_data = {
            'violation_type': violation.violation_type.value,
            'risk_level': violation.risk_level.value,
            'timestamp': violation.timestamp.isoformat(),
            'user_id': violation.user_id,
            'resource': violation.resource,
            'description': violation.description,
            'evidence': violation.evidence,
            'regulatory_framework': violation.regulatory_framework,
            'remediation_required': violation.remediation_required,
            'acknowledged': violation.acknowledged,
            'resolved': violation.resolved
        }
        
        self.redis_client.lpush('compliance_violations', json.dumps(violation_data, default=str))
        self.redis_client.ltrim('compliance_violations', 0, 10000)  # Keep last 10k violations
        
        # Send notifications for critical violations
        if violation.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            await self._send_violation_alert(violation)
        
        # Log violation
        self.logger.warning(
            f"Compliance violation detected: {violation.violation_type.value} "
            f"(Risk: {violation.risk_level.value}) - {violation.description}"
        )
    
    async def _send_violation_alert(self, violation: ComplianceViolation):
        """Send alert for compliance violation"""
        
        if not self.notification_settings.get('email_alerts'):
            return
        
        try:
            subject = f"COMPLIANCE ALERT: {violation.violation_type.value} ({violation.risk_level.value})"
            
            body = f"""
            Compliance Violation Detected
            
            Type: {violation.violation_type.value}
            Risk Level: {violation.risk_level.value}
            Time: {violation.timestamp.isoformat()}
            User: {violation.user_id or 'System'}
            Resource: {violation.resource}
            
            Description: {violation.description}
            
            Regulatory Framework: {', '.join(violation.regulatory_framework)}
            
            Evidence: {json.dumps(violation.evidence, indent=2, default=str)}
            
            This violation requires immediate attention.
            """
            
            # Send email (simplified - in production, use proper email service)
            self.logger.info(f"Compliance alert: {subject}")
            # In production: send_email(to=self.notification_settings['alert_email'], subject=subject, body=body)
            
        except Exception as e:
            self.logger.error(f"Failed to send compliance alert: {e}")
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance monitoring dashboard data"""
        
        # Get recent violations
        recent_violations = [
            v for v in self.violations
            if v.timestamp > datetime.now() - timedelta(days=30)
        ]
        
        # Calculate metrics
        violation_by_type = {}
        violation_by_risk = {}
        
        for violation in recent_violations:
            # By type
            vtype = violation.violation_type.value
            violation_by_type[vtype] = violation_by_type.get(vtype, 0) + 1
            
            # By risk level
            risk = violation.risk_level.value
            violation_by_risk[risk] = violation_by_risk.get(risk, 0) + 1
        
        return {
            "monitoring_status": "active" if self.running else "stopped",
            "last_check": datetime.now().isoformat(),
            "total_violations": len(self.violations),
            "recent_violations": len(recent_violations),
            "violations_by_type": violation_by_type,
            "violations_by_risk": violation_by_risk,
            "unresolved_violations": len([v for v in recent_violations if not v.resolved]),
            "critical_violations": len([v for v in recent_violations if v.risk_level == RiskLevel.CRITICAL]),
            "compliance_frameworks": ["FINRA", "SEC", "GDPR", "SOC2"],
            "next_regulatory_report": (datetime.now() + timedelta(days=30)).isoformat()
        }
    
    async def generate_regulatory_report(self, framework: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        
        if framework.upper() == "FINRA":
            return await self.reporter.generate_finra_report(start_date, end_date)
        else:
            return {
                "error": f"Report generation not implemented for {framework}",
                "available_frameworks": ["FINRA"]
            }


# Example usage
async def main():
    """Example usage of Compliance Monitor"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize compliance monitor
    monitor = ComplianceMonitor()
    
    # Run monitoring for demonstration
    try:
        # Start monitoring (would run continuously)
        monitoring_task = asyncio.create_task(monitor.start_monitoring(interval=60))
        
        # Wait a bit then check dashboard
        await asyncio.sleep(5)
        
        dashboard = monitor.get_compliance_dashboard()
        print("Compliance Dashboard:")
        print(json.dumps(dashboard, indent=2))
        
        # Generate sample report
        report = await monitor.generate_regulatory_report(
            "FINRA",
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
        print("\nFINRA Report:")
        print(json.dumps(report, indent=2))
        
        # Cancel monitoring task for demo
        monitoring_task.cancel()
        
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("Compliance monitoring stopped")


if __name__ == "__main__":
    asyncio.run(main())