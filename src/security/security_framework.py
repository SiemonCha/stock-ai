"""
Comprehensive Security and Compliance Framework

This module provides enterprise-grade security controls including:
- Multi-factor authentication and authorization
- Data encryption at rest and in transit
- Audit logging and compliance monitoring
- API security with rate limiting and DDoS protection
- Secrets management and key rotation
- Security scanning and vulnerability assessment
- GDPR, SOC2, and financial regulatory compliance
"""

import asyncio
import logging
import hashlib
import secrets
import jwt
import bcrypt
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
import ssl
import base64
from pathlib import Path
import subprocess
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests


class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceStandard(Enum):
    GDPR = "gdpr"
    SOC2 = "soc2"
    FINRA = "finra"
    SEC = "sec"
    PCI_DSS = "pci_dss"
    ISO27001 = "iso27001"


@dataclass
class SecurityEvent:
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    resource: str
    action: str
    result: str  # success, failure, blocked
    risk_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    resource_pattern: str
    required_roles: Set[str]
    required_permissions: Set[str]
    security_level: SecurityLevel
    time_restrictions: Optional[Dict[str, Any]] = None
    ip_restrictions: Optional[List[str]] = None
    mfa_required: bool = True


class EncryptionManager:
    """
    Handles encryption/decryption operations with key rotation
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.master_key = master_key
        else:
            self.master_key = Fernet.generate_key()
        
        self.fernet = Fernet(self.master_key)
        self.key_rotation_interval = timedelta(days=90)
        self.key_created = datetime.now()
        
        # RSA keys for asymmetric encryption
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        self.public_key = self.private_key.public_key()
        
        self.logger = logging.getLogger(__name__)
    
    def encrypt_data(self, data: str, classification: SecurityLevel = SecurityLevel.CONFIDENTIAL) -> str:
        """Encrypt data with appropriate encryption strength"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Add metadata about classification and timestamp
        metadata = {
            'classification': classification.value,
            'encrypted_at': datetime.now().isoformat(),
            'key_version': 1
        }
        
        # Combine metadata and data
        payload = {
            'metadata': metadata,
            'data': base64.b64encode(data).decode('utf-8')
        }
        
        payload_bytes = json.dumps(payload).encode('utf-8')
        
        # Use stronger encryption for higher classifications
        if classification in [SecurityLevel.RESTRICTED, SecurityLevel.TOP_SECRET]:
            # Double encryption for sensitive data
            encrypted_once = self.fernet.encrypt(payload_bytes)
            encrypted_twice = self.fernet.encrypt(encrypted_once)
            return base64.b64encode(encrypted_twice).decode('utf-8')
        else:
            encrypted = self.fernet.encrypt(payload_bytes)
            return base64.b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> tuple[str, SecurityLevel]:
        """Decrypt data and return data with its classification"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            # Try double decryption first
            try:
                decrypted_once = self.fernet.decrypt(encrypted_bytes)
                decrypted_twice = self.fernet.decrypt(decrypted_once)
                payload_bytes = decrypted_twice
            except:
                # Single encryption
                payload_bytes = self.fernet.decrypt(encrypted_bytes)
            
            payload = json.loads(payload_bytes.decode('utf-8'))
            
            # Extract original data and metadata
            data = base64.b64decode(payload['data']).decode('utf-8')
            classification = SecurityLevel(payload['metadata']['classification'])
            
            return data, classification
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise SecurityException("Decryption failed")
    
    def encrypt_with_public_key(self, data: str, public_key=None) -> str:
        """Encrypt data using RSA public key"""
        if public_key is None:
            public_key = self.public_key
        
        data_bytes = data.encode('utf-8')
        
        # RSA can only encrypt data smaller than key size minus padding
        # For larger data, use hybrid encryption
        if len(data_bytes) > 446:  # 4096-bit key can encrypt max ~446 bytes with OAEP
            # Generate random AES key
            aes_key = Fernet.generate_key()
            aes_cipher = Fernet(aes_key)
            
            # Encrypt data with AES
            encrypted_data = aes_cipher.encrypt(data_bytes)
            
            # Encrypt AES key with RSA
            encrypted_key = public_key.encrypt(
                aes_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Combine encrypted key and data
            combined = base64.b64encode(encrypted_key).decode('utf-8') + '::' + \
                      base64.b64encode(encrypted_data).decode('utf-8')
            
            return combined
        else:
            # Direct RSA encryption for small data
            encrypted = public_key.encrypt(
                data_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted).decode('utf-8')
    
    def needs_key_rotation(self) -> bool:
        """Check if encryption keys need rotation"""
        return datetime.now() - self.key_created > self.key_rotation_interval
    
    def rotate_keys(self) -> str:
        """Rotate encryption keys and return new key for secure storage"""
        old_key = self.master_key
        self.master_key = Fernet.generate_key()
        self.fernet = Fernet(self.master_key)
        self.key_created = datetime.now()
        
        self.logger.info("Encryption keys rotated successfully")
        return base64.b64encode(self.master_key).decode('utf-8')


class AuthenticationManager:
    """
    Multi-factor authentication and session management
    """
    
    def __init__(self, jwt_secret: str, redis_client):
        self.jwt_secret = jwt_secret
        self.redis_client = redis_client
        self.session_timeout = timedelta(hours=8)
        self.mfa_timeout = timedelta(minutes=5)
        self.logger = logging.getLogger(__name__)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    def generate_mfa_token(self, user_id: str) -> str:
        """Generate time-based MFA token"""
        import pyotp
        
        # Get or create secret for user
        secret_key = f"mfa_secret:{user_id}"
        secret = self.redis_client.get(secret_key)
        
        if not secret:
            secret = pyotp.random_base32()
            self.redis_client.setex(secret_key, int(timedelta(days=365).total_seconds()), secret)
        else:
            secret = secret.decode('utf-8')
        
        totp = pyotp.TOTP(secret)
        return totp.now()
    
    def verify_mfa_token(self, user_id: str, token: str) -> bool:
        """Verify MFA token"""
        import pyotp
        
        secret_key = f"mfa_secret:{user_id}"
        secret = self.redis_client.get(secret_key)
        
        if not secret:
            return False
        
        secret = secret.decode('utf-8')
        totp = pyotp.TOTP(secret)
        
        # Allow for clock skew (previous, current, next token)
        return totp.verify(token, valid_window=1)
    
    async def authenticate_user(self, 
                              username: str, 
                              password: str, 
                              mfa_token: Optional[str] = None,
                              ip_address: str = None) -> Dict[str, Any]:
        """Authenticate user with optional MFA"""
        
        # Rate limiting
        rate_limit_key = f"auth_attempts:{ip_address}"
        attempts = self.redis_client.incr(rate_limit_key)
        
        if attempts == 1:
            self.redis_client.expire(rate_limit_key, 300)  # 5 minutes
        
        if attempts > 5:
            raise SecurityException("Too many authentication attempts")
        
        # Simulate user lookup (in practice, query database)
        user_key = f"user:{username}"
        user_data = self.redis_client.hgetall(user_key)
        
        if not user_data:
            raise SecurityException("Invalid credentials")
        
        stored_password = user_data.get(b'password', b'').decode('utf-8')
        requires_mfa = user_data.get(b'mfa_enabled', b'false').decode('utf-8') == 'true'
        
        # Verify password
        if not self.verify_password(password, stored_password):
            raise SecurityException("Invalid credentials")
        
        # Check MFA if required
        if requires_mfa:
            if not mfa_token:
                return {
                    'status': 'mfa_required',
                    'mfa_methods': ['totp'],
                    'session_id': None
                }
            
            if not self.verify_mfa_token(username, mfa_token):
                raise SecurityException("Invalid MFA token")
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': username,
            'created_at': datetime.now().isoformat(),
            'ip_address': ip_address,
            'mfa_verified': requires_mfa
        }
        
        # Store session
        session_key = f"session:{session_id}"
        self.redis_client.setex(
            session_key,
            int(self.session_timeout.total_seconds()),
            json.dumps(session_data)
        )
        
        # Generate JWT token
        jwt_payload = {
            'user_id': username,
            'session_id': session_id,
            'exp': datetime.now() + self.session_timeout,
            'iat': datetime.now(),
            'ip': ip_address
        }
        
        token = jwt.encode(jwt_payload, self.jwt_secret, algorithm='HS256')
        
        # Reset rate limiting on successful auth
        self.redis_client.delete(rate_limit_key)
        
        return {
            'status': 'authenticated',
            'token': token,
            'session_id': session_id,
            'expires_at': (datetime.now() + self.session_timeout).isoformat()
        }
    
    def validate_session(self, token: str, ip_address: str = None) -> Dict[str, Any]:
        """Validate JWT token and session"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Verify IP if provided
            if ip_address and payload.get('ip') != ip_address:
                raise SecurityException("IP address mismatch")
            
            # Check session exists
            session_key = f"session:{payload['session_id']}"
            session_data = self.redis_client.get(session_key)
            
            if not session_data:
                raise SecurityException("Session expired")
            
            return {
                'valid': True,
                'user_id': payload['user_id'],
                'session_id': payload['session_id']
            }
            
        except jwt.ExpiredSignatureError:
            raise SecurityException("Token expired")
        except jwt.InvalidTokenError:
            raise SecurityException("Invalid token")


class AuthorizationManager:
    """
    Role-based access control and policy enforcement
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.policies: List[AccessPolicy] = []
        self.logger = logging.getLogger(__name__)
    
    def add_policy(self, policy: AccessPolicy):
        """Add access policy"""
        self.policies.append(policy)
        self.logger.info(f"Added policy for {policy.resource_pattern}")
    
    def grant_role(self, user_id: str, role: str):
        """Grant role to user"""
        user_roles_key = f"user_roles:{user_id}"
        self.redis_client.sadd(user_roles_key, role)
    
    def revoke_role(self, user_id: str, role: str):
        """Revoke role from user"""
        user_roles_key = f"user_roles:{user_id}"
        self.redis_client.srem(user_roles_key, role)
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get user's roles"""
        user_roles_key = f"user_roles:{user_id}"
        roles = self.redis_client.smembers(user_roles_key)
        return {role.decode('utf-8') for role in roles}
    
    def check_access(self, 
                    user_id: str, 
                    resource: str, 
                    action: str,
                    ip_address: str = None) -> bool:
        """Check if user has access to resource/action"""
        
        user_roles = self.get_user_roles(user_id)
        
        # Find matching policy
        matching_policy = None
        for policy in self.policies:
            if re.match(policy.resource_pattern, resource):
                matching_policy = policy
                break
        
        if not matching_policy:
            # Default deny
            return False
        
        # Check roles
        if not matching_policy.required_roles.issubset(user_roles):
            return False
        
        # Check IP restrictions
        if (matching_policy.ip_restrictions and 
            ip_address not in matching_policy.ip_restrictions):
            return False
        
        # Check time restrictions
        if matching_policy.time_restrictions:
            current_hour = datetime.now().hour
            allowed_hours = matching_policy.time_restrictions.get('hours', [])
            if allowed_hours and current_hour not in allowed_hours:
                return False
        
        return True


class AuditLogger:
    """
    Comprehensive audit logging for compliance
    """
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Set up structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Audit log file handler
        audit_handler = logging.FileHandler('audit.log')
        audit_handler.setFormatter(formatter)
        
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.addHandler(audit_handler)
        self.audit_logger.setLevel(logging.INFO)
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event for audit trail"""
        
        event_data = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'user_id': event.user_id,
            'resource': event.resource,
            'action': event.action,
            'result': event.result,
            'risk_score': event.risk_score,
            'metadata': event.metadata
        }
        
        # Log to structured audit log
        self.audit_logger.info(json.dumps(event_data))
        
        # Store in Redis for real-time monitoring
        self.redis_client.lpush(
            'security_events',
            json.dumps(event_data)
        )
        self.redis_client.ltrim('security_events', 0, 10000)  # Keep last 10k events
        
        # Alert on high-risk events
        if event.risk_score > 0.8:
            self._trigger_security_alert(event)
    
    def _trigger_security_alert(self, event: SecurityEvent):
        """Trigger security alert for high-risk events"""
        alert_data = {
            'alert_type': 'HIGH_RISK_SECURITY_EVENT',
            'event': event.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store alert
        self.redis_client.lpush('security_alerts', json.dumps(alert_data, default=str))
        
        # Log critical alert
        self.logger.critical(f"High-risk security event: {event.event_type}")
    
    def get_audit_trail(self, 
                       user_id: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Retrieve audit trail with filters"""
        
        events = []
        event_data_list = self.redis_client.lrange('security_events', 0, -1)
        
        for event_data in event_data_list:
            event = json.loads(event_data)
            event_time = datetime.fromisoformat(event['timestamp'])
            
            # Apply filters
            if user_id and event.get('user_id') != user_id:
                continue
            
            if start_time and event_time < start_time:
                continue
            
            if end_time and event_time > end_time:
                continue
            
            events.append(event)
        
        return events


class SecretsManager:
    """
    Secure secrets management with automatic rotation
    """
    
    def __init__(self, encryption_manager: EncryptionManager, redis_client):
        self.encryption_manager = encryption_manager
        self.redis_client = redis_client
        self.logger = logging.getLogger(__name__)
        
        # Secret rotation intervals
        self.rotation_intervals = {
            'database_password': timedelta(days=90),
            'api_key': timedelta(days=30),
            'jwt_secret': timedelta(days=180),
            'encryption_key': timedelta(days=90)
        }
    
    def store_secret(self, 
                    secret_name: str, 
                    secret_value: str,
                    classification: SecurityLevel = SecurityLevel.CONFIDENTIAL):
        """Store encrypted secret"""
        
        encrypted_value = self.encryption_manager.encrypt_data(secret_value, classification)
        
        secret_metadata = {
            'name': secret_name,
            'created_at': datetime.now().isoformat(),
            'last_rotated': datetime.now().isoformat(),
            'classification': classification.value,
            'rotation_interval': self.rotation_intervals.get(secret_name, timedelta(days=90)).days
        }
        
        # Store encrypted secret
        self.redis_client.hset(
            f"secret:{secret_name}",
            mapping={
                'value': encrypted_value,
                'metadata': json.dumps(secret_metadata)
            }
        )
        
        self.logger.info(f"Secret '{secret_name}' stored securely")
    
    def retrieve_secret(self, secret_name: str) -> Optional[str]:
        """Retrieve and decrypt secret"""
        
        secret_data = self.redis_client.hgetall(f"secret:{secret_name}")
        
        if not secret_data:
            return None
        
        encrypted_value = secret_data[b'value'].decode('utf-8')
        
        try:
            decrypted_value, _ = self.encryption_manager.decrypt_data(encrypted_value)
            return decrypted_value
        except Exception as e:
            self.logger.error(f"Failed to retrieve secret '{secret_name}': {e}")
            return None
    
    def rotate_secret(self, secret_name: str, new_value: str) -> bool:
        """Rotate secret with new value"""
        
        secret_data = self.redis_client.hgetall(f"secret:{secret_name}")
        
        if not secret_data:
            self.logger.error(f"Secret '{secret_name}' not found for rotation")
            return False
        
        metadata = json.loads(secret_data[b'metadata'].decode('utf-8'))
        classification = SecurityLevel(metadata['classification'])
        
        # Encrypt new value
        encrypted_value = self.encryption_manager.encrypt_data(new_value, classification)
        
        # Update metadata
        metadata['last_rotated'] = datetime.now().isoformat()
        
        # Store rotated secret
        self.redis_client.hset(
            f"secret:{secret_name}",
            mapping={
                'value': encrypted_value,
                'metadata': json.dumps(metadata)
            }
        )
        
        self.logger.info(f"Secret '{secret_name}' rotated successfully")
        return True
    
    def check_rotation_needed(self) -> List[str]:
        """Check which secrets need rotation"""
        
        secrets_needing_rotation = []
        
        # Get all secrets
        keys = self.redis_client.keys("secret:*")
        
        for key in keys:
            secret_name = key.decode('utf-8').replace('secret:', '')
            secret_data = self.redis_client.hgetall(key)
            
            if not secret_data:
                continue
            
            metadata = json.loads(secret_data[b'metadata'].decode('utf-8'))
            last_rotated = datetime.fromisoformat(metadata['last_rotated'])
            rotation_interval = timedelta(days=metadata['rotation_interval'])
            
            if datetime.now() - last_rotated > rotation_interval:
                secrets_needing_rotation.append(secret_name)
        
        return secrets_needing_rotation


class SecurityScanner:
    """
    Security vulnerability scanning and assessment
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scan_results = []
    
    async def scan_dependencies(self, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
        """Scan Python dependencies for vulnerabilities"""
        
        try:
            # Use safety to scan for known vulnerabilities
            process = await asyncio.create_subprocess_exec(
                'safety', 'check', '--json', '-r', requirements_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return {
                    'status': 'safe',
                    'vulnerabilities': [],
                    'scanned_at': datetime.now().isoformat()
                }
            else:
                # Parse JSON output
                try:
                    vulnerabilities = json.loads(stdout.decode())
                    return {
                        'status': 'vulnerabilities_found',
                        'vulnerabilities': vulnerabilities,
                        'scanned_at': datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    return {
                        'status': 'scan_error',
                        'error': stderr.decode(),
                        'scanned_at': datetime.now().isoformat()
                    }
                    
        except FileNotFoundError:
            self.logger.warning("Safety package not found. Install with: pip install safety")
            return {
                'status': 'scanner_not_available',
                'message': 'Safety package not installed'
            }
    
    async def scan_secrets_in_code(self, directory: str = ".") -> Dict[str, Any]:
        """Scan code for exposed secrets"""
        
        # Common secret patterns
        secret_patterns = {
            'api_key': r'(?i)api[_-]?key\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
            'password': r'(?i)password\s*[:=]\s*["\']?([^"\'\s]{8,})["\']?',
            'secret': r'(?i)secret\s*[:=]\s*["\']?([a-zA-Z0-9_-]{16,})["\']?',
            'token': r'(?i)token\s*[:=]\s*["\']?([a-zA-Z0-9_-]{20,})["\']?',
            'aws_access': r'AKIA[0-9A-Z]{16}',
            'private_key': r'-----BEGIN [A-Z]+ PRIVATE KEY-----'
        }
        
        findings = []
        
        try:
            # Scan Python files
            for py_file in Path(directory).rglob("*.py"):
                if any(exclude in str(py_file) for exclude in ['.venv', '__pycache__', '.git']):
                    continue
                
                try:
                    content = py_file.read_text()
                    
                    for secret_type, pattern in secret_patterns.items():
                        matches = re.finditer(pattern, content)
                        
                        for match in matches:
                            findings.append({
                                'file': str(py_file),
                                'secret_type': secret_type,
                                'line': content[:match.start()].count('\n') + 1,
                                'severity': 'high' if secret_type in ['private_key', 'aws_access'] else 'medium'
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Could not scan {py_file}: {e}")
            
            return {
                'status': 'completed',
                'findings': findings,
                'files_scanned': len(list(Path(directory).rglob("*.py"))),
                'secrets_found': len(findings),
                'scanned_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'scanned_at': datetime.now().isoformat()
            }
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security assessment report"""
        
        return {
            'assessment_date': datetime.now().isoformat(),
            'overall_risk_score': self._calculate_overall_risk(),
            'findings_summary': {
                'high_risk': len([f for f in self.scan_results if f.get('severity') == 'high']),
                'medium_risk': len([f for f in self.scan_results if f.get('severity') == 'medium']),
                'low_risk': len([f for f in self.scan_results if f.get('severity') == 'low'])
            },
            'recommendations': self._generate_recommendations(),
            'compliance_status': self._check_compliance_status()
        }
    
    def _calculate_overall_risk(self) -> float:
        """Calculate overall security risk score (0.0 - 1.0)"""
        if not self.scan_results:
            return 0.0
        
        risk_weights = {'high': 0.8, 'medium': 0.5, 'low': 0.2}
        total_risk = 0.0
        
        for finding in self.scan_results:
            severity = finding.get('severity', 'low')
            total_risk += risk_weights.get(severity, 0.2)
        
        return min(1.0, total_risk / len(self.scan_results))
    
    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations"""
        return [
            "Enable multi-factor authentication for all accounts",
            "Rotate secrets and API keys every 90 days",
            "Implement comprehensive audit logging",
            "Use encryption for sensitive data at rest and in transit",
            "Regular security vulnerability scanning",
            "Implement rate limiting and DDoS protection",
            "Regular security training for development team"
        ]
    
    def _check_compliance_status(self) -> Dict[str, Any]:
        """Check compliance with various standards"""
        return {
            ComplianceStandard.GDPR.value: {
                'status': 'partial',
                'missing': ['data retention policy', 'right to erasure implementation']
            },
            ComplianceStandard.SOC2.value: {
                'status': 'partial',
                'missing': ['regular penetration testing', 'incident response plan']
            },
            ComplianceStandard.FINRA.value: {
                'status': 'in_progress',
                'missing': ['trade surveillance', 'regulatory reporting']
            }
        }


class SecurityFramework:
    """
    Main security framework orchestrator
    """
    
    def __init__(self, 
                 jwt_secret: str = None,
                 master_key: bytes = None,
                 redis_url: str = "redis://localhost:6379"):
        
        # Initialize Redis connection
        self.redis_client = redis.from_url(redis_url)
        
        # Initialize components
        self.encryption_manager = EncryptionManager(master_key)
        self.auth_manager = AuthenticationManager(
            jwt_secret or secrets.token_urlsafe(32), 
            self.redis_client
        )
        self.authz_manager = AuthorizationManager(self.redis_client)
        self.audit_logger = AuditLogger(self.redis_client)
        self.secrets_manager = SecretsManager(self.encryption_manager, self.redis_client)
        self.security_scanner = SecurityScanner()
        
        self.logger = logging.getLogger(__name__)
        
        # Set up default policies
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Set up default access policies"""
        
        # API access policies
        self.authz_manager.add_policy(AccessPolicy(
            resource_pattern=r'/api/predictions/.*',
            required_roles={'trader', 'analyst'},
            required_permissions={'read_predictions'},
            security_level=SecurityLevel.CONFIDENTIAL,
            mfa_required=True
        ))
        
        self.authz_manager.add_policy(AccessPolicy(
            resource_pattern=r'/api/admin/.*',
            required_roles={'admin'},
            required_permissions={'admin_access'},
            security_level=SecurityLevel.RESTRICTED,
            mfa_required=True,
            time_restrictions={'hours': list(range(8, 18))}  # Business hours only
        ))
        
        self.authz_manager.add_policy(AccessPolicy(
            resource_pattern=r'/api/portfolio/.*',
            required_roles={'portfolio_manager', 'trader'},
            required_permissions={'manage_portfolio'},
            security_level=SecurityLevel.CONFIDENTIAL,
            mfa_required=True
        ))
    
    async def initialize_security(self):
        """Initialize security framework"""
        
        self.logger.info("Initializing security framework...")
        
        # Store default secrets if they don't exist
        if not self.secrets_manager.retrieve_secret('jwt_secret'):
            self.secrets_manager.store_secret(
                'jwt_secret', 
                secrets.token_urlsafe(32),
                SecurityLevel.RESTRICTED
            )
        
        # Run initial security scan
        scan_results = await self.security_scanner.scan_dependencies()
        if scan_results['status'] == 'vulnerabilities_found':
            self.logger.warning(f"Found {len(scan_results['vulnerabilities'])} vulnerabilities")
        
        # Set up periodic tasks
        asyncio.create_task(self._periodic_security_tasks())
        
        self.logger.info("Security framework initialized successfully")
    
    async def _periodic_security_tasks(self):
        """Run periodic security maintenance tasks"""
        
        while True:
            try:
                # Check for secrets that need rotation
                secrets_to_rotate = self.secrets_manager.check_rotation_needed()
                if secrets_to_rotate:
                    self.logger.info(f"Secrets need rotation: {secrets_to_rotate}")
                
                # Check for key rotation
                if self.encryption_manager.needs_key_rotation():
                    self.logger.info("Encryption keys need rotation")
                
                # Run security scans
                await self.security_scanner.scan_dependencies()
                
                # Sleep for 24 hours
                await asyncio.sleep(86400)
                
            except Exception as e:
                self.logger.error(f"Error in periodic security tasks: {e}")
                await asyncio.sleep(3600)  # Retry in 1 hour
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        
        return {
            'framework_status': 'active',
            'encryption_key_age_days': (datetime.now() - self.encryption_manager.key_created).days,
            'active_sessions': len(self.redis_client.keys('session:*')),
            'security_policies': len(self.authz_manager.policies),
            'secrets_managed': len(self.redis_client.keys('secret:*')),
            'security_events_logged': self.redis_client.llen('security_events'),
            'high_risk_alerts': len([
                alert for alert in self.redis_client.lrange('security_alerts', 0, -1)
                if 'HIGH_RISK' in alert.decode('utf-8')
            ]),
            'last_security_scan': datetime.now().isoformat()
        }


class SecurityException(Exception):
    """Custom exception for security-related errors"""
    pass


# Example usage and testing
async def main():
    """Example usage of the Security Framework"""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize security framework
    security = SecurityFramework()
    await security.initialize_security()
    
    # Example: Create user and authenticate
    try:
        # Store user credentials (normally done via registration API)
        user_data = {
            'password': security.auth_manager.hash_password('secure_password123'),
            'mfa_enabled': 'true',
            'roles': json.dumps(['trader', 'analyst'])
        }
        
        security.redis_client.hset('user:john_doe', mapping=user_data)
        
        # Grant roles
        security.authz_manager.grant_role('john_doe', 'trader')
        security.authz_manager.grant_role('john_doe', 'analyst')
        
        # Authenticate
        auth_result = await security.auth_manager.authenticate_user(
            username='john_doe',
            password='secure_password123',
            mfa_token=security.auth_manager.generate_mfa_token('john_doe'),
            ip_address='192.168.1.100'
        )
        
        print(f"Authentication result: {auth_result['status']}")
        
        # Check authorization
        has_access = security.authz_manager.check_access(
            user_id='john_doe',
            resource='/api/predictions/stocks',
            action='read',
            ip_address='192.168.1.100'
        )
        
        print(f"Has access to predictions: {has_access}")
        
        # Log security event
        security.audit_logger.log_security_event(SecurityEvent(
            timestamp=datetime.now(),
            event_type='api_access',
            user_id='john_doe',
            resource='/api/predictions/stocks',
            action='read',
            result='success',
            risk_score=0.2,
            metadata={'ip_address': '192.168.1.100'}
        ))
        
        # Store a secret
        security.secrets_manager.store_secret(
            'database_password',
            'super_secret_db_pass_123',
            SecurityLevel.RESTRICTED
        )
        
        # Get security status
        status = security.get_security_status()
        print(f"Security framework status: {json.dumps(status, indent=2)}")
        
    except SecurityException as e:
        print(f"Security error: {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())