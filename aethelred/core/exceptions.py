"""
Exceptions for the Aethelred SDK.

This module defines all custom exceptions used across the SDK,
with detailed error codes and messages for debugging.
"""

from enum import IntEnum
from typing import Any, Dict, Optional


class ErrorCode(IntEnum):
    """Error codes for Aethelred SDK exceptions."""
    
    # General errors (1000-1099)
    UNKNOWN = 1000
    INTERNAL = 1001
    INVALID_ARGUMENT = 1002
    NOT_FOUND = 1003
    ALREADY_EXISTS = 1004
    PERMISSION_DENIED = 1005
    
    # Connection errors (1100-1199)
    CONNECTION_FAILED = 1100
    CONNECTION_TIMEOUT = 1101
    CONNECTION_CLOSED = 1102
    DNS_RESOLUTION_FAILED = 1103
    SSL_ERROR = 1104
    
    # Authentication errors (1200-1299)
    AUTHENTICATION_REQUIRED = 1200
    INVALID_API_KEY = 1201
    EXPIRED_TOKEN = 1202
    INVALID_SIGNATURE = 1203
    
    # Rate limiting (1300-1399)
    RATE_LIMITED = 1300
    QUOTA_EXCEEDED = 1301
    
    # Transaction errors (1400-1499)
    TRANSACTION_FAILED = 1400
    INSUFFICIENT_FUNDS = 1401
    GAS_ESTIMATION_FAILED = 1402
    NONCE_TOO_LOW = 1403
    NONCE_TOO_HIGH = 1404
    TX_ALREADY_IN_MEMPOOL = 1405
    TX_NOT_FOUND = 1406
    
    # Job errors (1500-1599)
    JOB_NOT_FOUND = 1500
    JOB_ALREADY_EXISTS = 1501
    JOB_EXPIRED = 1502
    JOB_CANCELLED = 1503
    JOB_FAILED = 1504
    INVALID_JOB_STATUS = 1505
    MODEL_NOT_REGISTERED = 1506
    
    # Seal errors (1600-1699)
    SEAL_NOT_FOUND = 1600
    SEAL_ALREADY_EXISTS = 1601
    SEAL_REVOKED = 1602
    SEAL_EXPIRED = 1603
    SEAL_VERIFICATION_FAILED = 1604
    
    # Model errors (1700-1799)
    MODEL_NOT_FOUND = 1700
    MODEL_ALREADY_REGISTERED = 1701
    INVALID_MODEL_HASH = 1702
    MODEL_TOO_LARGE = 1703
    
    # Verification errors (1800-1899)
    VERIFICATION_FAILED = 1800
    INVALID_PROOF = 1801
    INVALID_ATTESTATION = 1802
    PROOF_EXPIRED = 1803
    CONSENSUS_NOT_REACHED = 1804
    
    # Validation errors (1900-1999)
    VALIDATION_FAILED = 1900
    INVALID_ADDRESS = 1901
    INVALID_HASH = 1902
    INVALID_INPUT = 1903


class AethelredError(Exception):
    """Base exception for all Aethelred SDK errors."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.cause = cause
    
    def __str__(self) -> str:
        parts = [f"[{self.code.name}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            "error": self.__class__.__name__,
            "code": self.code.value,
            "code_name": self.code.name,
            "message": self.message,
            "details": self.details,
        }


class ConnectionError(AethelredError):
    """Connection-related errors."""
    
    def __init__(
        self,
        message: str = "Failed to connect to Aethelred node",
        code: ErrorCode = ErrorCode.CONNECTION_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class AuthenticationError(AethelredError):
    """Authentication and authorization errors."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        code: ErrorCode = ErrorCode.AUTHENTICATION_REQUIRED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class RateLimitError(AethelredError):
    """Rate limit exceeded errors."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(message, ErrorCode.RATE_LIMITED, **kwargs)
        self.retry_after = retry_after


class TransactionError(AethelredError):
    """Transaction execution errors."""
    
    def __init__(
        self,
        message: str = "Transaction failed",
        code: ErrorCode = ErrorCode.TRANSACTION_FAILED,
        tx_hash: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)
        self.tx_hash = tx_hash


class JobError(AethelredError):
    """Job-related errors."""
    
    def __init__(
        self,
        message: str = "Job operation failed",
        code: ErrorCode = ErrorCode.JOB_FAILED,
        job_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)
        self.job_id = job_id


class SealError(AethelredError):
    """Seal-related errors."""
    
    def __init__(
        self,
        message: str = "Seal operation failed",
        code: ErrorCode = ErrorCode.SEAL_NOT_FOUND,
        seal_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)
        self.seal_id = seal_id


class ModelError(AethelredError):
    """Model-related errors."""
    
    def __init__(
        self,
        message: str = "Model operation failed",
        code: ErrorCode = ErrorCode.MODEL_NOT_FOUND,
        model_hash: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)
        self.model_hash = model_hash


class VerificationError(AethelredError):
    """Verification-related errors."""
    
    def __init__(
        self,
        message: str = "Verification failed",
        code: ErrorCode = ErrorCode.VERIFICATION_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


class ValidationError(AethelredError):
    """Input validation errors."""
    
    def __init__(
        self,
        message: str = "Validation failed",
        code: ErrorCode = ErrorCode.VALIDATION_FAILED,
        field: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)
        self.field = field


class TimeoutError(AethelredError):
    """Timeout errors."""
    
    def __init__(
        self,
        message: str = "Operation timed out",
        timeout_seconds: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(message, ErrorCode.CONNECTION_TIMEOUT, **kwargs)
        self.timeout_seconds = timeout_seconds


class NetworkError(AethelredError):
    """Network-related errors."""
    
    def __init__(
        self,
        message: str = "Network error",
        code: ErrorCode = ErrorCode.CONNECTION_FAILED,
        **kwargs,
    ):
        super().__init__(message, code, **kwargs)


# Backwards-compatible alias retained for older imports in sdk/python/aethelred/core/__init__.py
ProofError = VerificationError
