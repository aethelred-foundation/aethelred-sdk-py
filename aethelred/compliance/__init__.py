"""
Aethelred Compliance Module

Pre-flight compliance checks and data sanitization utilities to help
developers avoid accidentally uploading PII or sensitive data.

Features:
- Enterprise-grade PII detection with Microsoft Presidio
- Multi-language PII detection (50+ entity types)
- GDPR, HIPAA, PCI-DSS, CCPA compliance support
- Multiple anonymization operators (mask, hash, encrypt, replace)
- Custom entity recognizers for Aethelred addresses
- Compliance pre-flight checks
- Audit trail generation

Example:
    >>> from aethelred.compliance import PresidioPIIScrubber, ComplianceChecker
    >>>
    >>> # Enterprise-grade scrubbing with Presidio
    >>> scrubber = PresidioPIIScrubber(
    ...     compliance_frameworks=["HIPAA", "GDPR"],
    ...     score_threshold=0.7
    ... )
    >>> result = await scrubber.scrub("Patient John Smith, SSN: 123-45-6789")
    >>> print(result.scrubbed_text)
    >>> # "<PERSON>, SSN: <US_SSN>"
    >>>
    >>> # Pre-flight compliance check
    >>> checker = ComplianceChecker(frameworks=["HIPAA", "GDPR"])
    >>> result = checker.check(data, metadata)
    >>> if not result.passed:
    ...     raise ValueError(f"Compliance check failed: {result.violations}")
"""

from aethelred.compliance.sanitizer import (
    PIIScrubber,
    PIIType,
    PIIDetection,
    ScrubResult,
    DataClassification,
)
from aethelred.compliance.checker import (
    ComplianceChecker,
    ComplianceResult,
    ComplianceViolation,
    ViolationSeverity,
)

# Enterprise Presidio-based scrubber
from aethelred.compliance.presidio_scrubber import (
    PIIScrubber as PresidioPIIScrubber,
    PIIFinding,
    ScrubResult as PresidioScrubResult,
    BatchScrubResult,
    OperatorConfiguration,
    PIIEntityType,
    ComplianceFramework as PresidioComplianceFramework,
    AnonymizationOperator,
    CustomRecognizer,
    AethelredAddressRecognizer,
    AethelredTxHashRecognizer,
    create_hipaa_scrubber,
    create_gdpr_scrubber,
    create_pci_scrubber,
    PRESIDIO_AVAILABLE,
)

__all__ = [
    # Basic Sanitizer
    "PIIScrubber",
    "PIIType",
    "PIIDetection",
    "ScrubResult",
    "DataClassification",
    # Checker
    "ComplianceChecker",
    "ComplianceResult",
    "ComplianceViolation",
    "ViolationSeverity",
    # Enterprise Presidio Scrubber
    "PresidioPIIScrubber",
    "PIIFinding",
    "PresidioScrubResult",
    "BatchScrubResult",
    "OperatorConfiguration",
    "PIIEntityType",
    "PresidioComplianceFramework",
    "AnonymizationOperator",
    "CustomRecognizer",
    "AethelredAddressRecognizer",
    "AethelredTxHashRecognizer",
    # Factory functions
    "create_hipaa_scrubber",
    "create_gdpr_scrubber",
    "create_pci_scrubber",
    # Constants
    "PRESIDIO_AVAILABLE",
]
