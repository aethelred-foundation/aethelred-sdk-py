"""
Compliance Pre-Flight Checker

Validates data and job configurations against regulatory compliance
frameworks before submission to the network.

Supports:
- HIPAA (Healthcare)
- GDPR (EU Privacy)
- SOC2 (Security)
- PCI-DSS (Payment Cards)
- CCPA (California Privacy)

Example:
    >>> from aethelred.compliance import ComplianceChecker
    >>>
    >>> checker = ComplianceChecker(frameworks=["HIPAA", "GDPR"])
    >>> result = checker.check(data, sla)
    >>> if not result.passed:
    ...     for v in result.violations:
    ...         print(f"[{v.severity}] {v.message}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from aethelred.compliance.sanitizer import PIIScrubber, PIIType, DataClassification
from aethelred.core.types import SLA, PrivacyLevel, ComplianceFramework

logger = logging.getLogger(__name__)


class ViolationSeverity(str, Enum):
    """Severity levels for compliance violations."""

    INFO = "info"  # Informational, not blocking
    WARNING = "warning"  # Should be addressed
    ERROR = "error"  # Must be addressed before submission
    CRITICAL = "critical"  # Immediate security concern


@dataclass
class ComplianceViolation:
    """A compliance violation detected during pre-flight check."""

    framework: ComplianceFramework
    rule_id: str
    severity: ViolationSeverity
    message: str
    recommendation: str
    pii_types_involved: list[PIIType] = field(default_factory=list)
    field_path: Optional[str] = None  # Path to problematic field

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework.value,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "message": self.message,
            "recommendation": self.recommendation,
            "pii_types_involved": [p.value for p in self.pii_types_involved],
            "field_path": self.field_path,
        }


@dataclass
class ComplianceResult:
    """Result of compliance pre-flight check."""

    passed: bool
    frameworks_checked: list[ComplianceFramework]
    violations: list[ComplianceViolation]
    warnings: list[ComplianceViolation]
    data_classification: DataClassification
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def error_count(self) -> int:
        return len([v for v in self.violations if v.severity in [ViolationSeverity.ERROR, ViolationSeverity.CRITICAL]])

    @property
    def warning_count(self) -> int:
        return len([v for v in self.violations if v.severity == ViolationSeverity.WARNING])

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "frameworks_checked": [f.value for f in self.frameworks_checked],
            "violation_count": len(self.violations),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "data_classification": self.data_classification.value,
            "checked_at": self.checked_at.isoformat(),
            "violations": [v.to_dict() for v in self.violations],
        }


class ComplianceChecker:
    """
    Pre-flight compliance checker for data and job configurations.

    Validates against configured compliance frameworks before submission
    to help avoid rejected transactions and potential violations.

    Example:
        >>> checker = ComplianceChecker(frameworks=["HIPAA", "SOC2"])
        >>>
        >>> # Check data
        >>> result = checker.check(patient_data, sla)
        >>> if not result.passed:
        ...     print("Compliance check failed!")
        ...     for v in result.violations:
        ...         print(f"  [{v.severity}] {v.rule_id}: {v.message}")
        >>>
        >>> # Quick validation
        >>> is_valid = checker.validate(data, sla)
    """

    # Framework-specific rules
    FRAMEWORK_RULES = {
        ComplianceFramework.HIPAA: {
            "HIPAA-001": {
                "description": "PHI must be encrypted",
                "pii_types": [PIIType.SSN, PIIType.MEDICAL_RECORD, PIIType.DATE_OF_BIRTH],
                "requires_encryption": True,
                "min_privacy_level": PrivacyLevel.TEE,
            },
            "HIPAA-002": {
                "description": "No unencrypted PHI transmission",
                "pii_types": [PIIType.SSN, PIIType.MEDICAL_RECORD],
                "requires_privacy": True,
            },
            "HIPAA-003": {
                "description": "Access must be auditable",
                "requires_seal": True,
            },
        },
        ComplianceFramework.GDPR: {
            "GDPR-001": {
                "description": "Personal data must have legal basis",
                "pii_types": [PIIType.NAME, PIIType.EMAIL, PIIType.ADDRESS],
                "requires_consent": True,
            },
            "GDPR-002": {
                "description": "Data minimization required",
                "max_pii_types": 5,
            },
            "GDPR-003": {
                "description": "EU data residency",
                "allowed_regions": ["eu-west", "eu-central", "eu-north"],
            },
        },
        ComplianceFramework.SOC2: {
            "SOC2-001": {
                "description": "Encryption at rest and in transit",
                "requires_encryption": True,
            },
            "SOC2-002": {
                "description": "Access controls required",
                "requires_privacy": True,
            },
            "SOC2-003": {
                "description": "Audit logging required",
                "requires_seal": True,
            },
        },
        ComplianceFramework.PCI_DSS: {
            "PCI-001": {
                "description": "Cardholder data must be protected",
                "pii_types": [PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT],
                "requires_encryption": True,
                "min_privacy_level": PrivacyLevel.HYBRID,
            },
            "PCI-002": {
                "description": "No storage of sensitive authentication data",
                "prohibited_pii": [PIIType.CREDIT_CARD],  # After authorization
            },
        },
        ComplianceFramework.CCPA: {
            "CCPA-001": {
                "description": "Consumer opt-out rights",
                "pii_types": [PIIType.EMAIL, PIIType.NAME, PIIType.ADDRESS],
                "requires_consent": True,
            },
            "CCPA-002": {
                "description": "Data sale disclosure",
                "requires_disclosure": True,
            },
        },
    }

    def __init__(
        self,
        frameworks: Optional[list[Union[str, ComplianceFramework]]] = None,
        strict_mode: bool = False,
        scrubber: Optional[PIIScrubber] = None,
    ):
        """
        Initialize compliance checker.

        Args:
            frameworks: Compliance frameworks to check against
            strict_mode: Treat warnings as errors
            scrubber: Custom PII scrubber instance
        """
        # Parse frameworks
        if frameworks:
            self.frameworks = [
                ComplianceFramework(f) if isinstance(f, str) else f
                for f in frameworks
            ]
        else:
            self.frameworks = []

        self.strict_mode = strict_mode
        self.scrubber = scrubber or PIIScrubber()

        logger.info(f"ComplianceChecker initialized: frameworks={[f.value for f in self.frameworks]}")

    def supported_frameworks(self) -> list[ComplianceFramework]:
        """Return the list of all supported compliance frameworks."""
        return list(self.FRAMEWORK_RULES.keys())

    def check(
        self,
        data: Union[str, bytes, dict, list],
        sla: Optional[SLA] = None,
        metadata: Optional[dict] = None,
        framework: Optional[Union[str, ComplianceFramework]] = None,
    ) -> ComplianceResult:
        """
        Run compliance pre-flight check.

        Args:
            data: Data to be submitted
            sla: SLA configuration for the job
            metadata: Additional metadata

        Returns:
            Compliance check result
        """
        violations: list[ComplianceViolation] = []
        warnings: list[ComplianceViolation] = []

        # If a single framework was passed via the ``framework`` kwarg,
        # use it (temporarily) for this call instead of self.frameworks.
        frameworks_to_check = list(self.frameworks)
        if framework is not None:
            if isinstance(framework, str):
                framework = ComplianceFramework(framework)
            if framework not in frameworks_to_check:
                frameworks_to_check.append(framework)

        # Scan for PII
        pii_detections = self.scrubber.scan(data)
        pii_types_found = set(d.pii_type for d in pii_detections)
        data_classification = self.scrubber.classify(data)

        # Check each framework
        for framework in frameworks_to_check:
            if framework not in self.FRAMEWORK_RULES:
                continue

            rules = self.FRAMEWORK_RULES[framework]
            for rule_id, rule in rules.items():
                violation = self._check_rule(
                    framework=framework,
                    rule_id=rule_id,
                    rule=rule,
                    pii_types_found=pii_types_found,
                    data_classification=data_classification,
                    sla=sla,
                    metadata=metadata,
                )

                if violation:
                    if violation.severity in [ViolationSeverity.ERROR, ViolationSeverity.CRITICAL]:
                        violations.append(violation)
                    elif violation.severity == ViolationSeverity.WARNING:
                        if self.strict_mode:
                            violations.append(violation)
                        else:
                            warnings.append(violation)

        # Determine pass/fail
        passed = len(violations) == 0

        return ComplianceResult(
            passed=passed,
            frameworks_checked=frameworks_to_check,
            violations=violations,
            warnings=warnings,
            data_classification=data_classification,
        )

    def validate(
        self,
        data: Union[str, bytes, dict, list],
        sla: Optional[SLA] = None,
    ) -> bool:
        """Quick validation (returns True/False only)."""
        result = self.check(data, sla)
        return result.passed

    def _check_rule(
        self,
        framework: ComplianceFramework,
        rule_id: str,
        rule: dict[str, Any],
        pii_types_found: set[PIIType],
        data_classification: DataClassification,
        sla: Optional[SLA],
        metadata: Optional[dict],
    ) -> Optional[ComplianceViolation]:
        """Check a single compliance rule."""

        # Check PII-related rules
        if "pii_types" in rule:
            relevant_pii = pii_types_found & set(rule["pii_types"])

            if relevant_pii:
                # Check encryption requirement
                if rule.get("requires_encryption"):
                    if sla and sla.privacy_level == PrivacyLevel.NONE:
                        return ComplianceViolation(
                            framework=framework,
                            rule_id=rule_id,
                            severity=ViolationSeverity.ERROR,
                            message=f"{rule['description']}: Found {[p.value for p in relevant_pii]} without encryption",
                            recommendation=f"Enable privacy_level='tee' or 'tee_zk' in SLA",
                            pii_types_involved=list(relevant_pii),
                        )

                # Check minimum privacy level
                if "min_privacy_level" in rule:
                    min_level = rule["min_privacy_level"]
                    if sla:
                        current = sla.privacy_level
                        level_order = [PrivacyLevel.NONE, PrivacyLevel.TEE, PrivacyLevel.ZK, PrivacyLevel.HYBRID]
                        if level_order.index(current) < level_order.index(min_level):
                            return ComplianceViolation(
                                framework=framework,
                                rule_id=rule_id,
                                severity=ViolationSeverity.ERROR,
                                message=f"{rule['description']}: Privacy level {current.value} insufficient",
                                recommendation=f"Set privacy_level='{min_level.value}' or higher",
                                pii_types_involved=list(relevant_pii),
                            )

        # Check prohibited PII
        if "prohibited_pii" in rule:
            prohibited_found = pii_types_found & set(rule["prohibited_pii"])
            if prohibited_found:
                return ComplianceViolation(
                    framework=framework,
                    rule_id=rule_id,
                    severity=ViolationSeverity.CRITICAL,
                    message=f"{rule['description']}: Prohibited data found: {[p.value for p in prohibited_found]}",
                    recommendation="Remove prohibited data before submission",
                    pii_types_involved=list(prohibited_found),
                )

        # Check max PII types
        if "max_pii_types" in rule:
            if len(pii_types_found) > rule["max_pii_types"]:
                return ComplianceViolation(
                    framework=framework,
                    rule_id=rule_id,
                    severity=ViolationSeverity.WARNING,
                    message=f"{rule['description']}: Found {len(pii_types_found)} PII types, max is {rule['max_pii_types']}",
                    recommendation="Minimize data to only necessary fields",
                    pii_types_involved=list(pii_types_found),
                )

        # Check region restrictions
        if "allowed_regions" in rule:
            if sla and sla.allowed_regions:
                invalid_regions = set(sla.allowed_regions) - set(rule["allowed_regions"])
                if invalid_regions:
                    return ComplianceViolation(
                        framework=framework,
                        rule_id=rule_id,
                        severity=ViolationSeverity.ERROR,
                        message=f"{rule['description']}: Invalid regions: {invalid_regions}",
                        recommendation=f"Restrict to allowed regions: {rule['allowed_regions']}",
                    )

        # Check seal requirement
        if rule.get("requires_seal"):
            # Seals are created automatically by the network, so this is info only
            pass

        return None

    def add_framework(self, framework: Union[str, ComplianceFramework]) -> None:
        """Add a compliance framework to check."""
        if isinstance(framework, str):
            framework = ComplianceFramework(framework)
        if framework not in self.frameworks:
            self.frameworks.append(framework)

    def remove_framework(self, framework: Union[str, ComplianceFramework]) -> None:
        """Remove a compliance framework."""
        if isinstance(framework, str):
            framework = ComplianceFramework(framework)
        if framework in self.frameworks:
            self.frameworks.remove(framework)


# ============ Convenience Functions ============


def check_compliance(
    data: Union[str, bytes, dict, list],
    frameworks: list[str],
    sla: Optional[SLA] = None,
) -> ComplianceResult:
    """Quick compliance check."""
    checker = ComplianceChecker(frameworks=frameworks)
    return checker.check(data, sla)


def is_compliant(
    data: Union[str, bytes, dict, list],
    frameworks: list[str],
    sla: Optional[SLA] = None,
) -> bool:
    """Quick compliance validation."""
    checker = ComplianceChecker(frameworks=frameworks)
    return checker.validate(data, sla)
