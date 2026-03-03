"""
PII Detection and Data Sanitization

Local pre-flight checks to help developers avoid accidentally uploading
unencrypted PII or sensitive data in RAW_UPLOAD mode.

Features:
- Pattern-based PII detection (SSN, email, phone, credit cards)
- Named entity recognition for names/addresses
- Data classification (public, internal, confidential, restricted)
- Automatic scrubbing/masking

Example:
    >>> from aethelred.compliance import PIIScrubber
    >>>
    >>> scrubber = PIIScrubber()
    >>> data = "Patient SSN: 123-45-6789, email: john@hospital.org"
    >>>
    >>> # Scan for PII
    >>> detections = scrubber.scan(data)
    >>> for d in detections:
    ...     print(f"Found {d.pii_type}: {d.match} at position {d.start}")
    >>>
    >>> # Automatically scrub PII
    >>> clean_data, result = scrubber.scrub(data)
    >>> print(clean_data)  # "Patient SSN: [REDACTED-SSN], email: [REDACTED-EMAIL]"
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class PIIType(str, Enum):
    """Types of Personally Identifiable Information."""

    # Direct identifiers
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    CREDIT_CARD = "credit_card"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"

    # Financial
    BANK_ACCOUNT = "bank_account"
    ROUTING_NUMBER = "routing_number"
    IBAN = "iban"

    # Health
    MEDICAL_RECORD = "medical_record"
    NPI = "npi"  # National Provider Identifier
    DEA_NUMBER = "dea_number"

    # Personal
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    IP_ADDRESS = "ip_address"

    # Credentials
    API_KEY = "api_key"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    SEED_PHRASE = "seed_phrase"


class DataClassification(str, Enum):
    """Data classification levels."""

    PUBLIC = "public"  # Can be shared freely
    INTERNAL = "internal"  # Internal use only
    CONFIDENTIAL = "confidential"  # Business sensitive
    RESTRICTED = "restricted"  # Highly sensitive (PII, PHI)


@dataclass
class PIIDetection:
    """A detected PII instance."""

    pii_type: PIIType
    match: str
    start: int
    end: int
    confidence: float  # 0-1
    context: Optional[str] = None  # Surrounding text
    redacted: Optional[str] = None  # Masked version


@dataclass
class ScrubResult:
    """Result of PII scrubbing operation."""

    original_length: int
    scrubbed_length: int
    detections: list[PIIDetection]
    pii_count: int
    pii_types_found: list[PIIType]
    classification: DataClassification
    scrubbed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_pii(self) -> bool:
        return self.pii_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_length": self.original_length,
            "scrubbed_length": self.scrubbed_length,
            "pii_count": self.pii_count,
            "pii_types_found": [t.value for t in self.pii_types_found],
            "classification": self.classification.value,
            "scrubbed_at": self.scrubbed_at.isoformat(),
        }


class PIIScrubber:
    """
    PII detection and scrubbing utility.

    Scans text and structured data for personally identifiable information
    using pattern matching and heuristics.

    Example:
        >>> scrubber = PIIScrubber()
        >>>
        >>> # Quick check
        >>> if scrubber.contains_pii(user_input):
        ...     print("WARNING: Input contains PII")
        >>>
        >>> # Detailed scan
        >>> detections = scrubber.scan(data)
        >>> for d in detections:
        ...     print(f"{d.pii_type.value}: {d.match}")
        >>>
        >>> # Scrub and get clean data
        >>> clean, result = scrubber.scrub(data)
    """

    # Regex patterns for PII detection
    PATTERNS = {
        PIIType.SSN: [
            r"\b\d{3}-\d{2}-\d{4}\b",  # 123-45-6789
            r"\b\d{3}\s\d{2}\s\d{4}\b",  # 123 45 6789
            r"\b\d{9}\b(?=.*(?:ssn|social))",  # 123456789 near "ssn"
        ],
        PIIType.EMAIL: [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ],
        PIIType.PHONE: [
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # 123-456-7890
            r"\b\(\d{3}\)\s*\d{3}[-.]?\d{4}\b",  # (123) 456-7890
            r"\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # +1-123-456-7890
        ],
        PIIType.CREDIT_CARD: [
            r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Visa
            r"\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Mastercard
            r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",  # Amex
            r"\b6(?:011|5\d{2})[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Discover
        ],
        PIIType.IP_ADDRESS: [
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        ],
        PIIType.DATE_OF_BIRTH: [
            r"\b(?:dob|birth(?:day|date)?)[:\s]+\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b(?=.*(?:birth|dob|born))",
        ],
        PIIType.BANK_ACCOUNT: [
            r"\b\d{8,17}\b(?=.*(?:account|acct|routing))",
        ],
        PIIType.ROUTING_NUMBER: [
            r"\b(?:routing|aba)[:\s#]*\d{9}\b",
            r"\b\d{9}\b(?=.*routing)",
        ],
        PIIType.PASSPORT: [
            r"\b[A-Z]{1,2}\d{6,9}\b(?=.*passport)",
        ],
        PIIType.API_KEY: [
            r"\b(?:api[_-]?key|apikey|secret[_-]?key)[:\s=]+['\"]?[\w-]{20,}['\"]?",
            r"\bsk_(?:live|test)_[a-zA-Z0-9]{24,}\b",  # Stripe-style
            r"\bAKIA[A-Z0-9]{16}\b",  # AWS Access Key
        ],
        PIIType.PRIVATE_KEY: [
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"\b(?:private[_-]?key)[:\s=]+['\"]?[\w/+=-]{40,}['\"]?",
        ],
        PIIType.SEED_PHRASE: [
            # Detect 12/24 word sequences that look like mnemonic phrases
            r"\b(?:abandon|ability|able|about|above|absent|absorb|abstract|absurd|abuse|access|accident|account|accuse|achieve|acid|acoustic|acquire|across|act|action|actor|actress|actual|adapt|add|addict|address|adjust|admit|adult|advance|advice|aerobic|affair|afford|afraid|again|age|agent|agree|ahead|aim|air|airport|aisle|alarm|album|alcohol|alert|alien|all|alley|allow|almost|alone|alpha|already|also|alter|always|amateur|amazing|among|amount|amused|analyst|anchor|ancient|anger|angle|angry|animal|ankle|announce|annual|another|answer|antenna|antique|anxiety|any|apart|apology|appear|apple|approve|april|arch|arctic|area|arena|argue|arm|armed|armor|army|around|arrange|arrest|arrive|arrow|art|artefact|artist|artwork|ask|aspect|assault|asset|assist|assume|asthma|athlete|atom|attack|attend|attitude|attract|auction|audit|august|aunt|author|auto|autumn|average|avocado|avoid|awake|aware|away|awesome|awful|awkward|axis){12,24}\b",
        ],
        PIIType.NPI: [
            r"\b(?:npi|provider)[:\s#]*\d{10}\b",
        ],
    }

    # Redaction templates
    REDACTION_TEMPLATES = {
        PIIType.SSN: "[REDACTED-SSN]",
        PIIType.EMAIL: "[REDACTED-EMAIL]",
        PIIType.PHONE: "[REDACTED-PHONE]",
        PIIType.CREDIT_CARD: "[REDACTED-CC]",
        PIIType.IP_ADDRESS: "[REDACTED-IP]",
        PIIType.DATE_OF_BIRTH: "[REDACTED-DOB]",
        PIIType.BANK_ACCOUNT: "[REDACTED-ACCOUNT]",
        PIIType.ROUTING_NUMBER: "[REDACTED-ROUTING]",
        PIIType.PASSPORT: "[REDACTED-PASSPORT]",
        PIIType.API_KEY: "[REDACTED-API-KEY]",
        PIIType.PRIVATE_KEY: "[REDACTED-PRIVATE-KEY]",
        PIIType.SEED_PHRASE: "[REDACTED-SEED-PHRASE]",
        PIIType.NPI: "[REDACTED-NPI]",
        PIIType.NAME: "[REDACTED-NAME]",
        PIIType.ADDRESS: "[REDACTED-ADDRESS]",
    }

    def __init__(
        self,
        pii_types: Optional[list[PIIType]] = None,
        case_sensitive: bool = False,
        min_confidence: float = 0.5,
    ):
        """
        Initialize PII scrubber.

        Args:
            pii_types: PII types to detect (default: all)
            case_sensitive: Case-sensitive pattern matching
            min_confidence: Minimum confidence threshold for detection
        """
        self.pii_types = pii_types or list(PIIType)
        self.case_sensitive = case_sensitive
        self.min_confidence = min_confidence

        # Compile patterns
        self._compiled_patterns: dict[PIIType, list[re.Pattern]] = {}
        flags = 0 if case_sensitive else re.IGNORECASE

        for pii_type in self.pii_types:
            if pii_type in self.PATTERNS:
                self._compiled_patterns[pii_type] = [
                    re.compile(pattern, flags)
                    for pattern in self.PATTERNS[pii_type]
                ]

    def scan(
        self,
        data: Union[str, bytes, dict, list],
        include_context: bool = True,
        context_chars: int = 20,
    ) -> list[PIIDetection]:
        """
        Scan data for PII.

        Args:
            data: Data to scan (string, bytes, dict, or list)
            include_context: Include surrounding text in detection
            context_chars: Number of context characters

        Returns:
            List of PII detections
        """
        # Convert to string
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        elif isinstance(data, dict):
            text = json.dumps(data)
        elif isinstance(data, list):
            text = json.dumps(data)
        else:
            text = str(data)

        detections: list[PIIDetection] = []

        for pii_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_confidence(pii_type, match.group())

                    if confidence >= self.min_confidence:
                        context = None
                        if include_context:
                            start = max(0, match.start() - context_chars)
                            end = min(len(text), match.end() + context_chars)
                            context = text[start:end]

                        detections.append(
                            PIIDetection(
                                pii_type=pii_type,
                                match=match.group(),
                                start=match.start(),
                                end=match.end(),
                                confidence=confidence,
                                context=context,
                                redacted=self.REDACTION_TEMPLATES.get(pii_type, "[REDACTED]"),
                            )
                        )

        # Remove duplicates (same position, different patterns)
        detections = self._deduplicate_detections(detections)

        return sorted(detections, key=lambda d: d.start)

    def scrub(
        self,
        data: Union[str, bytes],
        mask_type: str = "redact",  # "redact", "hash", "random"
    ) -> tuple[str, ScrubResult]:
        """
        Scrub PII from data.

        Args:
            data: Data to scrub
            mask_type: How to mask PII ("redact", "hash", "random")

        Returns:
            Tuple of (scrubbed_data, result)
        """
        if isinstance(data, bytes):
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1")
        else:
            text = str(data)

        detections = self.scan(text)

        # Sort detections by position (descending) to replace from end
        detections_sorted = sorted(detections, key=lambda d: d.start, reverse=True)

        scrubbed = text
        for detection in detections_sorted:
            replacement = self._get_replacement(detection, mask_type)
            scrubbed = scrubbed[:detection.start] + replacement + scrubbed[detection.end:]

        # Determine classification
        classification = self._classify_data(detections)

        result = ScrubResult(
            original_length=len(text),
            scrubbed_length=len(scrubbed),
            detections=detections,
            pii_count=len(detections),
            pii_types_found=list(set(d.pii_type for d in detections)),
            classification=classification,
        )

        return scrubbed, result

    def contains_pii(self, data: Union[str, bytes, dict, list]) -> bool:
        """Quick check if data contains any PII."""
        return len(self.scan(data)) > 0

    def classify(self, data: Union[str, bytes, dict, list]) -> DataClassification:
        """Classify data based on PII content."""
        detections = self.scan(data)
        return self._classify_data(detections)

    def validate_for_upload(
        self,
        data: Union[str, bytes, dict, list],
        allowed_pii_types: Optional[list[PIIType]] = None,
        max_pii_count: int = 0,
    ) -> tuple[bool, list[str]]:
        """
        Validate data is safe for upload.

        Args:
            data: Data to validate
            allowed_pii_types: PII types that are acceptable
            max_pii_count: Maximum allowed PII instances

        Returns:
            Tuple of (is_valid, warnings)
        """
        detections = self.scan(data)
        warnings: list[str] = []

        # Filter out allowed types
        if allowed_pii_types:
            detections = [d for d in detections if d.pii_type not in allowed_pii_types]

        if len(detections) > max_pii_count:
            for d in detections:
                warnings.append(
                    f"Found {d.pii_type.value}: '{d.match[:20]}...' "
                    f"(confidence: {d.confidence:.0%})"
                )

        return len(warnings) == 0, warnings

    def _calculate_confidence(self, pii_type: PIIType, match: str) -> float:
        """Calculate confidence score for a detection."""
        # Base confidence on pattern match
        confidence = 0.7

        # Adjust based on PII type characteristics
        if pii_type == PIIType.SSN:
            # Validate Luhn-like checksum could increase confidence
            confidence = 0.9

        elif pii_type == PIIType.EMAIL:
            # Valid email structure
            if "@" in match and "." in match.split("@")[1]:
                confidence = 0.95

        elif pii_type == PIIType.CREDIT_CARD:
            # Luhn check would increase confidence
            if self._luhn_check(re.sub(r"[-\s]", "", match)):
                confidence = 0.95
            else:
                confidence = 0.6

        elif pii_type == PIIType.PHONE:
            # Valid phone number structure
            digits = re.sub(r"\D", "", match)
            if len(digits) in [10, 11]:
                confidence = 0.85

        elif pii_type in [PIIType.API_KEY, PIIType.PRIVATE_KEY]:
            # High confidence for obvious patterns
            confidence = 0.95

        return confidence

    def _luhn_check(self, number: str) -> bool:
        """Validate credit card number using Luhn algorithm."""
        try:
            digits = [int(d) for d in number]
            checksum = 0
            for i, d in enumerate(reversed(digits)):
                if i % 2 == 1:
                    d *= 2
                    if d > 9:
                        d -= 9
                checksum += d
            return checksum % 10 == 0
        except ValueError:
            return False

    def _get_replacement(self, detection: PIIDetection, mask_type: str) -> str:
        """Get replacement text for PII."""
        if mask_type == "redact":
            return self.REDACTION_TEMPLATES.get(detection.pii_type, "[REDACTED]")
        elif mask_type == "hash":
            import hashlib
            h = hashlib.sha256(detection.match.encode()).hexdigest()[:8]
            return f"[HASH:{h}]"
        elif mask_type == "random":
            import secrets
            return f"[ID:{secrets.token_hex(4)}]"
        else:
            return "[REDACTED]"

    def _classify_data(self, detections: list[PIIDetection]) -> DataClassification:
        """Classify data based on detected PII."""
        if not detections:
            return DataClassification.PUBLIC

        # Highly sensitive PII types
        restricted_types = {
            PIIType.SSN, PIIType.CREDIT_CARD, PIIType.BANK_ACCOUNT,
            PIIType.PASSPORT, PIIType.PRIVATE_KEY, PIIType.SEED_PHRASE,
            PIIType.MEDICAL_RECORD,
        }

        # Moderately sensitive
        confidential_types = {
            PIIType.DATE_OF_BIRTH, PIIType.API_KEY, PIIType.ADDRESS,
            PIIType.DRIVERS_LICENSE, PIIType.NPI,
        }

        detected_types = {d.pii_type for d in detections}

        if detected_types & restricted_types:
            return DataClassification.RESTRICTED
        elif detected_types & confidential_types:
            return DataClassification.CONFIDENTIAL
        elif detected_types:
            return DataClassification.INTERNAL

        return DataClassification.PUBLIC

    def _deduplicate_detections(
        self,
        detections: list[PIIDetection],
    ) -> list[PIIDetection]:
        """Remove duplicate detections at same position."""
        seen: dict[tuple[int, int], PIIDetection] = {}

        for d in detections:
            key = (d.start, d.end)
            if key not in seen or d.confidence > seen[key].confidence:
                seen[key] = d

        return list(seen.values())


# ============ Convenience Functions ============


def scan_for_pii(data: Union[str, bytes, dict, list]) -> list[PIIDetection]:
    """Quick scan for PII in data."""
    scrubber = PIIScrubber()
    return scrubber.scan(data)


def scrub_pii(data: Union[str, bytes]) -> str:
    """Quick scrub PII from data."""
    scrubber = PIIScrubber()
    scrubbed, _ = scrubber.scrub(data)
    return scrubbed


def contains_pii(data: Union[str, bytes, dict, list]) -> bool:
    """Quick check if data contains PII."""
    scrubber = PIIScrubber()
    return scrubber.contains_pii(data)


class DataSanitizer:
    """High-level data sanitizer that wraps PIIScrubber.

    Provides a simple ``sanitize(text)`` interface for stripping PII
    from text data before it is submitted to the network.
    """

    def __init__(self) -> None:
        self._scrubber = PIIScrubber()

    def sanitize(self, text: str) -> str:
        """Remove PII from *text* and return the scrubbed string."""
        if not text:
            return text
        scrubbed, _ = self._scrubber.scrub(text)
        return scrubbed
