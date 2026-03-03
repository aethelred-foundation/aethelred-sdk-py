"""
Enterprise PII Scrubber with Microsoft Presidio Integration

Provides enterprise-grade Personally Identifiable Information (PII) detection
and redaction using Microsoft Presidio. Supports 50+ entity types across
multiple languages with configurable anonymization strategies.

Key Features:
- Multi-language PII detection (English, Arabic, Chinese, etc.)
- 50+ built-in entity recognizers
- Custom entity recognizer support
- Multiple anonymization operators (mask, hash, encrypt, replace)
- GDPR, HIPAA, PCI-DSS compliance support
- Batch processing with async support
- Audit trail generation

Example:
    >>> from aethelred.compliance.presidio_scrubber import PIIScrubber
    >>>
    >>> scrubber = PIIScrubber(
    ...     language="en",
    ...     score_threshold=0.7,
    ...     compliance_frameworks=["HIPAA", "GDPR"]
    ... )
    >>>
    >>> # Scrub sensitive data
    >>> result = await scrubber.scrub(
    ...     "John Smith's SSN is 123-45-6789 and email is john@example.com"
    ... )
    >>> print(result.scrubbed_text)
    # "<PERSON>'s SSN is <US_SSN> and email is <EMAIL_ADDRESS>"
    >>>
    >>> # Get detailed findings
    >>> for finding in result.findings:
    ...     print(f"{finding.entity_type}: {finding.original_text}")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple, Union

# Presidio imports (optional)
try:
    from presidio_analyzer import AnalyzerEngine, RecognizerResult, Pattern as PresidioPattern
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.recognizer_registry import RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig, RecognizerResult as AnonymResult
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    AnalyzerEngine = None
    AnonymizerEngine = None

logger = logging.getLogger(__name__)


# ============ Entity Types ============


class PIIEntityType(str, Enum):
    """Standard PII entity types supported by Presidio."""

    # Personal Identification
    PERSON = "PERSON"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    DATE_TIME = "DATE_TIME"

    # Government IDs
    US_SSN = "US_SSN"
    US_PASSPORT = "US_PASSPORT"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_ITIN = "US_ITIN"
    UK_NHS = "UK_NHS"
    UK_NINO = "UK_NINO"

    # Financial
    CREDIT_CARD = "CREDIT_CARD"
    CRYPTO = "CRYPTO"
    IBAN_CODE = "IBAN_CODE"
    US_BANK_NUMBER = "US_BANK_NUMBER"

    # Medical
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    NRP = "NRP"  # Nurse Registration Number

    # Technical
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    DOMAIN_NAME = "DOMAIN_NAME"
    MAC_ADDRESS = "MAC_ADDRESS"

    # Location
    LOCATION = "LOCATION"
    ADDRESS = "ADDRESS"

    # Other
    ORGANIZATION = "ORGANIZATION"
    AGE = "AGE"

    # Aethelred-specific
    AETHEL_ADDRESS = "AETHEL_ADDRESS"  # Aethelred wallet address
    AETHEL_TX_HASH = "AETHEL_TX_HASH"  # Transaction hash


class ComplianceFramework(str, Enum):
    """Regulatory compliance frameworks."""

    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI_DSS"
    CCPA = "CCPA"
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"


class AnonymizationOperator(str, Enum):
    """Anonymization operators for PII redaction."""

    REPLACE = "replace"  # Replace with placeholder
    MASK = "mask"  # Partially mask (e.g., ***-**-6789)
    HASH = "hash"  # Replace with hash
    ENCRYPT = "encrypt"  # Encrypt (reversible)
    REDACT = "redact"  # Remove entirely
    KEEP = "keep"  # Keep original (for audit)


# Entity types required for each compliance framework
COMPLIANCE_ENTITY_MAPPING: Dict[ComplianceFramework, Set[PIIEntityType]] = {
    ComplianceFramework.HIPAA: {
        PIIEntityType.PERSON,
        PIIEntityType.EMAIL_ADDRESS,
        PIIEntityType.PHONE_NUMBER,
        PIIEntityType.DATE_TIME,
        PIIEntityType.US_SSN,
        PIIEntityType.MEDICAL_LICENSE,
        PIIEntityType.IP_ADDRESS,
        PIIEntityType.ADDRESS,
        PIIEntityType.AGE,
    },
    ComplianceFramework.GDPR: {
        PIIEntityType.PERSON,
        PIIEntityType.EMAIL_ADDRESS,
        PIIEntityType.PHONE_NUMBER,
        PIIEntityType.DATE_TIME,
        PIIEntityType.LOCATION,
        PIIEntityType.IP_ADDRESS,
        PIIEntityType.CREDIT_CARD,
        PIIEntityType.IBAN_CODE,
    },
    ComplianceFramework.PCI_DSS: {
        PIIEntityType.CREDIT_CARD,
        PIIEntityType.US_BANK_NUMBER,
        PIIEntityType.IBAN_CODE,
    },
    ComplianceFramework.CCPA: {
        PIIEntityType.PERSON,
        PIIEntityType.EMAIL_ADDRESS,
        PIIEntityType.PHONE_NUMBER,
        PIIEntityType.US_SSN,
        PIIEntityType.US_DRIVER_LICENSE,
        PIIEntityType.CREDIT_CARD,
        PIIEntityType.IP_ADDRESS,
        PIIEntityType.LOCATION,
    },
}


# ============ Data Classes ============


@dataclass
class PIIFinding:
    """Single PII finding in analyzed text."""

    entity_type: str
    start: int
    end: int
    original_text: str
    score: float

    # Anonymization result
    anonymized_text: Optional[str] = None
    operator_used: Optional[AnonymizationOperator] = None

    # Metadata
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    recognizer_name: Optional[str] = None
    analysis_explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize finding for audit."""
        return {
            "finding_id": self.finding_id,
            "entity_type": self.entity_type,
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "operator_used": self.operator_used.value if self.operator_used else None,
            "recognizer_name": self.recognizer_name,
        }


@dataclass
class ScrubResult:
    """Result of PII scrubbing operation."""

    # Input/Output
    original_text: str
    scrubbed_text: str

    # Findings
    findings: List[PIIFinding]
    entity_counts: Dict[str, int]

    # Statistics
    total_findings: int
    unique_entity_types: int
    processing_time_ms: int

    # Compliance
    compliance_passed: bool
    compliance_frameworks_checked: List[str]
    compliance_violations: List[str] = field(default_factory=list)

    # Metadata
    scrub_id: str = field(default_factory=lambda: f"scrub_{uuid.uuid4().hex[:16]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"

    @property
    def has_pii(self) -> bool:
        """Check if any PII was found."""
        return self.total_findings > 0

    @property
    def highest_risk_score(self) -> float:
        """Get highest confidence PII score."""
        if not self.findings:
            return 0.0
        return max(f.score for f in self.findings)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize result for storage/transmission."""
        return {
            "scrub_id": self.scrub_id,
            "timestamp": self.timestamp.isoformat(),
            "language": self.language,
            "total_findings": self.total_findings,
            "unique_entity_types": self.unique_entity_types,
            "entity_counts": self.entity_counts,
            "compliance_passed": self.compliance_passed,
            "compliance_frameworks_checked": self.compliance_frameworks_checked,
            "compliance_violations": self.compliance_violations,
            "processing_time_ms": self.processing_time_ms,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_audit_report(self) -> str:
        """Generate audit report."""
        report = f"""
PII Scrubbing Audit Report
==========================
Scrub ID: {self.scrub_id}
Timestamp: {self.timestamp.isoformat()}
Language: {self.language}

Summary
-------
Total Findings: {self.total_findings}
Unique Entity Types: {self.unique_entity_types}
Highest Risk Score: {self.highest_risk_score:.2f}
Processing Time: {self.processing_time_ms}ms

Entity Breakdown
----------------
"""
        for entity_type, count in sorted(self.entity_counts.items()):
            report += f"  {entity_type}: {count}\n"

        report += f"""
Compliance Status
-----------------
Frameworks Checked: {', '.join(self.compliance_frameworks_checked) or 'None'}
Status: {'PASSED' if self.compliance_passed else 'FAILED'}
"""
        if self.compliance_violations:
            report += "Violations:\n"
            for v in self.compliance_violations:
                report += f"  - {v}\n"

        return report


@dataclass
class BatchScrubResult:
    """Result of batch scrubbing operation."""

    results: List[ScrubResult]
    total_documents: int
    documents_with_pii: int
    total_findings: int
    processing_time_ms: int

    # Aggregate statistics
    entity_counts: Dict[str, int] = field(default_factory=dict)
    compliance_passed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_documents": self.total_documents,
            "documents_with_pii": self.documents_with_pii,
            "total_findings": self.total_findings,
            "processing_time_ms": self.processing_time_ms,
            "entity_counts": self.entity_counts,
            "compliance_passed": self.compliance_passed,
        }


@dataclass
class OperatorConfiguration:
    """Configuration for anonymization operators."""

    default_operator: AnonymizationOperator = AnonymizationOperator.REPLACE

    # Entity-specific operators
    entity_operators: Dict[str, AnonymizationOperator] = field(default_factory=dict)

    # Masking configuration
    mask_char: str = "*"
    mask_from_end: bool = True
    mask_chars_to_show: int = 4

    # Hashing configuration
    hash_algorithm: str = "sha256"
    hash_truncate: int = 8

    # Encryption configuration (for reversible anonymization)
    encryption_key: Optional[bytes] = None

    def get_operator(self, entity_type: str) -> AnonymizationOperator:
        """Get operator for entity type."""
        return self.entity_operators.get(
            entity_type,
            self.default_operator
        )


# ============ Custom Recognizers ============


class CustomRecognizer(ABC):
    """Base class for custom PII recognizers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Recognizer name."""
        pass

    @property
    @abstractmethod
    def supported_entities(self) -> List[str]:
        """List of entity types this recognizer detects."""
        pass

    @property
    def supported_language(self) -> str:
        """Supported language code."""
        return "en"

    @abstractmethod
    def analyze(
        self,
        text: str,
        entities: List[str],
    ) -> List[PIIFinding]:
        """Analyze text for PII."""
        pass


class AethelredAddressRecognizer(CustomRecognizer):
    """Recognizer for Aethelred wallet addresses."""

    # Aethelred addresses: aethel1[39 alphanumeric characters]
    ADDRESS_PATTERN = re.compile(r'\baethel1[a-z0-9]{39}\b', re.IGNORECASE)

    @property
    def name(self) -> str:
        return "AethelredAddressRecognizer"

    @property
    def supported_entities(self) -> List[str]:
        return [PIIEntityType.AETHEL_ADDRESS.value]

    def analyze(
        self,
        text: str,
        entities: List[str],
    ) -> List[PIIFinding]:
        findings = []

        for match in self.ADDRESS_PATTERN.finditer(text):
            findings.append(PIIFinding(
                entity_type=PIIEntityType.AETHEL_ADDRESS.value,
                start=match.start(),
                end=match.end(),
                original_text=match.group(),
                score=0.95,
                recognizer_name=self.name,
            ))

        return findings


class AethelredTxHashRecognizer(CustomRecognizer):
    """Recognizer for Aethelred transaction hashes."""

    # Transaction hashes: 64 hex characters
    TX_HASH_PATTERN = re.compile(r'\b[A-Fa-f0-9]{64}\b')

    @property
    def name(self) -> str:
        return "AethelredTxHashRecognizer"

    @property
    def supported_entities(self) -> List[str]:
        return [PIIEntityType.AETHEL_TX_HASH.value]

    def analyze(
        self,
        text: str,
        entities: List[str],
    ) -> List[PIIFinding]:
        findings = []

        for match in self.TX_HASH_PATTERN.finditer(text):
            findings.append(PIIFinding(
                entity_type=PIIEntityType.AETHEL_TX_HASH.value,
                start=match.start(),
                end=match.end(),
                original_text=match.group(),
                score=0.85,
                recognizer_name=self.name,
            ))

        return findings


# ============ Main PIIScrubber Class ============


class PIIScrubber:
    """
    Enterprise-grade PII scrubber with Microsoft Presidio integration.

    Provides comprehensive PII detection and anonymization with support for
    multiple languages, compliance frameworks, and custom recognizers.

    Example:
        >>> scrubber = PIIScrubber(
        ...     language="en",
        ...     score_threshold=0.7,
        ...     compliance_frameworks=["HIPAA", "GDPR"],
        ... )
        >>>
        >>> result = await scrubber.scrub(
        ...     "Patient John Smith (SSN: 123-45-6789) visited on 2024-01-15"
        ... )
        >>>
        >>> print(result.scrubbed_text)
        >>> print(f"Found {result.total_findings} PII entities")
    """

    def __init__(
        self,
        language: str = "en",
        score_threshold: float = 0.5,
        entities: Optional[List[str]] = None,
        compliance_frameworks: Optional[List[str]] = None,
        operator_config: Optional[OperatorConfiguration] = None,
        custom_recognizers: Optional[List[CustomRecognizer]] = None,
        enable_audit: bool = True,
    ):
        """
        Initialize PII scrubber.

        Args:
            language: Language code for analysis (default: "en")
            score_threshold: Minimum confidence score (0-1) to report findings
            entities: Specific entity types to detect (None = all)
            compliance_frameworks: Compliance frameworks to enforce
            operator_config: Configuration for anonymization operators
            custom_recognizers: Additional custom recognizers
            enable_audit: Enable audit trail generation
        """
        self.language = language
        self.score_threshold = score_threshold
        self.enable_audit = enable_audit

        # Determine entities to detect
        self._entities = self._resolve_entities(entities, compliance_frameworks)

        # Parse compliance frameworks
        self._compliance_frameworks: List[ComplianceFramework] = []
        if compliance_frameworks:
            for fw in compliance_frameworks:
                try:
                    self._compliance_frameworks.append(ComplianceFramework(fw))
                except ValueError:
                    logger.warning(f"Unknown compliance framework: {fw}")

        # Operator configuration
        self._operator_config = operator_config or OperatorConfiguration()

        # Custom recognizers
        self._custom_recognizers: List[CustomRecognizer] = custom_recognizers or []
        self._custom_recognizers.extend([
            AethelredAddressRecognizer(),
            AethelredTxHashRecognizer(),
        ])

        # Initialize Presidio if available
        self._analyzer: Optional[Any] = None
        self._anonymizer: Optional[Any] = None

        if PRESIDIO_AVAILABLE:
            self._initialize_presidio()
        else:
            logger.warning(
                "Microsoft Presidio not installed. Using fallback pattern matching. "
                "Install with: pip install presidio-analyzer presidio-anonymizer"
            )

        # Fallback patterns (used when Presidio unavailable)
        self._fallback_patterns = self._build_fallback_patterns()

    def _resolve_entities(
        self,
        entities: Optional[List[str]],
        compliance_frameworks: Optional[List[str]],
    ) -> Set[str]:
        """Resolve which entities to detect based on config and compliance."""
        resolved: Set[str] = set()

        if entities:
            resolved.update(entities)

        if compliance_frameworks:
            for fw in compliance_frameworks:
                try:
                    framework = ComplianceFramework(fw)
                    if framework in COMPLIANCE_ENTITY_MAPPING:
                        resolved.update(
                            e.value for e in COMPLIANCE_ENTITY_MAPPING[framework]
                        )
                except ValueError:
                    pass

        # Default: detect all if nothing specified
        if not resolved:
            resolved = {e.value for e in PIIEntityType}

        return resolved

    def _initialize_presidio(self) -> None:
        """Initialize Presidio analyzer and anonymizer."""
        try:
            # Configure NLP engine
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": self.language, "model_name": f"{self.language}_core_web_lg"}],
            }

            # Try to create NLP engine provider
            try:
                provider = NlpEngineProvider(nlp_configuration=configuration)
                nlp_engine = provider.create_engine()
            except Exception:
                # Fall back to default
                nlp_engine = None

            # Create analyzer
            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

            # Create anonymizer
            self._anonymizer = AnonymizerEngine()

            logger.info("Presidio initialized successfully")

        except Exception as e:
            logger.warning(f"Failed to initialize Presidio: {e}")
            self._analyzer = None
            self._anonymizer = None

    def _build_fallback_patterns(self) -> Dict[str, Pattern]:
        """Build fallback regex patterns when Presidio unavailable."""
        return {
            PIIEntityType.EMAIL_ADDRESS.value: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIIEntityType.PHONE_NUMBER.value: re.compile(
                r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'
            ),
            PIIEntityType.US_SSN.value: re.compile(
                r'\b[0-9]{3}[-\s]?[0-9]{2}[-\s]?[0-9]{4}\b'
            ),
            PIIEntityType.CREDIT_CARD.value: re.compile(
                r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|'
                r'3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|'
                r'(?:2131|1800|35\d{3})\d{11})\b'
            ),
            PIIEntityType.IP_ADDRESS.value: re.compile(
                r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
                r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
            ),
            PIIEntityType.URL.value: re.compile(
                r'https?://[^\s<>"{}|\\^`\[\]]+'
            ),
            PIIEntityType.DATE_TIME.value: re.compile(
                r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
            ),
        }

    async def scrub(
        self,
        text: str,
        *,
        entities: Optional[List[str]] = None,
        operator: Optional[AnonymizationOperator] = None,
        return_original_on_error: bool = True,
    ) -> ScrubResult:
        """
        Scrub PII from text.

        Args:
            text: Text to scrub
            entities: Override entity types to detect
            operator: Override anonymization operator
            return_original_on_error: Return original text if scrubbing fails

        Returns:
            ScrubResult with scrubbed text and findings

        Example:
            >>> result = await scrubber.scrub(
            ...     "Call John at 555-123-4567 or john@email.com"
            ... )
            >>> print(result.scrubbed_text)
            # "Call <PERSON> at <PHONE_NUMBER> or <EMAIL_ADDRESS>"
        """
        import time
        start_time = time.time()

        target_entities = set(entities) if entities else self._entities

        try:
            # Analyze text for PII
            findings = await self._analyze(text, target_entities)

            # Anonymize
            scrubbed_text, anonymized_findings = await self._anonymize(
                text,
                findings,
                operator,
            )

            # Check compliance
            compliance_passed, violations = self._check_compliance(anonymized_findings)

            # Build entity counts
            entity_counts: Dict[str, int] = {}
            for f in anonymized_findings:
                entity_counts[f.entity_type] = entity_counts.get(f.entity_type, 0) + 1

            processing_time = int((time.time() - start_time) * 1000)

            return ScrubResult(
                original_text=text if self.enable_audit else "",
                scrubbed_text=scrubbed_text,
                findings=anonymized_findings,
                entity_counts=entity_counts,
                total_findings=len(anonymized_findings),
                unique_entity_types=len(entity_counts),
                processing_time_ms=processing_time,
                compliance_passed=compliance_passed,
                compliance_frameworks_checked=[fw.value for fw in self._compliance_frameworks],
                compliance_violations=violations,
                language=self.language,
            )

        except Exception as e:
            logger.error(f"Scrubbing failed: {e}")
            if return_original_on_error:
                return ScrubResult(
                    original_text=text if self.enable_audit else "",
                    scrubbed_text=text,
                    findings=[],
                    entity_counts={},
                    total_findings=0,
                    unique_entity_types=0,
                    processing_time_ms=int((time.time() - start_time) * 1000),
                    compliance_passed=False,
                    compliance_frameworks_checked=[fw.value for fw in self._compliance_frameworks],
                    compliance_violations=[f"Scrubbing error: {e}"],
                    language=self.language,
                )
            raise

    async def scrub_batch(
        self,
        texts: List[str],
        *,
        entities: Optional[List[str]] = None,
        max_concurrent: int = 10,
    ) -> BatchScrubResult:
        """
        Scrub PII from multiple texts concurrently.

        Args:
            texts: List of texts to scrub
            entities: Override entity types to detect
            max_concurrent: Maximum concurrent scrubbing operations

        Returns:
            BatchScrubResult with aggregated results
        """
        import time
        start_time = time.time()

        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrub_with_semaphore(text: str) -> ScrubResult:
            async with semaphore:
                return await self.scrub(text, entities=entities)

        # Run all scrubs concurrently
        results = await asyncio.gather(
            *[scrub_with_semaphore(t) for t in texts],
            return_exceptions=True,
        )

        # Process results
        successful_results: List[ScrubResult] = []
        for r in results:
            if isinstance(r, ScrubResult):
                successful_results.append(r)

        # Aggregate statistics
        total_findings = sum(r.total_findings for r in successful_results)
        documents_with_pii = sum(1 for r in successful_results if r.has_pii)

        entity_counts: Dict[str, int] = {}
        for r in successful_results:
            for entity, count in r.entity_counts.items():
                entity_counts[entity] = entity_counts.get(entity, 0) + count

        compliance_passed = all(r.compliance_passed for r in successful_results)

        processing_time = int((time.time() - start_time) * 1000)

        return BatchScrubResult(
            results=successful_results,
            total_documents=len(texts),
            documents_with_pii=documents_with_pii,
            total_findings=total_findings,
            processing_time_ms=processing_time,
            entity_counts=entity_counts,
            compliance_passed=compliance_passed,
        )

    async def _analyze(
        self,
        text: str,
        entities: Set[str],
    ) -> List[PIIFinding]:
        """Analyze text for PII entities."""
        findings: List[PIIFinding] = []

        # Use Presidio if available
        if self._analyzer:
            findings.extend(await self._analyze_presidio(text, entities))
        else:
            findings.extend(self._analyze_fallback(text, entities))

        # Run custom recognizers
        for recognizer in self._custom_recognizers:
            recognizer_entities = set(recognizer.supported_entities) & entities
            if recognizer_entities:
                try:
                    custom_findings = recognizer.analyze(text, list(recognizer_entities))
                    findings.extend(custom_findings)
                except Exception as e:
                    logger.warning(f"Custom recognizer {recognizer.name} failed: {e}")

        # Filter by score threshold and remove duplicates
        findings = self._filter_and_dedupe(findings)

        return findings

    async def _analyze_presidio(
        self,
        text: str,
        entities: Set[str],
    ) -> List[PIIFinding]:
        """Analyze using Presidio."""
        findings: List[PIIFinding] = []

        try:
            # Run analysis
            results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._analyzer.analyze(
                    text=text,
                    language=self.language,
                    entities=list(entities),
                    score_threshold=self.score_threshold,
                )
            )

            for result in results:
                findings.append(PIIFinding(
                    entity_type=result.entity_type,
                    start=result.start,
                    end=result.end,
                    original_text=text[result.start:result.end],
                    score=result.score,
                    recognizer_name=result.recognition_metadata.get(
                        "recognizer_name", "presidio"
                    ) if hasattr(result, 'recognition_metadata') else "presidio",
                ))

        except Exception as e:
            logger.error(f"Presidio analysis failed: {e}")

        return findings

    def _analyze_fallback(
        self,
        text: str,
        entities: Set[str],
    ) -> List[PIIFinding]:
        """Analyze using fallback regex patterns."""
        findings: List[PIIFinding] = []

        for entity_type, pattern in self._fallback_patterns.items():
            if entity_type in entities:
                for match in pattern.finditer(text):
                    findings.append(PIIFinding(
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        original_text=match.group(),
                        score=0.8,  # Default score for regex matches
                        recognizer_name="fallback_regex",
                    ))

        return findings

    def _filter_and_dedupe(self, findings: List[PIIFinding]) -> List[PIIFinding]:
        """Filter by score and remove overlapping findings."""
        # Filter by score
        filtered = [f for f in findings if f.score >= self.score_threshold]

        # Sort by score (descending) then by position
        filtered.sort(key=lambda f: (-f.score, f.start))

        # Remove overlapping findings (keep highest score)
        deduped: List[PIIFinding] = []
        for finding in filtered:
            overlaps = False
            for existing in deduped:
                if (finding.start < existing.end and finding.end > existing.start):
                    overlaps = True
                    break
            if not overlaps:
                deduped.append(finding)

        # Sort by position for consistent output
        deduped.sort(key=lambda f: f.start)

        return deduped

    async def _anonymize(
        self,
        text: str,
        findings: List[PIIFinding],
        operator_override: Optional[AnonymizationOperator],
    ) -> Tuple[str, List[PIIFinding]]:
        """Anonymize text based on findings."""
        if not findings:
            return text, []

        # Process from end to start to preserve positions
        sorted_findings = sorted(findings, key=lambda f: -f.start)
        result = text
        anonymized_findings: List[PIIFinding] = []

        for finding in sorted_findings:
            operator = operator_override or self._operator_config.get_operator(
                finding.entity_type
            )

            replacement = self._apply_operator(
                finding.original_text,
                finding.entity_type,
                operator,
            )

            result = result[:finding.start] + replacement + result[finding.end:]

            # Update finding
            finding.anonymized_text = replacement
            finding.operator_used = operator
            anonymized_findings.append(finding)

        # Reverse to maintain original order
        anonymized_findings.reverse()

        return result, anonymized_findings

    def _apply_operator(
        self,
        original: str,
        entity_type: str,
        operator: AnonymizationOperator,
    ) -> str:
        """Apply anonymization operator to text."""
        if operator == AnonymizationOperator.REPLACE:
            return f"<{entity_type}>"

        elif operator == AnonymizationOperator.MASK:
            if len(original) <= self._operator_config.mask_chars_to_show:
                return self._operator_config.mask_char * len(original)

            if self._operator_config.mask_from_end:
                visible = original[-self._operator_config.mask_chars_to_show:]
                masked = self._operator_config.mask_char * (len(original) - len(visible))
                return masked + visible
            else:
                visible = original[:self._operator_config.mask_chars_to_show]
                masked = self._operator_config.mask_char * (len(original) - len(visible))
                return visible + masked

        elif operator == AnonymizationOperator.HASH:
            hash_value = hashlib.sha256(original.encode()).hexdigest()
            truncated = hash_value[:self._operator_config.hash_truncate]
            return f"[{entity_type}:{truncated}]"

        elif operator == AnonymizationOperator.REDACT:
            return ""

        elif operator == AnonymizationOperator.ENCRYPT:
            # Simplified encryption (production would use proper encryption)
            if self._operator_config.encryption_key:
                import base64
                encoded = base64.b64encode(original.encode()).decode()
                return f"[ENC:{encoded}]"
            return f"<{entity_type}>"

        elif operator == AnonymizationOperator.KEEP:
            return original

        return f"<{entity_type}>"

    def _check_compliance(
        self,
        findings: List[PIIFinding],
    ) -> Tuple[bool, List[str]]:
        """Check if scrubbing meets compliance requirements."""
        if not self._compliance_frameworks:
            return True, []

        violations: List[str] = []

        for framework in self._compliance_frameworks:
            required_entities = COMPLIANCE_ENTITY_MAPPING.get(framework, set())

            for finding in findings:
                # Check if any unscrubbed PII violates compliance
                if finding.entity_type in {e.value for e in required_entities}:
                    if finding.operator_used == AnonymizationOperator.KEEP:
                        violations.append(
                            f"{framework.value}: Unscrubbed {finding.entity_type} found"
                        )

        return len(violations) == 0, violations

    def add_custom_recognizer(self, recognizer: CustomRecognizer) -> None:
        """Add a custom recognizer."""
        self._custom_recognizers.append(recognizer)

    @property
    def supported_entities(self) -> List[str]:
        """Get list of supported entity types."""
        entities = list(self._entities)

        # Add custom recognizer entities
        for recognizer in self._custom_recognizers:
            entities.extend(recognizer.supported_entities)

        return list(set(entities))


# ============ Factory Function ============


def create_hipaa_scrubber() -> PIIScrubber:
    """Create HIPAA-compliant scrubber."""
    return PIIScrubber(
        compliance_frameworks=["HIPAA"],
        score_threshold=0.5,
        operator_config=OperatorConfiguration(
            default_operator=AnonymizationOperator.REPLACE,
            entity_operators={
                PIIEntityType.DATE_TIME.value: AnonymizationOperator.MASK,
                PIIEntityType.AGE.value: AnonymizationOperator.REPLACE,
            }
        ),
    )


def create_gdpr_scrubber() -> PIIScrubber:
    """Create GDPR-compliant scrubber."""
    return PIIScrubber(
        compliance_frameworks=["GDPR"],
        score_threshold=0.6,
        operator_config=OperatorConfiguration(
            default_operator=AnonymizationOperator.HASH,
        ),
    )


def create_pci_scrubber() -> PIIScrubber:
    """Create PCI-DSS compliant scrubber for payment data."""
    return PIIScrubber(
        compliance_frameworks=["PCI_DSS"],
        score_threshold=0.7,
        entities=[
            PIIEntityType.CREDIT_CARD.value,
            PIIEntityType.US_BANK_NUMBER.value,
            PIIEntityType.IBAN_CODE.value,
        ],
        operator_config=OperatorConfiguration(
            default_operator=AnonymizationOperator.MASK,
            mask_chars_to_show=4,
        ),
    )


# ============ Exports ============

__all__ = [
    # Main class
    "PIIScrubber",
    # Data classes
    "PIIFinding",
    "ScrubResult",
    "BatchScrubResult",
    "OperatorConfiguration",
    # Enums
    "PIIEntityType",
    "ComplianceFramework",
    "AnonymizationOperator",
    # Custom recognizers
    "CustomRecognizer",
    "AethelredAddressRecognizer",
    "AethelredTxHashRecognizer",
    # Factory functions
    "create_hipaa_scrubber",
    "create_gdpr_scrubber",
    "create_pci_scrubber",
    # Constants
    "COMPLIANCE_ENTITY_MAPPING",
    "PRESIDIO_AVAILABLE",
]
