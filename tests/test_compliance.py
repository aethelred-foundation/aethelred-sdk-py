"""Tests for compliance module — checker, sanitizer, scrubber."""

from __future__ import annotations

import pytest

from aethelred.compliance.checker import ComplianceChecker
from aethelred.compliance.sanitizer import DataSanitizer


class TestComplianceChecker:
    """Test compliance framework checker."""

    def test_init(self) -> None:
        checker = ComplianceChecker()
        assert checker is not None

    def test_supported_frameworks(self) -> None:
        checker = ComplianceChecker()
        frameworks = checker.supported_frameworks()
        assert isinstance(frameworks, (list, set))
        assert len(frameworks) > 0

    def test_check_returns_result(self) -> None:
        checker = ComplianceChecker()
        result = checker.check(data=b"test data", framework="HIPAA")
        assert result is not None


class TestDataSanitizer:
    """Test data sanitization for PII removal."""

    def test_init(self) -> None:
        sanitizer = DataSanitizer()
        assert sanitizer is not None

    def test_sanitize_text(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("My name is John Doe and my email is john@example.com")
        assert isinstance(result, str)

    def test_sanitize_empty(self) -> None:
        sanitizer = DataSanitizer()
        result = sanitizer.sanitize("")
        assert result == ""
