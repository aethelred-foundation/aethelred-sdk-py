"""Tests for core.exceptions — error hierarchy."""

from __future__ import annotations

import pytest

from aethelred.core.exceptions import (
    AethelredError,
    ProofError,
    ValidationError,
)


class TestExceptionHierarchy:
    """Test error class hierarchy."""

    def test_base_error(self) -> None:
        with pytest.raises(AethelredError):
            raise AethelredError("base error")

    def test_proof_error_is_aethelred_error(self) -> None:
        with pytest.raises(AethelredError):
            raise ProofError("proof failed")

    def test_validation_error_is_aethelred_error(self) -> None:
        with pytest.raises(AethelredError):
            raise ValidationError("invalid input")

    def test_proof_error_message(self) -> None:
        error = ProofError("bad proof")
        assert "bad proof" in str(error)

    def test_validation_error_message(self) -> None:
        error = ValidationError("invalid field")
        assert "invalid field" in str(error)


class TestErrorCatching:
    """Test that errors can be caught at different levels."""

    def test_catch_specific(self) -> None:
        try:
            raise ProofError("specific")
        except ProofError as e:
            assert "specific" in str(e)

    def test_catch_base(self) -> None:
        try:
            raise ValidationError("caught by base")
        except AethelredError as e:
            assert "caught by base" in str(e)

    def test_not_caught_by_wrong_type(self) -> None:
        with pytest.raises(ProofError):
            try:
                raise ProofError("wrong catch")
            except ValidationError:
                pass  # Should not catch ProofError
