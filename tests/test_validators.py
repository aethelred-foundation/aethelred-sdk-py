"""Tests for validators module."""

from __future__ import annotations

import pytest

from aethelred.core.types import ValidatorStats, HardwareCapability, TEEPlatform


class TestValidatorStats:
    """Test validator data types."""

    def test_create(self) -> None:
        stats = ValidatorStats(
            address="aeth1val123",
            jobs_completed=100,
            reputation_score=0.95,
        )
        assert stats.jobs_completed == 100
        assert stats.reputation_score == 0.95

    def test_defaults(self) -> None:
        stats = ValidatorStats(address="aeth1v")
        assert stats.slashing_events == 0
        assert stats.average_latency_ms == 0


class TestHardwareCapability:
    """Test hardware capability model."""

    def test_create(self) -> None:
        cap = HardwareCapability(
            tee_platforms=[TEEPlatform.INTEL_SGX, TEEPlatform.AWS_NITRO],
            zkml_supported=True,
            max_model_size_mb=4096,
            gpu_memory_gb=80,
        )
        assert len(cap.tee_platforms) == 2
        assert cap.zkml_supported is True
        assert cap.gpu_memory_gb == 80

    def test_defaults(self) -> None:
        cap = HardwareCapability()
        assert cap.tee_platforms == []
        assert cap.zkml_supported is False
