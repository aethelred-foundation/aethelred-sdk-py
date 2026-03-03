"""Tests for core.types — Pydantic data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aethelred.core.types import (
    ComputeJob,
    DigitalSeal,
    JobStatus,
    SealStatus,
    ProofType,
    TEEPlatform,
    SubmitJobRequest,
    PageRequest,
    ValidatorStats,
    RegulatoryInfo,
)


class TestJobStatus:
    """Test job status enum."""

    def test_all_statuses_exist(self) -> None:
        assert len(JobStatus) >= 7

    def test_pending(self) -> None:
        assert JobStatus.PENDING.value == "JOB_STATUS_PENDING"


class TestComputeJob:
    """Test ComputeJob model."""

    def test_create_minimal(self) -> None:
        job = ComputeJob(
            id="job_1",
            creator="aeth1creator",
            model_hash=b"\x00" * 32,
            input_hash=b"\x01" * 32,
        )
        assert job.status == JobStatus.PENDING
        assert job.priority == 1

    def test_priority_validation(self) -> None:
        with pytest.raises(Exception):
            ComputeJob(
                id="j1", creator="a", model_hash=b"x",
                input_hash=b"y", priority=99,  # >10 should fail
            )

    def test_defaults(self) -> None:
        job = ComputeJob(id="j", creator="a", model_hash=b"x", input_hash=b"y")
        assert job.proof_type == ProofType.TEE
        assert job.timeout_blocks == 100


class TestDigitalSeal:
    """Test DigitalSeal model."""

    def test_create(self) -> None:
        seal = DigitalSeal(id="s1", job_id="j1", model_hash=b"\xaa" * 32)
        assert seal.status == SealStatus.ACTIVE
        assert seal.validators == []

    def test_expiry(self) -> None:
        seal = DigitalSeal(id="s1", job_id="j1", model_hash=b"\xaa" * 32)
        assert seal.expires_at is None  # No expiry by default


class TestSubmitJobRequest:
    """Test job submission request."""

    def test_defaults(self) -> None:
        req = SubmitJobRequest(model_hash=b"\x00" * 32, input_hash=b"\x01" * 32)
        assert req.proof_type == ProofType.TEE
        assert req.priority == 1
        assert req.metadata == {}


class TestPageRequest:
    """Test pagination."""

    def test_defaults(self) -> None:
        page = PageRequest()
        assert page.limit == 100
        assert page.offset == 0
        assert page.reverse is False


class TestRegulatoryInfo:
    """Test regulatory compliance types."""

    def test_create(self) -> None:
        info = RegulatoryInfo(
            jurisdiction="US",
            compliance_frameworks=["HIPAA", "SOC2"],
        )
        assert "HIPAA" in info.compliance_frameworks


class TestValidatorStats:
    """Test validator statistics."""

    def test_defaults(self) -> None:
        stats = ValidatorStats(address="aeth1validator")
        assert stats.jobs_completed == 0
        assert stats.reputation_score == 0.0
