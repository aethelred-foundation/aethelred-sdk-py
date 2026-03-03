"""Tests for verification module — VerificationResult types."""

from __future__ import annotations

import pytest

from aethelred.core.types import (
    VerificationResult,
    ProofType,
    TEEAttestation,
    TEEPlatform,
    ZKMLProof,
    ProofSystem,
)


class TestVerificationResult:
    """Test verification result model."""

    def test_valid_result(self) -> None:
        result = VerificationResult(
            valid=True,
            output_hash=b"\xaa" * 32,
            proof_type=ProofType.TEE,
            consensus_validators=3,
            total_voting_power=100,
        )
        assert result.valid is True
        assert result.consensus_validators == 3

    def test_invalid_result(self) -> None:
        result = VerificationResult(valid=False, output_hash=b"\x00" * 32)
        assert result.valid is False
        assert result.proof_type == ProofType.UNSPECIFIED


class TestTEEAttestation:
    """Test TEE attestation model."""

    def test_create(self) -> None:
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            quote=b"\x01" * 64,
            enclave_hash=b"\x02" * 32,
        )
        assert att.platform == TEEPlatform.INTEL_SGX

    def test_defaults(self) -> None:
        att = TEEAttestation()
        assert att.platform == TEEPlatform.UNSPECIFIED
        assert att.quote == b""


class TestZKMLProof:
    """Test ZKML proof model."""

    def test_create(self) -> None:
        proof = ZKMLProof(
            proof_system=ProofSystem.PLONK,
            proof=b"\x03" * 128,
            public_inputs=[b"\x04" * 32],
        )
        assert proof.proof_system == ProofSystem.PLONK
        assert len(proof.proof) == 128

    def test_defaults(self) -> None:
        proof = ZKMLProof()
        assert proof.proof_system == ProofSystem.UNSPECIFIED
        assert proof.proof == b""
        assert proof.public_inputs == []
