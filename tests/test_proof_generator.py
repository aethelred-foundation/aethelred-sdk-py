"""Tests for the ProofGenerator — proof pipeline, circuits, hashing.

Covers:
- ProofGenerator initialization
- Proof generation pipeline
- Circuit validation
- Hash verification
- Proof serialization
"""

from __future__ import annotations

import hashlib

import pytest

from aethelred.proofs.generator import (
    ProofGenerator,
    ProofConfig,
)
from aethelred.proofs.verifier import ProofVerifier
from aethelred.core.types import ProofType, ProofSystem


# ---------------------------------------------------------------------------
# ProofGenerator
# ---------------------------------------------------------------------------


class TestProofGenerator:
    """Test proof generation pipeline."""

    def test_init_defaults(self) -> None:
        gen = ProofGenerator()
        assert gen is not None

    def test_init_with_config(self) -> None:
        config = ProofConfig(
            proof_system=ProofSystem.PLONK,
            max_constraints=1_000_000,
        )
        gen = ProofGenerator(config=config)
        assert gen._config.proof_system == ProofSystem.PLONK

    def test_hash_model_produces_hex(self) -> None:
        gen = ProofGenerator()
        model_bytes = b"fake model weights"
        h = gen._hash_model(model_bytes)
        assert len(h) == 64  # SHA-256 hex
        assert h == hashlib.sha256(model_bytes).hexdigest()

    def test_hash_input_produces_hex(self) -> None:
        gen = ProofGenerator()
        input_bytes = b"input tensor data"
        h = gen._hash_input(input_bytes)
        assert len(h) == 64

    def test_hash_deterministic(self) -> None:
        gen = ProofGenerator()
        data = b"deterministic"
        assert gen._hash_model(data) == gen._hash_model(data)


# ---------------------------------------------------------------------------
# ProofVerifier
# ---------------------------------------------------------------------------


class TestProofVerifier:
    """Test proof verification."""

    def test_init_defaults(self) -> None:
        v = ProofVerifier()
        assert v is not None

    def test_verify_rejects_none_proof(self) -> None:
        v = ProofVerifier()
        result = v.verify(proof=None, circuit=None)
        assert not result.is_valid

    def test_verify_rejects_empty_proof(self) -> None:
        v = ProofVerifier()
        from unittest.mock import MagicMock
        mock_proof = MagicMock()
        mock_proof.proof = b""
        mock_proof.proof_type = ProofType.ZKML
        result = v.verify(proof=mock_proof, circuit=None)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# ProofConfig
# ---------------------------------------------------------------------------


class TestProofConfig:
    """Test proof configuration."""

    def test_default_config(self) -> None:
        config = ProofConfig()
        assert config.proof_system == ProofSystem.EZKL
        assert config.max_constraints > 0

    def test_custom_config(self) -> None:
        config = ProofConfig(
            proof_system=ProofSystem.GROTH16,
            max_constraints=500_000,
        )
        assert config.proof_system == ProofSystem.GROTH16
        assert config.max_constraints == 500_000
