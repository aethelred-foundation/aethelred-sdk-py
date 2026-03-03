"""Comprehensive tests for aethelred/verification/__init__.py — targeting 95%+ coverage.

Covers: VerifyZKProofResponse, VerifyTEEResponse, VerificationModule (verify_zk_proof,
verify_tee_attestation with/without expected_enclave_hash), and SyncVerificationModule.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from aethelred.core.types import ProofSystem, TEEAttestation, TEEPlatform
from aethelred.verification import (
    SyncVerificationModule,
    VerificationModule,
    VerifyTEEResponse,
    VerifyZKProofResponse,
)


# ---------------------------------------------------------------------------
# Response model tests
# ---------------------------------------------------------------------------


class TestVerifyZKProofResponse:
    """Test VerifyZKProofResponse construction."""

    def test_valid_response(self) -> None:
        resp = VerifyZKProofResponse(valid=True, verification_time_ms=42)
        assert resp.valid is True
        assert resp.verification_time_ms == 42
        assert resp.error is None

    def test_invalid_response_with_error(self) -> None:
        resp = VerifyZKProofResponse(valid=False, verification_time_ms=0, error="bad proof")
        assert resp.valid is False
        assert resp.error == "bad proof"

    def test_defaults(self) -> None:
        resp = VerifyZKProofResponse(valid=True)
        assert resp.verification_time_ms == 0
        assert resp.error is None


class TestVerifyTEEResponse:
    """Test VerifyTEEResponse construction."""

    def test_valid_response(self) -> None:
        resp = VerifyTEEResponse(valid=True, platform=TEEPlatform.INTEL_SGX)
        assert resp.valid is True
        assert resp.platform == TEEPlatform.INTEL_SGX
        assert resp.error is None

    def test_invalid_response(self) -> None:
        resp = VerifyTEEResponse(valid=False, error="bad attestation")
        assert resp.valid is False
        assert resp.error == "bad attestation"

    def test_defaults(self) -> None:
        resp = VerifyTEEResponse(valid=True)
        assert resp.platform == TEEPlatform.UNSPECIFIED
        assert resp.error is None


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------


class StubClient:
    """Minimal async stub for VerificationModule."""

    def __init__(self) -> None:
        self.posts: List[Tuple[str, Any]] = []

    async def post(self, path: str, json: Any = None) -> Dict[str, Any]:
        self.posts.append((path, json))
        if "zkproofs:verify" in path:
            return {"valid": True, "verification_time_ms": 15}
        if "tee/attestation:verify" in path:
            return {"valid": True, "platform": TEEPlatform.AWS_NITRO.value}
        return {}


# ---------------------------------------------------------------------------
# VerificationModule (async)
# ---------------------------------------------------------------------------


class TestVerificationModule:
    """Test async VerificationModule operations."""

    @pytest.fixture
    def stub(self) -> StubClient:
        return StubClient()

    @pytest.fixture
    def module(self, stub: StubClient) -> VerificationModule:
        return VerificationModule(stub)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_verify_zk_proof(self, module: VerificationModule, stub: StubClient) -> None:
        resp = await module.verify_zk_proof(
            proof=b"\x01\x02\x03",
            public_inputs=[b"\x04" * 32],
            verifying_key_hash=b"\x05" * 32,
            proof_system=ProofSystem.GROTH16,
        )
        assert isinstance(resp, VerifyZKProofResponse)
        assert resp.valid is True
        assert resp.verification_time_ms == 15
        # Verify the request payload
        posted = stub.posts[0][1]
        assert posted["proof_system"] == ProofSystem.GROTH16.value
        assert posted["proof"] == b"\x01\x02\x03".hex()

    @pytest.mark.asyncio
    async def test_verify_zk_proof_plonk(self, module: VerificationModule) -> None:
        resp = await module.verify_zk_proof(
            proof=b"\xaa",
            public_inputs=[b"\xbb" * 16, b"\xcc" * 16],
            verifying_key_hash=b"\xdd" * 32,
            proof_system=ProofSystem.PLONK,
        )
        assert resp.valid is True

    @pytest.mark.asyncio
    async def test_verify_tee_attestation_with_enclave_hash(self, module: VerificationModule, stub: StubClient) -> None:
        att = TEEAttestation(
            platform=TEEPlatform.AWS_NITRO,
            quote=b"\x05" * 64,
            enclave_hash=b"\x06" * 32,
        )
        resp = await module.verify_tee_attestation(
            attestation=att,
            expected_enclave_hash=b"\x07" * 32,
        )
        assert isinstance(resp, VerifyTEEResponse)
        assert resp.valid is True
        # Verify enclave hash was sent
        posted = stub.posts[0][1]
        assert posted["expected_enclave_hash"] == (b"\x07" * 32).hex()

    @pytest.mark.asyncio
    async def test_verify_tee_attestation_without_enclave_hash(self, module: VerificationModule, stub: StubClient) -> None:
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_TDX,
            quote=b"\x08" * 64,
            enclave_hash=b"\x09" * 32,
        )
        resp = await module.verify_tee_attestation(attestation=att)
        assert resp.valid is True
        # Verify enclave hash is None
        posted = stub.posts[0][1]
        assert posted["expected_enclave_hash"] is None


# ---------------------------------------------------------------------------
# SyncVerificationModule
# ---------------------------------------------------------------------------


class TestSyncVerificationModule:
    """Test synchronous wrapper for VerificationModule."""

    @pytest.fixture
    def sync_module(self) -> SyncVerificationModule:
        stub = StubClient()
        async_mod = VerificationModule(stub)  # type: ignore[arg-type]
        sync_client = MagicMock()
        sync_client._run = lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        return SyncVerificationModule(sync_client, async_mod)

    def test_verify_zk_proof(self, sync_module: SyncVerificationModule) -> None:
        resp = sync_module.verify_zk_proof(
            proof=b"\x01",
            public_inputs=[b"\x02" * 32],
            verifying_key_hash=b"\x03" * 32,
        )
        assert resp.valid is True

    def test_verify_tee_attestation(self, sync_module: SyncVerificationModule) -> None:
        att = TEEAttestation(
            platform=TEEPlatform.AMD_SEV,
            quote=b"\x04" * 64,
            enclave_hash=b"\x05" * 32,
        )
        resp = sync_module.verify_tee_attestation(attestation=att)
        assert resp.valid is True
