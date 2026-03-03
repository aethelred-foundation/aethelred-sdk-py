"""Verification module for Aethelred SDK."""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from aethelred.core.types import ProofSystem, TEEAttestation, TEEPlatform, ZKMLProof

if TYPE_CHECKING:
    from aethelred.core.client import AsyncAethelredClient, AethelredClient

class VerifyZKProofResponse:
    def __init__(self, valid: bool, verification_time_ms: int = 0, error: Optional[str] = None):
        self.valid = valid
        self.verification_time_ms = verification_time_ms
        self.error = error

class VerifyTEEResponse:
    def __init__(self, valid: bool, platform: TEEPlatform = TEEPlatform.UNSPECIFIED, error: Optional[str] = None):
        self.valid = valid
        self.platform = platform
        self.error = error

class VerificationModule:
    """Async module for verification operations."""
    BASE_PATH = "/aethelred/verify/v1"
    
    def __init__(self, client: "AsyncAethelredClient"):
        self._client = client
    
    async def verify_zk_proof(
        self, proof: bytes, public_inputs: list, verifying_key_hash: bytes,
        proof_system: ProofSystem = ProofSystem.GROTH16,
    ) -> VerifyZKProofResponse:
        """Verify a zero-knowledge proof."""
        data = await self._client.post(
            f"{self.BASE_PATH}/zkproofs:verify",
            json={
                "proof": proof.hex(), "public_inputs": [p.hex() for p in public_inputs],
                "verifying_key_hash": verifying_key_hash.hex(), "proof_system": proof_system.value,
            },
        )
        return VerifyZKProofResponse(**data)
    
    async def verify_tee_attestation(
        self, attestation: TEEAttestation, expected_enclave_hash: Optional[bytes] = None,
    ) -> VerifyTEEResponse:
        """Verify a TEE attestation."""
        data = await self._client.post(
            f"{self.BASE_PATH}/tee/attestation:verify",
            json={
                "attestation": attestation.model_dump(mode="json"),
                "expected_enclave_hash": expected_enclave_hash.hex() if expected_enclave_hash else None,
            },
        )
        return VerifyTEEResponse(**data)

class SyncVerificationModule:
    """Synchronous wrapper for VerificationModule."""
    def __init__(self, client: "AethelredClient", async_module: VerificationModule):
        self._client = client
        self._async = async_module
    def verify_zk_proof(self, *args, **kwargs) -> VerifyZKProofResponse:
        return self._client._run(self._async.verify_zk_proof(*args, **kwargs))
    def verify_tee_attestation(self, *args, **kwargs) -> VerifyTEEResponse:
        return self._client._run(self._async.verify_tee_attestation(*args, **kwargs))

__all__ = ["VerificationModule", "SyncVerificationModule", "VerifyZKProofResponse", "VerifyTEEResponse"]
