"""Modern SDK tests for current client/jobs/seals/verification surfaces."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from aethelred.core.client import AsyncAethelredClient
from aethelred.core.config import Config, SecretStr
from aethelred.core.types import ProofSystem, ProofType, TEEAttestation, TEEPlatform
from aethelred.jobs import JobsModule
from aethelred.seals import SealsModule
from aethelred.verification import VerificationModule


class StubAsyncClient:
    def __init__(self) -> None:
        self.posts: List[Tuple[str, Dict[str, Any] | None]] = []
        self.gets: List[Tuple[str, Dict[str, Any] | None]] = []

    async def post(self, path: str, json: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.posts.append((path, json))
        if path.endswith("/jobs"):
            return {
                "job_id": "job_123",
                "tx_hash": "AA" * 32,
                "estimated_blocks": 5,
            }
        if path.endswith("/seals"):
            return {
                "seal_id": "seal_123",
                "tx_hash": "BB" * 32,
            }
        if path.endswith("/zkproofs:verify"):
            return {"valid": True, "verification_time_ms": 21}
        if path.endswith("/tee/attestation:verify"):
            return {"valid": True, "platform": TEEPlatform.AWS_NITRO.value}
        return {}

    async def get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.gets.append((path, params))
        if path.endswith("/jobs/job_123"):
            return {
                "job": {
                    "id": "job_123",
                    "creator": "aeth1creator",
                    "model_hash": "11" * 32,
                    "input_hash": "22" * 32,
                    "status": "JOB_STATUS_PENDING",
                    "proof_type": "PROOF_TYPE_TEE",
                }
            }
        if path.endswith("/jobs"):
            return {"jobs": []}
        if path.endswith("/jobs/pending"):
            return {"jobs": []}
        if path.endswith("/jobs/job_123/result"):
            return {
                "job_id": "job_123",
                "output_hash": "33" * 32,
                "verified": True,
                "consensus_validators": 3,
                "total_voting_power": 100,
            }
        if path.endswith("/seals/seal_123"):
            return {
                "seal": {
                    "id": "seal_123",
                    "job_id": "job_123",
                    "model_hash": "44" * 32,
                    "status": "SEAL_STATUS_ACTIVE",
                    "requester": "aeth1req",
                }
            }
        if path.endswith("/seals"):
            return {"seals": []}
        if path.endswith("/seals/by_model"):
            return {"seals": []}
        if path.endswith("/seals/seal_123/verify"):
            return {"valid": True, "verification_details": {"hash_match": True}}
        if path.endswith("/seals/seal_123/export"):
            return {"data": "deadbeef"}
        return {}


def test_async_client_builds_urls_and_redacts_secret_headers() -> None:
    client = AsyncAethelredClient(
        Config(
            rpc_url="https://rpc.example.org/",
            api_key=SecretStr("super-secret"),
        )
    )

    assert client._build_url("/aethelred/pouw/v1/jobs") == "https://rpc.example.org/aethelred/pouw/v1/jobs"
    headers = client._get_headers()
    assert headers["X-API-Key"] == "super-secret"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_jobs_module_submit_get_and_result() -> None:
    stub = StubAsyncClient()
    jobs = JobsModule(stub)  # type: ignore[arg-type]

    submit = await jobs.submit(
        model_hash=bytes.fromhex("11" * 32),
        input_hash=bytes.fromhex("22" * 32),
        proof_type=ProofType.TEE,
    )
    assert submit.job_id == "job_123"

    job = await jobs.get("job_123")
    assert job.id == "job_123"

    result = await jobs.get_result("job_123")
    assert result.verified is True
    assert result.job_id == "job_123"


@pytest.mark.asyncio
async def test_jobs_module_wait_for_completion_returns_terminal_status() -> None:
    class TerminalStub(StubAsyncClient):
        async def get(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
            if path.endswith("/jobs/job_123"):
                return {
                    "job": {
                        "id": "job_123",
                        "creator": "aeth1creator",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": "JOB_STATUS_COMPLETED",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }
            return await super().get(path, params)

    jobs = JobsModule(TerminalStub())  # type: ignore[arg-type]
    job = await jobs.wait_for_completion("job_123", poll_interval=0.001, timeout=0.01)
    assert str(job.status).endswith("COMPLETED")


@pytest.mark.asyncio
async def test_seals_module_create_verify_export() -> None:
    stub = StubAsyncClient()
    seals = SealsModule(stub)  # type: ignore[arg-type]

    create = await seals.create(job_id="job_123")
    assert create.seal_id == "seal_123"

    seal = await seals.get("seal_123")
    assert seal.id == "seal_123"

    verify = await seals.verify("seal_123")
    assert verify.valid is True

    exported = await seals.export("seal_123")
    assert exported == "deadbeef"


@pytest.mark.asyncio
async def test_verification_module_zk_and_tee_calls() -> None:
    stub = StubAsyncClient()
    verification = VerificationModule(stub)  # type: ignore[arg-type]

    zk = await verification.verify_zk_proof(
        proof=b"\x01\x02",
        public_inputs=[b"\x03" * 32],
        verifying_key_hash=b"\x04" * 32,
        proof_system=ProofSystem.GROTH16,
    )
    assert zk.valid is True

    tee = await verification.verify_tee_attestation(
        TEEAttestation(
            platform=TEEPlatform.AWS_NITRO,
            quote=b"\x05" * 64,
            enclave_hash=b"\x06" * 32,
        )
    )
    assert tee.valid is True
    assert tee.platform in (TEEPlatform.AWS_NITRO, TEEPlatform.AWS_NITRO.value)
