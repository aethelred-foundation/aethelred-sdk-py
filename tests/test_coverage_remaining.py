"""
Comprehensive tests for remaining modules:
- aethelred/seals/__init__.py (SealsModule, SyncSealsModule)
- aethelred/jobs/__init__.py (JobsModule, SyncJobsModule)
- aethelred/validators/__init__.py (ValidatorsModule, SyncValidatorsModule)
- aethelred/verification/__init__.py (VerificationModule, SyncVerificationModule)
- aethelred/integrations/tensorflow.py (AethelredKerasCallback, create_keras_callback)
- aethelred/integrations/langchain.py (VerifiedLangChainRunnable, wrap_langchain_runnable)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# SealsModule
# ===========================================================================

class TestSealsModule:
    def test_init(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        module = SealsModule(mock_client)
        assert module._client is mock_client
        assert module.BASE_PATH == "/aethelred/seal/v1"

    @pytest.mark.asyncio
    async def test_create(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "seal_id": "seal-123",
            "tx_hash": "0xabc",
        })
        module = SealsModule(mock_client)
        result = await module.create(job_id="job-1")
        assert result.seal_id == "seal-123"
        assert result.tx_hash == "0xabc"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/seals" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_with_options(self):
        from aethelred.seals import SealsModule
        from aethelred.core.types import RegulatoryInfo
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "seal_id": "seal-456",
            "tx_hash": "0xdef",
        })
        module = SealsModule(mock_client)
        reg_info = RegulatoryInfo(jurisdiction="US", compliance_frameworks=["HIPAA"])
        result = await module.create(
            job_id="job-2",
            regulatory_info=reg_info,
            expires_in_blocks=100,
            metadata={"key": "val"},
        )
        assert result.seal_id == "seal-456"

    @pytest.mark.asyncio
    async def test_get(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "seal": {
                "id": "seal-123",
                "job_id": "job-1",
                "model_hash": b"\x01" * 32,
            }
        })
        module = SealsModule(mock_client)
        seal = await module.get("seal-123")
        assert seal.id == "seal-123"

    @pytest.mark.asyncio
    async def test_get_no_seal_wrapper(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "id": "seal-123",
            "job_id": "job-1",
            "model_hash": b"\x01" * 32,
        })
        module = SealsModule(mock_client)
        seal = await module.get("seal-123")
        assert seal.id == "seal-123"

    @pytest.mark.asyncio
    async def test_list_no_filters(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "seals": [
                {"id": "s1", "job_id": "j1", "model_hash": b"\x01" * 32},
                {"id": "s2", "job_id": "j2", "model_hash": b"\x02" * 32},
            ]
        })
        module = SealsModule(mock_client)
        seals = await module.list()
        assert len(seals) == 2

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        from aethelred.seals import SealsModule
        from aethelred.core.types import SealStatus, PageRequest
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={"seals": []})
        module = SealsModule(mock_client)
        result = await module.list(
            requester="aeth1abc",
            model_hash="hash123",
            status=SealStatus.ACTIVE,
            pagination=PageRequest(limit=10),
        )
        assert result == []
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["requester"] == "aeth1abc"
        assert params["model_hash"] == "hash123"
        assert params["status"] == SealStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_list_by_model(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "seals": [{"id": "s1", "job_id": "j1", "model_hash": b"\x01" * 32}]
        })
        module = SealsModule(mock_client)
        seals = await module.list_by_model("model-hash-abc")
        assert len(seals) == 1

    @pytest.mark.asyncio
    async def test_list_by_model_with_pagination(self):
        from aethelred.seals import SealsModule
        from aethelred.core.types import PageRequest
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={"seals": []})
        module = SealsModule(mock_client)
        await module.list_by_model("hash", pagination=PageRequest(limit=5))
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["model_hash"] == "hash"

    @pytest.mark.asyncio
    async def test_verify(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "valid": True,
            "verification_details": {"proof": True},
        })
        module = SealsModule(mock_client)
        result = await module.verify("seal-123")
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_revoke(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={})
        module = SealsModule(mock_client)
        result = await module.revoke("seal-123", "compromised")
        assert result is True
        call_args = mock_client.post.call_args
        assert "revoke" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_export(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={"data": b"seal-bytes"})
        module = SealsModule(mock_client)
        result = await module.export("seal-123", format="cbor")
        assert result == b"seal-bytes"

    @pytest.mark.asyncio
    async def test_export_empty(self):
        from aethelred.seals import SealsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={})
        module = SealsModule(mock_client)
        result = await module.export("seal-123")
        assert result == b""


class TestSyncSealsModule:
    def test_init(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_client = MagicMock()
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_client, async_module)
        assert module._client is mock_client
        assert module._async is async_module

    def test_create_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(seal_id="s1")
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.create(job_id="j1")
        assert result.seal_id == "s1"
        mock_sync_client._run.assert_called_once()

    def test_get_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(id="s1")
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.get("s1")
        assert result.id == "s1"

    def test_list_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = []
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.list()
        assert result == []

    def test_list_by_model_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = []
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.list_by_model(model_hash="abc")
        assert result == []

    def test_verify_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(valid=True)
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.verify("s1")
        assert result.valid is True

    def test_revoke_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = True
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.revoke("s1", "reason")
        assert result is True

    def test_export_delegates(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = b"data"
        async_module = SealsModule(MagicMock())
        module = SyncSealsModule(mock_sync_client, async_module)
        result = module.export(seal_id="s1", format="json")
        assert result == b"data"


# ===========================================================================
# JobsModule
# ===========================================================================

class TestJobsModule:
    def test_init(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        module = JobsModule(mock_client)
        assert module._client is mock_client
        assert module.BASE_PATH == "/aethelred/pouw/v1"

    @pytest.mark.asyncio
    async def test_submit(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "job_id": "job-123",
            "tx_hash": "0xabc",
            "estimated_blocks": 5,
        })
        module = JobsModule(mock_client)
        result = await module.submit(
            model_hash=b"\x01" * 32,
            input_hash=b"\x02" * 32,
        )
        assert result.job_id == "job-123"
        assert result.tx_hash == "0xabc"

    @pytest.mark.asyncio
    async def test_submit_with_options(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import ProofType
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "job_id": "job-456",
            "tx_hash": "0xdef",
        })
        module = JobsModule(mock_client)
        result = await module.submit(
            model_hash=b"\x01" * 32,
            input_hash=b"\x02" * 32,
            proof_type=ProofType.ZKML,
            priority=5,
            max_gas=1000,
            timeout_blocks=200,
            callback_url="https://example.com/cb",
            metadata={"key": "val"},
        )
        assert result.job_id == "job-456"

    @pytest.mark.asyncio
    async def test_get(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "job": {
                "id": "job-123",
                "creator": "aeth1abc",
                "model_hash": b"\x01" * 32,
                "input_hash": b"\x02" * 32,
            }
        })
        module = JobsModule(mock_client)
        job = await module.get("job-123")
        assert job.id == "job-123"

    @pytest.mark.asyncio
    async def test_get_no_job_wrapper(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "id": "job-123",
            "creator": "aeth1abc",
            "model_hash": b"\x01" * 32,
            "input_hash": b"\x02" * 32,
        })
        module = JobsModule(mock_client)
        job = await module.get("job-123")
        assert job.id == "job-123"

    @pytest.mark.asyncio
    async def test_list_no_filters(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "jobs": [
                {"id": "j1", "creator": "a", "model_hash": b"\x01" * 32, "input_hash": b"\x02" * 32},
            ]
        })
        module = JobsModule(mock_client)
        jobs = await module.list()
        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus, PageRequest
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={"jobs": []})
        module = JobsModule(mock_client)
        result = await module.list(
            status=JobStatus.PENDING,
            creator="aeth1abc",
            pagination=PageRequest(limit=5),
        )
        assert result == []
        call_args = mock_client.get.call_args
        params = call_args[1]["params"]
        assert params["status"] == JobStatus.PENDING.value
        assert params["creator"] == "aeth1abc"

    @pytest.mark.asyncio
    async def test_list_pending(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "jobs": [
                {"id": "j1", "creator": "a", "model_hash": b"\x01" * 32, "input_hash": b"\x02" * 32},
            ]
        })
        module = JobsModule(mock_client)
        jobs = await module.list_pending()
        assert len(jobs) == 1

    @pytest.mark.asyncio
    async def test_list_pending_with_pagination(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import PageRequest
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={"jobs": []})
        module = JobsModule(mock_client)
        await module.list_pending(pagination=PageRequest(limit=3))
        mock_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={})
        module = JobsModule(mock_client)
        result = await module.cancel("job-123")
        assert result is True
        call_args = mock_client.post.call_args
        assert "cancel" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_result(self):
        from aethelred.jobs import JobsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "job_id": "job-123",
            "output_hash": b"\x03" * 32,
            "verified": True,
        })
        module = JobsModule(mock_client)
        result = await module.get_result("job-123")
        assert result.job_id == "job-123"
        assert result.verified is True

    @pytest.mark.asyncio
    async def test_wait_for_completion_immediate(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "id": "job-123",
            "creator": "aeth1abc",
            "model_hash": b"\x01" * 32,
            "input_hash": b"\x02" * 32,
            "status": JobStatus.COMPLETED,
        })
        module = JobsModule(mock_client)
        job = await module.wait_for_completion("job-123", poll_interval=0.01, timeout=1.0)
        assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_for_completion_failed(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "id": "job-123",
            "creator": "aeth1abc",
            "model_hash": b"\x01" * 32,
            "input_hash": b"\x02" * 32,
            "status": JobStatus.FAILED,
        })
        module = JobsModule(mock_client)
        job = await module.wait_for_completion("job-123", poll_interval=0.01, timeout=1.0)
        assert job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_wait_for_completion_cancelled(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "id": "job-123",
            "creator": "aeth1abc",
            "model_hash": b"\x01" * 32,
            "input_hash": b"\x02" * 32,
            "status": JobStatus.CANCELLED,
        })
        module = JobsModule(mock_client)
        job = await module.wait_for_completion("job-123", poll_interval=0.01, timeout=1.0)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus
        from aethelred.core.exceptions import TimeoutError
        mock_client = MagicMock()
        # Always return PENDING
        mock_client.get = AsyncMock(return_value={
            "id": "job-123",
            "creator": "aeth1abc",
            "model_hash": b"\x01" * 32,
            "input_hash": b"\x02" * 32,
            "status": JobStatus.PENDING,
        })
        module = JobsModule(mock_client)
        with pytest.raises(TimeoutError, match="did not complete"):
            await module.wait_for_completion("job-123", poll_interval=0.01, timeout=0.05)

    @pytest.mark.asyncio
    async def test_wait_for_completion_transition(self):
        """Test job transitions from PENDING to COMPLETED after a few polls."""
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus

        call_count = 0

        async def mock_get(path, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return {
                    "id": "job-123",
                    "creator": "aeth1abc",
                    "model_hash": b"\x01" * 32,
                    "input_hash": b"\x02" * 32,
                    "status": JobStatus.COMPUTING,
                }
            return {
                "id": "job-123",
                "creator": "aeth1abc",
                "model_hash": b"\x01" * 32,
                "input_hash": b"\x02" * 32,
                "status": JobStatus.COMPLETED,
            }

        mock_client = MagicMock()
        mock_client.get = mock_get
        module = JobsModule(mock_client)
        job = await module.wait_for_completion("job-123", poll_interval=0.01, timeout=5.0)
        assert job.status == JobStatus.COMPLETED
        assert call_count == 3


class TestSyncJobsModule:
    def test_init(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_client = MagicMock()
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_client, async_module)
        assert module._client is mock_client
        assert module._async is async_module

    def test_submit_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(job_id="j1")
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.submit(model_hash=b"\x01" * 32, input_hash=b"\x02" * 32)
        assert result.job_id == "j1"

    def test_get_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(id="j1")
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.get("j1")
        assert result.id == "j1"

    def test_list_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = []
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.list()
        assert result == []

    def test_list_pending_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = []
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.list_pending()
        assert result == []

    def test_cancel_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = True
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.cancel("j1")
        assert result is True

    def test_get_result_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(job_id="j1")
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.get_result("j1")
        assert result.job_id == "j1"

    def test_wait_for_completion_delegates(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(id="j1")
        async_module = JobsModule(MagicMock())
        module = SyncJobsModule(mock_sync_client, async_module)
        result = module.wait_for_completion(job_id="j1")
        assert result.id == "j1"


# ===========================================================================
# ValidatorsModule
# ===========================================================================

class TestValidatorsModule:
    def test_init(self):
        from aethelred.validators import ValidatorsModule
        mock_client = MagicMock()
        module = ValidatorsModule(mock_client)
        assert module._client is mock_client
        assert module.BASE_PATH == "/aethelred/pouw/v1"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        from aethelred.validators import ValidatorsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "address": "aeth1validator",
            "jobs_completed": 100,
            "reputation_score": 0.95,
        })
        module = ValidatorsModule(mock_client)
        stats = await module.get_stats("aeth1validator")
        assert stats.address == "aeth1validator"
        assert stats.jobs_completed == 100

    @pytest.mark.asyncio
    async def test_list_no_pagination(self):
        from aethelred.validators import ValidatorsModule
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={
            "validators": [
                {"address": "v1", "jobs_completed": 10},
                {"address": "v2", "jobs_completed": 20},
            ]
        })
        module = ValidatorsModule(mock_client)
        validators = await module.list()
        assert len(validators) == 2

    @pytest.mark.asyncio
    async def test_list_with_pagination(self):
        from aethelred.validators import ValidatorsModule
        from aethelred.core.types import PageRequest
        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value={"validators": []})
        module = ValidatorsModule(mock_client)
        result = await module.list(pagination=PageRequest(limit=5))
        assert result == []

    @pytest.mark.asyncio
    async def test_register_capability(self):
        from aethelred.validators import ValidatorsModule
        from aethelred.core.types import HardwareCapability, TEEPlatform
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={})
        module = ValidatorsModule(mock_client)
        cap = HardwareCapability(
            tee_platforms=[TEEPlatform.INTEL_SGX],
            zkml_supported=True,
            max_model_size_mb=1024,
        )
        result = await module.register_capability("aeth1validator", cap)
        assert result is True
        call_args = mock_client.post.call_args
        assert "capability" in call_args[0][0]


class TestSyncValidatorsModule:
    def test_init(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        mock_client = MagicMock()
        async_module = ValidatorsModule(MagicMock())
        module = SyncValidatorsModule(mock_client, async_module)
        assert module._client is mock_client

    def test_get_stats_delegates(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(address="v1")
        async_module = ValidatorsModule(MagicMock())
        module = SyncValidatorsModule(mock_sync_client, async_module)
        result = module.get_stats("v1")
        assert result.address == "v1"

    def test_list_delegates(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = []
        async_module = ValidatorsModule(MagicMock())
        module = SyncValidatorsModule(mock_sync_client, async_module)
        result = module.list()
        assert result == []

    def test_register_capability_delegates(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = True
        async_module = ValidatorsModule(MagicMock())
        module = SyncValidatorsModule(mock_sync_client, async_module)
        result = module.register_capability(address="v1", capability=MagicMock())
        assert result is True


# ===========================================================================
# VerificationModule
# ===========================================================================

class TestVerifyZKProofResponse:
    def test_init_defaults(self):
        from aethelred.verification import VerifyZKProofResponse
        r = VerifyZKProofResponse(valid=True)
        assert r.valid is True
        assert r.verification_time_ms == 0
        assert r.error is None

    def test_init_with_values(self):
        from aethelred.verification import VerifyZKProofResponse
        r = VerifyZKProofResponse(valid=False, verification_time_ms=123, error="bad proof")
        assert r.valid is False
        assert r.verification_time_ms == 123
        assert r.error == "bad proof"


class TestVerifyTEEResponse:
    def test_init_defaults(self):
        from aethelred.verification import VerifyTEEResponse
        from aethelred.core.types import TEEPlatform
        r = VerifyTEEResponse(valid=True)
        assert r.valid is True
        assert r.platform == TEEPlatform.UNSPECIFIED
        assert r.error is None

    def test_init_with_values(self):
        from aethelred.verification import VerifyTEEResponse
        from aethelred.core.types import TEEPlatform
        r = VerifyTEEResponse(valid=False, platform=TEEPlatform.INTEL_SGX, error="failed")
        assert r.valid is False
        assert r.platform == TEEPlatform.INTEL_SGX
        assert r.error == "failed"


class TestVerificationModule:
    def test_init(self):
        from aethelred.verification import VerificationModule
        mock_client = MagicMock()
        module = VerificationModule(mock_client)
        assert module._client is mock_client
        assert module.BASE_PATH == "/aethelred/verify/v1"

    @pytest.mark.asyncio
    async def test_verify_zk_proof(self):
        from aethelred.verification import VerificationModule
        from aethelred.core.types import ProofSystem
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "valid": True,
            "verification_time_ms": 42,
        })
        module = VerificationModule(mock_client)
        result = await module.verify_zk_proof(
            proof=b"\x01" * 32,
            public_inputs=[b"\x02" * 16, b"\x03" * 16],
            verifying_key_hash=b"\x04" * 32,
            proof_system=ProofSystem.GROTH16,
        )
        assert result.valid is True
        assert result.verification_time_ms == 42
        call_args = mock_client.post.call_args
        assert "zkproofs:verify" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_verify_zk_proof_invalid(self):
        from aethelred.verification import VerificationModule
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "valid": False,
            "error": "invalid proof",
        })
        module = VerificationModule(mock_client)
        result = await module.verify_zk_proof(
            proof=b"\x01" * 32,
            public_inputs=[],
            verifying_key_hash=b"\x04" * 32,
        )
        assert result.valid is False
        assert result.error == "invalid proof"

    @pytest.mark.asyncio
    async def test_verify_tee_attestation(self):
        from aethelred.verification import VerificationModule
        from aethelred.core.types import TEEAttestation, TEEPlatform
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "valid": True,
            "platform": TEEPlatform.INTEL_SGX,
        })
        module = VerificationModule(mock_client)
        attestation = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            quote=b"\x01" * 64,
            enclave_hash=b"\x02" * 32,
        )
        result = await module.verify_tee_attestation(
            attestation=attestation,
            expected_enclave_hash=b"\x02" * 32,
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_verify_tee_attestation_no_expected_hash(self):
        from aethelred.verification import VerificationModule
        from aethelred.core.types import TEEAttestation, TEEPlatform
        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value={
            "valid": True,
            "platform": TEEPlatform.AMD_SEV,
        })
        module = VerificationModule(mock_client)
        attestation = TEEAttestation(
            platform=TEEPlatform.AMD_SEV,
            quote=b"\x01" * 64,
        )
        result = await module.verify_tee_attestation(attestation=attestation)
        assert result.valid is True
        call_args = mock_client.post.call_args
        json_body = call_args[1]["json"]
        assert json_body["expected_enclave_hash"] is None


class TestSyncVerificationModule:
    def test_init(self):
        from aethelred.verification import SyncVerificationModule, VerificationModule
        mock_client = MagicMock()
        async_module = VerificationModule(MagicMock())
        module = SyncVerificationModule(mock_client, async_module)
        assert module._client is mock_client

    def test_verify_zk_proof_delegates(self):
        from aethelred.verification import SyncVerificationModule, VerificationModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(valid=True)
        async_module = VerificationModule(MagicMock())
        module = SyncVerificationModule(mock_sync_client, async_module)
        result = module.verify_zk_proof(
            proof=b"\x01" * 32,
            public_inputs=[],
            verifying_key_hash=b"\x04" * 32,
        )
        assert result.valid is True

    def test_verify_tee_attestation_delegates(self):
        from aethelred.verification import SyncVerificationModule, VerificationModule
        mock_sync_client = MagicMock()
        mock_sync_client._run.return_value = MagicMock(valid=True)
        async_module = VerificationModule(MagicMock())
        module = SyncVerificationModule(mock_sync_client, async_module)
        result = module.verify_tee_attestation(attestation=MagicMock())
        assert result.valid is True


# ===========================================================================
# Module Exports
# ===========================================================================

class TestModuleExports:
    def test_seals_exports(self):
        from aethelred.seals import __all__
        assert "SealsModule" in __all__
        assert "SyncSealsModule" in __all__

    def test_jobs_exports(self):
        from aethelred.jobs import __all__
        assert "JobsModule" in __all__
        assert "SyncJobsModule" in __all__

    def test_validators_exports(self):
        from aethelred.validators import __all__
        assert "ValidatorsModule" in __all__
        assert "SyncValidatorsModule" in __all__

    def test_verification_exports(self):
        from aethelred.verification import __all__
        assert "VerificationModule" in __all__
        assert "SyncVerificationModule" in __all__
        assert "VerifyZKProofResponse" in __all__
        assert "VerifyTEEResponse" in __all__


# ===========================================================================
# TensorFlow / Keras integration
# ===========================================================================

class TestAethelredKerasCallback:
    def test_init_defaults(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        assert cb._capture_batch_events is False
        assert cb._include_log_keys == set()
        assert cb._extra_metadata == {}
        assert cb._model is None
        assert cb.last_verification is None

    def test_init_with_options(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        cb = AethelredKerasCallback(
            recorder=recorder,
            capture_batch_events=True,
            include_log_keys=["loss", "accuracy"],
            extra_metadata={"experiment": "test"},
        )
        assert cb._capture_batch_events is True
        assert cb._include_log_keys == {"loss", "accuracy"}
        assert cb._extra_metadata == {"experiment": "test"}
        assert cb._recorder is recorder

    def test_set_model(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        mock_model = MagicMock()
        cb.set_model(mock_model)
        assert cb._model is mock_model

    def test_filtered_logs_none(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        assert cb._filtered_logs(None) == {}

    def test_filtered_logs_no_filter(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        logs = {"loss": 0.5, "accuracy": 0.9}
        result = cb._filtered_logs(logs)
        assert result == {"loss": 0.5, "accuracy": 0.9}

    def test_filtered_logs_with_filter(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(include_log_keys=["loss"])
        logs = {"loss": 0.5, "accuracy": 0.9, "lr": 0.001}
        result = cb._filtered_logs(logs)
        assert result == {"loss": 0.5}

    def test_on_train_batch_end_no_capture(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=False)
        cb.on_train_batch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is None  # Should not record

    def test_on_train_batch_end_with_capture(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=True)
        cb.on_train_batch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is not None
        assert cb.last_verification.operation == "train_batch_end"

    def test_on_test_batch_end_no_capture(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=False)
        cb.on_test_batch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is None

    def test_on_test_batch_end_with_capture(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=True)
        cb.on_test_batch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is not None
        assert cb.last_verification.operation == "test_batch_end"

    def test_on_predict_batch_end_no_capture(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=False)
        cb.on_predict_batch_end(0)
        assert cb.last_verification is None

    def test_on_predict_batch_end_with_capture(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=True)
        cb.on_predict_batch_end(0, logs={"val": 1.0})
        assert cb.last_verification is not None
        assert cb.last_verification.operation == "predict_batch_end"

    def test_on_epoch_end(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        cb.on_epoch_end(0, logs={"loss": 0.3, "val_loss": 0.4})
        assert cb.last_verification is not None
        assert cb.last_verification.operation == "epoch_end"
        assert cb.last_verification.framework == "tensorflow.keras"

    def test_on_epoch_end_no_logs(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        cb.on_epoch_end(5)
        assert cb.last_verification is not None

    def test_record_model_name(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()

        class FakeModel:
            pass

        cb.set_model(FakeModel())
        cb.on_epoch_end(0, logs={"loss": 0.1})
        assert cb.last_verification is not None
        assert cb.last_verification.metadata.get("model") == "FakeModel"

    def test_record_model_name_none(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        # No model set
        cb.on_epoch_end(0, logs={"loss": 0.1})
        assert cb.last_verification is not None

    def test_extra_metadata_in_record(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(extra_metadata={"run_id": "abc"})
        cb.on_epoch_end(0)
        assert cb.last_verification.metadata.get("run_id") == "abc"


class TestCreateKerasCallback:
    def test_without_tensorflow(self):
        from aethelred.integrations.tensorflow import create_keras_callback
        with patch.dict("sys.modules", {"tensorflow": None}):
            cb = create_keras_callback()
            # Should return the core callback directly
            from aethelred.integrations.tensorflow import AethelredKerasCallback
            assert isinstance(cb, AethelredKerasCallback)

    def test_without_tensorflow_import_error(self):
        from aethelred.integrations.tensorflow import create_keras_callback
        with patch("builtins.__import__", side_effect=ImportError("no tensorflow")):
            # create_keras_callback catches Exception broadly
            cb = create_keras_callback()
            from aethelred.integrations.tensorflow import AethelredKerasCallback
            assert isinstance(cb, AethelredKerasCallback)

    def test_with_tensorflow(self):
        from aethelred.integrations.tensorflow import create_keras_callback

        # Mock TensorFlow
        mock_callback_base = type("Callback", (), {
            "__init__": lambda self: None,
        })
        mock_keras = MagicMock()
        mock_keras.callbacks.Callback = mock_callback_base

        mock_tf = MagicMock()
        mock_tf.keras = mock_keras

        with patch.dict("sys.modules", {"tensorflow": mock_tf}):
            cb = create_keras_callback(capture_batch_events=True)
            # Should return a wrapped TF callback, not just the core
            assert hasattr(cb, "_inner")

    def test_with_kwargs(self):
        from aethelred.integrations.tensorflow import create_keras_callback
        with patch.dict("sys.modules", {"tensorflow": None}):
            cb = create_keras_callback(
                capture_batch_events=True,
                include_log_keys=["loss"],
                extra_metadata={"test": True},
            )
            assert cb._capture_batch_events is True


# ===========================================================================
# LangChain integration
# ===========================================================================

class TestVerifiedLangChainRunnable:
    def test_init_defaults(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.__class__.__name__ = "FakeRunnable"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        assert wrapper._runnable is mock_runnable
        assert wrapper._component_name == "FakeRunnable"
        assert wrapper._extra_metadata == {}
        assert wrapper.last_verification is None

    def test_init_with_options(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        mock_runnable = MagicMock()
        wrapper = VerifiedLangChainRunnable(
            mock_runnable,
            recorder=recorder,
            component_name="MyChain",
            extra_metadata={"version": "1.0"},
        )
        assert wrapper._component_name == "MyChain"
        assert wrapper._extra_metadata == {"version": "1.0"}
        assert wrapper._recorder is recorder

    def test_getattr_delegates(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.custom_method.return_value = "custom"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.custom_method()
        assert result == "custom"

    def test_invoke_with_invoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "output"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.invoke("input_data")
        assert result == "output"
        assert wrapper.last_verification is not None
        assert wrapper.last_verification.operation == "invoke"
        mock_runnable.invoke.assert_called_once()

    def test_invoke_without_invoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock(spec=[])
        mock_runnable.return_value = "output_fallback"
        # No invoke method
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.invoke("input_data")
        assert result == "output_fallback"
        assert wrapper.last_verification is not None

    def test_invoke_with_config(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "output"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.invoke("input", config={"key": "val"})
        assert result == "output"

    @pytest.mark.asyncio
    async def test_ainvoke_with_ainvoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.ainvoke = AsyncMock(return_value="async_output")
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = await wrapper.ainvoke("input_data")
        assert result == "async_output"
        assert wrapper.last_verification is not None
        assert wrapper.last_verification.operation == "ainvoke"

    @pytest.mark.asyncio
    async def test_ainvoke_without_ainvoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock(spec=[])
        mock_runnable.return_value = "sync_fallback"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = await wrapper.ainvoke("input_data")
        assert result == "sync_fallback"

    def test_batch_with_batch(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.batch.return_value = ["out1", "out2"]
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.batch(["in1", "in2"])
        assert result == ["out1", "out2"]
        assert wrapper.last_verification is not None
        assert wrapper.last_verification.operation == "batch"
        assert wrapper.last_verification.metadata.get("batch_size") == 2

    def test_batch_without_batch(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock(spec=[])
        mock_runnable.side_effect = lambda x, **kw: f"processed_{x}"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.batch(["a", "b"])
        assert result == ["processed_a", "processed_b"]
        assert wrapper.last_verification is not None

    def test_batch_with_config(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.batch.return_value = ["out"]
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper.batch(["in"], config={"key": "val"})
        assert result == ["out"]

    @pytest.mark.asyncio
    async def test_abatch_with_abatch(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.abatch = AsyncMock(return_value=["out1", "out2"])
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = await wrapper.abatch(["in1", "in2"])
        assert result == ["out1", "out2"]
        assert wrapper.last_verification is not None
        assert wrapper.last_verification.operation == "abatch"

    @pytest.mark.asyncio
    async def test_abatch_with_ainvoke_fallback(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock(spec=["ainvoke"])
        mock_runnable.ainvoke = AsyncMock(side_effect=lambda x, **kw: f"async_{x}")
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = await wrapper.abatch(["a", "b"])
        assert result == ["async_a", "async_b"]

    @pytest.mark.asyncio
    async def test_abatch_sync_fallback(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock(spec=[])
        mock_runnable.side_effect = lambda x, **kw: f"sync_{x}"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = await wrapper.abatch(["a", "b"])
        assert result == ["sync_a", "sync_b"]

    def test_call_delegates_to_invoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "called"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper("input")
        assert result == "called"
        assert wrapper.last_verification is not None

    def test_call_with_args(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = "called_with_args"
        wrapper = VerifiedLangChainRunnable(mock_runnable)
        result = wrapper("input", "extra_arg", key="val")
        assert result == "called_with_args"
        # When args are provided, they should be packed into kwargs
        call_kwargs = mock_runnable.invoke.call_args[1]
        assert "_args" in call_kwargs or "key" in call_kwargs

    def test_metadata(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        wrapper = VerifiedLangChainRunnable(
            mock_runnable,
            component_name="MyChain",
            extra_metadata={"env": "test"},
        )
        meta = wrapper._metadata("invoke")
        assert meta["component"] == "MyChain"
        assert meta["operation_kind"] == "invoke"
        assert meta["env"] == "test"


class TestWrapLangchainRunnable:
    def test_basic(self):
        from aethelred.integrations.langchain import wrap_langchain_runnable, VerifiedLangChainRunnable
        mock_runnable = MagicMock()
        wrapper = wrap_langchain_runnable(mock_runnable)
        assert isinstance(wrapper, VerifiedLangChainRunnable)
        assert wrapper._runnable is mock_runnable

    def test_with_options(self):
        from aethelred.integrations.langchain import wrap_langchain_runnable
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        mock_runnable = MagicMock()
        wrapper = wrap_langchain_runnable(
            mock_runnable,
            recorder=recorder,
            component_name="TestChain",
            extra_metadata={"key": "val"},
        )
        assert wrapper._component_name == "TestChain"
        assert wrapper._recorder is recorder
        assert wrapper._extra_metadata == {"key": "val"}

    def test_exports(self):
        from aethelred.integrations.langchain import __all__
        assert "VerifiedLangChainRunnable" in __all__
        assert "wrap_langchain_runnable" in __all__


# ===========================================================================
# Integrations _common module
# ===========================================================================

class TestVerificationEnvelope:
    def test_creation(self):
        from aethelred.integrations._common import VerificationEnvelope
        env = VerificationEnvelope(
            trace_id="abc123",
            framework="test",
            operation="run",
            input_hash="inhash",
            output_hash="outhash",
            timestamp_ms=1000,
            metadata={"key": "val"},
        )
        assert env.trace_id == "abc123"
        assert env.framework == "test"
        assert env.operation == "run"
        assert env.input_hash == "inhash"
        assert env.output_hash == "outhash"
        assert env.timestamp_ms == 1000
        assert env.metadata == {"key": "val"}

    def test_frozen(self):
        from aethelred.integrations._common import VerificationEnvelope
        env = VerificationEnvelope(
            trace_id="abc",
            framework="test",
            operation="run",
            input_hash="in",
            output_hash="out",
            timestamp_ms=1000,
            metadata={},
        )
        with pytest.raises(AttributeError):
            env.trace_id = "changed"

    def test_to_headers(self):
        from aethelred.integrations._common import VerificationEnvelope
        env = VerificationEnvelope(
            trace_id="trace-1",
            framework="langchain",
            operation="invoke",
            input_hash="inhash",
            output_hash="outhash",
            timestamp_ms=12345,
            metadata={},
        )
        headers = env.to_headers()
        assert headers["x-aethelred-trace-id"] == "trace-1"
        assert headers["x-aethelred-framework"] == "langchain"
        assert headers["x-aethelred-operation"] == "invoke"
        assert headers["x-aethelred-input-hash"] == "inhash"
        assert headers["x-aethelred-output-hash"] == "outhash"
        assert headers["x-aethelred-ts-ms"] == "12345"

    def test_to_headers_custom_prefix(self):
        from aethelred.integrations._common import VerificationEnvelope
        env = VerificationEnvelope(
            trace_id="t1",
            framework="f",
            operation="o",
            input_hash="i",
            output_hash="o",
            timestamp_ms=1,
            metadata={},
        )
        headers = env.to_headers("x-custom-")
        assert "x-custom-trace-id" in headers


class TestVerificationRecorder:
    def test_init_defaults(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        assert recorder._sink is None
        assert recorder._default_metadata == {}
        assert recorder.header_prefix == "x-aethelred"

    def test_init_with_options(self):
        from aethelred.integrations._common import VerificationRecorder
        sink = MagicMock()
        recorder = VerificationRecorder(
            sink=sink,
            default_metadata={"env": "test"},
            header_prefix="x-custom",
        )
        assert recorder._sink is sink
        assert recorder._default_metadata == {"env": "test"}
        assert recorder.header_prefix == "x-custom"

    def test_record_no_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        envelope = recorder.record(
            framework="test",
            operation="op",
            input_data={"a": 1},
            output_data={"b": 2},
        )
        assert envelope.framework == "test"
        assert envelope.operation == "op"
        assert len(envelope.trace_id) > 0

    def test_record_with_sync_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        sink = MagicMock(return_value=None)
        recorder = VerificationRecorder(sink=sink)
        envelope = recorder.record(
            framework="test",
            operation="op",
            input_data={"a": 1},
            output_data={"b": 2},
        )
        sink.assert_called_once_with(envelope)

    def test_record_with_async_sink_raises(self):
        from aethelred.integrations._common import VerificationRecorder

        async def async_sink(env):
            pass

        recorder = VerificationRecorder(sink=async_sink)
        with pytest.raises(TypeError, match="async sink"):
            recorder.record(
                framework="test",
                operation="op",
                input_data={},
                output_data={},
            )

    def test_record_with_metadata(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder(default_metadata={"default_key": "default_val"})
        envelope = recorder.record(
            framework="test",
            operation="op",
            input_data={},
            output_data={},
            metadata={"extra_key": "extra_val"},
        )
        assert envelope.metadata.get("default_key") == "default_val"
        assert envelope.metadata.get("extra_key") == "extra_val"

    @pytest.mark.asyncio
    async def test_arecord_no_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        envelope = await recorder.arecord(
            framework="test",
            operation="async_op",
            input_data={"a": 1},
            output_data={"b": 2},
        )
        assert envelope.operation == "async_op"

    @pytest.mark.asyncio
    async def test_arecord_with_sync_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        sink = MagicMock(return_value=None)
        recorder = VerificationRecorder(sink=sink)
        envelope = await recorder.arecord(
            framework="test",
            operation="op",
            input_data={},
            output_data={},
        )
        sink.assert_called_once_with(envelope)

    @pytest.mark.asyncio
    async def test_arecord_with_async_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        sink = AsyncMock()
        recorder = VerificationRecorder(sink=sink)
        envelope = await recorder.arecord(
            framework="test",
            operation="op",
            input_data={},
            output_data={},
        )
        sink.assert_called_once_with(envelope)


class TestCanonicalJson:
    def test_basic_types(self):
        from aethelred.integrations._common import canonical_json
        assert canonical_json(None) == "null"
        assert canonical_json(True) == "true"
        assert canonical_json(42) == "42"
        assert canonical_json(3.14) == "3.14"
        assert canonical_json("hello") == '"hello"'

    def test_bytes(self):
        from aethelred.integrations._common import canonical_json
        import json
        result = json.loads(canonical_json(b"\x01\x02"))
        assert "__bytes__" in result

    def test_bytearray(self):
        from aethelred.integrations._common import canonical_json
        import json
        result = json.loads(canonical_json(bytearray(b"\x01\x02")))
        assert "__bytes__" in result

    def test_dict_sorted(self):
        from aethelred.integrations._common import canonical_json
        result = canonical_json({"b": 2, "a": 1})
        assert result.index('"a"') < result.index('"b"')

    def test_list(self):
        from aethelred.integrations._common import canonical_json
        result = canonical_json([1, 2, 3])
        assert result == "[1,2,3]"

    def test_tuple(self):
        from aethelred.integrations._common import canonical_json
        result = canonical_json((1, 2, 3))
        assert result == "[1,2,3]"

    def test_set(self):
        from aethelred.integrations._common import canonical_json
        result = canonical_json({1})
        assert "1" in result

    def test_dataclass(self):
        from aethelred.integrations._common import canonical_json
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: int
            y: int

        result = canonical_json(Point(1, 2))
        assert '"x":1' in result
        assert '"y":2' in result

    def test_object_with_dict(self):
        from aethelred.integrations._common import canonical_json

        class Obj:
            def __init__(self):
                self.value = 42
                self._private = "hidden"

        result = canonical_json(Obj())
        assert '"value":42' in result
        assert "_private" not in result

    def test_repr_fallback(self):
        from aethelred.integrations._common import canonical_json

        class Opaque:
            pass

        result = canonical_json(Opaque())
        assert "__repr__" in result


class TestHashPayload:
    def test_basic(self):
        from aethelred.integrations._common import hash_payload
        h = hash_payload({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_deterministic(self):
        from aethelred.integrations._common import hash_payload
        h1 = hash_payload({"a": 1, "b": 2})
        h2 = hash_payload({"b": 2, "a": 1})
        assert h1 == h2  # deterministic regardless of order


class TestCommonExports:
    def test_all(self):
        from aethelred.integrations._common import __all__
        assert "VerificationEnvelope" in __all__
        assert "VerificationRecorder" in __all__
        assert "canonical_json" in __all__
        assert "hash_payload" in __all__
