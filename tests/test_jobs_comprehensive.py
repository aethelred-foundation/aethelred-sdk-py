"""Comprehensive tests for aethelred/jobs/__init__.py — targeting 95%+ coverage.

Covers all JobsModule methods: submit, get, list (all filter combos),
list_pending, cancel, get_result, wait_for_completion (success, timeout, all terminal states).
Also covers SyncJobsModule wrapper methods.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from aethelred.core.exceptions import TimeoutError
from aethelred.core.types import (
    ComputeJob,
    JobResult,
    JobStatus,
    PageRequest,
    ProofType,
    SubmitJobResponse,
)
from aethelred.jobs import JobsModule, SyncJobsModule


# ---------------------------------------------------------------------------
# Stub async client
# ---------------------------------------------------------------------------


class StubClient:
    """Minimal stub that tracks calls and returns canned responses."""

    def __init__(self) -> None:
        self.posts: List[Tuple[str, Any]] = []
        self.gets: List[Tuple[str, Any]] = []

    async def post(self, path: str, json: Any = None) -> Dict[str, Any]:
        self.posts.append((path, json))
        if path.endswith("/jobs"):
            return {"job_id": "job_abc", "tx_hash": "DD" * 32, "estimated_blocks": 10}
        if "cancel" in path:
            return {}
        return {}

    async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
        self.gets.append((path, params))
        if "/jobs/job_abc/result" in path:
            return {
                "job_id": "job_abc",
                "output_hash": "ee" * 32,
                "verified": True,
                "consensus_validators": 5,
                "total_voting_power": 200,
            }
        if path.endswith("/jobs/job_abc"):
            return {
                "job": {
                    "id": "job_abc",
                    "creator": "aethel1creator",
                    "model_hash": "11" * 32,
                    "input_hash": "22" * 32,
                    "status": "JOB_STATUS_PENDING",
                    "proof_type": "PROOF_TYPE_TEE",
                }
            }
        if "/jobs/pending" in path:
            return {
                "jobs": [
                    {
                        "id": "job_p1",
                        "creator": "aethel1cr",
                        "model_hash": "33" * 32,
                        "input_hash": "44" * 32,
                        "status": "JOB_STATUS_PENDING",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                ]
            }
        if path.endswith("/jobs"):
            return {
                "jobs": [
                    {
                        "id": "job_l1",
                        "creator": "aethel1cr",
                        "model_hash": "55" * 32,
                        "input_hash": "66" * 32,
                        "status": "JOB_STATUS_COMPLETED",
                        "proof_type": "PROOF_TYPE_ZKML",
                    }
                ]
            }
        return {}


# ---------------------------------------------------------------------------
# JobsModule (async)
# ---------------------------------------------------------------------------


class TestJobsModule:
    """Test async JobsModule operations."""

    @pytest.fixture
    def stub(self) -> StubClient:
        return StubClient()

    @pytest.fixture
    def module(self, stub: StubClient) -> JobsModule:
        return JobsModule(stub)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_submit_minimal(self, module: JobsModule) -> None:
        resp = await module.submit(
            model_hash=bytes.fromhex("11" * 32),
            input_hash=bytes.fromhex("22" * 32),
        )
        assert isinstance(resp, SubmitJobResponse)
        assert resp.job_id == "job_abc"

    @pytest.mark.asyncio
    async def test_submit_with_all_options(self, module: JobsModule) -> None:
        resp = await module.submit(
            model_hash=bytes.fromhex("11" * 32),
            input_hash=bytes.fromhex("22" * 32),
            proof_type=ProofType.ZKML,
            priority=5,
            max_gas=100000,
            timeout_blocks=200,
            callback_url="https://callback.test/done",
            metadata={"key": "val"},
        )
        assert resp.job_id == "job_abc"
        assert resp.estimated_blocks == 10

    @pytest.mark.asyncio
    async def test_submit_with_none_metadata(self, module: JobsModule, stub: StubClient) -> None:
        await module.submit(
            model_hash=bytes.fromhex("11" * 32),
            input_hash=bytes.fromhex("22" * 32),
            metadata=None,
        )
        posted_json = stub.posts[0][1]
        assert posted_json["metadata"] == {}

    @pytest.mark.asyncio
    async def test_get(self, module: JobsModule) -> None:
        job = await module.get("job_abc")
        assert isinstance(job, ComputeJob)
        assert job.id == "job_abc"
        assert job.creator == "aethel1creator"

    @pytest.mark.asyncio
    async def test_list_no_filters(self, module: JobsModule) -> None:
        jobs = await module.list()
        assert len(jobs) == 1
        assert jobs[0].id == "job_l1"

    @pytest.mark.asyncio
    async def test_list_with_status(self, module: JobsModule, stub: StubClient) -> None:
        await module.list(status=JobStatus.COMPLETED)
        params = stub.gets[-1][1]
        assert params["status"] == JobStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_list_with_creator(self, module: JobsModule, stub: StubClient) -> None:
        await module.list(creator="aethel1creator")
        params = stub.gets[-1][1]
        assert params["creator"] == "aethel1creator"

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, module: JobsModule, stub: StubClient) -> None:
        page = PageRequest(offset=5, limit=10, count_total=True)
        await module.list(pagination=page)
        params = stub.gets[-1][1]
        assert params["offset"] == 5
        assert params["limit"] == 10

    @pytest.mark.asyncio
    async def test_list_with_all_filters(self, module: JobsModule, stub: StubClient) -> None:
        page = PageRequest(offset=0, limit=20)
        await module.list(
            status=JobStatus.PENDING,
            creator="aethel1user",
            pagination=page,
        )
        params = stub.gets[-1][1]
        assert params["status"] == JobStatus.PENDING.value
        assert params["creator"] == "aethel1user"
        assert params["limit"] == 20

    @pytest.mark.asyncio
    async def test_list_pending_without_pagination(self, module: JobsModule) -> None:
        jobs = await module.list_pending()
        assert len(jobs) == 1
        assert jobs[0].id == "job_p1"

    @pytest.mark.asyncio
    async def test_list_pending_with_pagination(self, module: JobsModule, stub: StubClient) -> None:
        page = PageRequest(offset=0, limit=5)
        await module.list_pending(pagination=page)
        params = stub.gets[-1][1]
        assert params["limit"] == 5

    @pytest.mark.asyncio
    async def test_cancel(self, module: JobsModule, stub: StubClient) -> None:
        result = await module.cancel("job_abc")
        assert result is True
        assert "cancel" in stub.posts[-1][0]

    @pytest.mark.asyncio
    async def test_get_result(self, module: JobsModule) -> None:
        result = await module.get_result("job_abc")
        assert isinstance(result, JobResult)
        assert result.job_id == "job_abc"
        assert result.verified is True
        assert result.consensus_validators == 5

    @pytest.mark.asyncio
    async def test_wait_for_completion_immediate(self) -> None:
        """Job already in COMPLETED status returns immediately."""
        call_count = 0

        class ImmediateStub:
            async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
                nonlocal call_count
                call_count += 1
                return {
                    "job": {
                        "id": "job_done",
                        "creator": "aethel1cr",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": "JOB_STATUS_COMPLETED",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }

        module = JobsModule(ImmediateStub())  # type: ignore[arg-type]
        job = await module.wait_for_completion("job_done", poll_interval=0.001, timeout=0.01)
        assert job.status == JobStatus.COMPLETED
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_wait_for_completion_failed_state(self) -> None:
        """Job in FAILED status also terminates waiting."""

        class FailedStub:
            async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
                return {
                    "job": {
                        "id": "job_f",
                        "creator": "aethel1cr",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": "JOB_STATUS_FAILED",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }

        module = JobsModule(FailedStub())  # type: ignore[arg-type]
        job = await module.wait_for_completion("job_f", poll_interval=0.001, timeout=0.01)
        assert job.status == JobStatus.FAILED

    @pytest.mark.asyncio
    async def test_wait_for_completion_cancelled_state(self) -> None:
        """Job in CANCELLED status also terminates waiting."""

        class CancelledStub:
            async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
                return {
                    "job": {
                        "id": "job_c",
                        "creator": "aethel1cr",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": "JOB_STATUS_CANCELLED",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }

        module = JobsModule(CancelledStub())  # type: ignore[arg-type]
        job = await module.wait_for_completion("job_c", poll_interval=0.001, timeout=0.01)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self) -> None:
        """Raises TimeoutError when job stays pending."""

        class PendingForeverStub:
            async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
                return {
                    "job": {
                        "id": "job_p",
                        "creator": "aethel1cr",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": "JOB_STATUS_COMPUTING",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }

        module = JobsModule(PendingForeverStub())  # type: ignore[arg-type]
        with pytest.raises(TimeoutError, match="did not complete"):
            await module.wait_for_completion("job_p", poll_interval=0.001, timeout=0.005)

    @pytest.mark.asyncio
    async def test_wait_for_completion_polling(self) -> None:
        """Job transitions from COMPUTING to COMPLETED after a few polls."""
        poll_count = 0

        class TransitionStub:
            async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
                nonlocal poll_count
                poll_count += 1
                status = "JOB_STATUS_COMPLETED" if poll_count >= 3 else "JOB_STATUS_COMPUTING"
                return {
                    "job": {
                        "id": "job_t",
                        "creator": "aethel1cr",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": status,
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }

        module = JobsModule(TransitionStub())  # type: ignore[arg-type]
        job = await module.wait_for_completion("job_t", poll_interval=0.001, timeout=1.0)
        assert job.status == JobStatus.COMPLETED
        assert poll_count == 3


# ---------------------------------------------------------------------------
# SyncJobsModule
# ---------------------------------------------------------------------------


class TestSyncJobsModule:
    """Test synchronous wrapper for JobsModule."""

    @pytest.fixture
    def sync_module(self) -> SyncJobsModule:
        stub = StubClient()
        async_module = JobsModule(stub)  # type: ignore[arg-type]
        sync_client = MagicMock()
        sync_client._run = lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        return SyncJobsModule(sync_client, async_module)

    def test_submit(self, sync_module: SyncJobsModule) -> None:
        resp = sync_module.submit(
            model_hash=bytes.fromhex("11" * 32),
            input_hash=bytes.fromhex("22" * 32),
        )
        assert resp.job_id == "job_abc"

    def test_get(self, sync_module: SyncJobsModule) -> None:
        job = sync_module.get("job_abc")
        assert job.id == "job_abc"

    def test_list(self, sync_module: SyncJobsModule) -> None:
        jobs = sync_module.list()
        assert len(jobs) >= 1

    def test_list_pending(self, sync_module: SyncJobsModule) -> None:
        jobs = sync_module.list_pending()
        assert len(jobs) >= 1

    def test_cancel(self, sync_module: SyncJobsModule) -> None:
        result = sync_module.cancel("job_abc")
        assert result is True

    def test_get_result(self, sync_module: SyncJobsModule) -> None:
        result = sync_module.get_result("job_abc")
        assert result.job_id == "job_abc"

    def test_wait_for_completion(self) -> None:
        """SyncJobsModule.wait_for_completion with immediately completed job."""

        class CompletedStub:
            async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
                return {
                    "job": {
                        "id": "job_sync",
                        "creator": "aethel1cr",
                        "model_hash": "11" * 32,
                        "input_hash": "22" * 32,
                        "status": "JOB_STATUS_COMPLETED",
                        "proof_type": "PROOF_TYPE_TEE",
                    }
                }

        stub = CompletedStub()
        async_module = JobsModule(stub)  # type: ignore[arg-type]
        sync_client = MagicMock()
        sync_client._run = lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        sync_module = SyncJobsModule(sync_client, async_module)
        job = sync_module.wait_for_completion("job_sync", poll_interval=0.001, timeout=0.01)
        assert job.status == JobStatus.COMPLETED
