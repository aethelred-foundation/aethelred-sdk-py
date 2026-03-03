"""
Jobs module for Aethelred SDK.

Provides functionality for submitting and managing AI compute jobs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from aethelred.core.types import (
    ComputeJob,
    JobResult,
    JobStatus,
    PageRequest,
    ProofType,
    SubmitJobRequest,
    SubmitJobResponse,
)

if TYPE_CHECKING:
    from aethelred.core.client import AsyncAethelredClient, AethelredClient


class JobsModule:
    """Async module for job operations."""
    
    BASE_PATH = "/aethelred/pouw/v1"
    
    def __init__(self, client: "AsyncAethelredClient"):
        self._client = client
    
    async def submit(
        self,
        model_hash: bytes,
        input_hash: bytes,
        proof_type: ProofType = ProofType.TEE,
        priority: int = 1,
        max_gas: int = 0,
        timeout_blocks: int = 100,
        callback_url: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SubmitJobResponse:
        """Submit a new compute job.
        
        Args:
            model_hash: SHA-256 hash of the model
            input_hash: SHA-256 hash of input data
            proof_type: Type of proof required (TEE, ZKML, HYBRID)
            priority: Job priority (1-10)
            max_gas: Maximum gas to spend
            timeout_blocks: Number of blocks before timeout
            callback_url: URL to call when job completes
            metadata: Additional metadata
        
        Returns:
            SubmitJobResponse with job_id and tx_hash
        """
        request = SubmitJobRequest(
            model_hash=model_hash,
            input_hash=input_hash,
            proof_type=proof_type,
            priority=priority,
            max_gas=max_gas,
            timeout_blocks=timeout_blocks,
            callback_url=callback_url,
            metadata=metadata or {},
        )
        
        data = await self._client.post(
            f"{self.BASE_PATH}/jobs",
            json=request.model_dump(mode="json"),
        )
        return SubmitJobResponse(**data)
    
    async def get(self, job_id: str) -> ComputeJob:
        """Get a job by ID.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            ComputeJob details
        """
        data = await self._client.get(f"{self.BASE_PATH}/jobs/{job_id}")
        return ComputeJob(**data.get("job", data))
    
    async def list(
        self,
        status: Optional[JobStatus] = None,
        creator: Optional[str] = None,
        pagination: Optional[PageRequest] = None,
    ) -> List[ComputeJob]:
        """List jobs with optional filters.
        
        Args:
            status: Filter by job status
            creator: Filter by creator address
            pagination: Pagination parameters
        
        Returns:
            List of ComputeJob objects
        """
        params = {}
        if status:
            params["status"] = status.value
        if creator:
            params["creator"] = creator
        if pagination:
            params.update(pagination.model_dump(exclude_none=True))
        
        data = await self._client.get(f"{self.BASE_PATH}/jobs", params=params)
        return [ComputeJob(**job) for job in data.get("jobs", [])]
    
    async def list_pending(self, pagination: Optional[PageRequest] = None) -> List[ComputeJob]:
        """List pending jobs.
        
        Args:
            pagination: Pagination parameters
        
        Returns:
            List of pending ComputeJob objects
        """
        params = pagination.model_dump(exclude_none=True) if pagination else {}
        data = await self._client.get(f"{self.BASE_PATH}/jobs/pending", params=params)
        return [ComputeJob(**job) for job in data.get("jobs", [])]
    
    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending job.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            True if cancelled successfully
        """
        await self._client.post(f"{self.BASE_PATH}/jobs/{job_id}/cancel")
        return True
    
    async def get_result(self, job_id: str) -> JobResult:
        """Get the result of a completed job.
        
        Args:
            job_id: Unique job identifier
        
        Returns:
            JobResult with output and proof
        """
        data = await self._client.get(f"{self.BASE_PATH}/jobs/{job_id}/result")
        return JobResult(**data)
    
    async def wait_for_completion(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> ComputeJob:
        """Wait for a job to complete.
        
        Args:
            job_id: Unique job identifier
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
        
        Returns:
            Completed ComputeJob
        
        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        import asyncio
        from aethelred.core.exceptions import TimeoutError
        
        elapsed = 0.0
        while elapsed < timeout:
            job = await self.get(job_id)
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


class SyncJobsModule:
    """Synchronous wrapper for JobsModule."""
    
    def __init__(self, client: "AethelredClient", async_module: JobsModule):
        self._client = client
        self._async = async_module
    
    def submit(self, *args, **kwargs) -> SubmitJobResponse:
        return self._client._run(self._async.submit(*args, **kwargs))
    
    def get(self, job_id: str) -> ComputeJob:
        return self._client._run(self._async.get(job_id))
    
    def list(self, *args, **kwargs) -> List[ComputeJob]:
        return self._client._run(self._async.list(*args, **kwargs))
    
    def list_pending(self, *args, **kwargs) -> List[ComputeJob]:
        return self._client._run(self._async.list_pending(*args, **kwargs))
    
    def cancel(self, job_id: str) -> bool:
        return self._client._run(self._async.cancel(job_id))
    
    def get_result(self, job_id: str) -> JobResult:
        return self._client._run(self._async.get_result(job_id))
    
    def wait_for_completion(self, *args, **kwargs) -> ComputeJob:
        return self._client._run(self._async.wait_for_completion(*args, **kwargs))


__all__ = ["JobsModule", "SyncJobsModule"]
