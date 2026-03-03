"""
Seals module for Aethelred SDK.

Provides functionality for creating, verifying, and managing digital seals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from aethelred.core.types import (
    CreateSealRequest,
    CreateSealResponse,
    DigitalSeal,
    PageRequest,
    RegulatoryInfo,
    SealStatus,
    VerifySealResponse,
)

if TYPE_CHECKING:
    from aethelred.core.client import AsyncAethelredClient, AethelredClient


class SealsModule:
    """Async module for digital seal operations."""
    
    BASE_PATH = "/aethelred/seal/v1"
    
    def __init__(self, client: "AsyncAethelredClient"):
        self._client = client
    
    async def create(
        self,
        job_id: str,
        regulatory_info: Optional[RegulatoryInfo] = None,
        expires_in_blocks: int = 0,
        metadata: Optional[dict] = None,
    ) -> CreateSealResponse:
        """Create a digital seal for a completed job.
        
        Args:
            job_id: ID of the completed job
            regulatory_info: Regulatory compliance information
            expires_in_blocks: Number of blocks until expiration (0 = never)
            metadata: Additional metadata
        
        Returns:
            CreateSealResponse with seal_id and tx_hash
        """
        request = CreateSealRequest(
            job_id=job_id,
            regulatory_info=regulatory_info,
            expires_in_blocks=expires_in_blocks,
            metadata=metadata or {},
        )
        
        data = await self._client.post(
            f"{self.BASE_PATH}/seals",
            json=request.model_dump(mode="json"),
        )
        return CreateSealResponse(**data)
    
    async def get(self, seal_id: str) -> DigitalSeal:
        """Get a seal by ID.
        
        Args:
            seal_id: Unique seal identifier
        
        Returns:
            DigitalSeal details
        """
        data = await self._client.get(f"{self.BASE_PATH}/seals/{seal_id}")
        return DigitalSeal(**data.get("seal", data))
    
    async def list(
        self,
        requester: Optional[str] = None,
        model_hash: Optional[str] = None,
        status: Optional[SealStatus] = None,
        pagination: Optional[PageRequest] = None,
    ) -> List[DigitalSeal]:
        """List seals with optional filters.
        
        Args:
            requester: Filter by requester address
            model_hash: Filter by model hash
            status: Filter by seal status
            pagination: Pagination parameters
        
        Returns:
            List of DigitalSeal objects
        """
        params = {}
        if requester:
            params["requester"] = requester
        if model_hash:
            params["model_hash"] = model_hash
        if status:
            params["status"] = status.value
        if pagination:
            params.update(pagination.model_dump(exclude_none=True))
        
        data = await self._client.get(f"{self.BASE_PATH}/seals", params=params)
        return [DigitalSeal(**seal) for seal in data.get("seals", [])]
    
    async def list_by_model(
        self,
        model_hash: str,
        pagination: Optional[PageRequest] = None,
    ) -> List[DigitalSeal]:
        """List seals for a specific model.
        
        Args:
            model_hash: Model hash to filter by
            pagination: Pagination parameters
        
        Returns:
            List of DigitalSeal objects
        """
        params = {"model_hash": model_hash}
        if pagination:
            params.update(pagination.model_dump(exclude_none=True))
        
        data = await self._client.get(f"{self.BASE_PATH}/seals/by_model", params=params)
        return [DigitalSeal(**seal) for seal in data.get("seals", [])]
    
    async def verify(self, seal_id: str) -> VerifySealResponse:
        """Verify a seal's validity.
        
        Args:
            seal_id: Unique seal identifier
        
        Returns:
            VerifySealResponse with validity and details
        """
        data = await self._client.get(f"{self.BASE_PATH}/seals/{seal_id}/verify")
        return VerifySealResponse(**data)
    
    async def revoke(self, seal_id: str, reason: str) -> bool:
        """Revoke a seal.
        
        Args:
            seal_id: Unique seal identifier
            reason: Reason for revocation
        
        Returns:
            True if revoked successfully
        """
        await self._client.post(
            f"{self.BASE_PATH}/seals/{seal_id}/revoke",
            json={"reason": reason},
        )
        return True
    
    async def export(self, seal_id: str, format: str = "json") -> bytes:
        """Export a seal in the specified format.
        
        Args:
            seal_id: Unique seal identifier
            format: Export format (json, cbor, protobuf)
        
        Returns:
            Serialized seal data
        """
        data = await self._client.get(
            f"{self.BASE_PATH}/seals/{seal_id}/export",
            params={"format": format},
        )
        return data.get("data", b"")


class SyncSealsModule:
    """Synchronous wrapper for SealsModule."""
    
    def __init__(self, client: "AethelredClient", async_module: SealsModule):
        self._client = client
        self._async = async_module
    
    def create(self, *args, **kwargs) -> CreateSealResponse:
        return self._client._run(self._async.create(*args, **kwargs))
    
    def get(self, seal_id: str) -> DigitalSeal:
        return self._client._run(self._async.get(seal_id))
    
    def list(self, *args, **kwargs) -> List[DigitalSeal]:
        return self._client._run(self._async.list(*args, **kwargs))
    
    def list_by_model(self, *args, **kwargs) -> List[DigitalSeal]:
        return self._client._run(self._async.list_by_model(*args, **kwargs))
    
    def verify(self, seal_id: str) -> VerifySealResponse:
        return self._client._run(self._async.verify(seal_id))
    
    def revoke(self, seal_id: str, reason: str) -> bool:
        return self._client._run(self._async.revoke(seal_id, reason))
    
    def export(self, *args, **kwargs) -> bytes:
        return self._client._run(self._async.export(*args, **kwargs))


__all__ = ["SealsModule", "SyncSealsModule"]
