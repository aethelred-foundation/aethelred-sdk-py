"""Validators module for Aethelred SDK."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from aethelred.core.types import HardwareCapability, PageRequest, ValidatorStats

if TYPE_CHECKING:
    from aethelred.core.client import AsyncAethelredClient, AethelredClient

class ValidatorsModule:
    """Async module for validator operations."""
    BASE_PATH = "/aethelred/pouw/v1"
    
    def __init__(self, client: "AsyncAethelredClient"):
        self._client = client
    
    async def get_stats(self, address: str) -> ValidatorStats:
        """Get validator statistics."""
        data = await self._client.get(f"{self.BASE_PATH}/validators/{address}/stats")
        return ValidatorStats(**data)
    
    async def list(self, pagination: Optional[PageRequest] = None) -> List[ValidatorStats]:
        """List all validators."""
        params = pagination.model_dump(exclude_none=True) if pagination else {}
        data = await self._client.get(f"{self.BASE_PATH}/validators", params=params)
        return [ValidatorStats(**v) for v in data.get("validators", [])]
    
    async def register_capability(self, address: str, capability: HardwareCapability) -> bool:
        """Register validator hardware capability."""
        await self._client.post(
            f"{self.BASE_PATH}/validators/{address}/capability",
            json={"hardware_capabilities": capability.model_dump()},
        )
        return True

class SyncValidatorsModule:
    """Synchronous wrapper for ValidatorsModule."""
    def __init__(self, client: "AethelredClient", async_module: ValidatorsModule):
        self._client = client
        self._async = async_module
    def get_stats(self, address: str) -> ValidatorStats:
        return self._client._run(self._async.get_stats(address))
    def list(self, *args, **kwargs) -> List[ValidatorStats]:
        return self._client._run(self._async.list(*args, **kwargs))
    def register_capability(self, *args, **kwargs) -> bool:
        return self._client._run(self._async.register_capability(*args, **kwargs))

__all__ = ["ValidatorsModule", "SyncValidatorsModule"]
