"""Models module for Aethelred SDK."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional
from aethelred.core.types import (
    PageRequest, RegisteredModel, RegisterModelRequest, RegisterModelResponse, UtilityCategory,
)

if TYPE_CHECKING:
    from aethelred.core.client import AsyncAethelredClient, AethelredClient

class ModelsModule:
    """Async module for model registry operations."""
    BASE_PATH = "/aethelred/pouw/v1"
    
    def __init__(self, client: "AsyncAethelredClient"):
        self._client = client
    
    async def register(
        self, model_hash: bytes, name: str, architecture: str = "", version: str = "1.0.0",
        category: UtilityCategory = UtilityCategory.GENERAL, input_schema: str = "",
        output_schema: str = "", storage_uri: str = "", metadata: Optional[dict] = None,
    ) -> RegisterModelResponse:
        """Register a new model."""
        request = RegisterModelRequest(
            model_hash=model_hash, name=name, architecture=architecture, version=version,
            category=category, input_schema=input_schema, output_schema=output_schema,
            storage_uri=storage_uri, metadata=metadata or {},
        )
        data = await self._client.post(f"{self.BASE_PATH}/models", json=request.model_dump(mode="json"))
        return RegisterModelResponse(**data)
    
    async def get(self, model_hash: str) -> RegisteredModel:
        """Get a model by hash."""
        data = await self._client.get(f"{self.BASE_PATH}/models/{model_hash}")
        return RegisteredModel(**data.get("model", data))
    
    async def list(
        self, owner: Optional[str] = None, category: Optional[UtilityCategory] = None,
        pagination: Optional[PageRequest] = None,
    ) -> List[RegisteredModel]:
        """List models with optional filters."""
        params = {}
        if owner: params["owner"] = owner
        if category: params["category"] = category.value
        if pagination: params.update(pagination.model_dump(exclude_none=True))
        data = await self._client.get(f"{self.BASE_PATH}/models", params=params)
        return [RegisteredModel(**m) for m in data.get("models", [])]

class SyncModelsModule:
    """Synchronous wrapper for ModelsModule."""
    def __init__(self, client: "AethelredClient", async_module: ModelsModule):
        self._client = client
        self._async = async_module
    def register(self, *args, **kwargs) -> RegisterModelResponse:
        return self._client._run(self._async.register(*args, **kwargs))
    def get(self, model_hash: str) -> RegisteredModel:
        return self._client._run(self._async.get(model_hash))
    def list(self, *args, **kwargs) -> List[RegisteredModel]:
        return self._client._run(self._async.list(*args, **kwargs))

__all__ = ["ModelsModule", "SyncModelsModule"]
