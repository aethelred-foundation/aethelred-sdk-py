"""
Model Registry for Aethelred

Manages registered models and their associated circuits on the network.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from aethelred.core.types import Circuit, ModelInfo

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Model registration status."""

    PENDING = "pending"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"


@dataclass
class RegisteredModel:
    """A model registered on the Aethelred network."""

    model_id: str
    name: str
    version: str
    owner: str

    # Hashes
    model_hash: str
    circuit_hash: str
    verification_key_hash: str

    # Circuit info
    circuit: Optional[Circuit] = None

    # Metadata
    description: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    category: Optional[str] = None

    # Architecture
    framework: str = "pytorch"
    architecture: Optional[str] = None
    input_shape: tuple[int, ...] = ()
    output_shape: tuple[int, ...] = ()
    parameter_count: int = 0

    # Performance
    accuracy: Optional[float] = None
    inference_time_ms: Optional[int] = None
    proving_time_ms: Optional[int] = None

    # Status
    status: ModelStatus = ModelStatus.PENDING
    registered_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    # Blockchain
    tx_hash: Optional[str] = None
    block_height: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "owner": self.owner,
            "model_hash": self.model_hash,
            "circuit_hash": self.circuit_hash,
            "status": self.status.value,
            "framework": self.framework,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
        }


class ModelRegistry:
    """
    Client for the Aethelred Model Registry.

    Manages model registration, lookup, and verification.

    Example:
        >>> registry = ModelRegistry(client)
        >>>
        >>> # Register a new model
        >>> registered = await registry.register(
        ...     circuit=circuit,
        ...     name="credit-score-v1",
        ...     description="Credit scoring model for loan applications",
        ... )
        >>>
        >>> # Look up a model
        >>> model = await registry.get("model_abc123")
        >>>
        >>> # List models
        >>> models = await registry.list(owner="aethel1...")
    """

    def __init__(self, client: Any):
        """
        Initialize the model registry.

        Args:
            client: Aethelred client instance
        """
        self._client = client

    async def register(
        self,
        circuit: Circuit,
        name: str,
        *,
        version: str = "1.0.0",
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        category: Optional[str] = None,
        accuracy: Optional[float] = None,
        public: bool = True,
    ) -> RegisteredModel:
        """
        Register a model circuit on the network.

        Args:
            circuit: Compiled circuit to register
            name: Human-readable model name
            version: Model version string
            description: Model description
            tags: Searchable tags
            category: Model category (e.g., "credit-scoring", "fraud-detection")
            accuracy: Model accuracy/performance metric
            public: Whether model is publicly discoverable

        Returns:
            Registered model with on-chain ID
        """
        logger.info(f"Registering model: {name} v{version}")

        # Prepare registration message
        msg = {
            "type": "aethelred/MsgRegisterModel",
            "value": {
                "name": name,
                "version": version,
                "model_hash": circuit.model_hash,
                "circuit_hash": self._compute_circuit_hash(circuit),
                "verification_key_hash": self._compute_vk_hash(circuit),
                "input_shape": list(circuit.input_shape),
                "output_shape": list(circuit.output_shape),
                "framework": circuit.framework,
                "description": description or "",
                "tags": tags or [],
                "category": category or "",
                "public": public,
            },
        }

        # Submit transaction
        result = await self._client._submit_tx(msg)

        return RegisteredModel(
            model_id=result.get("model_id", f"model_{circuit.circuit_id}"),
            name=name,
            version=version,
            owner=self._client.address,
            model_hash=circuit.model_hash,
            circuit_hash=msg["value"]["circuit_hash"],
            verification_key_hash=msg["value"]["verification_key_hash"],
            circuit=circuit,
            description=description,
            tags=tags or [],
            category=category,
            framework=circuit.framework,
            input_shape=circuit.input_shape,
            output_shape=circuit.output_shape,
            accuracy=accuracy,
            status=ModelStatus.ACTIVE,
            registered_at=datetime.now(timezone.utc),
            tx_hash=result.get("tx_hash"),
            block_height=result.get("block_height"),
        )

    async def get(self, model_id: str) -> Optional[RegisteredModel]:
        """
        Get a registered model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Registered model or None if not found
        """
        response = await self._client._query(f"/aethelred/model/v1/model/{model_id}")

        if not response or "model" not in response:
            return None

        return self._parse_model(response["model"])

    async def get_by_hash(self, model_hash: str) -> Optional[RegisteredModel]:
        """
        Get a registered model by model hash.

        Args:
            model_hash: SHA-256 hash of model file

        Returns:
            Registered model or None if not found
        """
        response = await self._client._query(
            f"/aethelred/model/v1/model/hash/{model_hash}"
        )

        if not response or "model" not in response:
            return None

        return self._parse_model(response["model"])

    async def list(
        self,
        *,
        owner: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: Optional[ModelStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RegisteredModel]:
        """
        List registered models with optional filters.

        Args:
            owner: Filter by owner address
            category: Filter by category
            tags: Filter by tags (any match)
            status: Filter by status
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of registered models
        """
        params = {
            "pagination.limit": limit,
            "pagination.offset": offset,
        }

        if owner:
            params["owner"] = owner
        if category:
            params["category"] = category
        if tags:
            params["tags"] = ",".join(tags)
        if status:
            params["status"] = status.value

        response = await self._client._query(
            "/aethelred/model/v1/models",
            params=params,
        )

        models = []
        for model_data in response.get("models", []):
            models.append(self._parse_model(model_data))

        return models

    async def deprecate(
        self,
        model_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Deprecate a registered model.

        Args:
            model_id: Model identifier
            reason: Deprecation reason

        Returns:
            True if successful
        """
        msg = {
            "type": "aethelred/MsgDeprecateModel",
            "value": {
                "model_id": model_id,
                "reason": reason or "",
            },
        }

        await self._client._submit_tx(msg)
        return True

    async def update_metadata(
        self,
        model_id: str,
        *,
        description: Optional[str] = None,
        tags: Optional[list[str]] = None,
        accuracy: Optional[float] = None,
    ) -> RegisteredModel:
        """
        Update model metadata.

        Args:
            model_id: Model identifier
            description: New description
            tags: New tags
            accuracy: Updated accuracy metric

        Returns:
            Updated model
        """
        msg = {
            "type": "aethelred/MsgUpdateModelMetadata",
            "value": {
                "model_id": model_id,
                "description": description,
                "tags": tags,
                "accuracy": str(accuracy) if accuracy else None,
            },
        }

        await self._client._submit_tx(msg)
        return await self.get(model_id)

    def _parse_model(self, data: dict[str, Any]) -> RegisteredModel:
        """Parse model data from chain response."""
        return RegisteredModel(
            model_id=data["model_id"],
            name=data["name"],
            version=data["version"],
            owner=data["owner"],
            model_hash=data["model_hash"],
            circuit_hash=data["circuit_hash"],
            verification_key_hash=data.get("verification_key_hash", ""),
            description=data.get("description"),
            tags=data.get("tags", []),
            category=data.get("category"),
            framework=data.get("framework", "pytorch"),
            input_shape=tuple(data.get("input_shape", [])),
            output_shape=tuple(data.get("output_shape", [])),
            parameter_count=data.get("parameter_count", 0),
            accuracy=float(data["accuracy"]) if data.get("accuracy") else None,
            status=ModelStatus(data.get("status", "active")),
            registered_at=datetime.fromisoformat(data["registered_at"]) if data.get("registered_at") else None,
            tx_hash=data.get("tx_hash"),
            block_height=data.get("block_height"),
        )

    def _compute_circuit_hash(self, circuit: Circuit) -> str:
        """Compute hash of circuit binary."""
        import hashlib
        return hashlib.sha256(circuit.circuit_binary).hexdigest()

    def _compute_vk_hash(self, circuit: Circuit) -> str:
        """Compute hash of verification key."""
        import hashlib
        return hashlib.sha256(circuit.verification_key).hexdigest()
