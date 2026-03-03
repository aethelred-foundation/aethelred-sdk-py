"""Tests for the ModelRegistry — registration, lookup, deprecation.

Covers:
- Model registration via mock client
- Model lookup by ID and hash
- Model listing with filters
- Deprecation and metadata updates
- Hash computation helpers
"""

from __future__ import annotations

import pytest

from aethelred.models.registry import (
    ModelRegistry,
    ModelStatus,
    RegisteredModel,
)
from aethelred.core.types import Circuit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(mock_client) -> ModelRegistry:
    return ModelRegistry(mock_client)


@pytest.fixture
def mock_circuit() -> Circuit:
    """Create a minimal mock Circuit."""
    return Circuit(
        circuit_id="test_circuit",
        model_hash="aa" * 32,
        circuit_binary=b"\x00" * 64,
        verification_key=b"\x01" * 32,
        input_shape=(1, 28, 28),
        output_shape=(1, 10),
        framework="pytorch",
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Test model registration."""

    @pytest.mark.asyncio
    async def test_register_returns_model(
        self, registry: ModelRegistry, mock_circuit: Circuit
    ) -> None:
        result = await registry.register(
            circuit=mock_circuit,
            name="test-model",
            version="1.0.0",
            tags=["test"],
        )
        assert isinstance(result, RegisteredModel)
        assert result.name == "test-model"
        assert result.status == ModelStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_register_records_tx(
        self, registry: ModelRegistry, mock_circuit: Circuit
    ) -> None:
        result = await registry.register(circuit=mock_circuit, name="m1")
        assert result.tx_hash is not None
        assert result.block_height is not None


# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------


class TestLookup:
    """Test model lookup methods."""

    @pytest.mark.asyncio
    async def test_get_by_id(self, registry: ModelRegistry) -> None:
        model = await registry.get("model_mock123")
        assert model is not None
        assert model.model_id == "model_mock123"

    @pytest.mark.asyncio
    async def test_get_by_hash(self, registry: ModelRegistry) -> None:
        model = await registry.get_by_hash("aa" * 32)
        assert model is not None

    @pytest.mark.asyncio
    async def test_list_models(self, registry: ModelRegistry) -> None:
        # Mock client returns models under "models" key
        models = await registry.list()
        assert isinstance(models, list)


# ---------------------------------------------------------------------------
# Deprecation
# ---------------------------------------------------------------------------


class TestDeprecation:
    """Test model deprecation."""

    @pytest.mark.asyncio
    async def test_deprecate(self, registry: ModelRegistry) -> None:
        result = await registry.deprecate("model_mock123", reason="outdated")
        assert result is True


# ---------------------------------------------------------------------------
# Hash Computation
# ---------------------------------------------------------------------------


class TestHashComputation:
    """Test internal hash helpers."""

    def test_circuit_hash_deterministic(
        self, registry: ModelRegistry, mock_circuit: Circuit
    ) -> None:
        h1 = registry._compute_circuit_hash(mock_circuit)
        h2 = registry._compute_circuit_hash(mock_circuit)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_vk_hash_deterministic(
        self, registry: ModelRegistry, mock_circuit: Circuit
    ) -> None:
        h1 = registry._compute_vk_hash(mock_circuit)
        h2 = registry._compute_vk_hash(mock_circuit)
        assert h1 == h2


# ---------------------------------------------------------------------------
# RegisteredModel Serialization
# ---------------------------------------------------------------------------


class TestRegisteredModelSerialization:
    """Test RegisteredModel.to_dict()."""

    def test_to_dict_contains_required_fields(self) -> None:
        model = RegisteredModel(
            model_id="m1",
            name="test",
            version="1.0.0",
            owner="aethel1owner",
            model_hash="aa" * 32,
            circuit_hash="bb" * 32,
            verification_key_hash="cc" * 32,
        )
        d = model.to_dict()
        assert d["model_id"] == "m1"
        assert d["name"] == "test"
        assert d["status"] == "pending"
