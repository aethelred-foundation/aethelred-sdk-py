"""Shared test fixtures for Aethelred SDK tests.

Provides mock clients, wallets, and data factories to avoid
duplicating setup code across test modules.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Mock Client
# ---------------------------------------------------------------------------


class MockClient:
    """Mock Aethelred client for testing without network access."""

    def __init__(self) -> None:
        self.address = "aeth1mockaddress1234567890abcdef"
        self._config = MagicMock()
        self._config.endpoint = "https://mock.aethelred.local"
        self._requests: list[dict] = []

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        """Record and return a mock response."""
        self._requests.append({"method": method, "path": path, **kwargs})
        return self._default_response(path)

    async def _query(self, path: str, **kwargs: Any) -> dict:
        self._requests.append({"method": "QUERY", "path": path, **kwargs})
        return self._default_response(path)

    async def _submit_tx(self, msg: dict) -> dict:
        self._requests.append({"method": "TX", "msg": msg})
        return {
            "tx_hash": "AABB" * 16,
            "block_height": 42,
            "model_id": "model_mock123",
        }

    def _default_response(self, path: str) -> dict:
        if "feeds" in path and "history" not in path and "verify" not in path:
            return _mock_feed_response()
        if "verify" in path:
            return {"is_valid": True}
        if "nodes" in path:
            return {"nodes": [_mock_oracle_node()]}
        if "model" in path:
            return {"model": _mock_model_data()}
        if "history" in path:
            return {"values": [{"value": 42000, "timestamp": "2026-01-01T00:00:00+00:00"}]}
        return {}


@pytest.fixture
def mock_client() -> MockClient:
    """Provide a mock Aethelred client."""
    return MockClient()


# ---------------------------------------------------------------------------
# Data Factories
# ---------------------------------------------------------------------------


def _mock_feed_response() -> dict:
    return {
        "value": 42000.0,
        "value_hash": hashlib.sha256(b'42000.0').hexdigest(),
        "timestamp": "2026-01-15T12:00:00+00:00",
        "metadata": {
            "name": "BTC/USD",
            "description": "Bitcoin price",
            "feed_type": "market_data",
            "value_type": "number",
            "unit": "USD",
            "source": "aggregator",
        },
        "attestation": {
            "attestation_id": "att_mock123",
            "data_hash": "deadbeef" * 8,
            "method": "tee",
            "oracle_node_id": "node_1",
            "timestamp": "2026-01-15T12:00:00+00:00",
            "tee_platform": "TEE_PLATFORM_INTEL_SGX",
            "tx_hash": "AABB" * 16,
            "block_height": 100,
        },
        "feeds": [
            {
                "feed_id": "market_data/btc_usd",
                "name": "BTC/USD",
                "description": "Bitcoin price feed",
                "feed_type": "market_data",
                "value_type": "number",
                "unit": "USD",
            }
        ],
    }


def _mock_oracle_node() -> dict:
    return {
        "node_id": "node_1",
        "name": "TestOracle",
        "operator": "aeth1operator",
        "endpoint": "https://oracle.test",
        "supported_feeds": ["market_data/btc_usd"],
        "tee_platform": "TEE_PLATFORM_INTEL_SGX",
        "stake": 1000000,
        "uptime": 0.995,
        "is_active": True,
    }


def _mock_model_data() -> dict:
    return {
        "model_id": "model_mock123",
        "name": "test-model",
        "version": "1.0.0",
        "owner": "aeth1mockowner",
        "model_hash": "aa" * 32,
        "circuit_hash": "bb" * 32,
        "verification_key_hash": "cc" * 32,
        "framework": "pytorch",
        "input_shape": [1, 28, 28],
        "output_shape": [1, 10],
        "status": "active",
        "registered_at": "2026-01-01T00:00:00+00:00",
    }
