"""Comprehensive tests for aethelred/seals/__init__.py — targeting 95%+ coverage.

Covers all SealsModule methods: create, get, list (all filter combos),
list_by_model (with/without pagination), verify, revoke, export.
Also covers SyncSealsModule wrapper methods.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest

from aethelred.core.types import (
    CreateSealResponse,
    DigitalSeal,
    PageRequest,
    RegulatoryInfo,
    SealStatus,
    VerifySealResponse,
)
from aethelred.seals import SealsModule, SyncSealsModule


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
        if path.endswith("/seals"):
            return {"seal_id": "seal_abc", "tx_hash": "CC" * 32}
        if "revoke" in path:
            return {}
        return {}

    async def get(self, path: str, params: Any = None) -> Dict[str, Any]:
        self.gets.append((path, params))
        if "/seals/seal_abc/verify" in path:
            return {
                "valid": True,
                "verification_details": {"hash_match": True, "attestation_valid": True},
                "errors": [],
            }
        if "/seals/seal_abc/export" in path:
            return {"data": "cafebabe"}
        if path.endswith("/seals/seal_abc"):
            return {
                "seal": {
                    "id": "seal_abc",
                    "job_id": "job_1",
                    "model_hash": "aa" * 32,
                    "status": "SEAL_STATUS_ACTIVE",
                    "requester": "aethel1user",
                }
            }
        if "/seals/by_model" in path:
            return {
                "seals": [
                    {
                        "id": "seal_m1",
                        "job_id": "job_m1",
                        "model_hash": "bb" * 32,
                        "status": "SEAL_STATUS_ACTIVE",
                        "requester": "aethel1user",
                    }
                ]
            }
        if path.endswith("/seals"):
            return {
                "seals": [
                    {
                        "id": "seal_list_1",
                        "job_id": "job_l1",
                        "model_hash": "cc" * 32,
                        "status": "SEAL_STATUS_ACTIVE",
                        "requester": "aethel1user",
                    }
                ]
            }
        return {}


# ---------------------------------------------------------------------------
# SealsModule (async)
# ---------------------------------------------------------------------------


class TestSealsModule:
    """Test async SealsModule operations."""

    @pytest.fixture
    def stub(self) -> StubClient:
        return StubClient()

    @pytest.fixture
    def module(self, stub: StubClient) -> SealsModule:
        return SealsModule(stub)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_create_minimal(self, module: SealsModule) -> None:
        resp = await module.create(job_id="job_1")
        assert isinstance(resp, CreateSealResponse)
        assert resp.seal_id == "seal_abc"

    @pytest.mark.asyncio
    async def test_create_with_all_options(self, module: SealsModule) -> None:
        reg = RegulatoryInfo(jurisdiction="US", compliance_frameworks=["HIPAA"])
        resp = await module.create(
            job_id="job_1",
            regulatory_info=reg,
            expires_in_blocks=1000,
            metadata={"key": "val"},
        )
        assert resp.seal_id == "seal_abc"

    @pytest.mark.asyncio
    async def test_create_with_none_metadata(self, module: SealsModule, stub: StubClient) -> None:
        resp = await module.create(job_id="job_1", metadata=None)
        assert resp.seal_id == "seal_abc"
        # Verify metadata was defaulted to {}
        posted_json = stub.posts[0][1]
        assert posted_json["metadata"] == {}

    @pytest.mark.asyncio
    async def test_get(self, module: SealsModule) -> None:
        seal = await module.get("seal_abc")
        assert isinstance(seal, DigitalSeal)
        assert seal.id == "seal_abc"
        assert seal.job_id == "job_1"

    @pytest.mark.asyncio
    async def test_list_no_filters(self, module: SealsModule) -> None:
        seals = await module.list()
        assert len(seals) == 1
        assert seals[0].id == "seal_list_1"

    @pytest.mark.asyncio
    async def test_list_with_requester(self, module: SealsModule, stub: StubClient) -> None:
        await module.list(requester="aethel1user")
        params = stub.gets[-1][1]
        assert params["requester"] == "aethel1user"

    @pytest.mark.asyncio
    async def test_list_with_model_hash(self, module: SealsModule, stub: StubClient) -> None:
        await module.list(model_hash="aa" * 32)
        params = stub.gets[-1][1]
        assert params["model_hash"] == "aa" * 32

    @pytest.mark.asyncio
    async def test_list_with_status(self, module: SealsModule, stub: StubClient) -> None:
        await module.list(status=SealStatus.ACTIVE)
        params = stub.gets[-1][1]
        assert params["status"] == SealStatus.ACTIVE.value

    @pytest.mark.asyncio
    async def test_list_with_pagination(self, module: SealsModule, stub: StubClient) -> None:
        page = PageRequest(offset=10, limit=5, count_total=True)
        await module.list(pagination=page)
        params = stub.gets[-1][1]
        assert params["offset"] == 10
        assert params["limit"] == 5

    @pytest.mark.asyncio
    async def test_list_with_all_filters(self, module: SealsModule, stub: StubClient) -> None:
        page = PageRequest(offset=0, limit=10)
        await module.list(
            requester="aethel1user",
            model_hash="dd" * 32,
            status=SealStatus.REVOKED,
            pagination=page,
        )
        params = stub.gets[-1][1]
        assert params["requester"] == "aethel1user"
        assert params["model_hash"] == "dd" * 32
        assert params["status"] == SealStatus.REVOKED.value

    @pytest.mark.asyncio
    async def test_list_by_model_without_pagination(self, module: SealsModule) -> None:
        seals = await module.list_by_model(model_hash="bb" * 32)
        assert len(seals) == 1
        assert seals[0].id == "seal_m1"

    @pytest.mark.asyncio
    async def test_list_by_model_with_pagination(self, module: SealsModule, stub: StubClient) -> None:
        page = PageRequest(offset=0, limit=5)
        await module.list_by_model(model_hash="bb" * 32, pagination=page)
        params = stub.gets[-1][1]
        assert params["model_hash"] == "bb" * 32
        assert params["limit"] == 5

    @pytest.mark.asyncio
    async def test_verify(self, module: SealsModule) -> None:
        resp = await module.verify("seal_abc")
        assert isinstance(resp, VerifySealResponse)
        assert resp.valid is True

    @pytest.mark.asyncio
    async def test_revoke(self, module: SealsModule, stub: StubClient) -> None:
        result = await module.revoke("seal_abc", reason="Policy violation")
        assert result is True
        posted = stub.posts[-1]
        assert "revoke" in posted[0]
        assert posted[1]["reason"] == "Policy violation"

    @pytest.mark.asyncio
    async def test_export_default_format(self, module: SealsModule) -> None:
        data = await module.export("seal_abc")
        assert data == "cafebabe"

    @pytest.mark.asyncio
    async def test_export_custom_format(self, module: SealsModule, stub: StubClient) -> None:
        await module.export("seal_abc", format="cbor")
        params = stub.gets[-1][1]
        assert params["format"] == "cbor"


# ---------------------------------------------------------------------------
# SyncSealsModule
# ---------------------------------------------------------------------------


class TestSyncSealsModule:
    """Test synchronous wrapper for SealsModule."""

    @pytest.fixture
    def sync_module(self) -> SyncSealsModule:
        stub = StubClient()
        async_module = SealsModule(stub)  # type: ignore[arg-type]
        sync_client = MagicMock()
        sync_client._run = lambda coro: asyncio.get_event_loop().run_until_complete(coro)
        return SyncSealsModule(sync_client, async_module)

    def test_create(self, sync_module: SyncSealsModule) -> None:
        resp = sync_module.create(job_id="job_1")
        assert resp.seal_id == "seal_abc"

    def test_get(self, sync_module: SyncSealsModule) -> None:
        seal = sync_module.get("seal_abc")
        assert seal.id == "seal_abc"

    def test_list(self, sync_module: SyncSealsModule) -> None:
        seals = sync_module.list()
        assert len(seals) >= 1

    def test_list_by_model(self, sync_module: SyncSealsModule) -> None:
        seals = sync_module.list_by_model(model_hash="bb" * 32)
        assert len(seals) >= 1

    def test_verify(self, sync_module: SyncSealsModule) -> None:
        resp = sync_module.verify("seal_abc")
        assert resp.valid is True

    def test_revoke(self, sync_module: SyncSealsModule) -> None:
        result = sync_module.revoke("seal_abc", reason="test")
        assert result is True

    def test_export(self, sync_module: SyncSealsModule) -> None:
        data = sync_module.export("seal_abc")
        assert data == "cafebabe"
