"""Comprehensive tests for aethelred/core/client.py — targeting 95%+ coverage.

Covers:
- BaseClient: __init__, _setup_logging, _build_url, _get_headers (api_key branch)
- AsyncAethelredClient: all init patterns, context manager, connect/close,
  client property, module properties (jobs/seals/models/validators/verification),
  _request (logging, rate-limit, HTTP errors, timeout, connect error),
  get/post/put/delete helpers, get_node_info, get_latest_block, get_block,
  health_check (success + failure)
- AethelredClient (sync wrapper): all init patterns, context manager,
  _get_loop (reuse + closed-loop + RuntimeError), connect/close,
  module properties, utility methods, health_check
"""

from __future__ import annotations

import asyncio
import json as json_mod
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from aethelred.core.client import (
    AethelredClient,
    AsyncAethelredClient,
    BaseClient,
)
from aethelred.core.config import Config, Network, SecretStr
from aethelred.core.exceptions import (
    AethelredError,
    ConnectionError,
    RateLimitError,
    TimeoutError,
)


def _make_response(
    status_code: int = 200,
    json_data: Optional[dict] = None,
    text: Optional[str] = None,
    headers: Optional[dict] = None,
) -> MagicMock:
    """Create a mock httpx Response with the needed attributes."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = httpx.Headers(headers or {})
    if json_data is not None:
        resp.json.return_value = json_data
        resp.text = json_mod.dumps(json_data)
    elif text is not None:
        resp.text = text
        if text:
            resp.json.return_value = json_mod.loads(text)
        else:
            resp.json.side_effect = Exception("No JSON")
    else:
        resp.text = ""
        resp.json.return_value = {}
    return resp


# ---------------------------------------------------------------------------
# BaseClient
# ---------------------------------------------------------------------------


class TestBaseClient:
    """Tests for BaseClient shared helpers."""

    def test_init_stores_config_and_sets_up_logging(self) -> None:
        config = Config(rpc_url="https://node.test:26657", log_level="DEBUG")
        base = BaseClient(config)
        assert base.config is config

    def test_build_url_strips_and_joins(self) -> None:
        config = Config(rpc_url="https://node.test:26657/")
        base = BaseClient(config)
        assert base._build_url("/aethelred/v1/jobs") == "https://node.test:26657/aethelred/v1/jobs"

    def test_build_url_no_trailing_slash(self) -> None:
        config = Config(rpc_url="https://node.test:26657")
        base = BaseClient(config)
        assert base._build_url("v1/jobs") == "https://node.test:26657/v1/jobs"

    def test_get_headers_without_api_key(self) -> None:
        config = Config(rpc_url="https://node.test")
        base = BaseClient(config)
        headers = base._get_headers()
        assert "X-API-Key" not in headers
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers

    def test_get_headers_with_secret_str_api_key(self) -> None:
        config = Config(rpc_url="https://node.test", api_key=SecretStr("my-secret"))
        base = BaseClient(config)
        headers = base._get_headers()
        assert headers["X-API-Key"] == "my-secret"

    def test_get_headers_with_plain_string_api_key(self) -> None:
        config = Config(rpc_url="https://node.test")
        config.api_key = "plain-key"  # type: ignore[assignment]
        base = BaseClient(config)
        headers = base._get_headers()
        assert headers["X-API-Key"] == "plain-key"


# ---------------------------------------------------------------------------
# AsyncAethelredClient — initialization patterns
# ---------------------------------------------------------------------------


class TestAsyncClientInit:
    """Test all constructor paths for AsyncAethelredClient."""

    def test_init_with_config_object(self) -> None:
        config = Config(rpc_url="https://node.test")
        client = AsyncAethelredClient(config)
        assert client.config.rpc_url == "https://node.test"

    def test_init_with_url_string(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        assert client.config.rpc_url == "https://node.test"

    def test_init_with_network_enum(self) -> None:
        client = AsyncAethelredClient(network=Network.TESTNET)
        assert "testnet" in client.config.rpc_url

    def test_init_with_no_args_uses_defaults(self) -> None:
        client = AsyncAethelredClient()
        assert client.config.rpc_url is not None

    def test_init_with_kwargs_forwarded(self) -> None:
        client = AsyncAethelredClient("https://node.test", log_level="DEBUG")
        assert client.config.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# AsyncAethelredClient — connect / close / context manager
# ---------------------------------------------------------------------------


class TestAsyncClientConnection:
    """Test connect, close, and async context manager."""

    @pytest.mark.asyncio
    async def test_connect_creates_httpx_client(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        assert client._client is None
        await client.connect()
        assert client._client is not None
        await client.close()

    @pytest.mark.asyncio
    async def test_connect_idempotent(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        await client.connect()
        first = client._client
        await client.connect()
        assert client._client is first  # same instance
        await client.close()

    @pytest.mark.asyncio
    async def test_close_when_not_connected_is_noop(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        await client.close()  # no error

    @pytest.mark.asyncio
    async def test_close_sets_client_to_none(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        await client.connect()
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        async with AsyncAethelredClient("https://node.test") as client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio
    async def test_client_property_raises_when_not_connected(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        with pytest.raises(ConnectionError, match="not connected"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_client_property_returns_httpx_client(self) -> None:
        async with AsyncAethelredClient("https://node.test") as client:
            assert isinstance(client.client, httpx.AsyncClient)


# ---------------------------------------------------------------------------
# AsyncAethelredClient — module lazy properties
# ---------------------------------------------------------------------------


class TestAsyncClientModules:
    """Test lazy module property accessors."""

    def test_jobs_property_creates_module(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        jobs = client.jobs
        assert jobs is not None
        # Second access returns the same object
        assert client.jobs is jobs

    def test_seals_property_creates_module(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        seals = client.seals
        assert seals is not None
        assert client.seals is seals

    def test_models_property_creates_module(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        models = client.models
        assert models is not None
        assert client.models is models

    def test_validators_property_creates_module(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        validators = client.validators
        assert validators is not None
        assert client.validators is validators

    def test_verification_property_creates_module(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        verification = client.verification
        assert verification is not None
        assert client.verification is verification


# ---------------------------------------------------------------------------
# AsyncAethelredClient — _request and HTTP helpers
# ---------------------------------------------------------------------------


class TestAsyncClientRequest:
    """Test _request with mocked httpx transport."""

    @pytest.mark.asyncio
    async def test_get_helper(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"ok": True}))

        result = await client.get("/api/test", params={"k": "v"})
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_post_helper(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"created": True}))

        result = await client.post("/api/test", json={"data": "x"})
        assert result == {"created": True}

    @pytest.mark.asyncio
    async def test_put_helper(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"updated": True}))

        result = await client.put("/api/test", json={"data": "y"})
        assert result == {"updated": True}

    @pytest.mark.asyncio
    async def test_delete_helper(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"deleted": True}))

        result = await client.delete("/api/test", params={"id": "1"})
        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_request_rate_limit_with_retry_after(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_make_response(429, {}, headers={"Retry-After": "30"})
        )

        with pytest.raises(RateLimitError) as exc_info:
            await client._request("GET", "/api/test")
        assert exc_info.value.retry_after == 30

    @pytest.mark.asyncio
    async def test_request_rate_limit_without_retry_after(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(429, {}))

        with pytest.raises(RateLimitError) as exc_info:
            await client._request("GET", "/api/test")
        assert exc_info.value.retry_after is None

    @pytest.mark.asyncio
    async def test_request_http_error_with_body(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_make_response(404, {"message": "Not found", "code": 1003})
        )

        with pytest.raises(AethelredError, match="Not found"):
            await client._request("GET", "/api/missing")

    @pytest.mark.asyncio
    async def test_request_http_error_empty_body(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(500, text=""))

        with pytest.raises(AethelredError, match="HTTP 500"):
            await client._request("GET", "/api/error")

    @pytest.mark.asyncio
    async def test_request_timeout_exception(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with pytest.raises(TimeoutError, match="timed out"):
            await client._request("GET", "/api/slow")

    @pytest.mark.asyncio
    async def test_request_connect_error(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(ConnectionError, match="Connection failed"):
            await client._request("GET", "/api/down")

    @pytest.mark.asyncio
    async def test_request_with_log_requests_enabled(self) -> None:
        config = Config(rpc_url="https://node.test", log_requests=True)
        client = AsyncAethelredClient(config)
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"ok": True}))

        result = await client._request("POST", "/api/test", params={"a": 1}, json={"b": 2})
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_with_log_responses_enabled(self) -> None:
        config = Config(rpc_url="https://node.test", log_responses=True)
        client = AsyncAethelredClient(config)
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"ok": True}))

        result = await client._request("GET", "/api/test")
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_request_with_log_requests_no_params_no_json(self) -> None:
        config = Config(rpc_url="https://node.test", log_requests=True)
        client = AsyncAethelredClient(config)
        client._client = AsyncMock()
        client._client.request = AsyncMock(return_value=_make_response(200, {"ok": True}))

        # No params, no json - tests the None branches in log_requests
        result = await client._request("GET", "/api/test")
        assert result == {"ok": True}


# ---------------------------------------------------------------------------
# AsyncAethelredClient — utility methods
# ---------------------------------------------------------------------------


class TestAsyncClientUtility:
    """Test get_node_info, get_latest_block, get_block, health_check."""

    @pytest.mark.asyncio
    async def test_get_node_info(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_make_response(200, {
                "default_node_info": {
                    "default_node_id": "abc123",
                    "listen_addr": "tcp://0.0.0.0:26656",
                    "network": "aethelred-1",
                    "version": "0.37.0",
                    "moniker": "validator-1",
                }
            })
        )

        info = await client.get_node_info()
        assert info.network == "aethelred-1"
        assert info.moniker == "validator-1"

    @pytest.mark.asyncio
    async def test_get_latest_block(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_make_response(200, {"block_id": {"hash": "abc"}})
        )

        block = await client.get_latest_block()
        assert block.block_id == {"hash": "abc"}

    @pytest.mark.asyncio
    async def test_get_block_by_height(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_make_response(200, {"block_id": {"hash": "def"}})
        )

        block = await client.get_block(42)
        assert block.block_id == {"hash": "def"}

    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            return_value=_make_response(200, {
                "default_node_info": {"network": "aethelred-1"}
            })
        )

        assert await client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self) -> None:
        client = AsyncAethelredClient("https://node.test")
        client._client = AsyncMock()
        client._client.request = AsyncMock(
            side_effect=httpx.ConnectError("refused")
        )

        assert await client.health_check() is False


# ---------------------------------------------------------------------------
# AethelredClient (sync wrapper)
# ---------------------------------------------------------------------------


class TestSyncClient:
    """Tests for the synchronous AethelredClient wrapper."""

    def test_init_with_config_object(self) -> None:
        config = Config(rpc_url="https://node.test")
        client = AethelredClient(config)
        assert client.config.rpc_url == "https://node.test"

    def test_init_with_url_string(self) -> None:
        client = AethelredClient("https://node.test")
        assert client.config.rpc_url == "https://node.test"

    def test_init_with_network(self) -> None:
        client = AethelredClient(network=Network.TESTNET)
        assert "testnet" in client.config.rpc_url

    def test_init_with_no_args(self) -> None:
        client = AethelredClient()
        assert client.config.rpc_url is not None

    def test_get_loop_creates_loop(self) -> None:
        client = AethelredClient("https://node.test")
        loop = client._get_loop()
        assert loop is not None
        assert not loop.is_closed()

    def test_get_loop_reuses_existing(self) -> None:
        client = AethelredClient("https://node.test")
        loop1 = client._get_loop()
        loop2 = client._get_loop()
        assert loop1 is loop2

    def test_get_loop_handles_closed_loop(self) -> None:
        client = AethelredClient("https://node.test")
        loop = client._get_loop()
        loop.close()
        new_loop = client._get_loop()
        assert new_loop is not None
        assert not new_loop.is_closed()

    def test_context_manager(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.connect = AsyncMock()
        client._async_client.close = AsyncMock()

        with client as c:
            assert c is client

    def test_connect_and_close(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.connect = AsyncMock()
        client._async_client.close = AsyncMock()

        client.connect()
        client.close()

    def test_jobs_property(self) -> None:
        client = AethelredClient("https://node.test")
        jobs = client.jobs
        assert jobs is not None

    def test_seals_property(self) -> None:
        client = AethelredClient("https://node.test")
        seals = client.seals
        assert seals is not None

    def test_models_property(self) -> None:
        client = AethelredClient("https://node.test")
        models = client.models
        assert models is not None

    def test_validators_property(self) -> None:
        client = AethelredClient("https://node.test")
        validators = client.validators
        assert validators is not None

    def test_verification_property(self) -> None:
        client = AethelredClient("https://node.test")
        verification = client.verification
        assert verification is not None

    def test_get_node_info(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.get_node_info = AsyncMock(
            return_value=MagicMock(network="aethelred-1")
        )
        info = client.get_node_info()
        assert info.network == "aethelred-1"

    def test_get_latest_block(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.get_latest_block = AsyncMock(
            return_value=MagicMock(block_id={"hash": "abc"})
        )
        block = client.get_latest_block()
        assert block.block_id == {"hash": "abc"}

    def test_get_block(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.get_block = AsyncMock(
            return_value=MagicMock(block_id={"hash": "def"})
        )
        block = client.get_block(42)
        assert block.block_id == {"hash": "def"}

    def test_health_check_true(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.health_check = AsyncMock(return_value=True)
        assert client.health_check() is True

    def test_health_check_false(self) -> None:
        client = AethelredClient("https://node.test")
        client._async_client.health_check = AsyncMock(return_value=False)
        assert client.health_check() is False

    def test_get_loop_handles_runtime_error(self) -> None:
        """When asyncio.get_event_loop() raises RuntimeError, a new loop is created."""
        client = AethelredClient("https://node.test")
        client._loop = None

        with patch("asyncio.get_event_loop", side_effect=RuntimeError("no current loop")):
            loop = client._get_loop()
            assert loop is not None
            assert not loop.is_closed()

    def test_run(self) -> None:
        """Test _run executes coroutine synchronously."""
        client = AethelredClient("https://node.test")

        async def coro():
            return 42

        result = client._run(coro())
        assert result == 42
