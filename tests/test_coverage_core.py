"""
Comprehensive tests for core client module:
- aethelred/core/client.py (BaseClient, AsyncAethelredClient, AethelredClient)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import httpx
import pytest

from aethelred.core.client import BaseClient, AsyncAethelredClient, AethelredClient
from aethelred.core.config import Config, Network, SecretStr, TimeoutConfig
from aethelred.core.exceptions import (
    AethelredError,
    ConnectionError,
    RateLimitError,
    TimeoutError,
)


class TestBaseClient:
    def test_init(self):
        config = Config(rpc_url="http://localhost:26657")
        client = BaseClient(config)
        assert client.config is config

    def test_build_url(self):
        config = Config(rpc_url="http://localhost:26657")
        client = BaseClient(config)
        assert client._build_url("/api/test") == "http://localhost:26657/api/test"

    def test_build_url_strips_trailing_slash(self):
        config = Config(rpc_url="http://localhost:26657/")
        client = BaseClient(config)
        assert client._build_url("/api/test") == "http://localhost:26657/api/test"

    def test_build_url_strips_leading_slash(self):
        config = Config(rpc_url="http://localhost:26657")
        client = BaseClient(config)
        assert client._build_url("api/test") == "http://localhost:26657/api/test"

    def test_get_headers_no_api_key(self):
        config = Config(rpc_url="http://localhost:26657")
        client = BaseClient(config)
        headers = client._get_headers()
        assert headers["Content-Type"] == "application/json"
        assert "User-Agent" in headers
        assert "X-API-Key" not in headers

    def test_get_headers_with_api_key_secret_str(self):
        config = Config(rpc_url="http://localhost:26657", api_key=SecretStr("test-key"))
        client = BaseClient(config)
        headers = client._get_headers()
        assert headers["X-API-Key"] == "test-key"

    def test_get_headers_with_api_key_string(self):
        config = Config(rpc_url="http://localhost:26657")
        config.api_key = "plain-key"
        client = BaseClient(config)
        headers = client._get_headers()
        assert headers["X-API-Key"] == "plain-key"


class TestAsyncAethelredClient:
    def test_init_with_config(self):
        config = Config(rpc_url="http://localhost:26657")
        client = AsyncAethelredClient(config)
        assert client.config.rpc_url == "http://localhost:26657"

    def test_init_with_url_string(self):
        client = AsyncAethelredClient("http://localhost:26657")
        assert client.config.rpc_url == "http://localhost:26657"

    def test_init_with_network(self):
        client = AsyncAethelredClient(network=Network.TESTNET)
        assert "testnet" in client.config.rpc_url

    def test_init_defaults(self):
        client = AsyncAethelredClient()
        assert client.config is not None

    def test_client_property_not_connected(self):
        client = AsyncAethelredClient("http://localhost:26657")
        with pytest.raises(ConnectionError, match="Client not connected"):
            _ = client.client

    @pytest.mark.asyncio
    async def test_connect_and_close(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        assert client._client is not None
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_connect_idempotent(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        first = client._client
        await client.connect()
        assert client._client is first  # Same client
        await client.close()

    @pytest.mark.asyncio
    async def test_close_when_not_connected(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self):
        async with AsyncAethelredClient("http://localhost:26657") as client:
            assert client._client is not None
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"result": "ok"}
            mock_response.text = '{"result":"ok"}'

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                result = await client.get("/test")
                assert result == {"result": "ok"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_post(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": "123"}
            mock_response.text = '{"id":"123"}'

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                result = await client.post("/test", json={"data": "value"})
                assert result == {"id": "123"}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_put(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"updated": True}
            mock_response.text = '{"updated":true}'

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                result = await client.put("/test", json={"data": "value"})
                assert result == {"updated": True}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_delete(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"deleted": True}
            mock_response.text = '{"deleted":true}'

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                result = await client.delete("/test")
                assert result == {"deleted": True}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_rate_limit(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "30"}

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                with pytest.raises(RateLimitError):
                    await client._request("GET", "/test")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_rate_limit_no_retry_after(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {}

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                with pytest.raises(RateLimitError):
                    await client._request("GET", "/test")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_http_error(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.text = '{"message":"not found"}'
            mock_response.json.return_value = {"message": "not found"}

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                with pytest.raises(AethelredError, match="not found"):
                    await client._request("GET", "/test")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_http_error_empty_body(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = ""

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                with pytest.raises(AethelredError, match="HTTP 500"):
                    await client._request("GET", "/test")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_timeout(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            with patch.object(
                client._client, "request",
                new_callable=AsyncMock,
                side_effect=httpx.TimeoutException("timeout"),
            ):
                with pytest.raises(TimeoutError, match="Request timed out"):
                    await client._request("GET", "/test")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_connect_error(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            with patch.object(
                client._client, "request",
                new_callable=AsyncMock,
                side_effect=httpx.ConnectError("connection refused"),
            ):
                with pytest.raises(ConnectionError, match="Connection failed"):
                    await client._request("GET", "/test")
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_request_with_logging(self):
        config = Config(rpc_url="http://localhost:26657", log_requests=True, log_responses=True)
        client = AsyncAethelredClient(config)
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"ok": True}
            mock_response.text = '{"ok":true}'

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                result = await client._request("GET", "/test", params={"a": "b"}, json={"c": "d"})
                assert result == {"ok": True}
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_node_info(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "default_node_info": {
                    "network": "aethelred-1",
                    "version": "0.1.0",
                }
            }
            mock_response.text = "{}"

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                info = await client.get_node_info()
                assert info is not None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_latest_block(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "block_id": {"hash": "abc"},
                "block": {"header": {"height": "100"}},
            }
            mock_response.text = "{}"

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                block = await client.get_latest_block()
                assert block is not None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_get_block(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "block_id": {"hash": "abc"},
                "block": {"header": {"height": "42"}},
            }
            mock_response.text = "{}"

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                block = await client.get_block(42)
                assert block is not None
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"default_node_info": {}}
            mock_response.text = "{}"

            with patch.object(client._client, "request", new_callable=AsyncMock, return_value=mock_response):
                assert await client.health_check() is True
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        client = AsyncAethelredClient("http://localhost:26657")
        await client.connect()
        try:
            with patch.object(
                client._client, "request",
                new_callable=AsyncMock,
                side_effect=Exception("fail"),
            ):
                assert await client.health_check() is False
        finally:
            await client.close()

    # Module property tests
    def test_jobs_property(self):
        client = AsyncAethelredClient("http://localhost:26657")
        jobs = client.jobs
        assert jobs is not None
        # Should be cached
        assert client.jobs is jobs

    def test_seals_property(self):
        client = AsyncAethelredClient("http://localhost:26657")
        seals = client.seals
        assert seals is not None
        assert client.seals is seals

    def test_models_property(self):
        client = AsyncAethelredClient("http://localhost:26657")
        models = client.models
        assert models is not None

    def test_validators_property(self):
        client = AsyncAethelredClient("http://localhost:26657")
        validators = client.validators
        assert validators is not None

    def test_verification_property(self):
        client = AsyncAethelredClient("http://localhost:26657")
        verification = client.verification
        assert verification is not None


class TestAethelredClient:
    def test_init_with_config(self):
        config = Config(rpc_url="http://localhost:26657")
        client = AethelredClient(config)
        assert client.config.rpc_url == "http://localhost:26657"

    def test_init_with_url_string(self):
        client = AethelredClient("http://localhost:26657")
        assert client.config.rpc_url == "http://localhost:26657"

    def test_init_with_network(self):
        client = AethelredClient(network=Network.TESTNET)
        assert "testnet" in client.config.rpc_url

    def test_init_defaults(self):
        client = AethelredClient()
        assert client.config is not None

    def test_get_loop(self):
        client = AethelredClient("http://localhost:26657")
        loop = client._get_loop()
        assert loop is not None

    def test_context_manager(self):
        client = AethelredClient("http://localhost:26657")
        with patch.object(client, "connect"):
            with patch.object(client, "close"):
                with client as c:
                    assert c is client

    def test_module_properties(self):
        client = AethelredClient("http://localhost:26657")
        # These create sync wrapper modules
        jobs = client.jobs
        assert jobs is not None
        seals = client.seals
        assert seals is not None
        models = client.models
        assert models is not None
        validators = client.validators
        assert validators is not None
        verification = client.verification
        assert verification is not None
