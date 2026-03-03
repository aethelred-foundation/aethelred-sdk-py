"""
Main client for Aethelred SDK.

This module provides sync and async clients for interacting
with the Aethelred blockchain.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aethelred.core.config import Config, Network
from aethelred.core.exceptions import (
    AethelredError,
    ConnectionError,
    RateLimitError,
    TimeoutError,
)
from aethelred.core.types import Block, NodeInfo

T = TypeVar("T")

logger = logging.getLogger(__name__)


class BaseClient:
    """Base client with shared functionality."""
    
    def __init__(self, config: Config):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging based on config."""
        logging.basicConfig(level=getattr(logging, self.config.log_level))
    
    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        base_url = self.config.rpc_url.rstrip("/")
        return f"{base_url}/{path.lstrip('/')}"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "aethelred-sdk-python/1.0.0",
        }
        if self.config.api_key:
            headers["X-API-Key"] = (
                self.config.api_key.get_secret_value()
                if hasattr(self.config.api_key, "get_secret_value")
                else str(self.config.api_key)
            )
        return headers


class AsyncAethelredClient(BaseClient):
    """Async client for Aethelred blockchain."""
    
    def __init__(
        self,
        config_or_url: Union[Config, str, None] = None,
        network: Optional[Network] = None,
        **kwargs,
    ):
        # Handle different initialization patterns
        if isinstance(config_or_url, Config):
            config = config_or_url
        elif isinstance(config_or_url, str):
            config = Config(rpc_url=config_or_url, **kwargs)
        elif network:
            config = Config.from_network(network, **kwargs)
        else:
            config = Config(**kwargs)
        
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        
        # Initialize modules (lazy)
        self._jobs: Optional["JobsModule"] = None
        self._seals: Optional["SealsModule"] = None
        self._models: Optional["ModelsModule"] = None
        self._validators: Optional["ValidatorsModule"] = None
        self._verification: Optional["VerificationModule"] = None
    
    async def __aenter__(self) -> "AsyncAethelredClient":
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def connect(self) -> None:
        """Establish connection to the node."""
        if self._client is None:
            timeout = httpx.Timeout(
                connect=self.config.timeout.connect_timeout,
                read=self.config.timeout.read_timeout,
                write=self.config.timeout.write_timeout,
                pool=self.config.timeout.pool_timeout,
            )
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_connections // 2,
            )
            self._client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                headers=self._get_headers(),
            )
            logger.info(f"Connected to {self.config.rpc_url}")
    
    async def close(self) -> None:
        """Close the connection."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Connection closed")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, connecting if necessary."""
        if self._client is None:
            raise ConnectionError("Client not connected. Use 'await client.connect()' first.")
        return self._client
    
    # Module properties
    @property
    def jobs(self) -> "JobsModule":
        """Get jobs module."""
        if self._jobs is None:
            from aethelred.jobs import JobsModule
            self._jobs = JobsModule(self)
        return self._jobs
    
    @property
    def seals(self) -> "SealsModule":
        """Get seals module."""
        if self._seals is None:
            from aethelred.seals import SealsModule
            self._seals = SealsModule(self)
        return self._seals
    
    @property
    def models(self) -> "ModelsModule":
        """Get models module."""
        if self._models is None:
            from aethelred.models import ModelsModule
            self._models = ModelsModule(self)
        return self._models
    
    @property
    def validators(self) -> "ValidatorsModule":
        """Get validators module."""
        if self._validators is None:
            from aethelred.validators import ValidatorsModule
            self._validators = ValidatorsModule(self)
        return self._validators
    
    @property
    def verification(self) -> "VerificationModule":
        """Get verification module."""
        if self._verification is None:
            from aethelred.verification import VerificationModule
            self._verification = VerificationModule(self)
        return self._verification
    
    # HTTP methods
    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=10),
    )
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        url = self._build_url(path)
        
        if self.config.log_requests:
            # Redact request body to prevent secret leakage (PY-12 fix)
            safe_params = "<redacted>" if params else None
            safe_json = "<redacted>" if json else None
            logger.debug(f"Request: {method} {url} params={safe_params} json={safe_json}")
        
        try:
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                json=json,
            )
            
            if self.config.log_responses:
                # Truncate and redact response body (PY-12 fix)
                logger.debug(f"Response: {response.status_code} [body redacted, {len(response.text)} chars]")
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=int(retry_after) if retry_after else None,
                )
            
            # Handle errors
            if response.status_code >= 400:
                error_data = response.json() if response.text else {}
                raise AethelredError(
                    message=error_data.get("message", f"HTTP {response.status_code}"),
                    details=error_data,
                )
            
            return response.json()
        
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {e}")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Connection failed: {e}")
    
    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a GET request."""
        return await self._request("GET", path, params=params)
    
    async def post(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a POST request."""
        return await self._request("POST", path, json=json)
    
    async def put(self, path: str, json: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a PUT request."""
        return await self._request("PUT", path, json=json)
    
    async def delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a DELETE request."""
        return await self._request("DELETE", path, params=params)
    
    # Utility methods
    async def get_node_info(self) -> NodeInfo:
        """Get information about the connected node."""
        data = await self.get("/cosmos/base/tendermint/v1beta1/node_info")
        return NodeInfo(**data.get("default_node_info", {}))
    
    async def get_latest_block(self) -> Block:
        """Get the latest block."""
        data = await self.get("/cosmos/base/tendermint/v1beta1/blocks/latest")
        return Block(**data)
    
    async def get_block(self, height: int) -> Block:
        """Get a block by height."""
        data = await self.get(f"/cosmos/base/tendermint/v1beta1/blocks/{height}")
        return Block(**data)
    
    async def health_check(self) -> bool:
        """Check if the node is healthy."""
        try:
            await self.get_node_info()
            return True
        except Exception:
            return False


class AethelredClient(BaseClient):
    """Synchronous client for Aethelred blockchain.
    
    This is a wrapper around AsyncAethelredClient that runs
    async methods synchronously.
    """
    
    def __init__(
        self,
        config_or_url: Union[Config, str, None] = None,
        network: Optional[Network] = None,
        **kwargs,
    ):
        # Handle different initialization patterns
        if isinstance(config_or_url, Config):
            config = config_or_url
        elif isinstance(config_or_url, str):
            config = Config(rpc_url=config_or_url, **kwargs)
        elif network:
            config = Config.from_network(network, **kwargs)
        else:
            config = Config(**kwargs)
        
        super().__init__(config)
        self._async_client = AsyncAethelredClient(config)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    raise RuntimeError("Event loop is closed")
                self._loop = loop
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop
    
    def _run(self, coro):
        """Run a coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)
    
    def __enter__(self) -> "AethelredClient":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    def connect(self) -> None:
        """Establish connection to the node."""
        self._run(self._async_client.connect())
    
    def close(self) -> None:
        """Close the connection."""
        self._run(self._async_client.close())
    
    # Module properties
    @property
    def jobs(self) -> "SyncJobsModule":
        """Get jobs module."""
        from aethelred.jobs import SyncJobsModule
        return SyncJobsModule(self, self._async_client.jobs)
    
    @property
    def seals(self) -> "SyncSealsModule":
        """Get seals module."""
        from aethelred.seals import SyncSealsModule
        return SyncSealsModule(self, self._async_client.seals)
    
    @property
    def models(self) -> "SyncModelsModule":
        """Get models module."""
        from aethelred.models import SyncModelsModule
        return SyncModelsModule(self, self._async_client.models)
    
    @property
    def validators(self) -> "SyncValidatorsModule":
        """Get validators module."""
        from aethelred.validators import SyncValidatorsModule
        return SyncValidatorsModule(self, self._async_client.validators)
    
    @property
    def verification(self) -> "SyncVerificationModule":
        """Get verification module."""
        from aethelred.verification import SyncVerificationModule
        return SyncVerificationModule(self, self._async_client.verification)
    
    # Utility methods
    def get_node_info(self) -> NodeInfo:
        """Get information about the connected node."""
        return self._run(self._async_client.get_node_info())
    
    def get_latest_block(self) -> Block:
        """Get the latest block."""
        return self._run(self._async_client.get_latest_block())
    
    def get_block(self, height: int) -> Block:
        """Get a block by height."""
        return self._run(self._async_client.get_block(height))
    
    def health_check(self) -> bool:
        """Check if the node is healthy."""
        return self._run(self._async_client.health_check())
