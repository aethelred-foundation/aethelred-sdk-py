"""
Configuration for Aethelred SDK.

This module provides configuration classes for connecting to
Aethelred networks with customizable options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# PY-21 fix: Explicit public API surface
__all__ = [
    "Config",
    "AethelredConfig",
    "Network",
    "NetworkConfig",
    "RetryConfig",
    "TimeoutConfig",
    "SecretStr",
]


class SecretStr:
    """A string wrapper that redacts its value in repr/str/logging.

    Prevents accidental exposure of secrets (private keys, mnemonics,
    API keys) in debug output, stack traces, and serialization.

    Use ``.get_secret_value()`` to access the underlying string.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        """Return the underlying secret string."""
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('******')"

    def __str__(self) -> str:
        return "******"

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


class Network(str, Enum):
    """Predefined Aethelred networks."""
    
    MAINNET = "mainnet"
    TESTNET = "testnet"
    DEVNET = "devnet"
    LOCAL = "local"


@dataclass
class NetworkConfig:
    """Network-specific configuration."""
    
    rpc_url: str
    chain_id: str
    ws_url: Optional[str] = None
    grpc_url: Optional[str] = None
    rest_url: Optional[str] = None
    explorer_url: Optional[str] = None


# Predefined network configurations
NETWORK_CONFIGS: Dict[Network, NetworkConfig] = {
    Network.MAINNET: NetworkConfig(
        rpc_url="https://rpc.mainnet.aethelred.org",
        chain_id="aethelred-1",
        ws_url="wss://ws.mainnet.aethelred.org",
        grpc_url="grpc.mainnet.aethelred.org:9090",
        rest_url="https://api.mainnet.aethelred.org",
        explorer_url="https://explorer.aethelred.org",
    ),
    Network.TESTNET: NetworkConfig(
        rpc_url="https://rpc.testnet.aethelred.org",
        chain_id="aethelred-testnet-1",
        ws_url="wss://ws.testnet.aethelred.org",
        grpc_url="grpc.testnet.aethelred.org:9090",
        rest_url="https://api.testnet.aethelred.org",
        explorer_url="https://testnet.explorer.aethelred.org",
    ),
    Network.DEVNET: NetworkConfig(
        rpc_url="https://rpc.devnet.aethelred.org",
        chain_id="aethelred-devnet-1",
        ws_url="wss://ws.devnet.aethelred.org",
        grpc_url="grpc.devnet.aethelred.org:9090",
        rest_url="https://api.devnet.aethelred.org",
        explorer_url="https://devnet.explorer.aethelred.org",
    ),
    Network.LOCAL: NetworkConfig(
        rpc_url="http://127.0.0.1:26657",
        chain_id="aethelred-local",
        ws_url="ws://127.0.0.1:26657/websocket",
        grpc_url="127.0.0.1:9090",
        rest_url="http://127.0.0.1:1317",
        explorer_url=None,
    ),
}


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    max_retries: int = 3
    initial_delay: float = 0.5
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True


@dataclass
class TimeoutConfig:
    """Configuration for timeout behavior."""
    
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0


class Config:
    """Main configuration for Aethelred client.

    Supports both ``rpc_url`` and ``endpoint`` as synonyms for the
    RPC endpoint URL, for backwards compatibility.
    """

    def __init__(
        self,
        *,
        network: Network = Network.MAINNET,
        rpc_url: Optional[str] = None,
        endpoint: Optional[str] = None,
        chain_id: Optional[str] = None,
        api_key: Optional[SecretStr] = None,
        private_key: Optional[SecretStr] = None,
        mnemonic: Optional[SecretStr] = None,
        timeout: Optional[TimeoutConfig] = None,
        retry: Optional[RetryConfig] = None,
        max_connections: int = 10,
        keepalive_timeout: float = 30.0,
        ws_enabled: bool = True,
        ws_heartbeat_interval: float = 30.0,
        ws_reconnect_delay: float = 1.0,
        ws_max_reconnect_attempts: int = 10,
        log_level: str = "INFO",
        log_requests: bool = False,
        log_responses: bool = False,
        cache_enabled: bool = True,
        cache_ttl: float = 60.0,
        cache_max_size: int = 1000,
    ):
        self.network = network
        # Accept either rpc_url or endpoint (endpoint takes precedence for compat)
        self.rpc_url = endpoint or rpc_url
        self.chain_id = chain_id
        self.api_key = api_key
        self.private_key = private_key
        self.mnemonic = mnemonic
        self.timeout = timeout or TimeoutConfig()
        self.retry = retry or RetryConfig()
        self.max_connections = max_connections
        self.keepalive_timeout = keepalive_timeout
        self.ws_enabled = ws_enabled
        self.ws_heartbeat_interval = ws_heartbeat_interval
        self.ws_reconnect_delay = ws_reconnect_delay
        self.ws_max_reconnect_attempts = ws_max_reconnect_attempts
        self.log_level = log_level
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.cache_max_size = cache_max_size

        # Derive from network defaults when not provided
        network_config = NETWORK_CONFIGS.get(self.network)
        if self.rpc_url is None and network_config:
            self.rpc_url = network_config.rpc_url
        if self.chain_id is None and network_config:
            self.chain_id = network_config.chain_id

    @property
    def endpoint(self) -> Optional[str]:
        """Alias for rpc_url (backwards compatibility)."""
        return self.rpc_url

    @endpoint.setter
    def endpoint(self, value: Optional[str]) -> None:
        self.rpc_url = value
    
    @classmethod
    def from_network(cls, network: Network, **kwargs) -> "Config":
        """Create configuration from a predefined network."""
        return cls(network=network, **kwargs)
    
    @classmethod
    def mainnet(cls, **kwargs) -> "Config":
        """Create mainnet configuration."""
        return cls.from_network(Network.MAINNET, **kwargs)
    
    @classmethod
    def testnet(cls, **kwargs) -> "Config":
        """Create testnet configuration."""
        return cls.from_network(Network.TESTNET, **kwargs)
    
    @classmethod
    def devnet(cls, **kwargs) -> "Config":
        """Create devnet configuration."""
        return cls.from_network(Network.DEVNET, **kwargs)
    
    @classmethod
    def local(cls, **kwargs) -> "Config":
        """Create local development configuration."""
        return cls.from_network(Network.LOCAL, **kwargs)
    
    @classmethod
    def custom(cls, rpc_url: str, chain_id: str, **kwargs) -> "Config":
        """Create custom configuration."""
        return cls(rpc_url=rpc_url, chain_id=chain_id, **kwargs)
    
    def get_network_config(self) -> Optional[NetworkConfig]:
        """Get the full network configuration."""
        return NETWORK_CONFIGS.get(self.network)
    
    @property
    def ws_url(self) -> Optional[str]:
        """Get WebSocket URL."""
        network_config = self.get_network_config()
        return network_config.ws_url if network_config else None
    
    @property
    def grpc_url(self) -> Optional[str]:
        """Get gRPC URL."""
        network_config = self.get_network_config()
        return network_config.grpc_url if network_config else None
    
    @property
    def rest_url(self) -> Optional[str]:
        """Get REST API URL."""
        network_config = self.get_network_config()
        return network_config.rest_url if network_config else None


# Backwards-compatible alias used by older SDK call sites and tests.
AethelredConfig = Config
