"""
Oracle Network Client

Interface to the Aethelred Oracle Network for querying attested data feeds
and referencing verified data in compute jobs.

The Oracle Network provides:
- Trusted data feeds (market data, weather, etc.)
- Data attestation with TEE verification
- Decentralized Identifiers (DIDs) for data references
- Provenance tracking for audit compliance

Example:
    >>> from aethelred.oracles import OracleClient
    >>>
    >>> oracle = OracleClient(client)
    >>>
    >>> # Get attested data feed
    >>> feed = oracle.get_feed("did:aethelred:oracle:market_data/btc_usd")
    >>>
    >>> # Use in job submission
    >>> job_input = oracle.create_trusted_input(feed.did)
    >>> job = client.submit_job(circuit, job_input, sla=sla)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Union

from aethelred.core.types import (
    DataSourceType,
    JobInput,
    DataProvenance,
    TEEPlatform,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Parsing Helpers (Code Reusability — auditor feedback)
# =============================================================================


def _parse_attestation_data(attestation_data: dict, feed_id: str) -> "AttestationRecord":
    """Parse attestation data from API response.

    Shared between OracleClient and AsyncOracleClient to avoid
    duplicating parsing logic (auditor feedback: code reusability).
    """
    return AttestationRecord(
        attestation_id=attestation_data.get("attestation_id", ""),
        feed_id=feed_id,
        data_hash=attestation_data.get("data_hash", ""),
        method=AttestationMethod(attestation_data.get("method", "tee")),
        oracle_node_id=attestation_data.get("oracle_node_id", ""),
        timestamp=datetime.fromisoformat(
            attestation_data.get("timestamp", datetime.now(timezone.utc).isoformat())
        ),
        tee_platform=(
            TEEPlatform(attestation_data["tee_platform"])
            if attestation_data.get("tee_platform")
            else None
        ),
        tx_hash=attestation_data.get("tx_hash"),
        block_height=attestation_data.get("block_height"),
    )


def _parse_feed_metadata(feed_data: dict) -> "FeedMetadata":
    """Parse feed metadata from API response.

    Shared between OracleClient and AsyncOracleClient.
    """
    return FeedMetadata(
        feed_id=feed_data.get("feed_id", ""),
        name=feed_data.get("name", ""),
        description=feed_data.get("description", ""),
        feed_type=FeedType(feed_data.get("feed_type", "custom")),
        value_type=feed_data.get("value_type", "json"),
        unit=feed_data.get("unit"),
        update_interval_seconds=feed_data.get("update_interval_seconds", 60),
        source=feed_data.get("source", ""),
        is_public=feed_data.get("is_public", True),
    )


def _build_data_feed(
    feed_id: str, response: dict, attestation: "AttestationRecord"
) -> "DataFeed":
    """Build a DataFeed from API response and parsed attestation.

    Shared between OracleClient and AsyncOracleClient.
    """
    provenance = DataProvenance(
        oracle_node_id=attestation.oracle_node_id,
        attestation_hash=response.get("attestation", {}).get("data_hash", ""),
        attestation_timestamp=attestation.timestamp,
        data_hash=response.get("value_hash", ""),
        verification_method=attestation.method.value,
    )

    metadata_data = response.get("metadata", {})
    metadata = FeedMetadata(
        feed_id=feed_id,
        name=metadata_data.get("name", feed_id),
        description=metadata_data.get("description", ""),
        feed_type=FeedType(metadata_data.get("feed_type", "custom")),
        value_type=metadata_data.get("value_type", "json"),
        unit=metadata_data.get("unit"),
        source=metadata_data.get("source", ""),
    )

    did = f"did:aethelred:oracle:{feed_id}"

    return DataFeed(
        did=did,
        feed_id=feed_id,
        metadata=metadata,
        value=response.get("value"),
        value_hash=response.get("value_hash", ""),
        timestamp=datetime.fromisoformat(
            response.get("timestamp", datetime.now(timezone.utc).isoformat())
        ),
        attestation=attestation,
        provenance=provenance,
    )


class FeedType(str, Enum):
    """Types of oracle data feeds."""

    MARKET_DATA = "market_data"
    WEATHER = "weather"
    SPORTS = "sports"
    FINANCIAL = "financial"
    HEALTHCARE = "healthcare"
    IDENTITY = "identity"
    CUSTOM = "custom"


class AttestationMethod(str, Enum):
    """Methods used to attest data."""

    TEE = "tee"  # Trusted Execution Environment
    CONSENSUS = "consensus"  # Multi-oracle consensus
    MULTISIG = "multisig"  # Multi-signature
    ZK = "zk"  # Zero-knowledge proof


@dataclass
class OracleNode:
    """Information about an oracle node."""

    node_id: str
    name: str
    operator: str
    endpoint: str

    # Capabilities
    supported_feeds: list[str]
    tee_platform: Optional[TEEPlatform] = None

    # Reputation
    stake: int = 0
    uptime: float = 0.99
    attestations_count: int = 0

    # Status
    is_active: bool = True
    last_attestation: Optional[datetime] = None


@dataclass
class FeedMetadata:
    """Metadata about a data feed."""

    feed_id: str
    name: str
    description: str
    feed_type: FeedType

    # Data schema
    value_type: str  # "number", "string", "json", "binary"
    unit: Optional[str] = None  # e.g., "USD", "celsius"
    precision: Optional[int] = None

    # Update frequency
    update_interval_seconds: int = 60
    min_confirmations: int = 1

    # Provenance
    source: str = ""
    oracle_nodes: list[str] = field(default_factory=list)

    # Access
    is_public: bool = True
    required_stake: int = 0


@dataclass
class AttestationRecord:
    """Record of data attestation by oracle."""

    attestation_id: str
    feed_id: str
    data_hash: str

    # Attestation details
    method: AttestationMethod
    oracle_node_id: str
    timestamp: datetime

    # TEE details (if applicable)
    tee_platform: Optional[TEEPlatform] = None
    tee_measurement: Optional[str] = None
    tee_signature: Optional[bytes] = None

    # Consensus details (if applicable)
    consensus_nodes: Optional[list[str]] = None
    consensus_threshold: Optional[int] = None

    # On-chain reference
    tx_hash: Optional[str] = None
    block_height: Optional[int] = None


@dataclass
class DataFeed:
    """
    A data feed from the Oracle Network.

    Contains the attested data value along with provenance information.
    """

    # Identification
    did: str  # Decentralized Identifier
    feed_id: str
    metadata: FeedMetadata

    # Current value
    value: Any
    value_hash: str
    timestamp: datetime

    # Attestation
    attestation: AttestationRecord

    # Provenance
    provenance: DataProvenance

    def to_job_input(self) -> JobInput:
        """Convert to JobInput for compute job submission."""
        return JobInput(
            source_type=DataSourceType.ORACLE_DID,
            payload=self.did,
            proof_of_provenance=self.provenance,
            did=self.did,
        )

    def verify_hash(self) -> bool:
        """Verify the value hash matches the data."""
        import json

        if isinstance(self.value, bytes):
            computed = hashlib.sha256(self.value).hexdigest()
        else:
            serialized = json.dumps(self.value, sort_keys=True)
            computed = hashlib.sha256(serialized.encode()).hexdigest()

        return computed == self.value_hash


class OracleClient:
    """
    Synchronous client for the Aethelred Oracle Network.

    Provides methods to query data feeds, verify attestations,
    and create job inputs from oracle data.

    Example:
        >>> from aethelred import Client
        >>> from aethelred.oracles import OracleClient
        >>>
        >>> client = Client(endpoint="https://mainnet.aethelred.org")
        >>> oracle = OracleClient(client)
        >>>
        >>> # List available feeds
        >>> feeds = oracle.list_feeds(feed_type="market_data")
        >>> for f in feeds:
        ...     print(f"{f.name}: {f.feed_id}")
        >>>
        >>> # Get a specific feed
        >>> btc_feed = oracle.get_feed("market_data/btc_usd")
        >>> print(f"BTC/USD: ${btc_feed.value}")
        >>>
        >>> # Create job input from oracle data
        >>> job_input = oracle.create_trusted_input(btc_feed.did)
        >>> job = client.submit_job(circuit, job_input)
    """

    def __init__(self, client: Any):
        """
        Initialize Oracle client.

        Args:
            client: Aethelred client instance
        """
        self._client = client
        self._cache: dict[str, DataFeed] = {}

    def list_feeds(
        self,
        feed_type: Optional[Union[str, FeedType]] = None,
        is_public: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> list[FeedMetadata]:
        """
        List available data feeds.

        Args:
            feed_type: Filter by feed type
            is_public: Only show public feeds
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of feed metadata
        """
        params = {
            "is_public": is_public,
            "limit": limit,
            "offset": offset,
        }

        if feed_type:
            if isinstance(feed_type, FeedType):
                feed_type = feed_type.value
            params["feed_type"] = feed_type

        response = self._client._request("GET", "/oracle/v1/feeds", params=params)

        return [_parse_feed_metadata(fd) for fd in response.get("feeds", [])]

    def get_feed(
        self,
        feed_id: str,
        version: Optional[str] = None,
        at_timestamp: Optional[datetime] = None,
    ) -> DataFeed:
        """
        Get a data feed by ID.

        Args:
            feed_id: Feed identifier or DID
            version: Specific version (default: latest)
            at_timestamp: Get value at specific time

        Returns:
            DataFeed with current value and attestation
        """
        # Handle DID format
        if feed_id.startswith("did:aethelred:oracle:"):
            feed_id = feed_id.replace("did:aethelred:oracle:", "")

        params = {}
        if version:
            params["version"] = version
        if at_timestamp:
            params["at_timestamp"] = at_timestamp.isoformat()

        response = self._client._request(
            "GET",
            f"/oracle/v1/feeds/{feed_id}",
            params=params
        )

        # Use shared parsing helpers (auditor feedback: code reusability)
        attestation_data = response.get("attestation", {})
        attestation = _parse_attestation_data(attestation_data, feed_id)
        feed = _build_data_feed(feed_id, response, attestation)

        # Cache the feed
        self._cache[feed.did] = feed

        return feed

    def get_feed_by_did(self, did: str) -> DataFeed:
        """
        Get a data feed by its Decentralized Identifier.

        Args:
            did: DID in format "did:aethelred:oracle:<feed_id>"

        Returns:
            DataFeed
        """
        # Check cache first
        if did in self._cache:
            cached = self._cache[did]
            # Check if still fresh (< 60 seconds old)
            age = (datetime.now(timezone.utc) - cached.timestamp).total_seconds()
            if age < 60:
                return cached

        return self.get_feed(did)

    def verify_attestation(
        self,
        feed: DataFeed,
        verify_tee: bool = True,
        verify_consensus: bool = True,
    ) -> bool:
        """
        Verify the attestation of a data feed.

        Args:
            feed: DataFeed to verify
            verify_tee: Verify TEE attestation if applicable
            verify_consensus: Verify consensus if applicable

        Returns:
            True if attestation is valid
        """
        # Verify value hash
        if not feed.verify_hash():
            logger.warning(f"Feed {feed.feed_id} value hash mismatch")
            return False

        # Request on-chain verification
        response = self._client._request(
            "GET",
            f"/oracle/v1/feeds/{feed.feed_id}/verify",
            params={
                "attestation_id": feed.attestation.attestation_id,
                "verify_tee": verify_tee,
                "verify_consensus": verify_consensus,
            }
        )

        return response.get("is_valid", False)

    def create_trusted_input(
        self,
        did_or_feed: Union[str, DataFeed],
        encrypted: bool = True,
    ) -> JobInput:
        """
        Create a JobInput that references oracle-attested data.

        This allows submitting compute jobs that use data verified
        by the Oracle Network, without uploading the raw data.

        Args:
            did_or_feed: DID string or DataFeed object
            encrypted: Whether the data reference should be encrypted

        Returns:
            JobInput configured for oracle data reference
        """
        if isinstance(did_or_feed, str):
            # It's a DID, fetch the feed
            feed = self.get_feed_by_did(did_or_feed)
        else:
            feed = did_or_feed

        return JobInput(
            source_type=DataSourceType.ORACLE_DID,
            payload=feed.did,
            proof_of_provenance=feed.provenance,
            did=feed.did,
            encrypted=encrypted,
        )

    def get_oracle_nodes(
        self,
        active_only: bool = True,
        tee_required: bool = False,
    ) -> list[OracleNode]:
        """
        List oracle nodes in the network.

        Args:
            active_only: Only return active nodes
            tee_required: Only nodes with TEE capability

        Returns:
            List of oracle nodes
        """
        params = {
            "active_only": active_only,
            "tee_required": tee_required,
        }

        response = self._client._request("GET", "/oracle/v1/nodes", params=params)

        nodes = []
        for node_data in response.get("nodes", []):
            nodes.append(OracleNode(
                node_id=node_data["node_id"],
                name=node_data["name"],
                operator=node_data.get("operator", ""),
                endpoint=node_data.get("endpoint", ""),
                supported_feeds=node_data.get("supported_feeds", []),
                tee_platform=TEEPlatform(node_data["tee_platform"]) if node_data.get("tee_platform") else None,
                stake=node_data.get("stake", 0),
                uptime=node_data.get("uptime", 0.99),
                is_active=node_data.get("is_active", True),
            ))

        return nodes

    def get_historical_values(
        self,
        feed_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        """
        Get historical values for a feed.

        Args:
            feed_id: Feed identifier
            start_time: Start of time range
            end_time: End of time range (default: now)
            limit: Maximum values to return

        Returns:
            List of historical values with timestamps
        """
        params = {
            "start_time": start_time.isoformat(),
            "limit": limit,
        }
        if end_time:
            params["end_time"] = end_time.isoformat()

        response = self._client._request(
            "GET",
            f"/oracle/v1/feeds/{feed_id}/history",
            params=params
        )

        return response.get("values", [])

    def subscribe_to_feed(
        self,
        feed_id: str,
        callback: Callable[[DataFeed], None],
    ) -> "FeedSubscription":
        """
        Subscribe to real-time feed updates.

        Args:
            feed_id: Feed to subscribe to
            callback: Called on each update

        Returns:
            FeedSubscription for managing subscription
        """
        subscription = FeedSubscription(
            oracle_client=self,
            feed_id=feed_id,
            callback=callback,
        )
        subscription.start()
        return subscription


class AsyncOracleClient:
    """
    Asynchronous client for the Aethelred Oracle Network.

    Same interface as OracleClient but with async methods.
    """

    def __init__(self, client: Any):
        """Initialize async Oracle client."""
        self._client = client
        self._cache: dict[str, DataFeed] = {}

    async def list_feeds(
        self,
        feed_type: Optional[Union[str, FeedType]] = None,
        is_public: bool = True,
        limit: int = 100,
    ) -> list[FeedMetadata]:
        """List available data feeds asynchronously."""
        params = {"is_public": is_public, "limit": limit}

        if feed_type:
            if isinstance(feed_type, FeedType):
                feed_type = feed_type.value
            params["feed_type"] = feed_type

        response = await self._client._request("GET", "/oracle/v1/feeds", params=params)

        return [_parse_feed_metadata(fd) for fd in response.get("feeds", [])]

    async def get_feed(self, feed_id: str) -> DataFeed:
        """Get a data feed asynchronously."""
        if feed_id.startswith("did:aethelred:oracle:"):
            feed_id = feed_id.replace("did:aethelred:oracle:", "")

        response = await self._client._request("GET", f"/oracle/v1/feeds/{feed_id}")

        # Use shared parsing helpers (auditor feedback: code reusability)
        attestation_data = response.get("attestation", {})
        attestation = _parse_attestation_data(attestation_data, feed_id)

        return _build_data_feed(feed_id, response, attestation)

    async def verify_attestation(self, feed: DataFeed) -> bool:
        """Verify attestation asynchronously."""
        if not feed.verify_hash():
            return False

        response = await self._client._request(
            "GET",
            f"/oracle/v1/feeds/{feed.feed_id}/verify"
        )

        return response.get("is_valid", False)

    def create_trusted_input(
        self,
        did_or_feed: Union[str, DataFeed],
        encrypted: bool = True,
    ) -> JobInput:
        """Create trusted input (sync method)."""
        if isinstance(did_or_feed, str):
            did = did_or_feed
            provenance = None
        else:
            did = did_or_feed.did
            provenance = did_or_feed.provenance

        return JobInput(
            source_type=DataSourceType.ORACLE_DID,
            payload=did,
            proof_of_provenance=provenance,
            did=did,
            encrypted=encrypted,
        )


class FeedSubscription:
    """WebSocket subscription for oracle feed updates."""

    def __init__(
        self,
        oracle_client: OracleClient,
        feed_id: str,
        callback: Callable[[DataFeed], None],
    ):
        self.oracle_client = oracle_client
        self.feed_id = feed_id
        self.callback = callback
        self._running = False
        self._thread = None

    def start(self) -> None:
        """Start the subscription."""
        import threading

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Run subscription loop."""
        import json
        import time

        try:
            import websockets.sync.client as ws_sync

            ws_url = self.oracle_client._client._config.endpoint
            ws_url = ws_url.replace("https://", "wss://").replace("http://", "ws://")
            ws_url = f"{ws_url}/ws/oracle/feeds/{self.feed_id}"

            with ws_sync.connect(ws_url) as websocket:
                websocket.send(json.dumps({
                    "type": "subscribe",
                    "feed_id": self.feed_id,
                }))

                while self._running:
                    try:
                        message = websocket.recv(timeout=1.0)
                        data = json.loads(message)

                        if data.get("type") == "update":
                            # Fetch full feed data
                            feed = self.oracle_client.get_feed(self.feed_id)
                            self.callback(feed)

                    except TimeoutError:
                        continue

        except Exception as e:
            logger.error(f"Feed subscription error: {e}")

    def unsubscribe(self) -> None:
        """Stop the subscription."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    @property
    def is_active(self) -> bool:
        return self._running
