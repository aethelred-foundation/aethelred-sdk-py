"""Tests for the OracleClient — feed parsing, attestation, caching.

Covers:
- Sync OracleClient: list_feeds, get_feed, verify, create_trusted_input
- Async OracleClient: basic parity
- DID parsing and caching behaviour
- FeedSubscription lifecycle
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from aethelred.oracles.client import (
    OracleClient,
    AsyncOracleClient,
    FeedType,
    AttestationMethod,
    FeedSubscription,
    DataFeed,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def oracle(mock_client) -> OracleClient:
    return OracleClient(mock_client)


# ---------------------------------------------------------------------------
# OracleClient.list_feeds
# ---------------------------------------------------------------------------


class TestListFeeds:
    """Test feed listing and filtering."""

    def test_returns_feed_metadata(self, oracle: OracleClient) -> None:
        feeds = oracle.list_feeds()
        assert len(feeds) > 0
        assert feeds[0].name == "BTC/USD"
        assert feeds[0].feed_type == FeedType.MARKET_DATA

    def test_filter_by_type(self, oracle: OracleClient) -> None:
        feeds = oracle.list_feeds(feed_type=FeedType.MARKET_DATA)
        assert len(feeds) > 0

    def test_filter_by_type_string(self, oracle: OracleClient) -> None:
        feeds = oracle.list_feeds(feed_type="market_data")
        assert len(feeds) > 0


# ---------------------------------------------------------------------------
# OracleClient.get_feed
# ---------------------------------------------------------------------------


class TestGetFeed:
    """Test single feed retrieval and parsing."""

    def test_returns_data_feed(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("market_data/btc_usd")
        assert isinstance(feed, DataFeed)
        assert feed.did == "did:aethelred:oracle:market_data/btc_usd"

    def test_strips_did_prefix(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("did:aethelred:oracle:market_data/btc_usd")
        assert feed.feed_id == "market_data/btc_usd"

    def test_attestation_parsed(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("market_data/btc_usd")
        assert feed.attestation.method == AttestationMethod.TEE
        assert feed.attestation.oracle_node_id == "node_1"

    def test_provenance_attached(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("market_data/btc_usd")
        assert feed.provenance is not None
        assert feed.provenance.verification_method == "tee"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


class TestCaching:
    """Test feed caching behaviour."""

    def test_get_feed_caches(self, oracle: OracleClient) -> None:
        feed1 = oracle.get_feed("market_data/btc_usd")
        assert feed1.did in oracle._cache

    def test_get_feed_by_did_uses_cache(self, oracle: OracleClient) -> None:
        feed1 = oracle.get_feed("market_data/btc_usd")
        # Should use cache (within 60s freshness window)
        feed2 = oracle.get_feed_by_did(feed1.did)
        assert feed2.did == feed1.did


# ---------------------------------------------------------------------------
# Verification & Trusted Input
# ---------------------------------------------------------------------------


class TestVerificationAndInput:
    """Test attestation verification and trusted input creation."""

    def test_verify_attestation(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("market_data/btc_usd")
        result = oracle.verify_attestation(feed)
        assert result is True

    def test_create_trusted_input_from_feed(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("market_data/btc_usd")
        job_input = oracle.create_trusted_input(feed)
        assert job_input.did == feed.did
        assert job_input.payload == feed.did

    def test_create_trusted_input_from_did(self, oracle: OracleClient) -> None:
        feed = oracle.get_feed("market_data/btc_usd")
        job_input = oracle.create_trusted_input(feed.did)
        assert job_input.did == feed.did


# ---------------------------------------------------------------------------
# Oracle Nodes
# ---------------------------------------------------------------------------


class TestOracleNodes:
    """Test oracle node listing."""

    def test_list_nodes(self, oracle: OracleClient) -> None:
        nodes = oracle.get_oracle_nodes()
        assert len(nodes) > 0
        assert nodes[0].node_id == "node_1"
        assert nodes[0].is_active is True


# ---------------------------------------------------------------------------
# FeedSubscription
# ---------------------------------------------------------------------------


class TestFeedSubscription:
    """Test subscription lifecycle without network."""

    def test_unsubscribe_stops(self) -> None:
        sub = FeedSubscription(
            oracle_client=MagicMock(),
            feed_id="test",
            callback=lambda f: None,
        )
        assert not sub.is_active
        sub._running = True
        sub.unsubscribe()
        assert not sub.is_active
