"""
Comprehensive tests for the realtime WebSocket module:
- aethelred/core/realtime.py (0% covered, 554 stmts)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# We need to mock websockets before importing the module if it's not available
import sys


# ============ Event and Data Class Tests ============

from aethelred.core.realtime import (
    EventType,
    JobEvent,
    SealEvent,
    BlockEvent,
    ConnectionConfig,
    ConnectionState,
    ConnectionMetrics,
    JobSubscription,
    SealSubscription,
    BlockSubscription,
)
from aethelred.core.types import JobStatus, ProofType


class TestEventType:
    def test_connection_events(self):
        assert EventType.CONNECTED == "connected"
        assert EventType.DISCONNECTED == "disconnected"
        assert EventType.RECONNECTING == "reconnecting"
        assert EventType.ERROR == "error"

    def test_job_events(self):
        assert EventType.JOB_CREATED == "job.created"
        assert EventType.JOB_COMPLETED == "job.completed"
        assert EventType.JOB_FAILED == "job.failed"
        assert EventType.JOB_CANCELLED == "job.cancelled"

    def test_seal_events(self):
        assert EventType.SEAL_CREATED == "seal.created"
        assert EventType.SEAL_VERIFIED == "seal.verified"
        assert EventType.SEAL_REVOKED == "seal.revoked"

    def test_system_events(self):
        assert EventType.HEARTBEAT == "heartbeat"
        assert EventType.SUBSCRIPTION_CONFIRMED == "subscription.confirmed"


class TestJobEvent:
    def test_from_message_full(self):
        data = {
            "job_id": "job_123",
            "type": "job.completed",
            "status": "JOB_STATUS_COMPLETED",
            "timestamp": "2024-01-01T00:00:00Z",
            "progress_percent": 100,
            "current_step": "done",
            "output_hash": "abc123",
            "proof_id": "proof_1",
            "seal_id": "seal_1",
            "assigned_validators": ["v1", "v2"],
            "completed_validators": ["v1"],
            "execution_time_ms": 500,
            "total_time_ms": 1000,
        }
        event = JobEvent.from_message(data)
        assert event.job_id == "job_123"
        assert event.event_type == EventType.JOB_COMPLETED
        assert event.status == JobStatus.COMPLETED
        assert event.progress_percent == 100
        assert event.output_hash == "abc123"
        assert event.assigned_validators == ["v1", "v2"]
        assert event.execution_time_ms == 500

    def test_from_message_defaults(self):
        data = {}
        event = JobEvent.from_message(data)
        assert event.job_id == ""
        assert event.event_type == EventType.JOB_EXECUTING  # fallback for unknown type
        assert event.status == JobStatus.PENDING  # fallback

    def test_from_message_unknown_event_type(self):
        data = {"type": "unknown.event", "status": "pending"}
        event = JobEvent.from_message(data)
        assert event.event_type == EventType.JOB_EXECUTING

    def test_from_message_unknown_status(self):
        data = {"type": "job.created", "status": "unknown_status"}
        event = JobEvent.from_message(data)
        assert event.status == JobStatus.PENDING

    def test_from_message_with_timestamp(self):
        data = {"timestamp": "2024-06-15T12:30:00Z"}
        event = JobEvent.from_message(data)
        assert event.timestamp.year == 2024

    def test_from_message_without_timestamp(self):
        data = {}
        event = JobEvent.from_message(data)
        assert event.timestamp is not None

    def test_is_complete_completed(self):
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_COMPLETED,
            status=JobStatus.COMPLETED, timestamp=datetime.now(timezone.utc)
        )
        assert event.is_complete is True

    def test_is_complete_failed(self):
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_FAILED,
            status=JobStatus.FAILED, timestamp=datetime.now(timezone.utc)
        )
        assert event.is_complete is True

    def test_is_complete_cancelled(self):
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_CANCELLED,
            status=JobStatus.CANCELLED, timestamp=datetime.now(timezone.utc)
        )
        assert event.is_complete is True

    def test_is_complete_pending(self):
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_QUEUED,
            status=JobStatus.PENDING, timestamp=datetime.now(timezone.utc)
        )
        assert event.is_complete is False

    def test_is_success(self):
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_COMPLETED,
            status=JobStatus.COMPLETED, timestamp=datetime.now(timezone.utc)
        )
        assert event.is_success is True

    def test_is_not_success(self):
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_FAILED,
            status=JobStatus.FAILED, timestamp=datetime.now(timezone.utc)
        )
        assert event.is_success is False


class TestSealEvent:
    def test_from_message_full(self):
        data = {
            "seal_id": "seal_1",
            "type": "seal.created",
            "timestamp": "2024-01-01T00:00:00Z",
            "job_id": "job_1",
            "model_hash": "mhash",
            "input_hash": "ihash",
            "output_hash": "ohash",
            "proof_type": "tee",
            "proof_hash": "phash",
            "block_height": 100,
            "tx_hash": "txhash",
            "validators": ["v1"],
        }
        event = SealEvent.from_message(data)
        assert event.seal_id == "seal_1"
        assert event.event_type == EventType.SEAL_CREATED
        assert event.job_id == "job_1"
        assert event.proof_type == ProofType.TEE
        assert event.block_height == 100

    def test_from_message_defaults(self):
        data = {}
        event = SealEvent.from_message(data)
        assert event.seal_id == ""
        assert event.event_type == EventType.SEAL_CREATED
        assert event.proof_type == ProofType.TEE
        assert event.block_height == 0

    def test_from_message_unknown_event_type(self):
        data = {"type": "unknown"}
        event = SealEvent.from_message(data)
        assert event.event_type == EventType.SEAL_CREATED

    def test_from_message_unknown_proof_type(self):
        data = {"proof_type": "unknown_proof"}
        event = SealEvent.from_message(data)
        assert event.proof_type == ProofType.TEE

    def test_from_message_without_timestamp(self):
        data = {}
        event = SealEvent.from_message(data)
        assert event.timestamp is not None


class TestBlockEvent:
    def test_from_message_full(self):
        data = {
            "block_height": 42,
            "block_hash": "blockhash",
            "timestamp": "2024-01-01T00:00:00Z",
            "num_transactions": 10,
            "num_seals": 3,
            "proposer": "val1",
        }
        event = BlockEvent.from_message(data)
        assert event.block_height == 42
        assert event.block_hash == "blockhash"
        assert event.num_transactions == 10
        assert event.num_seals == 3
        assert event.proposer == "val1"

    def test_from_message_defaults(self):
        data = {}
        event = BlockEvent.from_message(data)
        assert event.block_height == 0
        assert event.block_hash == ""
        assert event.num_transactions == 0

    def test_from_message_without_timestamp(self):
        data = {}
        event = BlockEvent.from_message(data)
        assert event.timestamp is not None


class TestConnectionConfig:
    def test_defaults(self):
        cc = ConnectionConfig()
        assert cc.reconnect_enabled is True
        assert cc.reconnect_max_attempts == 10
        assert cc.reconnect_base_delay_ms == 1000
        assert cc.reconnect_max_delay_ms == 30000
        assert cc.heartbeat_enabled is True
        assert cc.heartbeat_interval_s == 30.0
        assert cc.max_queue_size == 1000
        assert cc.queue_overflow_strategy == "drop_oldest"


class TestConnectionState:
    def test_values(self):
        assert ConnectionState.DISCONNECTED == "disconnected"
        assert ConnectionState.CONNECTING == "connecting"
        assert ConnectionState.CONNECTED == "connected"
        assert ConnectionState.RECONNECTING == "reconnecting"
        assert ConnectionState.CLOSING == "closing"
        assert ConnectionState.CLOSED == "closed"


class TestConnectionMetrics:
    def test_defaults(self):
        cm = ConnectionMetrics(state=ConnectionState.DISCONNECTED)
        assert cm.reconnect_attempts == 0
        assert cm.messages_sent == 0
        assert cm.messages_received == 0
        assert cm.messages_dropped == 0

    def test_uptime_when_connected(self):
        cm = ConnectionMetrics(
            state=ConnectionState.CONNECTED,
            connected_at=datetime.now(timezone.utc),
        )
        uptime = cm.uptime_seconds
        assert uptime is not None
        assert uptime >= 0

    def test_uptime_when_disconnected(self):
        cm = ConnectionMetrics(state=ConnectionState.DISCONNECTED)
        assert cm.uptime_seconds is None

    def test_uptime_when_connected_but_no_timestamp(self):
        cm = ConnectionMetrics(state=ConnectionState.CONNECTED)
        assert cm.uptime_seconds is None


# ============ Subscription Tests ============


class TestJobSubscription:
    def test_init_with_job_ids(self):
        sub = JobSubscription("sub_1", job_ids=["j1", "j2"])
        assert sub.subscription_id == "sub_1"
        assert sub.channel == "jobs"
        assert sub.job_ids == {"j1", "j2"}

    def test_init_without_job_ids(self):
        sub = JobSubscription("sub_1")
        assert sub.job_ids is None

    def test_init_with_status_filter(self):
        sub = JobSubscription("sub_1", status_filter=[JobStatus.COMPLETED])
        assert sub.status_filter == {JobStatus.COMPLETED}

    def test_matches_correct_channel(self):
        sub = JobSubscription("sub_1", job_ids=["j1"])
        assert sub.matches({"channel": "jobs", "job_id": "j1"}) is True

    def test_matches_wrong_channel(self):
        sub = JobSubscription("sub_1")
        assert sub.matches({"channel": "seals"}) is False

    def test_matches_wrong_job_id(self):
        sub = JobSubscription("sub_1", job_ids=["j1"])
        assert sub.matches({"channel": "jobs", "job_id": "j2"}) is False

    def test_matches_no_job_filter(self):
        sub = JobSubscription("sub_1")
        assert sub.matches({"channel": "jobs", "job_id": "any"}) is True

    def test_matches_status_filter_pass(self):
        sub = JobSubscription("sub_1", status_filter=[JobStatus.COMPLETED])
        assert sub.matches({"channel": "jobs", "status": JobStatus.COMPLETED.value}) is True

    def test_matches_status_filter_fail(self):
        sub = JobSubscription("sub_1", status_filter=[JobStatus.COMPLETED])
        assert sub.matches({"channel": "jobs", "status": JobStatus.PENDING.value}) is False

    def test_matches_invalid_status(self):
        sub = JobSubscription("sub_1", status_filter=[JobStatus.COMPLETED])
        assert sub.matches({"channel": "jobs", "status": "invalid_status_xyz"}) is True

    def test_parse_event(self):
        sub = JobSubscription("sub_1")
        event = sub.parse_event({"job_id": "j1", "status": "completed"})
        assert isinstance(event, JobEvent)

    def test_add_job_id(self):
        sub = JobSubscription("sub_1")
        sub.add_job_id("j1")
        assert sub.job_ids == {"j1"}
        sub.add_job_id("j2")
        assert sub.job_ids == {"j1", "j2"}

    def test_add_job_id_with_existing(self):
        sub = JobSubscription("sub_1", job_ids=["j1"])
        sub.add_job_id("j2")
        assert "j2" in sub.job_ids

    def test_remove_job_id(self):
        sub = JobSubscription("sub_1", job_ids=["j1", "j2"])
        sub.remove_job_id("j1")
        assert sub.job_ids == {"j2"}

    def test_remove_job_id_nonexistent(self):
        sub = JobSubscription("sub_1", job_ids=["j1"])
        sub.remove_job_id("j99")  # Should not raise
        assert sub.job_ids == {"j1"}

    def test_remove_job_id_none(self):
        sub = JobSubscription("sub_1")
        sub.remove_job_id("j1")  # Should not raise when job_ids is None

    def test_activate_deactivate(self):
        sub = JobSubscription("sub_1")
        assert sub.is_active is False
        sub.activate()
        assert sub.is_active is True
        sub.deactivate()
        assert sub.is_active is False

    @pytest.mark.asyncio
    async def test_put_and_get_event(self):
        sub = JobSubscription("sub_1")
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_COMPLETED,
            status=JobStatus.COMPLETED, timestamp=datetime.now(timezone.utc)
        )
        await sub.put_event(event)
        result = await sub.get_event(timeout=1.0)
        assert result.job_id == "j1"

    @pytest.mark.asyncio
    async def test_get_event_timeout(self):
        sub = JobSubscription("sub_1")
        with pytest.raises(asyncio.TimeoutError):
            await sub.get_event(timeout=0.01)

    @pytest.mark.asyncio
    async def test_get_event_no_timeout(self):
        sub = JobSubscription("sub_1")
        event = JobEvent(
            job_id="j1", event_type=EventType.JOB_COMPLETED,
            status=JobStatus.COMPLETED, timestamp=datetime.now(timezone.utc)
        )
        await sub.put_event(event)
        result = await sub.get_event()
        assert result.job_id == "j1"


class TestSealSubscription:
    def test_init(self):
        sub = SealSubscription("sub_1", model_hash_filter="hash1", requester_filter="req1")
        assert sub.channel == "seals"
        assert sub.model_hash_filter == "hash1"
        assert sub.requester_filter == "req1"

    def test_matches_correct(self):
        sub = SealSubscription("sub_1")
        assert sub.matches({"channel": "seals"}) is True

    def test_matches_wrong_channel(self):
        sub = SealSubscription("sub_1")
        assert sub.matches({"channel": "jobs"}) is False

    def test_matches_model_hash_filter_pass(self):
        sub = SealSubscription("sub_1", model_hash_filter="hash1")
        assert sub.matches({"channel": "seals", "model_hash": "hash1"}) is True

    def test_matches_model_hash_filter_fail(self):
        sub = SealSubscription("sub_1", model_hash_filter="hash1")
        assert sub.matches({"channel": "seals", "model_hash": "hash2"}) is False

    def test_matches_requester_filter_pass(self):
        sub = SealSubscription("sub_1", requester_filter="req1")
        assert sub.matches({"channel": "seals", "requester": "req1"}) is True

    def test_matches_requester_filter_fail(self):
        sub = SealSubscription("sub_1", requester_filter="req1")
        assert sub.matches({"channel": "seals", "requester": "req2"}) is False

    def test_parse_event(self):
        sub = SealSubscription("sub_1")
        event = sub.parse_event({"seal_id": "s1"})
        assert isinstance(event, SealEvent)


class TestBlockSubscription:
    def test_init(self):
        sub = BlockSubscription("sub_1")
        assert sub.channel == "blocks"

    def test_matches_correct(self):
        sub = BlockSubscription("sub_1")
        assert sub.matches({"channel": "blocks"}) is True

    def test_matches_wrong_channel(self):
        sub = BlockSubscription("sub_1")
        assert sub.matches({"channel": "jobs"}) is False

    def test_parse_event(self):
        sub = BlockSubscription("sub_1")
        event = sub.parse_event({"block_height": 42})
        assert isinstance(event, BlockEvent)


# ============ RealtimeClient Tests ============
# These require websockets to be importable

class TestRealtimeClient:
    """Test RealtimeClient initialization and methods with mocked websockets."""

    @pytest.fixture
    def mock_config(self):
        from aethelred.core.config import Config
        return Config(
            rpc_url="http://localhost:26657",
            endpoint="https://api.aethelred.org",
            chain_id="aethelred-1",
            api_key=None,
        )

    def test_import_check(self):
        """Test that WEBSOCKETS_AVAILABLE flag works."""
        from aethelred.core.realtime import WEBSOCKETS_AVAILABLE
        # This is just a boolean check, either True or False depending on environment
        assert isinstance(WEBSOCKETS_AVAILABLE, bool)

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    def test_init(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        assert client.state == ConnectionState.DISCONNECTED
        assert client.is_connected is False

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    def test_init_custom_connection_config(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        conn_config = ConnectionConfig(
            reconnect_max_attempts=5,
            heartbeat_enabled=False,
        )
        client = RealtimeClient(mock_config, connection_config=conn_config)
        assert client._conn_config.reconnect_max_attempts == 5
        assert client._conn_config.heartbeat_enabled is False

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    def test_properties(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        assert client.state == ConnectionState.DISCONNECTED
        assert isinstance(client.metrics, ConnectionMetrics)
        assert client.is_connected is False

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    def test_build_ws_url(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        assert "ws" in client._ws_url or "wss" in client._ws_url
        assert client._ws_url.endswith("/ws/v1")

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    def test_get_headers_no_api_key(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        headers = client._get_headers()
        assert "X-Chain-ID" in headers
        assert "Authorization" not in headers

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    def test_get_headers_with_api_key(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        mock_config.api_key = "test-api-key"
        client = RealtimeClient(mock_config)
        headers = client._get_headers()
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_connect_already_connected(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        client._state = ConnectionState.CONNECTED
        await client.connect()  # Should return early

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_disconnect_already_disconnected(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        await client.disconnect()  # Should return early
        assert client._state == ConnectionState.DISCONNECTED

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_send_not_connected(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        from aethelred.core.exceptions import ConnectionError
        client = RealtimeClient(mock_config)
        with pytest.raises(ConnectionError, match="Not connected"):
            await client._send({"type": "test"})

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_route_message(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)

        # Add a subscription
        sub = JobSubscription("sub_1", job_ids=["j1"])
        sub.activate()
        client._subscriptions["sub_1"] = sub

        msg = {"channel": "jobs", "job_id": "j1", "status": "completed"}
        await client._route_message(msg)

        # Event should be in queue
        event = await sub.get_event(timeout=1.0)
        assert event.job_id == "j1"

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_route_message_inactive_subscription(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)

        sub = JobSubscription("sub_1", job_ids=["j1"])
        # Not activated
        client._subscriptions["sub_1"] = sub

        msg = {"channel": "jobs", "job_id": "j1", "status": "completed"}
        await client._route_message(msg)

        assert sub._queue.empty()

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_subscribe_job(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        sub_id = await client.subscribe_job("job_123")
        assert sub_id.startswith("job_")
        assert sub_id in client._subscriptions
        assert client._subscriptions[sub_id].is_active is True

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_subscribe_jobs(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        sub_id = await client.subscribe_jobs(["j1", "j2", "j3"])
        assert sub_id.startswith("jobs_")
        assert sub_id in client._subscriptions

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_subscribe_seals(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        sub_id = await client.subscribe_seals(model_hash="hash1", requester="req1")
        assert sub_id.startswith("seals_")
        assert sub_id in client._subscriptions

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_subscribe_blocks(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        sub_id = await client.subscribe_blocks()
        assert sub_id.startswith("blocks_")
        assert sub_id in client._subscriptions

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_unsubscribe(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        sub_id = await client.subscribe_job("job_1")
        await client.unsubscribe(sub_id)
        assert sub_id not in client._subscriptions

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_handle_disconnect_no_reconnect(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        client._conn_config.reconnect_enabled = False
        await client._handle_disconnect()
        assert client._state == ConnectionState.DISCONNECTED

    @pytest.mark.skipif(
        not __import__("aethelred.core.realtime", fromlist=["WEBSOCKETS_AVAILABLE"]).WEBSOCKETS_AVAILABLE,
        reason="websockets not installed"
    )
    @pytest.mark.asyncio
    async def test_handle_disconnect_with_reconnect(self, mock_config):
        from aethelred.core.realtime import RealtimeClient
        client = RealtimeClient(mock_config)
        client._conn_config.reconnect_enabled = True

        # Mock connect to fail quickly
        with patch.object(client, "connect", new_callable=AsyncMock, side_effect=Exception("fail")):
            client._conn_config.reconnect_max_attempts = 1
            client._conn_config.reconnect_base_delay_ms = 1
            client._conn_config.reconnect_max_delay_ms = 1
            await client._handle_disconnect()
            # Let the task run briefly
            await asyncio.sleep(0.05)
            if client._reconnect_task:
                client._reconnect_task.cancel()
                try:
                    await client._reconnect_task
                except asyncio.CancelledError:
                    pass
