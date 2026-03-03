"""Tests for jobs module — JobsModule submit, get, cancel."""

from __future__ import annotations

import pytest

from aethelred.jobs import JobsModule
from aethelred.core.types import JobStatus


class TestJobsModule:
    """Test job operations via mock client."""

    @pytest.fixture
    def module(self, mock_client) -> JobsModule:
        return JobsModule(mock_client)

    def test_init(self, module: JobsModule) -> None:
        assert module.BASE_PATH == "/aethelred/pouw/v1"
