"""Tests for seals module — SealsModule create, verify, revoke."""

from __future__ import annotations

import pytest

from aethelred.seals import SealsModule
from aethelred.core.types import SealStatus


class TestSealsModule:
    """Test seal operations via mock client."""

    @pytest.fixture
    def module(self, mock_client) -> SealsModule:
        return SealsModule(mock_client)

    def test_init(self, module: SealsModule) -> None:
        assert module.BASE_PATH == "/aethelred/seal/v1"
