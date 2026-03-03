"""Tests for core.config — AethelredConfig and SecretStr."""

from __future__ import annotations

import pytest

from aethelred.core.config import AethelredConfig, SecretStr


class TestSecretStr:
    """Test SecretStr redaction behaviour."""

    def test_value_accessible(self) -> None:
        s = SecretStr("my-secret-key")
        assert s.get_secret_value() == "my-secret-key"

    def test_repr_redacted(self) -> None:
        s = SecretStr("password123")
        assert "password123" not in repr(s)
        assert "***" in repr(s) or "REDACTED" in repr(s).upper() or "SecretStr" in repr(s)

    def test_str_redacted(self) -> None:
        s = SecretStr("password123")
        assert "password123" not in str(s)

    def test_equality(self) -> None:
        a = SecretStr("same")
        b = SecretStr("same")
        assert a.get_secret_value() == b.get_secret_value()


class TestAethelredConfig:
    """Test SDK configuration."""

    def test_defaults(self) -> None:
        config = AethelredConfig()
        assert config is not None

    def test_custom_endpoint(self) -> None:
        config = AethelredConfig(endpoint="https://custom.aethelred.io")
        assert config.endpoint == "https://custom.aethelred.io"

    def test_chain_id(self) -> None:
        config = AethelredConfig(chain_id="aethelred-testnet-1")
        assert config.chain_id == "aethelred-testnet-1"
