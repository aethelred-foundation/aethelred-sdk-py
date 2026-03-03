"""Tests for core.client — client initialization and module wiring."""

from __future__ import annotations

import pytest

from aethelred.core.config import AethelredConfig


class TestClientConfig:
    """Test client can be configured."""

    def test_config_default(self) -> None:
        config = AethelredConfig()
        assert config.endpoint is not None

    def test_config_custom_endpoint(self) -> None:
        config = AethelredConfig(endpoint="https://testnet.aethelred.io:26657")
        assert "testnet" in config.endpoint

    def test_config_custom_chain(self) -> None:
        config = AethelredConfig(chain_id="aethelred-devnet-1")
        assert config.chain_id == "aethelred-devnet-1"


class TestClientImports:
    """Test that all client modules can be imported."""

    def test_import_core(self) -> None:
        from aethelred.core import DualKeyWallet, AethelredConfig
        assert DualKeyWallet is not None
        assert AethelredConfig is not None

    def test_import_crypto(self) -> None:
        from aethelred.crypto import HybridSigner, DilithiumSigner, KyberKEM
        assert HybridSigner is not None
        assert DilithiumSigner is not None
        assert KyberKEM is not None

    def test_import_proofs(self) -> None:
        from aethelred.proofs import ProofGenerator, ProofVerifier
        assert ProofGenerator is not None
        assert ProofVerifier is not None

    def test_import_oracles(self) -> None:
        from aethelred.oracles import OracleClient, AsyncOracleClient, DataFeed
        assert OracleClient is not None
        assert AsyncOracleClient is not None
        assert DataFeed is not None

    def test_import_models(self) -> None:
        from aethelred.models import ModelsModule
        assert ModelsModule is not None

    def test_import_seals(self) -> None:
        from aethelred.seals import SealsModule
        assert SealsModule is not None

    def test_import_jobs(self) -> None:
        from aethelred.jobs import JobsModule
        assert JobsModule is not None

    def test_import_validators(self) -> None:
        from aethelred.validators import ValidatorsModule
        assert ValidatorsModule is not None

    def test_import_verification(self) -> None:
        from aethelred.verification import VerificationModule
        assert VerificationModule is not None

    def test_import_utils(self) -> None:
        from aethelred.utils import sha256_hex, retry, format_size
        assert sha256_hex is not None
        assert retry is not None
        assert format_size is not None
