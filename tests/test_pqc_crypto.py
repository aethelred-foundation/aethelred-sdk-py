"""Tests for PQC crypto modules — Dilithium and Kyber."""

from __future__ import annotations

import pytest

from aethelred.crypto.pqc.dilithium import (
    DilithiumSigner,
    DilithiumSecurityLevel,
    DilithiumSignature,
)
from aethelred.crypto.pqc.kyber import (
    KyberKEM,
    KyberSecurityLevel,
)


class TestDilithiumSigner:
    """Test Dilithium post-quantum signing."""

    def test_create_default_level(self) -> None:
        signer = DilithiumSigner()
        assert signer is not None

    def test_create_all_levels(self) -> None:
        for level in DilithiumSecurityLevel:
            signer = DilithiumSigner(level=level)
            assert signer is not None

    def test_sign_produces_signature(self) -> None:
        signer = DilithiumSigner()
        sig = signer.sign(b"hello world")
        assert isinstance(sig, DilithiumSignature)
        assert len(sig.signature) > 0

    def test_sign_verify_roundtrip(self) -> None:
        signer = DilithiumSigner()
        message = b"test message for dilithium"
        sig = signer.sign(message)
        assert signer.verify(message, sig)

    def test_verify_wrong_message_fails(self) -> None:
        signer = DilithiumSigner()
        sig = signer.sign(b"correct")
        assert not signer.verify(b"wrong", sig)

    def test_public_key_bytes(self) -> None:
        signer = DilithiumSigner()
        pk = signer.public_key_bytes()
        assert isinstance(pk, bytes)
        assert len(pk) > 0

    def test_different_signers_different_keys(self) -> None:
        s1 = DilithiumSigner()
        s2 = DilithiumSigner()
        assert s1.public_key_bytes() != s2.public_key_bytes()


class TestKyberKEM:
    """Test Kyber key encapsulation mechanism."""

    def test_create_default(self) -> None:
        kem = KyberKEM()
        assert kem is not None

    def test_create_all_levels(self) -> None:
        for level in KyberSecurityLevel:
            kem = KyberKEM(level=level)
            assert kem is not None

    def test_encapsulate_decapsulate_roundtrip(self) -> None:
        kem = KyberKEM()
        ciphertext, shared_secret_enc = kem.encapsulate()
        shared_secret_dec = kem.decapsulate(ciphertext)
        assert shared_secret_enc == shared_secret_dec

    def test_public_key_available(self) -> None:
        kem = KyberKEM()
        pk = kem.public_key_bytes()
        assert isinstance(pk, bytes)
        assert len(pk) > 0
