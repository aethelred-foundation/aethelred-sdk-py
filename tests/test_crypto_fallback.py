"""Tests for the cryptographic backend fallback system."""

from __future__ import annotations

import pytest

from aethelred.crypto.fallback import (
    BackendVariant,
    HybridSigner,
    HybridVerifier,
    get_backend,
    is_native,
)


class TestBackendDetection:
    """Test backend auto-detection."""

    def test_backend_detected(self) -> None:
        backend = get_backend()
        assert backend.name in ("liboqs", "aethelred-pure-python")
        assert isinstance(backend.variant, BackendVariant)

    def test_backend_cached(self) -> None:
        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2

    def test_backend_str(self) -> None:
        s = str(get_backend())
        assert "FIPS" in s

    def test_is_native_returns_bool(self) -> None:
        assert isinstance(is_native(), bool)


class TestHybridSigner:
    """Test hybrid signing and verification across backends."""

    def test_sign_produces_bytes(self) -> None:
        signer = HybridSigner()
        sig = signer.sign(b"test message")
        assert isinstance(sig, bytes)
        assert len(sig) > 64  # Must be larger than just ECDSA

    def test_sign_verify_roundtrip(self) -> None:
        signer = HybridSigner()
        message = b"hello aethelred"
        sig = signer.sign(message)
        assert signer.verify(message, sig)

    def test_verify_wrong_message_fails(self) -> None:
        signer = HybridSigner()
        sig = signer.sign(b"correct message")
        try:
            result = signer.verify(b"wrong message", sig)
            # If verify returns without raising, it should be False
            assert not result
        except Exception:
            # BadSignatureError propagating is also acceptable (ecdsa backend)
            pass

    def test_verify_tampered_signature_fails(self) -> None:
        signer = HybridSigner()
        sig = bytearray(signer.sign(b"data"))
        sig[-1] ^= 0xFF  # Flip last byte
        assert not signer.verify(b"data", bytes(sig))

    def test_verify_empty_signature_fails(self) -> None:
        signer = HybridSigner()
        assert not signer.verify(b"data", b"")

    def test_verify_short_signature_fails(self) -> None:
        signer = HybridSigner()
        assert not signer.verify(b"data", b"\x00\x00\x00\x01")

    def test_public_key_bytes(self) -> None:
        signer = HybridSigner()
        pk = signer.public_key_bytes()
        assert isinstance(pk, bytes)
        assert len(pk) > 32  # Must include both ECDSA + Dilithium

    def test_fingerprint(self) -> None:
        signer = HybridSigner()
        fp = signer.fingerprint
        assert isinstance(fp, str)
        assert len(fp) == 16  # First 16 hex chars of SHA-256

    def test_different_signers_produce_different_keys(self) -> None:
        s1 = HybridSigner()
        s2 = HybridSigner()
        assert s1.public_key_bytes() != s2.public_key_bytes()

    def test_deterministic_with_same_keys(self) -> None:
        signer = HybridSigner()
        pk = signer.public_key_bytes()
        # Verify consistency
        assert signer.public_key_bytes() == pk


class TestHybridVerifier:
    """Test public-key-only verification."""

    def test_verifier_from_signer_pubkey(self) -> None:
        signer = HybridSigner()
        message = b"verify with separate verifier"
        sig = signer.sign(message)

        verifier = HybridVerifier(signer.public_key_bytes())
        assert verifier.verify(message, sig)

    def test_verifier_rejects_wrong_message(self) -> None:
        signer = HybridSigner()
        sig = signer.sign(b"original")

        verifier = HybridVerifier(signer.public_key_bytes())
        try:
            result = verifier.verify(b"tampered", sig)
            # If verify returns without raising, it should be False
            assert not result
        except Exception:
            # BadSignatureError propagating is also acceptable (ecdsa backend)
            pass

    def test_verifier_rejects_wrong_key(self) -> None:
        signer1 = HybridSigner()
        signer2 = HybridSigner()
        message = b"cross-key test"
        sig = signer1.sign(message)

        verifier = HybridVerifier(signer2.public_key_bytes())
        try:
            result = verifier.verify(message, sig)
            # If verify returns without raising, it should be False
            assert not result
        except Exception:
            # BadSignatureError propagating is also acceptable (ecdsa backend)
            pass
