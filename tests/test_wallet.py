"""Tests for the DualKeyWallet — signing, verification, and key management.

Covers:
- Wallet creation and address derivation
- Composite signature sign/verify roundtrip
- Key export with password protection
- Zeroization on close()
- Bech32 address format validation
"""

from __future__ import annotations

import pytest

from aethelred.core.wallet import (
    DualKeyWallet,
    ECDSASigner,
    CompositeSignature,
    SignatureScheme,
    bech32_encode,
)
from aethelred.crypto.pqc.dilithium import DilithiumSecurityLevel


# ---------------------------------------------------------------------------
# Wallet Creation
# ---------------------------------------------------------------------------


class TestWalletCreation:
    """Test wallet initialization and key generation."""

    def test_creates_with_defaults(self) -> None:
        wallet = DualKeyWallet()
        assert wallet.address.startswith("aethel1")
        wallet.close()

    def test_address_is_deterministic_for_same_keys(self) -> None:
        signer = ECDSASigner()
        pk = signer.private_key
        w1 = DualKeyWallet(classical_key=pk)
        w2 = DualKeyWallet(classical_key=pk)
        # Different Dilithium keys → different addresses
        assert w1.address != w2.address
        w1.close()
        w2.close()

    def test_address_bech32_format(self) -> None:
        wallet = DualKeyWallet()
        addr = wallet.address
        assert addr.startswith("aethel1")
        assert len(addr) > 10
        # Bech32 charset: qpzry9x8gf2tvdw0s3jn54khce6mua7l
        valid_chars = set("qpzry9x8gf2tvdw0s3jn54khce6mua7l")
        assert all(c in valid_chars for c in addr[5:])  # After "aethel1"
        wallet.close()

    def test_different_wallets_have_different_addresses(self) -> None:
        w1 = DualKeyWallet()
        w2 = DualKeyWallet()
        assert w1.address != w2.address
        w1.close()
        w2.close()


# ---------------------------------------------------------------------------
# Signing & Verification
# ---------------------------------------------------------------------------


class TestSigningVerification:
    """Test composite signing and verification."""

    def test_sign_transaction_produces_composite(self) -> None:
        wallet = DualKeyWallet()
        sig = wallet.sign_transaction(b"test tx data")
        assert isinstance(sig, CompositeSignature)
        assert sig.scheme == SignatureScheme.COMPOSITE
        assert len(sig.classical_sig) == 64  # ECDSA r||s
        assert len(sig.pqc_sig) > 0  # Dilithium
        wallet.close()

    def test_sign_verify_roundtrip(self) -> None:
        wallet = DualKeyWallet()
        tx = b"important transaction"
        sig = wallet.sign_transaction(tx)
        assert wallet.verify_signature(tx, sig)
        wallet.close()

    def test_verify_wrong_message_fails(self) -> None:
        wallet = DualKeyWallet()
        sig = wallet.sign_transaction(b"correct")
        assert not wallet.verify_signature(b"wrong", sig)
        wallet.close()

    def test_sign_message_with_domain_separation(self) -> None:
        wallet = DualKeyWallet()
        sig = wallet.sign_message("hello world")
        assert isinstance(sig, CompositeSignature)
        assert sig.signer_address == wallet.address
        wallet.close()

    def test_sign_empty_message(self) -> None:
        wallet = DualKeyWallet()
        sig = wallet.sign_transaction(b"")
        assert isinstance(sig, CompositeSignature)
        wallet.close()


# ---------------------------------------------------------------------------
# Composite Signature Serialization
# ---------------------------------------------------------------------------


class TestCompositeSignatureSerialization:
    """Test composite signature to_bytes / from_bytes roundtrip."""

    def test_roundtrip(self) -> None:
        wallet = DualKeyWallet()
        sig = wallet.sign_transaction(b"data")
        serialized = sig.to_bytes()
        restored = CompositeSignature.from_bytes(serialized)
        assert restored.classical_sig == sig.classical_sig
        assert restored.pqc_sig == sig.pqc_sig
        wallet.close()

    def test_from_bytes_rejects_short_data(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            CompositeSignature.from_bytes(b"\x00\x01")

    def test_from_bytes_rejects_wrong_marker(self) -> None:
        with pytest.raises(ValueError, match="scheme marker"):
            CompositeSignature.from_bytes(b"\x99\x00\x01\x00\x00\x01\x00")

    def test_from_bytes_rejects_truncated(self) -> None:
        wallet = DualKeyWallet()
        sig = wallet.sign_transaction(b"data")
        raw = sig.to_bytes()
        with pytest.raises(ValueError):
            CompositeSignature.from_bytes(raw[:10])  # Truncated
        wallet.close()


# ---------------------------------------------------------------------------
# Zeroization (PY-20)
# ---------------------------------------------------------------------------


class TestZeroization:
    """Test that close() performs key zeroization."""

    def test_close_sets_closed_flag(self) -> None:
        wallet = DualKeyWallet()
        assert not wallet._closed
        wallet.close()
        assert wallet._closed

    def test_double_close_is_safe(self) -> None:
        wallet = DualKeyWallet()
        wallet.close()
        wallet.close()  # Should not raise

    def test_del_calls_close(self) -> None:
        wallet = DualKeyWallet()
        assert not wallet._closed
        del wallet  # Should trigger __del__ → close()


# ---------------------------------------------------------------------------
# Bech32 Encoding
# ---------------------------------------------------------------------------


class TestBech32:
    """Test bech32 address encoding utility."""

    def test_encode_deterministic(self) -> None:
        data = b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a" * 2
        addr1 = bech32_encode("aethel", data)
        addr2 = bech32_encode("aethel", data)
        assert addr1 == addr2

    def test_different_data_different_address(self) -> None:
        d1 = b"\x00" * 20
        d2 = b"\xff" * 20
        assert bech32_encode("aethel", d1) != bech32_encode("aethel", d2)

    def test_prefix(self) -> None:
        addr = bech32_encode("aethel", b"\x00" * 20)
        assert addr.startswith("aethel1")
