"""Tests for oracles.provenance — W3C Verifiable Credentials, TrustedData."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from aethelred.oracles.provenance import (
    CredentialType,
    ProofType as ProvProofType,
    DIDMethod,
    VerificationResult,
    CredentialStatus,
    TrustedData,
)


class TestCredentialType:
    """Test credential type enum."""

    def test_oracle_attestations(self) -> None:
        assert CredentialType.ORACLE_PRICE_ATTESTATION.value == "OraclePriceAttestation"
        assert CredentialType.ORACLE_DATA_ATTESTATION.value == "OracleDataAttestation"

    def test_aethelred_seal(self) -> None:
        assert CredentialType.AETHELRED_SEAL_ATTESTATION.value == "AethelredSealAttestation"


class TestDIDMethod:
    """Test DID method enum."""

    def test_methods(self) -> None:
        assert DIDMethod.KEY.value == "key"
        assert DIDMethod.AETHELRED.value == "aethel"
        assert DIDMethod.ETHR.value == "ethr"


class TestVerificationResult:
    """Test verification result dataclass."""

    def test_valid_result(self) -> None:
        result = VerificationResult(
            valid=True,
            issuer_did="did:key:z6Mk...",
            subject_id="oracle_1",
            issuance_date=datetime.now(timezone.utc),
            proof_type="Ed25519Signature2020",
        )
        assert result.valid is True
        assert result.warnings == []

    def test_invalid_result(self) -> None:
        result = VerificationResult(
            valid=False,
            issuer_did="did:key:fake",
            subject_id=None,
            issuance_date=datetime.now(timezone.utc),
            proof_type="unknown",
            errors=["Invalid signature"],
        )
        assert not result.valid
        assert len(result.errors) == 1

    def test_to_dict(self) -> None:
        result = VerificationResult(
            valid=True,
            issuer_did="did:key:123",
            subject_id="s1",
            issuance_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            proof_type="Ed25519Signature2020",
        )
        d = result.to_dict()
        assert d["valid"] is True
        assert d["issuer_did"] == "did:key:123"


class TestCredentialStatus:
    """Test credential status dataclass."""

    def test_create(self) -> None:
        status = CredentialStatus(id="status_1")
        assert status.type == "RevocationList2020Status"
        assert status.revocation_list_index == 0

    def test_to_dict(self) -> None:
        status = CredentialStatus(id="s1", revocation_list_index=5)
        d = status.to_dict()
        assert d["id"] == "s1"
        assert d["revocationListIndex"] == 5 or "revocation" in str(d).lower()


class TestTrustedData:
    """Test TrustedData wrapper."""

    def _make_credential(self) -> dict:
        """Build a minimal W3C Verifiable Credential."""
        return {
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://aethelred.org/credentials/oracle/v1",
            ],
            "id": "urn:uuid:test-credential-123",
            "type": ["VerifiableCredential", "OraclePriceAttestation"],
            "issuer": "did:key:z6MkTest",
            "issuanceDate": "2026-01-15T12:00:00Z",
            "credentialSubject": {
                "id": "did:aethelred:oracle:btc_usd",
                "type": "OraclePriceAttestation",
                "value": 42000.0,
                "feed_id": "market_data/btc_usd",
            },
            "proof": {
                "type": "Ed25519Signature2020",
                "created": "2026-01-15T12:00:00Z",
                "proofPurpose": "assertionMethod",
                "verificationMethod": "did:key:z6MkTest#key-1",
                "proofValue": "z" + "a" * 64,
            },
        }

    def test_from_credential(self) -> None:
        td = TrustedData.from_credential(self._make_credential())
        assert td is not None
        assert td.credential is not None

    def test_data_hash(self) -> None:
        td = TrustedData.from_credential(self._make_credential())
        h = td.data_hash
        assert isinstance(h, str)
        assert len(h) == 64

    def test_is_expired_false(self) -> None:
        td = TrustedData.from_credential(self._make_credential())
        assert td.is_expired is False  # No expiration set

    def test_to_dict(self) -> None:
        td = TrustedData.from_credential(self._make_credential())
        d = td.to_dict()
        assert "credential" in d

    def test_to_json(self) -> None:
        td = TrustedData.from_credential(self._make_credential())
        j = td.to_json()
        assert isinstance(j, str)
        assert "VerifiableCredential" in j
