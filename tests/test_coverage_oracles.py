"""
Comprehensive tests for oracles/provenance module:
- aethelred/oracles/provenance.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aethelred.oracles.provenance import (
    CredentialType,
    ProofType,
    DIDMethod,
    VerificationResult,
    CredentialStatus,
    TrustedData,
    ProvenanceIssuer,
    ProvenanceVerifier,
    OracleProvenanceRegistry,
    wrap_oracle_data,
    extract_data_if_valid,
    DIDKIT_AVAILABLE,
)


class TestCredentialType:
    def test_values(self):
        assert CredentialType.ORACLE_PRICE_ATTESTATION == "OraclePriceAttestation"
        assert CredentialType.ORACLE_DATA_ATTESTATION == "OracleDataAttestation"
        assert CredentialType.AETHELRED_SEAL_ATTESTATION == "AethelredSealAttestation"


class TestProofType:
    def test_values(self):
        assert ProofType.ED25519_SIGNATURE_2020 == "Ed25519Signature2020"
        assert ProofType.JSON_WEB_SIGNATURE_2020 == "JsonWebSignature2020"


class TestDIDMethod:
    def test_values(self):
        assert DIDMethod.KEY == "key"
        assert DIDMethod.WEB == "web"
        assert DIDMethod.AETHELRED == "aethel"


class TestVerificationResult:
    def test_creation(self):
        vr = VerificationResult(
            valid=True,
            issuer_did="did:key:abc",
            subject_id="sub1",
            issuance_date=datetime.now(timezone.utc),
            proof_type="Ed25519Signature2020",
        )
        assert vr.valid is True
        assert vr.errors == []
        assert vr.warnings == []

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        vr = VerificationResult(
            valid=True,
            issuer_did="did:key:abc",
            subject_id="sub1",
            issuance_date=now,
            proof_type="Ed25519Signature2020",
            expiration_date=now + timedelta(hours=24),
            errors=["err1"],
            warnings=["warn1"],
        )
        d = vr.to_dict()
        assert d["valid"] is True
        assert d["issuer_did"] == "did:key:abc"
        assert d["errors"] == ["err1"]
        assert d["warnings"] == ["warn1"]
        assert d["expiration_date"] is not None

    def test_to_dict_no_expiration(self):
        vr = VerificationResult(
            valid=False,
            issuer_did="did:key:x",
            subject_id=None,
            issuance_date=datetime.now(timezone.utc),
            proof_type="unknown",
        )
        d = vr.to_dict()
        assert d["expiration_date"] is None


class TestCredentialStatus:
    def test_defaults(self):
        cs = CredentialStatus(id="status_1")
        assert cs.type == "RevocationList2020Status"
        assert cs.revocation_list_index == 0

    def test_to_dict(self):
        cs = CredentialStatus(
            id="status_1",
            revocation_list_index=5,
            revocation_list_credential="https://example.com/list",
        )
        d = cs.to_dict()
        assert d["id"] == "status_1"
        assert d["revocationListIndex"] == "5"
        assert d["revocationListCredential"] == "https://example.com/list"


class TestTrustedData:
    def _make_trusted_data(self, **kwargs):
        now = datetime.now(timezone.utc)
        defaults = {
            "credential": {"@context": [], "type": ["VerifiableCredential"]},
            "data": {"price": 45000},
            "issuer_did": "did:key:test",
            "issuance_date": now,
            "proof": {
                "type": "Ed25519Signature2020",
                "created": now.isoformat(),
                "verificationMethod": "did:key:test#key-1",
                "proofPurpose": "assertionMethod",
            },
        }
        defaults.update(kwargs)
        return TrustedData(**defaults)

    def test_creation(self):
        td = self._make_trusted_data()
        assert td.issuer_did == "did:key:test"
        assert td.data == {"price": 45000}
        assert td.credential_type == CredentialType.ORACLE_DATA_ATTESTATION

    def test_is_expired_not_expired(self):
        td = self._make_trusted_data(
            expiration_date=datetime.now(timezone.utc) + timedelta(hours=24)
        )
        assert td.is_expired is False

    def test_is_expired_expired(self):
        td = self._make_trusted_data(
            expiration_date=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        assert td.is_expired is True

    def test_is_expired_no_expiration(self):
        td = self._make_trusted_data()
        assert td.is_expired is False

    def test_data_hash(self):
        td = self._make_trusted_data(data={"price": 45000})
        expected = hashlib.sha256(
            json.dumps({"price": 45000}, sort_keys=True).encode()
        ).hexdigest()
        assert td.data_hash == expected

    def test_verification_result_property(self):
        td = self._make_trusted_data()
        assert td.verification_result is None

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        td = self._make_trusted_data(
            expiration_date=now + timedelta(hours=24),
            oracle_id="chainlink",
        )
        d = td.to_dict()
        assert d["issuer_did"] == "did:key:test"
        assert d["oracle_id"] == "chainlink"
        assert d["data_hash"] is not None
        assert d["expiration_date"] is not None

    def test_to_dict_no_expiration(self):
        td = self._make_trusted_data()
        d = td.to_dict()
        assert d["expiration_date"] is None

    def test_to_json(self):
        td = self._make_trusted_data()
        j = td.to_json()
        parsed = json.loads(j)
        assert parsed["issuer_did"] == "did:key:test"

    @pytest.mark.asyncio
    async def test_verify_fallback_valid(self):
        td = self._make_trusted_data()
        result = await td.verify()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_verify_cached(self):
        td = self._make_trusted_data()
        result1 = await td.verify()
        result2 = await td.verify()  # Should use cache
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_verify_with_options_skips_cache(self):
        td = self._make_trusted_data()
        await td.verify()
        result = await td.verify(options={"custom": True})
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_verify_expired_credential(self):
        td = self._make_trusted_data(
            expiration_date=datetime.now(timezone.utc) - timedelta(hours=1)
        )
        result = await td.verify()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_future_issuance(self):
        td = self._make_trusted_data(
            issuance_date=datetime.now(timezone.utc) + timedelta(hours=1)
        )
        result = await td.verify()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_no_proof(self):
        td = self._make_trusted_data(proof=None)
        result = await td.verify()
        # Without didkit, fallback checks for proof
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_invalid_proof_structure(self):
        td = self._make_trusted_data(proof={"type": "test"})  # Missing required fields
        result = await td.verify()
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_exception_handling(self):
        td = self._make_trusted_data()
        with patch.object(td, "_verify_credential", side_effect=Exception("boom")):
            result = await td.verify()
            assert result is False
            assert td._verification_result is not None
            assert len(td._verification_result.errors) > 0

    @pytest.mark.asyncio
    async def test_verify_exception_no_proof(self):
        td = self._make_trusted_data(proof=None)
        with patch.object(td, "_verify_credential", side_effect=Exception("error")):
            result = await td.verify()
            assert result is False

    def test_verify_proof_structure_valid(self):
        td = self._make_trusted_data()
        assert td._verify_proof_structure() is True

    def test_verify_proof_structure_missing_fields(self):
        td = self._make_trusted_data(proof={"type": "test"})
        assert td._verify_proof_structure() is False

    def test_verify_proof_structure_no_proof(self):
        td = self._make_trusted_data(proof=None)
        assert td._verify_proof_structure() is False

    def test_from_credential_basic(self):
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {"price": 100},
            "issuanceDate": "2024-01-01T00:00:00Z",
            "type": ["VerifiableCredential"],
            "proof": {"type": "Ed25519Signature2020"},
        }
        td = TrustedData.from_credential(cred)
        assert td.issuer_did == "did:key:abc"
        assert td.data == {"price": 100}

    def test_from_credential_issuer_dict(self):
        cred = {
            "issuer": {"id": "did:key:xyz", "name": "Test Oracle"},
            "credentialSubject": {"data": "value"},
            "type": ["VerifiableCredential", "OraclePriceAttestation"],
        }
        td = TrustedData.from_credential(cred)
        assert td.issuer_did == "did:key:xyz"
        assert td.credential_type == CredentialType.ORACLE_PRICE_ATTESTATION

    def test_from_credential_with_expiration(self):
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "issuanceDate": "2024-01-01T00:00:00Z",
            "expirationDate": "2024-12-31T23:59:59Z",
            "type": [],
        }
        td = TrustedData.from_credential(cred)
        assert td.expiration_date is not None
        assert td.expiration_date.year == 2024

    def test_from_credential_no_matching_type(self):
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "type": ["VerifiableCredential", "UnknownType"],
        }
        td = TrustedData.from_credential(cred)
        assert td.credential_type == CredentialType.ORACLE_DATA_ATTESTATION


class TestProvenanceIssuer:
    def test_init(self):
        issuer = ProvenanceIssuer(
            did="did:key:abc",
            private_key_jwk='{"kty":"OKP"}',
        )
        assert issuer.did == "did:key:abc"
        assert issuer.proof_type == ProofType.ED25519_SIGNATURE_2020
        assert issuer.verification_method == "did:key:abc#key-1"

    def test_init_custom_verification_method(self):
        issuer = ProvenanceIssuer(
            did="did:key:abc",
            private_key_jwk='{"kty":"OKP"}',
            verification_method="did:key:abc#custom",
        )
        assert issuer.verification_method == "did:key:abc#custom"

    def test_from_jwk_string(self):
        jwk = '{"kty":"OKP","crv":"Ed25519","x":"test","d":"secret"}'
        issuer = ProvenanceIssuer.from_jwk(jwk)
        assert issuer.did.startswith("did:key:")

    def test_from_jwk_dict(self):
        jwk = {"kty": "OKP", "crv": "Ed25519", "x": "test", "d": "secret"}
        issuer = ProvenanceIssuer.from_jwk(jwk)
        assert issuer.did.startswith("did:key:")

    def test_from_jwk_custom_method(self):
        jwk = '{"kty":"OKP"}'
        issuer = ProvenanceIssuer.from_jwk(jwk, did_method=DIDMethod.WEB)
        assert "web" in issuer.did

    @pytest.mark.asyncio
    async def test_issue_credential(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await issuer.issue_credential(
            subject_data={"price": 45000, "asset": "BTC/USD"},
            oracle_id="chainlink",
            credential_type=CredentialType.ORACLE_PRICE_ATTESTATION,
        )
        assert isinstance(td, TrustedData)
        assert td.oracle_id == "chainlink"
        assert td.data["price"] == 45000
        assert td.issuer_did == "did:key:test"
        assert td.credential_type == CredentialType.ORACLE_PRICE_ATTESTATION

    @pytest.mark.asyncio
    async def test_issue_credential_with_subject_id(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await issuer.issue_credential(
            subject_data={"data": "value"},
            oracle_id="test",
            subject_id="did:key:subject",
        )
        assert td.credential["credentialSubject"]["id"] == "did:key:subject"

    @pytest.mark.asyncio
    async def test_issue_credential_no_expiration(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await issuer.issue_credential(
            subject_data={"data": "value"},
            oracle_id="test",
            expiration_hours=None,
        )
        assert td.expiration_date is None
        assert "expirationDate" not in td.credential

    @pytest.mark.asyncio
    async def test_issue_credential_with_status(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        status = CredentialStatus(id="status_1")
        td = await issuer.issue_credential(
            subject_data={"data": "value"},
            oracle_id="test",
            credential_status=status,
        )
        assert "credentialStatus" in td.credential

    @pytest.mark.asyncio
    async def test_issue_credential_additional_context(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await issuer.issue_credential(
            subject_data={"data": "value"},
            oracle_id="test",
            additional_context=["https://example.com/v1"],
        )
        assert "https://example.com/v1" in td.credential["@context"]

    @pytest.mark.asyncio
    async def test_issue_presentation(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await issuer.issue_credential(
            subject_data={"data": "v1"},
            oracle_id="test1",
        )
        presentation = await issuer.issue_presentation(
            credentials=[td],
            holder_did="did:key:holder",
            challenge="challenge123",
            domain="example.com",
        )
        assert "verifiableCredential" in presentation
        assert len(presentation["verifiableCredential"]) == 1
        assert presentation["holder"] == "did:key:holder"

    @pytest.mark.asyncio
    async def test_issue_presentation_no_holder(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await issuer.issue_credential(
            subject_data={"data": "v1"},
            oracle_id="test1",
        )
        presentation = await issuer.issue_presentation(
            credentials=[td],
        )
        assert "holder" not in presentation

    @pytest.mark.asyncio
    async def test_generate_with_cryptography(self):
        try:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            issuer = await ProvenanceIssuer.generate()
            assert issuer.did is not None
            assert issuer.did.startswith("did:key:")
        except ImportError:
            pytest.skip("cryptography not installed")

    @pytest.mark.asyncio
    async def test_generate_no_libs(self):
        with patch("aethelred.oracles.provenance.DIDKIT_AVAILABLE", False):
            with patch("aethelred.oracles.provenance.CRYPTOGRAPHY_AVAILABLE", False):
                with pytest.raises(RuntimeError, match="Neither didkit nor cryptography"):
                    await ProvenanceIssuer.generate()


class TestProvenanceVerifier:
    def test_init_defaults(self):
        verifier = ProvenanceVerifier()
        assert verifier.trusted_issuers is None
        assert verifier.allow_expired is False
        assert verifier.max_age_seconds is None

    def test_init_with_trusted_issuers(self):
        verifier = ProvenanceVerifier(trusted_issuers=["did:key:a", "did:key:b"])
        assert verifier.trusted_issuers == {"did:key:a", "did:key:b"}

    @pytest.mark.asyncio
    async def test_verify_credential_dict(self):
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {"data": "value"},
            "issuanceDate": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "proof": {"type": "Ed25519Signature2020"},
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_credential(cred)
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_credential_string(self):
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "issuanceDate": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "proof": {"type": "test"},
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_credential(json.dumps(cred))
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_credential_trusted_data(self):
        now = datetime.now(timezone.utc)
        td = TrustedData(
            credential={
                "issuer": "did:key:abc",
                "credentialSubject": {},
                "issuanceDate": now.isoformat().replace("+00:00", "Z"),
                "proof": {"type": "test"},
            },
            data={},
            issuer_did="did:key:abc",
            issuance_date=now,
        )
        verifier = ProvenanceVerifier()
        result = await verifier.verify_credential(td)
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_credential_untrusted_issuer(self):
        cred = {
            "issuer": "did:key:untrusted",
            "credentialSubject": {},
            "issuanceDate": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "proof": {"type": "test"},
        }
        verifier = ProvenanceVerifier(trusted_issuers=["did:key:trusted"])
        result = await verifier.verify_credential(cred)
        assert result.valid is False
        assert any("not in trusted list" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_verify_credential_expired(self):
        past = datetime.now(timezone.utc) - timedelta(hours=24)
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "issuanceDate": past.isoformat().replace("+00:00", "Z"),
            "expirationDate": (past + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "proof": {"type": "test"},
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_credential(cred)
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_verify_credential_expired_allowed(self):
        past = datetime.now(timezone.utc) - timedelta(hours=24)
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "issuanceDate": past.isoformat().replace("+00:00", "Z"),
            "expirationDate": (past + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "proof": {"type": "test"},
        }
        verifier = ProvenanceVerifier(allow_expired=True)
        result = await verifier.verify_credential(cred)
        assert any("allow_expired" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_verify_credential_max_age(self):
        old = datetime.now(timezone.utc) - timedelta(days=30)
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "issuanceDate": old.isoformat().replace("+00:00", "Z"),
            "proof": {"type": "test"},
        }
        verifier = ProvenanceVerifier(max_age_seconds=3600)
        result = await verifier.verify_credential(cred)
        assert result.valid is False
        assert any("too old" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_verify_credential_no_proof(self):
        cred = {
            "issuer": "did:key:abc",
            "credentialSubject": {},
            "issuanceDate": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_credential(cred)
        assert result.valid is False

    @pytest.mark.asyncio
    async def test_verify_presentation(self):
        now = datetime.now(timezone.utc)
        presentation = {
            "holder": "did:key:holder",
            "verifiableCredential": [
                {
                    "issuer": "did:key:abc",
                    "credentialSubject": {},
                    "issuanceDate": now.isoformat().replace("+00:00", "Z"),
                    "proof": {"type": "test"},
                }
            ],
            "proof": {
                "type": "Ed25519Signature2020",
                "challenge": "ch1",
                "domain": "example.com",
            },
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_presentation(
            presentation, challenge="ch1", domain="example.com"
        )
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_presentation_string(self):
        now = datetime.now(timezone.utc)
        presentation = json.dumps({
            "holder": "did:key:holder",
            "verifiableCredential": [],
            "proof": {"type": "test"},
        })
        verifier = ProvenanceVerifier()
        result = await verifier.verify_presentation(presentation)
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_presentation_challenge_mismatch(self):
        presentation = {
            "holder": "did:key:holder",
            "verifiableCredential": [],
            "proof": {
                "type": "test",
                "challenge": "wrong",
                "domain": "example.com",
            },
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_presentation(
            presentation, challenge="expected", domain="example.com"
        )
        assert result.valid is False
        assert any("Challenge mismatch" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_verify_presentation_domain_mismatch(self):
        presentation = {
            "holder": "did:key:holder",
            "verifiableCredential": [],
            "proof": {
                "type": "test",
                "challenge": "ch1",
                "domain": "wrong.com",
            },
        }
        verifier = ProvenanceVerifier()
        result = await verifier.verify_presentation(
            presentation, challenge="ch1", domain="example.com"
        )
        assert result.valid is False
        assert any("Domain mismatch" in e for e in result.errors)


class TestOracleProvenanceRegistry:
    def test_init(self):
        registry = OracleProvenanceRegistry()
        assert registry.list_oracles() == []

    def test_register_oracle(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle(
            "chainlink",
            "did:key:chainlink",
            name="Chainlink",
            description="Price feeds",
            supported_feeds=["BTC/USD", "ETH/USD"],
        )
        assert registry.is_registered("chainlink")
        info = registry.get_oracle_info("chainlink")
        assert info["name"] == "Chainlink"
        assert info["supported_feeds"] == ["BTC/USD", "ETH/USD"]

    def test_register_oracle_defaults(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("test", "did:key:test")
        info = registry.get_oracle_info("test")
        assert info["name"] == "test"
        assert info["supported_feeds"] == []

    def test_is_registered(self):
        registry = OracleProvenanceRegistry()
        assert registry.is_registered("chainlink") is False
        registry.register_oracle("chainlink", "did:key:cl")
        assert registry.is_registered("chainlink") is True

    def test_get_oracle_info_not_registered(self):
        registry = OracleProvenanceRegistry()
        assert registry.get_oracle_info("nonexistent") is None

    def test_list_oracles(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("a", "did:key:a")
        registry.register_oracle("b", "did:key:b")
        oracles = registry.list_oracles()
        assert len(oracles) == 2

    def test_get_trusted_issuers_all(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("a", "did:key:a")
        registry.register_oracle("b", "did:key:b")
        issuers = registry.get_trusted_issuers()
        assert set(issuers) == {"did:key:a", "did:key:b"}

    def test_get_trusted_issuers_filtered(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("a", "did:key:a")
        registry.register_oracle("b", "did:key:b")
        issuers = registry.get_trusted_issuers(["a"])
        assert issuers == ["did:key:a"]

    def test_get_trusted_issuers_missing(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("a", "did:key:a")
        issuers = registry.get_trusted_issuers(["nonexistent"])
        assert issuers == []

    def test_get_verifier_all(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("a", "did:key:a")
        verifier = registry.get_verifier()
        assert isinstance(verifier, ProvenanceVerifier)

    def test_get_verifier_filtered(self):
        registry = OracleProvenanceRegistry()
        registry.register_oracle("a", "did:key:a")
        verifier = registry.get_verifier(["a"])
        assert verifier.trusted_issuers == {"did:key:a"}

    def test_get_verifier_empty(self):
        registry = OracleProvenanceRegistry()
        verifier = registry.get_verifier()
        assert verifier.trusted_issuers is None


class TestUtilityFunctions:
    @pytest.mark.asyncio
    async def test_wrap_oracle_data(self):
        issuer = ProvenanceIssuer(
            did="did:key:test",
            private_key_jwk='{"kty":"OKP"}',
        )
        td = await wrap_oracle_data(
            data={"price": 100},
            oracle_id="test",
            issuer=issuer,
        )
        assert isinstance(td, TrustedData)
        assert td.data["price"] == 100

    def test_extract_data_if_valid_verified(self):
        td = TrustedData(
            credential={}, data={"price": 100},
            issuer_did="did:key:x",
            issuance_date=datetime.now(timezone.utc),
        )
        td._verified = True
        result = extract_data_if_valid(td)
        assert result == {"price": 100}

    def test_extract_data_if_valid_not_verified(self):
        td = TrustedData(
            credential={}, data={"price": 100},
            issuer_did="did:key:x",
            issuance_date=datetime.now(timezone.utc),
        )
        result = extract_data_if_valid(td)
        assert result is None

    def test_extract_data_if_valid_failed(self):
        td = TrustedData(
            credential={}, data={"price": 100},
            issuer_did="did:key:x",
            issuance_date=datetime.now(timezone.utc),
        )
        td._verified = False
        result = extract_data_if_valid(td)
        assert result is None
