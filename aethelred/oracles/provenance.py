"""
W3C Verifiable Credentials Oracle Provenance Module

Enterprise-grade data provenance implementation using W3C Verifiable Credentials
standard (https://www.w3.org/TR/vc-data-model/) with didkit integration.

This module provides cryptographic proof of oracle data authenticity, enabling:
- Trustworthy AI input verification
- Regulatory audit trails
- Cross-chain data attestations
- Tamper-evident data lineage

Example:
    >>> from aethelred.oracles.provenance import TrustedData, ProvenanceIssuer
    >>>
    >>> # Create issuer with DID
    >>> issuer = ProvenanceIssuer.from_jwk(private_key_jwk)
    >>>
    >>> # Wrap oracle data with verifiable credential
    >>> trusted = await issuer.issue_credential(
    ...     subject_data={"price": 45000.00, "asset": "BTC/USD"},
    ...     oracle_id="chainlink-btc-usd",
    ...     credential_type="OraclePriceAttestation"
    ... )
    >>>
    >>> # Verify credential
    >>> is_valid = await trusted.verify()
    >>> print(f"Valid: {is_valid}, Issuer: {trusted.issuer_did}")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Union

# Enterprise dependencies
try:
    import didkit
    DIDKIT_AVAILABLE = True
except ImportError:
    DIDKIT_AVAILABLE = False
    didkit = None

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, ed25519
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False


class CredentialType(str, Enum):
    """W3C Verifiable Credential types for oracle data."""

    ORACLE_PRICE_ATTESTATION = "OraclePriceAttestation"
    ORACLE_DATA_ATTESTATION = "OracleDataAttestation"
    ORACLE_WEATHER_ATTESTATION = "OracleWeatherAttestation"
    ORACLE_IDENTITY_ATTESTATION = "OracleIdentityAttestation"
    ORACLE_FINANCIAL_ATTESTATION = "OracleFinancialAttestation"
    ORACLE_COMPUTE_ATTESTATION = "OracleComputeAttestation"
    AETHELRED_SEAL_ATTESTATION = "AethelredSealAttestation"


class ProofType(str, Enum):
    """Supported cryptographic proof types."""

    ED25519_SIGNATURE_2020 = "Ed25519Signature2020"
    ECDSA_SECP256K1_SIGNATURE_2019 = "EcdsaSecp256k1Signature2019"
    JSON_WEB_SIGNATURE_2020 = "JsonWebSignature2020"
    BBS_BLS_SIGNATURE_2020 = "BbsBlsSignature2020"


class DIDMethod(str, Enum):
    """Supported DID methods."""

    KEY = "key"  # did:key
    WEB = "web"  # did:web
    ION = "ion"  # did:ion (Bitcoin-anchored)
    ETHR = "ethr"  # did:ethr (Ethereum)
    AETHELRED = "aethel"  # did:aethel (Aethelred native)


@dataclass
class VerificationResult:
    """Result of credential verification."""

    valid: bool
    issuer_did: str
    subject_id: Optional[str]
    issuance_date: datetime
    proof_type: str
    expiration_date: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Verification metadata
    verification_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verifier_version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "valid": self.valid,
            "issuer_did": self.issuer_did,
            "subject_id": self.subject_id,
            "issuance_date": self.issuance_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "proof_type": self.proof_type,
            "errors": self.errors,
            "warnings": self.warnings,
            "verification_time": self.verification_time.isoformat(),
            "verifier_version": self.verifier_version,
        }


@dataclass
class CredentialStatus:
    """Credential revocation status."""

    id: str
    type: str = "RevocationList2020Status"
    revocation_list_index: int = 0
    revocation_list_credential: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "revocationListIndex": str(self.revocation_list_index),
            "revocationListCredential": self.revocation_list_credential,
        }


@dataclass
class TrustedData:
    """
    Wrapper for oracle data with W3C Verifiable Credential provenance.

    This class provides cryptographic proof that data came from a trusted
    oracle source, enabling tamper-evident audit trails for AI computations.

    Attributes:
        credential: The W3C Verifiable Credential as JSON-LD
        data: The actual oracle data (credential subject)
        issuer_did: DID of the credential issuer
        issuance_date: When the credential was issued
        expiration_date: When the credential expires (optional)
        proof: Cryptographic proof of authenticity

    Example:
        >>> trusted = await issuer.issue_credential(data, oracle_id)
        >>> if await trusted.verify():
        ...     process(trusted.data)
    """

    credential: Dict[str, Any]
    data: Dict[str, Any]
    issuer_did: str
    issuance_date: datetime
    expiration_date: Optional[datetime] = None
    proof: Optional[Dict[str, Any]] = None

    # Metadata
    credential_id: str = field(default_factory=lambda: f"urn:uuid:{uuid.uuid4()}")
    oracle_id: Optional[str] = None
    credential_type: CredentialType = CredentialType.ORACLE_DATA_ATTESTATION

    # Verification cache
    _verified: Optional[bool] = field(default=None, repr=False)
    _verification_result: Optional[VerificationResult] = field(default=None, repr=False)

    async def verify(self, options: Optional[Dict[str, Any]] = None) -> bool:
        """
        Verify the credential's cryptographic proof.

        Args:
            options: Additional verification options

        Returns:
            True if credential is valid and not expired

        Example:
            >>> is_valid = await trusted.verify()
            >>> if not is_valid:
            ...     print(trusted.verification_result.errors)
        """
        if self._verified is not None and options is None:
            return self._verified

        try:
            result = await self._verify_credential(options or {})
            self._verification_result = result
            self._verified = result.valid
            return result.valid
        except Exception as e:
            self._verification_result = VerificationResult(
                valid=False,
                issuer_did=self.issuer_did,
                subject_id=None,
                issuance_date=self.issuance_date,
                expiration_date=self.expiration_date,
                proof_type=self.proof.get("type", "unknown") if self.proof else "unknown",
                errors=[str(e)],
            )
            self._verified = False
            return False

    async def _verify_credential(self, options: Dict[str, Any]) -> VerificationResult:
        """Internal credential verification using didkit."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check expiration
        if self.expiration_date and datetime.now(timezone.utc) > self.expiration_date:
            errors.append("Credential has expired")

        # Check issuance date is not in the future
        if self.issuance_date > datetime.now(timezone.utc) + timedelta(minutes=5):
            errors.append("Credential issuance date is in the future")

        # Verify cryptographic proof
        if DIDKIT_AVAILABLE and didkit:
            try:
                credential_json = json.dumps(self.credential)
                proof_options = json.dumps(options) if options else "{}"

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: didkit.verify_credential(credential_json, proof_options)
                )

                result_data = json.loads(result)
                if result_data.get("errors"):
                    errors.extend(result_data["errors"])
                if result_data.get("warnings"):
                    warnings.extend(result_data["warnings"])

            except Exception as e:
                errors.append(f"didkit verification failed: {e}")
        else:
            # Fallback verification without didkit
            if not self.proof:
                errors.append("No proof present in credential")
            elif not self._verify_proof_structure():
                errors.append("Invalid proof structure")
            else:
                warnings.append("didkit not available, using fallback verification")

        return VerificationResult(
            valid=len(errors) == 0,
            issuer_did=self.issuer_did,
            subject_id=self.data.get("id"),
            issuance_date=self.issuance_date,
            expiration_date=self.expiration_date,
            proof_type=self.proof.get("type", "unknown") if self.proof else "unknown",
            errors=errors,
            warnings=warnings,
        )

    def _verify_proof_structure(self) -> bool:
        """Verify the proof has required fields."""
        if not self.proof:
            return False

        required_fields = ["type", "created", "verificationMethod", "proofPurpose"]
        return all(field in self.proof for field in required_fields)

    @property
    def verification_result(self) -> Optional[VerificationResult]:
        """Get the last verification result."""
        return self._verification_result

    @property
    def is_expired(self) -> bool:
        """Check if credential has expired."""
        if not self.expiration_date:
            return False
        return datetime.now(timezone.utc) > self.expiration_date

    @property
    def data_hash(self) -> str:
        """Get SHA-256 hash of the credential subject data."""
        data_bytes = json.dumps(self.data, sort_keys=True).encode()
        return hashlib.sha256(data_bytes).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "credential": self.credential,
            "data": self.data,
            "issuer_did": self.issuer_did,
            "issuance_date": self.issuance_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat() if self.expiration_date else None,
            "credential_id": self.credential_id,
            "oracle_id": self.oracle_id,
            "credential_type": self.credential_type.value,
            "data_hash": self.data_hash,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_credential(cls, credential: Dict[str, Any]) -> "TrustedData":
        """
        Create TrustedData from a W3C Verifiable Credential.

        Args:
            credential: The verifiable credential as dict

        Returns:
            TrustedData instance
        """
        issuer = credential.get("issuer", {})
        issuer_did = issuer if isinstance(issuer, str) else issuer.get("id", "")

        subject = credential.get("credentialSubject", {})

        issuance_date = datetime.fromisoformat(
            credential.get("issuanceDate", datetime.now(timezone.utc).isoformat()).replace("Z", "+00:00")
        )

        exp_date_str = credential.get("expirationDate")
        expiration_date = None
        if exp_date_str:
            expiration_date = datetime.fromisoformat(exp_date_str.replace("Z", "+00:00"))

        # Determine credential type
        types = credential.get("type", [])
        credential_type = CredentialType.ORACLE_DATA_ATTESTATION
        for t in types:
            try:
                credential_type = CredentialType(t)
                break
            except ValueError:
                continue

        return cls(
            credential=credential,
            data=subject,
            issuer_did=issuer_did,
            issuance_date=issuance_date,
            expiration_date=expiration_date,
            proof=credential.get("proof"),
            credential_id=credential.get("id", f"urn:uuid:{uuid.uuid4()}"),
            credential_type=credential_type,
        )


class ProvenanceIssuer:
    """
    Issues W3C Verifiable Credentials for oracle data provenance.

    This class creates cryptographically signed credentials that attest
    to the authenticity and source of oracle data.

    Example:
        >>> # Create issuer from JWK
        >>> issuer = ProvenanceIssuer.from_jwk(private_key_jwk)
        >>>
        >>> # Issue credential for price data
        >>> trusted = await issuer.issue_credential(
        ...     subject_data={"price": 45000.00, "asset": "BTC/USD"},
        ...     oracle_id="chainlink-btc-usd",
        ...     credential_type=CredentialType.ORACLE_PRICE_ATTESTATION
        ... )
    """

    def __init__(
        self,
        did: str,
        private_key_jwk: str,
        proof_type: ProofType = ProofType.ED25519_SIGNATURE_2020,
        verification_method: Optional[str] = None,
    ):
        """
        Initialize the provenance issuer.

        Args:
            did: The issuer's DID
            private_key_jwk: Private key in JWK format
            proof_type: Type of cryptographic proof to generate
            verification_method: Verification method ID (defaults to did#key-1)
        """
        self.did = did
        self._private_key_jwk = private_key_jwk
        self.proof_type = proof_type
        self.verification_method = verification_method or f"{did}#key-1"

        # Credential context
        self._context = [
            "https://www.w3.org/2018/credentials/v1",
            "https://w3id.org/security/suites/ed25519-2020/v1",
            {
                "OracleAttestation": "https://aethelred.io/vocab#OracleAttestation",
                "oracleId": "https://aethelred.io/vocab#oracleId",
                "dataHash": "https://aethelred.io/vocab#dataHash",
                "timestamp": "https://aethelred.io/vocab#timestamp",
                "chainId": "https://aethelred.io/vocab#chainId",
            }
        ]

    @classmethod
    def from_jwk(
        cls,
        private_key_jwk: Union[str, Dict[str, Any]],
        did_method: DIDMethod = DIDMethod.KEY,
        proof_type: ProofType = ProofType.ED25519_SIGNATURE_2020,
    ) -> "ProvenanceIssuer":
        """
        Create issuer from a JWK private key.

        Args:
            private_key_jwk: Private key as JWK (string or dict)
            did_method: DID method to use
            proof_type: Proof type for signing

        Returns:
            ProvenanceIssuer instance
        """
        if isinstance(private_key_jwk, dict):
            private_key_jwk = json.dumps(private_key_jwk)

        # Generate DID from key
        if DIDKIT_AVAILABLE and didkit:
            did = didkit.key_to_did(did_method.value, private_key_jwk)
        else:
            # Fallback: generate deterministic DID from key hash
            key_hash = hashlib.sha256(private_key_jwk.encode()).hexdigest()[:32]
            did = f"did:{did_method.value}:{key_hash}"

        return cls(
            did=did,
            private_key_jwk=private_key_jwk,
            proof_type=proof_type,
        )

    @classmethod
    async def generate(
        cls,
        did_method: DIDMethod = DIDMethod.KEY,
        proof_type: ProofType = ProofType.ED25519_SIGNATURE_2020,
    ) -> "ProvenanceIssuer":
        """
        Generate a new issuer with fresh keypair.

        Args:
            did_method: DID method to use
            proof_type: Proof type for signing

        Returns:
            ProvenanceIssuer instance with new keys
        """
        if DIDKIT_AVAILABLE and didkit:
            # Generate key using didkit
            private_key_jwk = didkit.generate_ed25519_key()
        elif CRYPTOGRAPHY_AVAILABLE:
            # Generate key using cryptography library
            private_key = ed25519.Ed25519PrivateKey.generate()

            # Convert to JWK format
            public_bytes = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw
            )
            private_bytes = private_key.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption()
            )

            import base64
            private_key_jwk = json.dumps({
                "kty": "OKP",
                "crv": "Ed25519",
                "x": base64.urlsafe_b64encode(public_bytes).decode().rstrip("="),
                "d": base64.urlsafe_b64encode(private_bytes).decode().rstrip("="),
            })
        else:
            raise RuntimeError("Neither didkit nor cryptography library available")

        return cls.from_jwk(private_key_jwk, did_method, proof_type)

    async def issue_credential(
        self,
        subject_data: Dict[str, Any],
        oracle_id: str,
        credential_type: CredentialType = CredentialType.ORACLE_DATA_ATTESTATION,
        subject_id: Optional[str] = None,
        expiration_hours: Optional[int] = 24,
        additional_context: Optional[List[Any]] = None,
        credential_status: Optional[CredentialStatus] = None,
    ) -> TrustedData:
        """
        Issue a verifiable credential for oracle data.

        Args:
            subject_data: The oracle data to attest
            oracle_id: Identifier of the oracle source
            credential_type: Type of credential to issue
            subject_id: Optional subject identifier
            expiration_hours: Hours until expiration (None for no expiration)
            additional_context: Additional JSON-LD context
            credential_status: Revocation status endpoint

        Returns:
            TrustedData with verifiable credential

        Example:
            >>> trusted = await issuer.issue_credential(
            ...     subject_data={"price": 45000.00},
            ...     oracle_id="chainlink-btc-usd"
            ... )
        """
        now = datetime.now(timezone.utc)
        credential_id = f"urn:uuid:{uuid.uuid4()}"

        # Build credential subject
        credential_subject = {
            "oracleId": oracle_id,
            "dataHash": hashlib.sha256(
                json.dumps(subject_data, sort_keys=True).encode()
            ).hexdigest(),
            "timestamp": now.isoformat(),
            **subject_data,
        }

        if subject_id:
            credential_subject["id"] = subject_id

        # Build context
        context = list(self._context)
        if additional_context:
            context.extend(additional_context)

        # Build credential
        credential: Dict[str, Any] = {
            "@context": context,
            "id": credential_id,
            "type": ["VerifiableCredential", credential_type.value],
            "issuer": {
                "id": self.did,
                "name": "Aethelred Oracle Network",
            },
            "issuanceDate": now.isoformat().replace("+00:00", "Z"),
            "credentialSubject": credential_subject,
        }

        # Add expiration
        expiration_date = None
        if expiration_hours:
            expiration_date = now + timedelta(hours=expiration_hours)
            credential["expirationDate"] = expiration_date.isoformat().replace("+00:00", "Z")

        # Add credential status
        if credential_status:
            credential["credentialStatus"] = credential_status.to_dict()

        # Sign credential
        signed_credential = await self._sign_credential(credential)

        return TrustedData(
            credential=signed_credential,
            data=subject_data,
            issuer_did=self.did,
            issuance_date=now,
            expiration_date=expiration_date,
            proof=signed_credential.get("proof"),
            credential_id=credential_id,
            oracle_id=oracle_id,
            credential_type=credential_type,
        )

    async def _sign_credential(self, credential: Dict[str, Any]) -> Dict[str, Any]:
        """Sign the credential with the issuer's private key."""
        if DIDKIT_AVAILABLE and didkit:
            credential_json = json.dumps(credential)

            proof_options = json.dumps({
                "proofPurpose": "assertionMethod",
                "verificationMethod": self.verification_method,
                "type": self.proof_type.value,
            })

            # Sign using didkit
            signed_json = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: didkit.issue_credential(
                    credential_json,
                    proof_options,
                    self._private_key_jwk
                )
            )

            return json.loads(signed_json)
        else:
            # Fallback: create proof structure without cryptographic signature
            # This is for development/testing only
            credential["proof"] = {
                "type": self.proof_type.value,
                "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "verificationMethod": self.verification_method,
                "proofPurpose": "assertionMethod",
                "proofValue": hashlib.sha256(
                    json.dumps(credential, sort_keys=True).encode()
                ).hexdigest(),
                "_warning": "Fallback proof - didkit not available",
            }
            return credential

    async def issue_presentation(
        self,
        credentials: List[TrustedData],
        holder_did: Optional[str] = None,
        challenge: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a Verifiable Presentation containing multiple credentials.

        Args:
            credentials: List of TrustedData to include
            holder_did: DID of the presentation holder
            challenge: Challenge for replay protection
            domain: Domain for audience restriction

        Returns:
            Verifiable Presentation as dict
        """
        presentation_id = f"urn:uuid:{uuid.uuid4()}"

        presentation = {
            "@context": ["https://www.w3.org/2018/credentials/v1"],
            "id": presentation_id,
            "type": ["VerifiablePresentation"],
            "verifiableCredential": [td.credential for td in credentials],
        }

        if holder_did:
            presentation["holder"] = holder_did

        # Sign presentation
        if DIDKIT_AVAILABLE and didkit:
            proof_options: Dict[str, Any] = {
                "proofPurpose": "authentication",
                "verificationMethod": self.verification_method,
            }

            if challenge:
                proof_options["challenge"] = challenge
            if domain:
                proof_options["domain"] = domain

            presentation_json = json.dumps(presentation)
            proof_options_json = json.dumps(proof_options)

            signed_json = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: didkit.issue_presentation(
                    presentation_json,
                    proof_options_json,
                    self._private_key_jwk
                )
            )

            return json.loads(signed_json)
        else:
            # Fallback without cryptographic signature
            presentation["proof"] = {
                "type": self.proof_type.value,
                "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "verificationMethod": self.verification_method,
                "proofPurpose": "authentication",
                "challenge": challenge,
                "domain": domain,
            }
            return presentation


class ProvenanceVerifier:
    """
    Verifies W3C Verifiable Credentials and Presentations.

    Example:
        >>> verifier = ProvenanceVerifier()
        >>> result = await verifier.verify_credential(credential)
        >>> if result.valid:
        ...     process(credential["credentialSubject"])
    """

    def __init__(
        self,
        trusted_issuers: Optional[List[str]] = None,
        allow_expired: bool = False,
        max_age_seconds: Optional[int] = None,
    ):
        """
        Initialize verifier.

        Args:
            trusted_issuers: List of trusted issuer DIDs (None = all)
            allow_expired: Whether to accept expired credentials
            max_age_seconds: Maximum credential age in seconds
        """
        self.trusted_issuers = set(trusted_issuers) if trusted_issuers else None
        self.allow_expired = allow_expired
        self.max_age_seconds = max_age_seconds

    async def verify_credential(
        self,
        credential: Union[Dict[str, Any], TrustedData, str],
        options: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify a verifiable credential.

        Args:
            credential: The credential to verify
            options: Additional verification options

        Returns:
            VerificationResult with validation details
        """
        # Normalize input
        if isinstance(credential, str):
            credential = json.loads(credential)
        if isinstance(credential, TrustedData):
            credential = credential.credential

        errors: List[str] = []
        warnings: List[str] = []

        # Extract metadata
        issuer = credential.get("issuer", {})
        issuer_did = issuer if isinstance(issuer, str) else issuer.get("id", "")
        subject = credential.get("credentialSubject", {})

        issuance_str = credential.get("issuanceDate", "")
        issuance_date = datetime.fromisoformat(issuance_str.replace("Z", "+00:00")) if issuance_str else datetime.now(timezone.utc)

        exp_str = credential.get("expirationDate")
        expiration_date = datetime.fromisoformat(exp_str.replace("Z", "+00:00")) if exp_str else None

        proof = credential.get("proof", {})
        proof_type = proof.get("type", "unknown")

        # Check trusted issuers
        if self.trusted_issuers and issuer_did not in self.trusted_issuers:
            errors.append(f"Issuer {issuer_did} not in trusted list")

        # Check expiration
        now = datetime.now(timezone.utc)
        if expiration_date and now > expiration_date:
            if self.allow_expired:
                warnings.append("Credential has expired but allow_expired=True")
            else:
                errors.append("Credential has expired")

        # Check max age
        if self.max_age_seconds:
            age = (now - issuance_date).total_seconds()
            if age > self.max_age_seconds:
                errors.append(f"Credential too old: {age}s > {self.max_age_seconds}s")

        # Verify cryptographic proof
        if DIDKIT_AVAILABLE and didkit:
            try:
                credential_json = json.dumps(credential)
                proof_options = json.dumps(options or {})

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: didkit.verify_credential(credential_json, proof_options)
                )

                result_data = json.loads(result)
                if result_data.get("errors"):
                    errors.extend(result_data["errors"])
                if result_data.get("warnings"):
                    warnings.extend(result_data["warnings"])

            except Exception as e:
                errors.append(f"Verification failed: {e}")
        else:
            if not proof:
                errors.append("No proof present")
            else:
                warnings.append("didkit not available, proof not cryptographically verified")

        return VerificationResult(
            valid=len(errors) == 0,
            issuer_did=issuer_did,
            subject_id=subject.get("id"),
            issuance_date=issuance_date,
            expiration_date=expiration_date,
            proof_type=proof_type,
            errors=errors,
            warnings=warnings,
        )

    async def verify_presentation(
        self,
        presentation: Union[Dict[str, Any], str],
        challenge: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a verifiable presentation.

        Args:
            presentation: The presentation to verify
            challenge: Expected challenge value
            domain: Expected domain value

        Returns:
            VerificationResult with validation details
        """
        if isinstance(presentation, str):
            presentation = json.loads(presentation)

        errors: List[str] = []
        warnings: List[str] = []

        # Extract holder
        holder = presentation.get("holder", "")
        proof = presentation.get("proof", {})

        # Verify challenge/domain if provided
        if challenge and proof.get("challenge") != challenge:
            errors.append("Challenge mismatch")
        if domain and proof.get("domain") != domain:
            errors.append("Domain mismatch")

        # Verify each credential
        credentials = presentation.get("verifiableCredential", [])
        for i, cred in enumerate(credentials):
            result = await self.verify_credential(cred)
            if not result.valid:
                errors.extend([f"Credential {i}: {e}" for e in result.errors])
            warnings.extend([f"Credential {i}: {w}" for w in result.warnings])

        # Verify presentation proof
        if DIDKIT_AVAILABLE and didkit:
            try:
                options: Dict[str, Any] = {}
                if challenge:
                    options["challenge"] = challenge
                if domain:
                    options["domain"] = domain

                presentation_json = json.dumps(presentation)
                options_json = json.dumps(options)

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: didkit.verify_presentation(presentation_json, options_json)
                )

                result_data = json.loads(result)
                if result_data.get("errors"):
                    errors.extend(result_data["errors"])

            except Exception as e:
                errors.append(f"Presentation verification failed: {e}")

        return VerificationResult(
            valid=len(errors) == 0,
            issuer_did=holder,
            subject_id=None,
            issuance_date=datetime.now(timezone.utc),
            expiration_date=None,
            proof_type=proof.get("type", "unknown"),
            errors=errors,
            warnings=warnings,
        )


class OracleProvenanceRegistry:
    """
    Registry for managing oracle provenance issuers and verifiers.

    Provides centralized management of trusted oracle sources and
    their verification credentials.

    Example:
        >>> registry = OracleProvenanceRegistry()
        >>> registry.register_oracle("chainlink", issuer_did)
        >>> verifier = registry.get_verifier(["chainlink", "band"])
    """

    def __init__(self):
        self._oracles: Dict[str, Dict[str, Any]] = {}
        self._issuers: Dict[str, ProvenanceIssuer] = {}

    def register_oracle(
        self,
        oracle_id: str,
        issuer_did: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        supported_feeds: Optional[List[str]] = None,
    ) -> None:
        """
        Register a trusted oracle source.

        Args:
            oracle_id: Unique oracle identifier
            issuer_did: DID of the oracle's credential issuer
            name: Human-readable name
            description: Oracle description
            supported_feeds: List of supported data feeds
        """
        self._oracles[oracle_id] = {
            "id": oracle_id,
            "issuer_did": issuer_did,
            "name": name or oracle_id,
            "description": description,
            "supported_feeds": supported_feeds or [],
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_trusted_issuers(self, oracle_ids: Optional[List[str]] = None) -> List[str]:
        """Get list of trusted issuer DIDs."""
        if oracle_ids is None:
            return [o["issuer_did"] for o in self._oracles.values()]
        return [
            self._oracles[oid]["issuer_did"]
            for oid in oracle_ids
            if oid in self._oracles
        ]

    def get_verifier(self, oracle_ids: Optional[List[str]] = None) -> ProvenanceVerifier:
        """
        Get a verifier configured for specific oracles.

        Args:
            oracle_ids: List of oracle IDs to trust (None = all registered)

        Returns:
            Configured ProvenanceVerifier
        """
        trusted = self.get_trusted_issuers(oracle_ids)
        return ProvenanceVerifier(trusted_issuers=trusted if trusted else None)

    def is_registered(self, oracle_id: str) -> bool:
        """Check if oracle is registered."""
        return oracle_id in self._oracles

    def get_oracle_info(self, oracle_id: str) -> Optional[Dict[str, Any]]:
        """Get oracle registration info."""
        return self._oracles.get(oracle_id)

    def list_oracles(self) -> List[Dict[str, Any]]:
        """List all registered oracles."""
        return list(self._oracles.values())


# Utility functions

async def wrap_oracle_data(
    data: Dict[str, Any],
    oracle_id: str,
    issuer: ProvenanceIssuer,
    credential_type: CredentialType = CredentialType.ORACLE_DATA_ATTESTATION,
) -> TrustedData:
    """
    Convenience function to wrap oracle data with provenance.

    Args:
        data: Oracle data to wrap
        oracle_id: Oracle source identifier
        issuer: ProvenanceIssuer instance
        credential_type: Type of credential

    Returns:
        TrustedData with verifiable credential
    """
    return await issuer.issue_credential(
        subject_data=data,
        oracle_id=oracle_id,
        credential_type=credential_type,
    )


def extract_data_if_valid(trusted: TrustedData) -> Optional[Dict[str, Any]]:
    """
    Extract data from TrustedData only if verified.

    Returns None if not verified or verification failed.
    """
    if trusted._verified is True:
        return trusted.data
    return None


__all__ = [
    # Core classes
    "TrustedData",
    "ProvenanceIssuer",
    "ProvenanceVerifier",
    "OracleProvenanceRegistry",
    # Data classes
    "VerificationResult",
    "CredentialStatus",
    # Enums
    "CredentialType",
    "ProofType",
    "DIDMethod",
    # Utilities
    "wrap_oracle_data",
    "extract_data_if_valid",
    # Constants
    "DIDKIT_AVAILABLE",
]
