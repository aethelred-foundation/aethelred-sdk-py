"""
Aethelred Oracle Network Module

Interface to query Oracle-attested data feeds and reference
verified data in compute jobs with W3C Verifiable Credentials provenance.

Features:
- Query oracle data feeds
- Reference attested data by DID
- W3C Verifiable Credentials for data provenance
- Verify data authenticity with didkit
- Subscribe to oracle updates

Example:
    >>> from aethelred.oracles import OracleClient, DataFeed, TrustedData
    >>>
    >>> oracle = OracleClient(client)
    >>>
    >>> # Get a specific data feed
    >>> feed = await oracle.get_feed("market_data/btc_usd")
    >>> print(f"BTC/USD: {feed.value} (attested at {feed.timestamp})")
    >>>
    >>> # Wrap with W3C Verifiable Credentials
    >>> from aethelred.oracles.provenance import ProvenanceIssuer
    >>> issuer = await ProvenanceIssuer.generate()
    >>> trusted = await issuer.issue_credential(
    ...     subject_data={"price": feed.value},
    ...     oracle_id="chainlink-btc-usd"
    ... )
    >>> assert await trusted.verify()
"""

from aethelred.oracles.client import (
    OracleClient,
    AsyncOracleClient,
    DataFeed,
    FeedMetadata,
    OracleNode,
    AttestationRecord,
)

from aethelred.oracles.provenance import (
    TrustedData,
    ProvenanceIssuer,
    ProvenanceVerifier,
    OracleProvenanceRegistry,
    VerificationResult,
    CredentialStatus,
    CredentialType,
    ProofType as ProvProofType,
    DIDMethod,
    wrap_oracle_data,
    extract_data_if_valid,
)

__all__ = [
    # Client
    "OracleClient",
    "AsyncOracleClient",
    "DataFeed",
    "FeedMetadata",
    "OracleNode",
    "AttestationRecord",
    # Provenance (W3C Verifiable Credentials)
    "TrustedData",
    "ProvenanceIssuer",
    "ProvenanceVerifier",
    "OracleProvenanceRegistry",
    "VerificationResult",
    "CredentialStatus",
    "CredentialType",
    "ProvProofType",
    "DIDMethod",
    "wrap_oracle_data",
    "extract_data_if_valid",
]
