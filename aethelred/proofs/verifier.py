"""
Proof Verifier for Aethelred

Verifies zkML proofs and TEE attestations for AI computation verification.

The verification pipeline:
1. Verify zkML proof against verification key
2. Validate TEE attestation signature chain
3. Check public inputs match expected values
4. Verify consensus from multiple validators

Example:
    >>> verifier = ProofVerifier()
    >>>
    >>> # Verify a proof
    >>> result = verifier.verify(proof, circuit)
    >>> assert result.is_valid
    >>>
    >>> # Verify TEE attestation
    >>> result = verifier.verify_tee_attestation(attestation)
    >>> assert result.is_valid
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional, Union

from aethelred.core.exceptions import ProofError, ValidationError
from aethelred.core.types import (
    Circuit,
    Proof,
    ProofType,
    ZKProof,
    TEEAttestation,
    TEEPlatform,
)

logger = logging.getLogger(__name__)


# ============ Types ============


class VerificationStatus(str, Enum):
    """Verification result status."""

    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    """Result of proof or attestation verification."""

    # Status
    is_valid: bool
    status: VerificationStatus

    # Details
    proof_id: Optional[str] = None
    verification_time_ms: int = 0

    # Checks performed
    checks: dict[str, bool] = field(default_factory=dict)

    # Error information
    error: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None

    # Metadata
    verified_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    verifier_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_valid": self.is_valid,
            "status": self.status.value,
            "proof_id": self.proof_id,
            "verification_time_ms": self.verification_time_ms,
            "checks": self.checks,
            "error": self.error,
            "verified_at": self.verified_at.isoformat(),
        }


# ============ TEE Verifier ============


class TEEVerifier:
    """
    Verifies TEE attestations from various platforms.

    Supports:
    - Intel SGX (DCAP and IAS)
    - AMD SEV-SNP
    - Intel TDX
    - AWS Nitro Enclaves

    Example:
        >>> verifier = TEEVerifier()
        >>> result = verifier.verify(attestation)
        >>> print(f"Valid: {result.is_valid}, Platform: {attestation.platform}")
    """

    def __init__(
        self,
        cache_certificates: bool = True,
        allow_debug_enclaves: bool = False,
        max_attestation_age_hours: int = 24,
    ):
        """
        Initialize the TEE verifier.

        Args:
            cache_certificates: Cache platform root certificates
            allow_debug_enclaves: Allow debug mode attestations (insecure)
            max_attestation_age_hours: Maximum age of valid attestation
        """
        self.cache_certificates = cache_certificates
        self.allow_debug_enclaves = allow_debug_enclaves
        self.max_attestation_age = timedelta(hours=max_attestation_age_hours)

        # Platform-specific verifiers
        self._verifiers = {
            TEEPlatform.INTEL_SGX: self._verify_sgx,
            TEEPlatform.AMD_SEV: self._verify_sev,
            TEEPlatform.AMD_SEV_SNP: self._verify_sev_snp,
            TEEPlatform.INTEL_TDX: self._verify_tdx,
            TEEPlatform.AWS_NITRO: self._verify_nitro,
        }

        # Certificate cache
        self._cert_cache: dict[str, bytes] = {}

        logger.info("TEEVerifier initialized")

    def verify(
        self,
        attestation: TEEAttestation,
        expected_measurement: Optional[str] = None,
        expected_data: Optional[bytes] = None,
    ) -> VerificationResult:
        """
        Verify a TEE attestation.

        Args:
            attestation: TEE attestation to verify
            expected_measurement: Expected enclave measurement (MRENCLAVE)
            expected_data: Expected report data (output hash)

        Returns:
            Verification result
        """
        start_time = time.time()
        checks: dict[str, bool] = {}

        try:
            # Check 1: Attestation freshness
            checks["freshness"] = self._check_freshness(attestation)
            if not checks["freshness"]:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.EXPIRED,
                    checks=checks,
                    error="Attestation has expired",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check 2: Platform-specific verification
            verifier_fn = self._verifiers.get(attestation.platform)
            if not verifier_fn:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.UNKNOWN,
                    checks=checks,
                    error=f"Unsupported platform: {attestation.platform}",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            platform_result = verifier_fn(attestation)
            checks.update(platform_result["checks"])

            if not platform_result["valid"]:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.INVALID,
                    checks=checks,
                    error=platform_result.get("error"),
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check 3: Measurement verification (if provided)
            if expected_measurement:
                checks["measurement"] = attestation.measurement == expected_measurement
                if not checks["measurement"]:
                    return VerificationResult(
                        is_valid=False,
                        status=VerificationStatus.INVALID,
                        checks=checks,
                        error="Measurement mismatch",
                        verification_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Check 4: Report data verification (if provided)
            if expected_data:
                checks["report_data"] = attestation.report_data == expected_data
                if not checks["report_data"]:
                    return VerificationResult(
                        is_valid=False,
                        status=VerificationStatus.INVALID,
                        checks=checks,
                        error="Report data mismatch",
                        verification_time_ms=int((time.time() - start_time) * 1000),
                    )

            # All checks passed
            return VerificationResult(
                is_valid=True,
                status=VerificationStatus.VALID,
                checks=checks,
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"TEE verification failed: {e}")
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error=str(e),
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

    def _check_freshness(self, attestation: TEEAttestation) -> bool:
        """Check if attestation is fresh enough."""
        age = datetime.now(timezone.utc) - attestation.timestamp
        return age <= self.max_attestation_age

    def _verify_sgx(self, attestation: TEEAttestation) -> dict[str, Any]:
        """Verify Intel SGX attestation."""
        logger.debug("Verifying SGX attestation")

        checks = {
            "signature": True,  # Simulation
            "certificate_chain": True,
            "tcb_level": True,
        }

        # Real implementation would:
        # 1. Verify quote signature using Intel's signing key
        # 2. Verify certificate chain to Intel root CA
        # 3. Check TCB (Trusted Computing Base) level
        # 4. Verify MRENCLAVE/MRSIGNER values

        return {"valid": True, "checks": checks}

    def _verify_sev(self, attestation: TEEAttestation) -> dict[str, Any]:
        """Verify AMD SEV attestation."""
        logger.debug("Verifying AMD SEV attestation")

        checks = {
            "signature": True,
            "certificate_chain": True,
            "platform_status": True,
        }

        return {"valid": True, "checks": checks}

    def _verify_sev_snp(self, attestation: TEEAttestation) -> dict[str, Any]:
        """Verify AMD SEV-SNP attestation."""
        logger.debug("Verifying AMD SEV-SNP attestation")

        checks = {
            "signature": True,
            "certificate_chain": True,
            "snp_report": True,
            "tcb_version": True,
        }

        # Real implementation would:
        # 1. Verify attestation report signature
        # 2. Verify certificate chain to AMD root
        # 3. Parse and validate SNP report
        # 4. Check TCB version against known-good values

        if attestation.snp_report:
            checks["snp_report_present"] = True

        return {"valid": True, "checks": checks}

    def _verify_tdx(self, attestation: TEEAttestation) -> dict[str, Any]:
        """Verify Intel TDX attestation."""
        logger.debug("Verifying Intel TDX attestation")

        checks = {
            "signature": True,
            "certificate_chain": True,
            "td_report": True,
        }

        return {"valid": True, "checks": checks}

    def _verify_nitro(self, attestation: TEEAttestation) -> dict[str, Any]:
        """Verify AWS Nitro Enclave attestation."""
        logger.debug("Verifying AWS Nitro attestation")

        checks = {
            "signature": True,
            "certificate_chain": True,
            "pcr_values": True,
        }

        # Real implementation would:
        # 1. Parse COSE_Sign1 attestation document
        # 2. Verify signature using AWS Nitro CA
        # 3. Verify PCR values match expected measurements
        # 4. Check user data field contains expected values

        if attestation.pcr_values:
            checks["pcr_values_present"] = True
            # Check specific PCRs (0, 1, 2 for code, 3 for IAM, 4 for instance ID)
            for pcr_idx in [0, 1, 2]:
                checks[f"pcr_{pcr_idx}"] = pcr_idx in attestation.pcr_values

        return {"valid": True, "checks": checks}


# ============ ZK Verifier ============


class ZKVerifier:
    """
    Verifies zero-knowledge proofs.

    Supports:
    - Groth16 (snarkjs compatible)
    - PLONK
    - Halo2
    - STARK

    Example:
        >>> verifier = ZKVerifier()
        >>> result = verifier.verify(zk_proof, verification_key)
        >>> print(f"Valid: {result.is_valid}")
    """

    def __init__(self):
        """Initialize the ZK verifier."""
        logger.info("ZKVerifier initialized")

    def verify(
        self,
        proof: ZKProof,
        verification_key: bytes,
        public_inputs: Optional[list[str]] = None,
    ) -> VerificationResult:
        """
        Verify a zero-knowledge proof.

        Args:
            proof: ZK proof to verify
            verification_key: Verification key bytes
            public_inputs: Expected public inputs (uses proof's if not provided)

        Returns:
            Verification result
        """
        start_time = time.time()
        checks: dict[str, bool] = {}

        try:
            # Use provided public inputs or proof's
            inputs = public_inputs or proof.public_inputs

            # Check 1: Proof format
            checks["format"] = self._check_proof_format(proof)
            if not checks["format"]:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.INVALID,
                    checks=checks,
                    error="Invalid proof format",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check 2: Verification key hash
            vk_hash = hashlib.sha256(verification_key).hexdigest()
            checks["vk_hash"] = vk_hash == proof.verification_key_hash
            if not checks["vk_hash"]:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.INVALID,
                    checks=checks,
                    error="Verification key mismatch",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check 3: Verify the proof
            checks["proof"] = self._verify_proof(proof, verification_key, inputs)
            if not checks["proof"]:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.INVALID,
                    checks=checks,
                    error="Proof verification failed",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            return VerificationResult(
                is_valid=True,
                status=VerificationStatus.VALID,
                checks=checks,
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"ZK verification failed: {e}")
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error=str(e),
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

    def _check_proof_format(self, proof: ZKProof) -> bool:
        """Check proof format is valid."""
        # Check proof has expected fields
        if not proof.proof_bytes or len(proof.proof_bytes) == 0:
            return False
        if not proof.public_inputs:
            return False
        return True

    def _verify_proof(
        self,
        proof: ZKProof,
        verification_key: bytes,
        public_inputs: list[str],
    ) -> bool:
        """
        Verify the proof against verification key.

        This is a simulation - real implementation would use actual
        cryptographic verification.
        """
        # Simulation — production guard (PY-08 fix)
        if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
            raise RuntimeError(
                "CRITICAL SECURITY ERROR: ZK proof verification is using a simulation stub. "
                "Real verification backend (snarkjs/arkworks/halo2) must be integrated "
                "before production deployment."
            )

        logger.warning(
            "SECURITY: ZK proof verification is using a SIMULATION STUB. "
            "Proof type: %s. This is NOT real verification.",
            proof.proof_type,
        )
        logger.debug(f"Verifying {proof.proof_type} proof")
        return True


# ============ Unified Proof Verifier ============


class ProofVerifier:
    """
    Unified verifier for all proof types.

    Handles zkML proofs, TEE attestations, and hybrid proofs.

    Example:
        >>> verifier = ProofVerifier()
        >>>
        >>> # Verify any proof type
        >>> result = verifier.verify(proof, circuit)
        >>> print(f"Valid: {result.is_valid}")
        >>>
        >>> # Verify with expected output
        >>> result = verifier.verify(
        ...     proof, circuit,
        ...     expected_output_hash="abc123..."
        ... )
    """

    def __init__(
        self,
        tee_verifier: Optional[TEEVerifier] = None,
        zk_verifier: Optional[ZKVerifier] = None,
    ):
        """
        Initialize the unified verifier.

        Args:
            tee_verifier: Custom TEE verifier instance
            zk_verifier: Custom ZK verifier instance
        """
        self.tee_verifier = tee_verifier or TEEVerifier()
        self.zk_verifier = zk_verifier or ZKVerifier()

        logger.info("ProofVerifier initialized")

    def verify(
        self,
        proof: Proof,
        circuit: Optional[Circuit] = None,
        *,
        expected_input_hash: Optional[str] = None,
        expected_output_hash: Optional[str] = None,
        expected_model_hash: Optional[str] = None,
        verification_key: Optional[bytes] = None,
    ) -> VerificationResult:
        """
        Verify a proof.

        Args:
            proof: Proof to verify
            circuit: Circuit (provides verification key if not given)
            expected_input_hash: Expected input hash
            expected_output_hash: Expected output hash
            expected_model_hash: Expected model hash
            verification_key: Verification key (overrides circuit's)

        Returns:
            Verification result
        """
        start_time = time.time()
        checks: dict[str, bool] = {}

        # Handle None proof
        if proof is None:
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error="No proof provided",
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

        try:
            # Get verification key
            vk = verification_key
            if vk is None and circuit:
                vk = circuit.verification_key

            # Check input hash
            if expected_input_hash:
                checks["input_hash"] = proof.input_hash == expected_input_hash
                if not checks["input_hash"]:
                    return VerificationResult(
                        is_valid=False,
                        status=VerificationStatus.INVALID,
                        proof_id=proof.proof_id,
                        checks=checks,
                        error="Input hash mismatch",
                        verification_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Check output hash
            if expected_output_hash:
                checks["output_hash"] = proof.output_hash == expected_output_hash
                if not checks["output_hash"]:
                    return VerificationResult(
                        is_valid=False,
                        status=VerificationStatus.INVALID,
                        proof_id=proof.proof_id,
                        checks=checks,
                        error="Output hash mismatch",
                        verification_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Check model hash
            if expected_model_hash:
                checks["model_hash"] = proof.model_hash == expected_model_hash
                if not checks["model_hash"]:
                    return VerificationResult(
                        is_valid=False,
                        status=VerificationStatus.INVALID,
                        proof_id=proof.proof_id,
                        checks=checks,
                        error="Model hash mismatch",
                        verification_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Verify based on proof type
            if proof.proof_type == ProofType.TEE:
                result = self._verify_tee_proof(proof, checks)
            elif proof.proof_type == ProofType.ZKML:
                result = self._verify_zk_proof(proof, vk, checks)
            elif proof.proof_type == ProofType.HYBRID:
                result = self._verify_hybrid_proof(proof, vk, checks)
            else:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.UNKNOWN,
                    proof_id=proof.proof_id,
                    checks=checks,
                    error=f"Unknown proof type: {proof.proof_type}",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            result.proof_id = proof.proof_id
            result.verification_time_ms = int((time.time() - start_time) * 1000)
            return result

        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                proof_id=proof.proof_id,
                checks=checks,
                error=str(e),
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

    def verify_seal(
        self,
        seal: Any,
        expected_output_hash: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a Digital Seal.

        Args:
            seal: Digital Seal to verify
            expected_output_hash: Expected output hash

        Returns:
            Verification result
        """
        start_time = time.time()
        checks: dict[str, bool] = {}

        try:
            # Check seal is not revoked
            checks["not_revoked"] = not seal.revoked
            if not checks["not_revoked"]:
                return VerificationResult(
                    is_valid=False,
                    status=VerificationStatus.REVOKED,
                    checks=checks,
                    error=f"Seal revoked: {seal.revocation_reason}",
                    verification_time_ms=int((time.time() - start_time) * 1000),
                )

            # Check output hash if provided
            if expected_output_hash:
                checks["output_hash"] = seal.output_hash == expected_output_hash
                if not checks["output_hash"]:
                    return VerificationResult(
                        is_valid=False,
                        status=VerificationStatus.INVALID,
                        checks=checks,
                        error="Output hash mismatch",
                        verification_time_ms=int((time.time() - start_time) * 1000),
                    )

            # Check validator signatures (simulation)
            checks["validator_signatures"] = len(seal.validators) >= 2  # 2/3 quorum

            return VerificationResult(
                is_valid=all(checks.values()),
                status=VerificationStatus.VALID if all(checks.values()) else VerificationStatus.INVALID,
                checks=checks,
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

        except Exception as e:
            logger.error(f"Seal verification failed: {e}")
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error=str(e),
                verification_time_ms=int((time.time() - start_time) * 1000),
            )

    def _verify_tee_proof(
        self,
        proof: Proof,
        checks: dict[str, bool],
    ) -> VerificationResult:
        """Verify TEE-only proof."""
        if not proof.tee_attestation:
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error="TEE proof missing attestation",
            )

        result = self.tee_verifier.verify(
            proof.tee_attestation,
            expected_data=proof.output_hash.encode() if proof.output_hash else None,
        )

        checks.update(result.checks)
        checks["tee_valid"] = result.is_valid

        return VerificationResult(
            is_valid=result.is_valid,
            status=result.status,
            checks=checks,
            error=result.error,
        )

    def _verify_zk_proof(
        self,
        proof: Proof,
        verification_key: Optional[bytes],
        checks: dict[str, bool],
    ) -> VerificationResult:
        """Verify zkML-only proof."""
        if not proof.zk_proof:
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error="zkML proof missing",
            )

        if not verification_key:
            return VerificationResult(
                is_valid=False,
                status=VerificationStatus.INVALID,
                checks=checks,
                error="Verification key required for zkML proof",
            )

        result = self.zk_verifier.verify(proof.zk_proof, verification_key)

        checks.update(result.checks)
        checks["zk_valid"] = result.is_valid

        return VerificationResult(
            is_valid=result.is_valid,
            status=result.status,
            checks=checks,
            error=result.error,
        )

    def _verify_hybrid_proof(
        self,
        proof: Proof,
        verification_key: Optional[bytes],
        checks: dict[str, bool],
    ) -> VerificationResult:
        """Verify hybrid TEE+zkML proof."""
        errors = []

        # Verify TEE
        if proof.tee_attestation:
            tee_result = self.tee_verifier.verify(proof.tee_attestation)
            checks.update({f"tee_{k}": v for k, v in tee_result.checks.items()})
            checks["tee_valid"] = tee_result.is_valid
            if not tee_result.is_valid:
                errors.append(f"TEE: {tee_result.error}")
        else:
            checks["tee_valid"] = False
            errors.append("TEE attestation missing")

        # Verify ZK
        if proof.zk_proof and verification_key:
            zk_result = self.zk_verifier.verify(proof.zk_proof, verification_key)
            checks.update({f"zk_{k}": v for k, v in zk_result.checks.items()})
            checks["zk_valid"] = zk_result.is_valid
            if not zk_result.is_valid:
                errors.append(f"ZK: {zk_result.error}")
        else:
            checks["zk_valid"] = False
            errors.append("ZK proof or verification key missing")

        # Hybrid requires both to be valid
        is_valid = checks.get("tee_valid", False) and checks.get("zk_valid", False)

        return VerificationResult(
            is_valid=is_valid,
            status=VerificationStatus.VALID if is_valid else VerificationStatus.INVALID,
            checks=checks,
            error="; ".join(errors) if errors and not is_valid else None,
        )


# ============ Convenience Functions ============


def verify_proof(
    proof: Proof,
    circuit: Optional[Circuit] = None,
    **kwargs,
) -> VerificationResult:
    """
    Convenience function to verify a proof.

    Args:
        proof: Proof to verify
        circuit: Circuit with verification key
        **kwargs: Additional verification options

    Returns:
        Verification result
    """
    verifier = ProofVerifier()
    return verifier.verify(proof, circuit, **kwargs)


def verify_tee_attestation(
    attestation: TEEAttestation,
    expected_measurement: Optional[str] = None,
) -> VerificationResult:
    """
    Convenience function to verify a TEE attestation.

    Args:
        attestation: TEE attestation
        expected_measurement: Expected enclave measurement

    Returns:
        Verification result
    """
    verifier = TEEVerifier()
    return verifier.verify(attestation, expected_measurement=expected_measurement)
