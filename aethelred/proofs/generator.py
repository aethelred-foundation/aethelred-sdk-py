"""
Proof Generator for Aethelred zkML

Generates zero-knowledge proofs for AI model inference verification.

The proof generation pipeline:
1. Load circuit and witness data
2. Generate witness from input
3. Create zkSNARK/PLONK/Halo2 proof
4. Serialize proof for on-chain verification

Example:
    >>> generator = ProofGenerator()
    >>> proof = generator.generate(
    ...     circuit=circuit,
    ...     input_data=input_tensor,
    ...     proof_system="halo2"
    ... )
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

from aethelred.core.exceptions import ProofError, ValidationError
from aethelred.core.types import (
    Circuit,
    Proof,
    ProofSystem,
    ProofType,
    ZKProof,
    TEEAttestation,
    TEEPlatform,
)

logger = logging.getLogger(__name__)


# ============ Configuration Types ============


class WitnessFormat(str, Enum):
    """Witness data formats."""

    JSON = "json"
    BINARY = "binary"
    FLATBUFFER = "flatbuffer"


@dataclass
class ProofConfig:
    """Configuration for proof generation."""

    # Proof system
    proof_system: ProofSystem = ProofSystem.EZKL

    # Performance
    use_gpu: bool = True
    gpu_device: int = 0
    num_threads: int = 8

    # Memory
    max_memory_gb: int = 32
    memory_mapped: bool = True

    # Constraints
    max_constraints: int = 10_000_000

    # Parallelization
    parallel_proving: bool = True
    batch_size: int = 1

    # Output
    compress_proof: bool = True
    include_public_inputs: bool = True

    # Timeout
    timeout_seconds: int = 300


@dataclass
class ProofRequest:
    """Request for proof generation."""

    circuit: Circuit
    input_data: Any
    config: ProofConfig = field(default_factory=ProofConfig)

    # Optional witness
    precomputed_witness: Optional[bytes] = None

    # Metadata
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    job_id: Optional[str] = None


@dataclass
class ProofResult:
    """Result of proof generation."""

    # Proof
    proof: Proof
    zk_proof: ZKProof

    # Timing
    witness_generation_ms: int
    proving_time_ms: int
    total_time_ms: int

    # Resources
    peak_memory_mb: int
    gpu_used: bool

    # Verification
    self_verified: bool = False

    # Request tracking
    request_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "proof_id": self.proof.proof_id,
            "proof_type": self.zk_proof.proof_type,
            "proving_time_ms": self.proving_time_ms,
            "total_time_ms": self.total_time_ms,
            "peak_memory_mb": self.peak_memory_mb,
            "gpu_used": self.gpu_used,
            "self_verified": self.self_verified,
        }


# ============ Proof Generator ============


class ProofGenerator:
    """
    Generates zero-knowledge proofs for AI inference verification.

    Supports multiple proof systems (Groth16, PLONK, Halo2) and
    integrates with TEE attestations for hybrid verification.

    Example:
        >>> generator = ProofGenerator()
        >>>
        >>> # Generate zkML proof
        >>> result = generator.generate(
        ...     circuit=circuit,
        ...     input_data=input_tensor,
        ...     proof_system="halo2"
        ... )
        >>>
        >>> # Generate with custom config
        >>> config = ProofConfig(
        ...     proof_system=ProofSystem.GROTH16,
        ...     use_gpu=True,
        ...     parallel_proving=True
        ... )
        >>> result = generator.generate(circuit, input_data, config=config)
        >>>
        >>> # Batch proof generation
        >>> results = generator.generate_batch(
        ...     circuit=circuit,
        ...     inputs=[input1, input2, input3]
        ... )
    """

    def __init__(
        self,
        default_proof_system: ProofSystem = ProofSystem.EZKL,
        gpu_enabled: bool = True,
        cache_dir: Optional[str] = None,
        config: Optional[ProofConfig] = None,
    ):
        """
        Initialize the proof generator.

        Args:
            default_proof_system: Default proof system to use
            gpu_enabled: Enable GPU acceleration
            cache_dir: Directory for caching intermediate data
            config: Optional ProofConfig to use
        """
        if config is not None:
            self._config = config
            self.default_proof_system = config.proof_system
            self.gpu_enabled = config.use_gpu if gpu_enabled else False
        else:
            self._config = ProofConfig(proof_system=default_proof_system)
            self.default_proof_system = default_proof_system
            self.gpu_enabled = gpu_enabled
        self.cache_dir = cache_dir

        self._backends = {
            ProofSystem.GROTH16: self._prove_groth16,
            ProofSystem.PLONK: self._prove_plonk,
            ProofSystem.EZKL: self._prove_halo2,
            ProofSystem.STARK: self._prove_stark,
        }

        logger.info(
            f"ProofGenerator initialized: system={default_proof_system.value}, "
            f"gpu={self.gpu_enabled}"
        )

    def generate(
        self,
        circuit: Circuit,
        input_data: Any,
        *,
        proof_system: Optional[Union[str, ProofSystem]] = None,
        config: Optional[ProofConfig] = None,
        output_data: Optional[Any] = None,
        job_id: Optional[str] = None,
    ) -> ProofResult:
        """
        Generate a zero-knowledge proof.

        Args:
            circuit: Compiled arithmetic circuit
            input_data: Input tensor/data for inference
            proof_system: Proof system to use (default: halo2)
            config: Proof generation configuration
            output_data: Pre-computed output (skip inference if provided)
            job_id: Associated job ID

        Returns:
            Proof generation result with proof and metrics

        Raises:
            ProofError: If proof generation fails
            ValidationError: If inputs are invalid
        """
        start_time = time.time()

        # Resolve proof system
        if isinstance(proof_system, str):
            proof_system = ProofSystem(proof_system)
        proof_system = proof_system or self.default_proof_system

        # Create config if not provided
        if config is None:
            config = ProofConfig(proof_system=proof_system)

        request_id = uuid.uuid4().hex
        logger.info(f"Generating {proof_system.value} proof (request={request_id[:8]})")

        try:
            # Step 1: Validate inputs
            self._validate_inputs(circuit, input_data)

            # Step 2: Serialize input data
            input_bytes = self._serialize_input(input_data)
            input_hash = hashlib.sha256(input_bytes).hexdigest()

            # Step 3: Generate witness
            witness_start = time.time()
            witness = self._generate_witness(circuit, input_data, output_data)
            witness_time = int((time.time() - witness_start) * 1000)
            logger.debug(f"Witness generated in {witness_time}ms")

            # Step 4: Generate proof using selected backend
            prove_start = time.time()
            backend_fn = self._backends[proof_system]
            proof_bytes, public_inputs = backend_fn(circuit, witness, config)
            proving_time = int((time.time() - prove_start) * 1000)
            logger.debug(f"Proof generated in {proving_time}ms")

            # Step 5: Compute output hash
            output_hash = hashlib.sha256(
                self._serialize_input(output_data or witness.get("output"))
            ).hexdigest()

            # Step 6: Create ZKProof object
            zk_proof = ZKProof(
                proof_type=proof_system.value,
                proof_bytes=proof_bytes,
                public_inputs=public_inputs,
                verification_key_hash=circuit.proving_key_hash,
                circuit_hash=hashlib.sha256(circuit.circuit_binary).hexdigest(),
                proving_time_ms=proving_time,
            )

            # Step 7: Create Proof wrapper
            proof_id = f"proof_{uuid.uuid4().hex[:16]}"
            proof = Proof(
                proof_id=proof_id,
                proof_type=ProofType.ZKML,
                job_id=job_id or "",
                zk_proof=zk_proof,
                input_hash=input_hash,
                output_hash=output_hash,
                model_hash=circuit.model_hash,
            )

            # Step 8: Self-verify if requested
            self_verified = False
            if config.include_public_inputs:
                self_verified = self._self_verify(proof, circuit)

            total_time = int((time.time() - start_time) * 1000)

            result = ProofResult(
                proof=proof,
                zk_proof=zk_proof,
                witness_generation_ms=witness_time,
                proving_time_ms=proving_time,
                total_time_ms=total_time,
                peak_memory_mb=self._estimate_memory_usage(circuit),
                gpu_used=config.use_gpu and self.gpu_enabled,
                self_verified=self_verified,
                request_id=request_id,
            )

            logger.info(
                f"Proof generated: id={proof_id[:16]}, time={total_time}ms, "
                f"verified={self_verified}"
            )

            return result

        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            raise ProofError(f"Proof generation failed: {e}") from e

    def generate_batch(
        self,
        circuit: Circuit,
        inputs: list[Any],
        *,
        config: Optional[ProofConfig] = None,
        parallel: bool = True,
    ) -> list[ProofResult]:
        """
        Generate proofs for multiple inputs in batch.

        Args:
            circuit: Compiled arithmetic circuit
            inputs: List of input data
            config: Proof generation configuration
            parallel: Enable parallel proof generation

        Returns:
            List of proof results
        """
        logger.info(f"Generating batch of {len(inputs)} proofs")

        results = []

        # For now, sequential generation
        # Real implementation would use multiprocessing or async
        for i, input_data in enumerate(inputs):
            logger.debug(f"Processing batch item {i+1}/{len(inputs)}")
            result = self.generate(circuit, input_data, config=config)
            results.append(result)

        return results

    def generate_hybrid(
        self,
        circuit: Circuit,
        input_data: Any,
        tee_attestation: TEEAttestation,
        *,
        config: Optional[ProofConfig] = None,
    ) -> ProofResult:
        """
        Generate hybrid proof combining zkML and TEE attestation.

        Args:
            circuit: Compiled arithmetic circuit
            input_data: Input tensor/data
            tee_attestation: TEE attestation from secure enclave
            config: Proof generation configuration

        Returns:
            Hybrid proof result
        """
        logger.info("Generating hybrid TEE+zkML proof")

        # Generate zkML proof
        result = self.generate(circuit, input_data, config=config)

        # Add TEE attestation to proof
        result.proof.tee_attestation = tee_attestation
        result.proof.proof_type = ProofType.HYBRID

        return result

    def estimate_proving_time(
        self,
        circuit: Circuit,
        proof_system: Optional[ProofSystem] = None,
    ) -> dict[str, int]:
        """
        Estimate proving time and resource usage.

        Args:
            circuit: Compiled circuit
            proof_system: Target proof system

        Returns:
            Estimated metrics (time, memory, etc.)
        """
        proof_system = proof_system or self.default_proof_system
        constraints = circuit.metrics.constraints

        # Rough estimates based on proof system
        # These would be calibrated based on actual benchmarks
        estimates = {
            ProofSystem.GROTH16: {
                "proving_time_ms": constraints // 50,  # ~50 constraints/ms with GPU
                "verification_time_ms": 10,  # Constant time
                "proof_size_bytes": 192,  # 3 group elements
                "memory_mb": constraints * 32 // (1024 * 1024),
            },
            ProofSystem.PLONK: {
                "proving_time_ms": constraints // 30,
                "verification_time_ms": 20,
                "proof_size_bytes": 800,  # Variable
                "memory_mb": constraints * 48 // (1024 * 1024),
            },
            ProofSystem.EZKL: {
                "proving_time_ms": constraints // 40,
                "verification_time_ms": 15,
                "proof_size_bytes": 600,
                "memory_mb": constraints * 40 // (1024 * 1024),
            },
            ProofSystem.STARK: {
                "proving_time_ms": constraints // 20,
                "verification_time_ms": 50,  # Larger proofs
                "proof_size_bytes": constraints // 10,  # Log-size proofs
                "memory_mb": constraints * 64 // (1024 * 1024),
            },
        }

        return estimates.get(proof_system, estimates[ProofSystem.EZKL])

    # ============ Internal Methods ============

    def _hash_model(self, model_bytes: bytes) -> str:
        """Compute SHA-256 hash of model bytes."""
        return hashlib.sha256(model_bytes).hexdigest()

    def _hash_input(self, input_bytes: bytes) -> str:
        """Compute SHA-256 hash of input bytes."""
        return hashlib.sha256(input_bytes).hexdigest()

    def _validate_inputs(self, circuit: Circuit, input_data: Any) -> None:
        """Validate inputs against circuit requirements."""
        # Check input shape if input is array-like
        if hasattr(input_data, "shape"):
            input_shape = tuple(input_data.shape)
            if input_shape != circuit.input_shape:
                raise ValidationError(
                    f"Input shape mismatch: expected {circuit.input_shape}, "
                    f"got {input_shape}"
                )

    def _serialize_input(self, input_data: Any) -> bytes:
        """Serialize input data to bytes."""
        import json

        if isinstance(input_data, bytes):
            return input_data
        elif hasattr(input_data, "tobytes"):
            # NumPy array
            return input_data.tobytes()
        elif hasattr(input_data, "numpy"):
            # PyTorch tensor
            return input_data.numpy().tobytes()
        else:
            # JSON serializable
            return json.dumps(input_data).encode()

    def _generate_witness(
        self,
        circuit: Circuit,
        input_data: Any,
        output_data: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Generate witness from input data."""
        # Simulation - real implementation would:
        # 1. Run the model inference to get output
        # 2. Record all intermediate values
        # 3. Format as witness for the proof system

        logger.debug("Generating witness")

        return {
            "input": input_data,
            "output": output_data or [0.5],  # Placeholder
            "intermediate": [],  # All intermediate values
        }

    def _prove_groth16(
        self,
        circuit: Circuit,
        witness: dict[str, Any],
        config: ProofConfig,
    ) -> tuple[bytes, list[str]]:
        """Generate Groth16 proof."""
        logger.debug("Generating Groth16 proof")

        # Simulation — production guard (PY-09 fix)
        if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
            raise RuntimeError(
                "CRITICAL: Groth16 proof generation is using a simulation stub. "
                "Integrate snarkjs or arkworks before production deployment."
            )

        logger.warning("SECURITY: Groth16 proof is SIMULATED (not real)")
        proof_bytes = b"GROTH16_PROOF_" + uuid.uuid4().bytes

        # Public inputs: input hash, output hash
        public_inputs = [
            hashlib.sha256(self._serialize_input(witness["input"])).hexdigest(),
            hashlib.sha256(self._serialize_input(witness["output"])).hexdigest(),
        ]

        return proof_bytes, public_inputs

    def _prove_plonk(
        self,
        circuit: Circuit,
        witness: dict[str, Any],
        config: ProofConfig,
    ) -> tuple[bytes, list[str]]:
        """Generate PLONK proof."""
        logger.debug("Generating PLONK proof")

        if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
            raise RuntimeError(
                "CRITICAL: PLONK proof generation is using a simulation stub. "
                "Integrate plonky2 or gnark before production deployment."
            )

        logger.warning("SECURITY: PLONK proof is SIMULATED (not real)")
        proof_bytes = b"PLONK_PROOF_" + uuid.uuid4().bytes

        public_inputs = [
            hashlib.sha256(self._serialize_input(witness["input"])).hexdigest(),
            hashlib.sha256(self._serialize_input(witness["output"])).hexdigest(),
        ]

        return proof_bytes, public_inputs

    def _prove_halo2(
        self,
        circuit: Circuit,
        witness: dict[str, Any],
        config: ProofConfig,
    ) -> tuple[bytes, list[str]]:
        """Generate Halo2 proof."""
        logger.debug("Generating Halo2 proof")

        if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
            raise RuntimeError(
                "CRITICAL: Halo2 proof generation is using a simulation stub. "
                "Integrate halo2 library before production deployment."
            )

        logger.warning("SECURITY: Halo2 proof is SIMULATED (not real)")
        proof_bytes = b"HALO2_PROOF_" + uuid.uuid4().bytes

        public_inputs = [
            hashlib.sha256(self._serialize_input(witness["input"])).hexdigest(),
            hashlib.sha256(self._serialize_input(witness["output"])).hexdigest(),
        ]

        return proof_bytes, public_inputs

    def _prove_stark(
        self,
        circuit: Circuit,
        witness: dict[str, Any],
        config: ProofConfig,
    ) -> tuple[bytes, list[str]]:
        """Generate STARK proof."""
        logger.debug("Generating STARK proof")

        if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
            raise RuntimeError(
                "CRITICAL: STARK proof generation is using a simulation stub. "
                "Integrate winterfell or stone before production deployment."
            )

        logger.warning("SECURITY: STARK proof is SIMULATED (not real)")
        proof_bytes = b"STARK_PROOF_" + uuid.uuid4().bytes

        public_inputs = [
            hashlib.sha256(self._serialize_input(witness["input"])).hexdigest(),
            hashlib.sha256(self._serialize_input(witness["output"])).hexdigest(),
        ]

        return proof_bytes, public_inputs

    def _self_verify(self, proof: Proof, circuit: Circuit) -> bool:
        """Self-verify the generated proof."""
        try:
            # Simulation — always passes in non-production mode
            if os.environ.get("AETHELRED_PRODUCTION_MODE") == "1":
                raise RuntimeError(
                    "Self-verification requires real proof verification backend."
                )
            return True
        except Exception as e:
            logger.warning(f"Self-verification failed: {e}")
            return False

    def _estimate_memory_usage(self, circuit: Circuit) -> int:
        """Estimate peak memory usage in MB."""
        # Rough estimate: 32-64 bytes per constraint
        constraints = circuit.metrics.constraints
        return max(100, constraints * 48 // (1024 * 1024))


# ============ Utility Functions ============


def create_proof_request(
    circuit: Circuit,
    input_data: Any,
    proof_system: Optional[Union[str, ProofSystem]] = None,
    **kwargs,
) -> ProofRequest:
    """
    Create a proof generation request.

    Args:
        circuit: Compiled circuit
        input_data: Input data
        proof_system: Proof system to use
        **kwargs: Additional config options

    Returns:
        Configured proof request
    """
    ps = ProofSystem.EZKL
    if proof_system is not None:
        if isinstance(proof_system, ProofSystem):
            ps = proof_system
        else:
            # Try to match by name
            ps = ProofSystem[proof_system.upper()] if proof_system.upper() in ProofSystem.__members__ else ProofSystem.EZKL
    config = ProofConfig(
        proof_system=ps,
        **kwargs,
    )

    return ProofRequest(
        circuit=circuit,
        input_data=input_data,
        config=config,
    )
