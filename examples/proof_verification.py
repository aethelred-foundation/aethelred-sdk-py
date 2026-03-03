#!/usr/bin/env python3
"""
Proof Generation and Verification Example

This example demonstrates how to:
1. Generate zero-knowledge proofs locally
2. Verify zkML proofs
3. Verify TEE attestations
4. Work with hybrid proofs (TEE + zkML)

Prerequisites:
    pip install -e /Users/rameshtamilselvan/Downloads/AethelredMVP/sdk/python[ml]

Usage:
    python proof_verification.py
"""

import numpy as np
from datetime import datetime

from aethelred import (
    ProofGenerator,
    ProofVerifier,
    TEEVerifier,
    ZKVerifier,
    ProofConfig,
    ProofSystem,
    setup_logging,
)
from aethelred.core.types import (
    Circuit,
    CircuitMetrics,
    Proof,
    ProofType,
    ZKProof,
    TEEAttestation,
    TEEPlatform,
)


def create_mock_circuit() -> Circuit:
    """Create a mock circuit for demonstration."""
    return Circuit(
        circuit_id="circuit_demo_001",
        model_hash="sha256:abc123def456...",
        version="1.0.0",
        circuit_binary=b"MOCK_CIRCUIT_BINARY",
        verification_key=b"MOCK_VERIFICATION_KEY",
        proving_key_hash="sha256:pk_hash_789...",
        input_shape=(1, 64),
        output_shape=(1, 1),
        quantization_bits=8,
        optimization_level=2,
        metrics=CircuitMetrics(
            constraints=1000000,
            public_inputs=2,
            private_inputs=64,
            gates=2000000,
            depth=1000,
            memory_bytes=32000000,
            estimated_proving_time_ms=1000,
        ),
        framework="pytorch",
    )


def create_mock_tee_attestation() -> TEEAttestation:
    """Create a mock TEE attestation."""
    return TEEAttestation(
        platform=TEEPlatform.INTEL_SGX,
        enclave_id="enclave_demo_001",
        measurement="mrenclave_abc123...",
        report_data=b"output_hash_demo",
        signature=b"ATTESTATION_SIGNATURE",
        certificate_chain=[b"CERT_1", b"CERT_2", b"ROOT_CERT"],
        timestamp=datetime.utcnow(),
    )


def proof_generation_example():
    """Demonstrate local proof generation."""
    print("\n" + "=" * 50)
    print("Proof Generation")
    print("=" * 50)

    # Initialize generator
    generator = ProofGenerator(
        default_proof_system=ProofSystem.HALO2,
        gpu_enabled=True
    )

    # Create mock circuit and input
    circuit = create_mock_circuit()
    input_data = np.random.randn(1, 64).astype(np.float32)

    print("\n1. Basic Proof Generation:")
    print(f"   Circuit: {circuit.circuit_id}")
    print(f"   Input shape: {input_data.shape}")
    print(f"   Proof system: Halo2")

    # Generate proof (simulated)
    print("\n   Generating proof...")
    print("   - Witness generation: 150ms")
    print("   - Proof generation: 2,100ms")
    print("   - Self-verification: PASSED")

    print("\n2. Different Proof Systems:")

    systems = [
        ("groth16", "Fastest verification, smallest proofs, trusted setup required"),
        ("plonk", "Universal setup, good balance of proof size and speed"),
        ("halo2", "No trusted setup, recursive friendly"),
        ("stark", "Transparent, quantum resistant, larger proofs"),
    ]

    for system, description in systems:
        print(f"\n   {system.upper()}:")
        print(f"     {description}")

        # Show example
        print(f"""
     result = generator.generate(
         circuit=circuit,
         input_data=input_data,
         proof_system="{system}"
     )
        """)

    print("3. Proof Configuration:")
    print("""
    config = ProofConfig(
        proof_system=ProofSystem.HALO2,
        use_gpu=True,
        gpu_device=0,
        num_threads=8,
        max_memory_gb=32,
        memory_mapped=True,
        parallel_proving=True,
        batch_size=1,
        compress_proof=True,
        include_public_inputs=True,
        timeout_seconds=300
    )

    result = generator.generate(
        circuit=circuit,
        input_data=input_data,
        config=config
    )
    """)

    print("4. Batch Proof Generation:")
    print("""
    inputs = [input1, input2, input3, input4]

    results = generator.generate_batch(
        circuit=circuit,
        inputs=inputs,
        parallel=True
    )

    for i, result in enumerate(results):
        print(f"Proof {i}: {result.proving_time_ms}ms")
    """)


def proof_verification_example():
    """Demonstrate proof verification."""
    print("\n" + "=" * 50)
    print("Proof Verification")
    print("=" * 50)

    verifier = ProofVerifier()
    circuit = create_mock_circuit()

    print("\n1. Unified Verification:")
    print("""
    # Verify any proof type
    result = verifier.verify(
        proof=proof,
        circuit=circuit,
        expected_input_hash="abc123...",
        expected_output_hash="def456...",
        expected_model_hash="789xyz..."
    )

    if result.is_valid:
        print("Proof verified successfully!")
        print(f"Checks: {result.checks}")
    else:
        print(f"Verification failed: {result.error}")
    """)

    print("2. Verification Result Details:")
    print("""
    result = verifier.verify(proof, circuit)

    print(f"Valid: {result.is_valid}")
    print(f"Status: {result.status}")
    print(f"Time: {result.verification_time_ms}ms")

    # Individual checks
    for check, passed in result.checks.items():
        print(f"  {check}: {'PASS' if passed else 'FAIL'}")
    """)

    print("\n   Example output:")
    print("   Valid: True")
    print("   Status: valid")
    print("   Time: 15ms")
    print("   Checks:")
    print("     input_hash: PASS")
    print("     output_hash: PASS")
    print("     model_hash: PASS")
    print("     zk_format: PASS")
    print("     zk_vk_hash: PASS")
    print("     zk_proof: PASS")


def tee_verification_example():
    """Demonstrate TEE attestation verification."""
    print("\n" + "=" * 50)
    print("TEE Attestation Verification")
    print("=" * 50)

    tee_verifier = TEEVerifier(
        cache_certificates=True,
        allow_debug_enclaves=False,
        max_attestation_age_hours=24
    )

    print("\n1. Intel SGX Verification:")
    print("""
    attestation = TEEAttestation(
        platform=TEEPlatform.INTEL_SGX,
        enclave_id="enclave_001",
        measurement="mrenclave_abc...",  # MRENCLAVE hash
        report_data=output_hash,
        signature=attestation_signature,
        certificate_chain=[cert1, cert2, root]
    )

    result = tee_verifier.verify(
        attestation,
        expected_measurement="mrenclave_abc..."
    )
    """)

    print("   Verification checks for SGX:")
    print("   - signature: Verify quote signature")
    print("   - certificate_chain: Verify chain to Intel root")
    print("   - tcb_level: Check Trusted Computing Base level")
    print("   - measurement: Verify MRENCLAVE matches expected")
    print("   - freshness: Attestation not expired")

    print("\n2. AMD SEV-SNP Verification:")
    print("""
    attestation = TEEAttestation(
        platform=TEEPlatform.AMD_SEV_SNP,
        enclave_id="vm_001",
        measurement="measurement_hash...",
        report_data=output_hash,
        signature=attestation_signature,
        certificate_chain=[vcek, ask, ark],
        snp_report=snp_report_bytes
    )

    result = tee_verifier.verify(attestation)
    """)

    print("   Verification checks for SEV-SNP:")
    print("   - signature: Verify attestation report")
    print("   - certificate_chain: Verify chain to AMD root")
    print("   - snp_report: Parse and validate SNP report")
    print("   - tcb_version: Check TCB version")
    print("   - freshness: Report not expired")

    print("\n3. AWS Nitro Verification:")
    print("""
    attestation = TEEAttestation(
        platform=TEEPlatform.AWS_NITRO,
        enclave_id="i-abc123-enc",
        measurement="pcr0_hash...",
        report_data=output_hash,
        signature=cose_sign1,
        certificate_chain=[cert_chain],
        pcr_values={
            0: "pcr0_hash...",  # Enclave image
            1: "pcr1_hash...",  # Kernel
            2: "pcr2_hash...",  # Application
            3: "pcr3_hash...",  # IAM role
            4: "pcr4_hash...",  # Instance ID
        }
    )

    result = tee_verifier.verify(
        attestation,
        expected_measurement="pcr0_hash..."
    )
    """)


def hybrid_verification_example():
    """Demonstrate hybrid proof verification."""
    print("\n" + "=" * 50)
    print("Hybrid Proof Verification (TEE + zkML)")
    print("=" * 50)

    print("""
    # Hybrid proofs combine TEE attestation and zkML proof
    # for maximum security

    1. TEE provides:
       - Hardware-based isolation
       - Fast execution
       - Confidentiality guarantees

    2. zkML provides:
       - Mathematical soundness
       - Publicly verifiable
       - No trusted hardware required

    3. Combined:
       - Defense in depth
       - Compromise of one doesn't break security
       - Strongest verification guarantees
    """)

    print("Verifying Hybrid Proof:")
    print("""
    verifier = ProofVerifier()

    # Hybrid proof contains both
    hybrid_proof = Proof(
        proof_id="proof_hybrid_001",
        proof_type=ProofType.HYBRID,
        job_id="job_001",
        tee_attestation=tee_attestation,
        zk_proof=zk_proof,
        input_hash="abc...",
        output_hash="def...",
        model_hash="xyz..."
    )

    result = verifier.verify(hybrid_proof, circuit)

    # Both must pass for hybrid verification
    print(f"TEE Valid: {result.checks.get('tee_valid')}")
    print(f"ZK Valid: {result.checks.get('zk_valid')}")
    print(f"Hybrid Valid: {result.is_valid}")
    """)


def digital_seal_verification_example():
    """Demonstrate Digital Seal verification."""
    print("\n" + "=" * 50)
    print("Digital Seal Verification")
    print("=" * 50)

    print("""
    # Digital Seals are on-chain records of verified computations

    verifier = ProofVerifier()

    # Verify a seal
    result = verifier.verify_seal(
        seal,
        expected_output_hash="abc123..."
    )

    print(f"Seal Valid: {result.is_valid}")
    print(f"Status: {result.status}")

    # Verification checks:
    # - not_revoked: Seal hasn't been revoked
    # - output_hash: Output matches expected
    # - validator_signatures: Quorum of validators signed
    """)

    print("\nSeal Properties:")
    print("""
    seal = client.get_seal("seal_abc123")

    print(f"Seal ID: {seal.seal_id}")
    print(f"Job ID: {seal.job_id}")
    print(f"Model Hash: {seal.model_hash}")
    print(f"Input Hash: {seal.input_hash}")
    print(f"Output Hash: {seal.output_hash}")
    print(f"Proof Type: {seal.proof_type}")
    print(f"Validators: {seal.validators}")
    print(f"Block Height: {seal.block_height}")
    print(f"TX Hash: {seal.tx_hash}")
    print(f"Timestamp: {seal.timestamp}")
    print(f"Status: {seal.status}")
    print(f"Revoked: {seal.revoked}")
    """)


def main():
    """Run all verification examples."""
    setup_logging(level="INFO")

    print("=" * 60)
    print("Aethelred Proof Verification Examples")
    print("=" * 60)

    proof_generation_example()
    proof_verification_example()
    tee_verification_example()
    hybrid_verification_example()
    digital_seal_verification_example()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Use ProofGenerator for local proof generation")
    print("  2. Use ProofVerifier for unified verification")
    print("  3. TEEVerifier handles platform-specific attestations")
    print("  4. Hybrid proofs provide maximum security")
    print("  5. Digital Seals provide immutable audit trails")


if __name__ == "__main__":
    main()
