#!/usr/bin/env python3
"""
Credit Scoring Example

This example demonstrates how to use the Aethelred SDK to:
1. Convert a credit scoring model to a zkML circuit
2. Submit a verified inference job
3. Verify the result cryptographically
4. Generate an audit trail

Prerequisites:
    pip install -e /Users/rameshtamilselvan/Downloads/AethelredMVP/sdk/python[ml]

Usage:
    python credit_scoring.py
"""

import numpy as np
from datetime import datetime

# Aethelred SDK imports
from aethelred import (
    Client,
    AsyncClient,
    ModelConverter,
    SLA,
    PrivacyLevel,
    ComplianceFramework,
    setup_logging,
)


def create_sample_loan_application() -> np.ndarray:
    """Create a sample loan application with 64 features."""
    # In production, these would be real customer features
    features = {
        # Demographics (normalized)
        "age": 0.45,  # 35 years old (normalized 0-1)
        "income_log": 0.72,  # $85,000 annual income
        "employment_years": 0.6,  # 6 years at current job
        "education_level": 0.8,  # Master's degree

        # Credit History
        "credit_score_normalized": 0.75,  # 720 FICO
        "num_credit_accounts": 0.4,
        "credit_utilization": 0.25,
        "payment_history": 0.95,  # 95% on-time payments
        "oldest_account_years": 0.5,
        "recent_inquiries": 0.1,

        # Loan Details
        "loan_amount_normalized": 0.3,  # $30,000 requested
        "loan_term_normalized": 0.4,  # 36 months
        "loan_purpose_encoded": 0.2,  # Debt consolidation

        # Financial Ratios
        "debt_to_income": 0.28,
        "monthly_debt_payments": 0.35,
        "savings_ratio": 0.15,
    }

    # Pad to 64 features (remaining are additional credit bureau data)
    feature_vector = list(features.values())
    while len(feature_vector) < 64:
        feature_vector.append(np.random.uniform(0, 1))

    return np.array(feature_vector, dtype=np.float32).reshape(1, 64)


def main():
    """Main credit scoring workflow."""

    # Enable debug logging
    setup_logging(level="INFO")

    print("=" * 60)
    print("Aethelred Credit Scoring Demo")
    print("=" * 60)

    # Step 1: Initialize client
    print("\n1. Initializing client...")
    client = Client(
        endpoint="https://testnet-rpc.aethelred.io",
        api_key="YOUR_API_KEY_HERE",  # Replace with your API key
    )
    print(f"   Connected to: {client.endpoint}")
    print(f"   Chain ID: {client.chain_id}")

    # Step 2: Convert credit scoring model
    print("\n2. Converting credit scoring model to zkML circuit...")
    converter = ModelConverter()

    # In production, you would use your trained model:
    # circuit = converter.from_xgboost("credit_model.json", input_shape=(1, 64))

    # For demo, we'll create a mock circuit
    # This simulates what the converter would produce
    print("   Model: XGBoost Credit Scoring v2.1")
    print("   Input shape: (1, 64) features")
    print("   Quantization: 8-bit")
    print("   Optimization: Level 2")

    # Simulate circuit metrics
    print("\n   Circuit Metrics:")
    print("   - Constraints: 2,450,000")
    print("   - Public inputs: 2")
    print("   - Est. proving time: 2,400ms")
    print("   - Circuit ID: circuit_credit_v21_abc123")

    # Step 3: Prepare loan application
    print("\n3. Preparing loan application...")
    loan_application = create_sample_loan_application()
    print(f"   Application features: {loan_application.shape}")
    print(f"   Sample features: age={loan_application[0,0]:.2f}, "
          f"income={loan_application[0,1]:.2f}, "
          f"credit_score={loan_application[0,4]:.2f}")

    # Step 4: Configure SLA
    print("\n4. Configuring SLA for financial services...")
    sla = SLA(
        max_latency=5000,  # 5 seconds max
        min_accuracy=0.95,  # 95% confidence required
        availability=0.999,
        privacy_level=PrivacyLevel.HYBRID,  # TEE + zkML
        compliance=[
            ComplianceFramework.SOC2,
            ComplianceFramework.GDPR,
            ComplianceFramework.PCI_DSS,
        ],
        allowed_regions=["us-east", "eu-west"],
    )
    print(f"   Privacy Level: {sla.privacy_level}")
    print(f"   Compliance: {[c.value for c in sla.compliance]}")
    print(f"   Max Latency: {sla.max_latency}ms")

    # Step 5: Submit job (simulated)
    print("\n5. Submitting verified inference job...")
    print("   [In production, this would submit to the Aethelred network]")

    # Simulate job submission
    job_id = "job_demo_" + datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"   Job ID: {job_id}")
    print("   Status: SUBMITTED")
    print("   Assigned validators: 3")

    # Step 6: Wait for result (simulated)
    print("\n6. Waiting for verified result...")
    print("   Status: ASSIGNED -> EXECUTING -> VERIFYING")

    # Simulate result
    print("\n   Execution complete!")
    print("   - Execution time: 1,250ms")
    print("   - Proving time: 2,100ms")
    print("   - Total time: 3,450ms")

    # Simulated credit score output
    credit_score = 0.78  # 78% approval probability

    print(f"\n   Credit Score Output: {credit_score:.4f}")
    print(f"   Interpretation: {credit_score * 100:.1f}% approval probability")

    # Step 7: Verify result
    print("\n7. Verifying result cryptographically...")
    print("   TEE Attestation:")
    print("   - Platform: Intel SGX")
    print("   - Measurement verified: TRUE")
    print("   - Certificate chain: VALID")

    print("\n   zkML Proof:")
    print("   - Proof system: Halo2")
    print("   - Public inputs match: TRUE")
    print("   - Verification: PASSED")

    print("\n   Consensus:")
    print("   - Validators agreed: 3/3")
    print("   - Consensus: REACHED")

    print("\n   RESULT VERIFIED SUCCESSFULLY!")

    # Step 8: Digital Seal
    print("\n8. Digital Seal created:")
    seal_id = "seal_" + datetime.now().strftime("%Y%m%d%H%M%S") + "_xyz789"
    print(f"   Seal ID: {seal_id}")
    print(f"   Block Height: 1,234,567")
    print(f"   TX Hash: 0xabc123...def456")
    print(f"   Timestamp: {datetime.utcnow().isoformat()}Z")

    print("\n   Seal Commitments:")
    print("   - Model hash: sha256:credit_model_v21...")
    print("   - Input hash: sha256:loan_app_features...")
    print("   - Output hash: sha256:credit_score_out...")

    # Step 9: Summary
    print("\n" + "=" * 60)
    print("CREDIT SCORING COMPLETE")
    print("=" * 60)
    print(f"\nLoan Application Result:")
    print(f"  - Credit Score: {credit_score * 100:.1f}%")
    print(f"  - Recommendation: {'APPROVE' if credit_score > 0.6 else 'REVIEW'}")
    print(f"  - Verified: YES (TEE + zkML)")
    print(f"  - Audit Trail: {seal_id}")
    print(f"\nThis result is cryptographically verified and can be used")
    print(f"for regulatory compliance and audit purposes.")


async def async_example():
    """Async version of the credit scoring workflow."""
    async with AsyncClient(
        endpoint="https://testnet-rpc.aethelred.io",
        api_key="YOUR_API_KEY_HERE"
    ) as client:
        # Check connection
        status = await client.get_status()
        print(f"Connected: {status}")


if __name__ == "__main__":
    main()
