"""Comprehensive tests to boost coverage to 95%+.

Covers uncovered paths in:
- core/exceptions.py (all exception subclasses, to_dict, repr)
- core/config.py (Config factory methods, properties, Network enum)
- core/types.py (Circuit, ZKProof, Proof, ModelInfo, DataProvenance, etc.)
- utils (keccak256, to_uaeth, from_uaeth, format_aeth, is_valid_address, base64, sha256)
- crypto/fallback.py (BackendInfo, protocol checks)
- crypto/pqc/kyber.py (KyberKeyPair, KyberCiphertext, utility functions)
- crypto/pqc/dilithium.py (DilithiumKeyPair, DilithiumSignature, utility functions)
- proofs/verifier.py (TEEVerifier, ZKVerifier, ProofVerifier)
- proofs/generator.py (ProofGenerator all systems, batch, hybrid, estimate)
- oracles/client.py (OracleClient, AsyncOracleClient, DataFeed, helpers)
- models/registry.py (RegisteredModel, ModelStatus)
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest


# ============================================================
# core/exceptions.py - All exception subclasses
# ============================================================


class TestExceptionSubclasses:
    """Test all exception subclasses for coverage."""

    def test_connection_error_defaults(self) -> None:
        from aethelred.core.exceptions import ConnectionError, ErrorCode
        err = ConnectionError()
        assert err.code == ErrorCode.CONNECTION_FAILED
        assert "connect" in err.message.lower()

    def test_connection_error_custom(self) -> None:
        from aethelred.core.exceptions import ConnectionError, ErrorCode
        err = ConnectionError("custom msg", code=ErrorCode.DNS_RESOLUTION_FAILED)
        assert err.code == ErrorCode.DNS_RESOLUTION_FAILED
        assert err.message == "custom msg"

    def test_authentication_error(self) -> None:
        from aethelred.core.exceptions import AuthenticationError, ErrorCode
        err = AuthenticationError()
        assert err.code == ErrorCode.AUTHENTICATION_REQUIRED

    def test_rate_limit_error(self) -> None:
        from aethelred.core.exceptions import RateLimitError
        err = RateLimitError(retry_after=30)
        assert err.retry_after == 30

    def test_transaction_error(self) -> None:
        from aethelred.core.exceptions import TransactionError
        err = TransactionError(tx_hash="0xabc")
        assert err.tx_hash == "0xabc"

    def test_job_error(self) -> None:
        from aethelred.core.exceptions import JobError
        err = JobError(job_id="job_123")
        assert err.job_id == "job_123"

    def test_seal_error(self) -> None:
        from aethelred.core.exceptions import SealError
        err = SealError(seal_id="seal_456")
        assert err.seal_id == "seal_456"

    def test_model_error(self) -> None:
        from aethelred.core.exceptions import ModelError
        err = ModelError(model_hash="abc123")
        assert err.model_hash == "abc123"

    def test_verification_error(self) -> None:
        from aethelred.core.exceptions import VerificationError
        err = VerificationError("proof invalid")
        assert "proof invalid" in str(err)

    def test_validation_error_with_field(self) -> None:
        from aethelred.core.exceptions import ValidationError
        err = ValidationError("bad field", field="model_hash")
        assert err.field == "model_hash"

    def test_timeout_error(self) -> None:
        from aethelred.core.exceptions import TimeoutError
        err = TimeoutError(timeout_seconds=30.0)
        assert err.timeout_seconds == 30.0

    def test_network_error(self) -> None:
        from aethelred.core.exceptions import NetworkError
        err = NetworkError("network down")
        assert "network down" in str(err)

    def test_aethelred_error_to_dict(self) -> None:
        from aethelred.core.exceptions import AethelredError, ErrorCode
        err = AethelredError("test", code=ErrorCode.INTERNAL, details={"key": "val"})
        d = err.to_dict()
        assert d["error"] == "AethelredError"
        assert d["code"] == ErrorCode.INTERNAL.value
        assert d["details"] == {"key": "val"}

    def test_aethelred_error_repr(self) -> None:
        from aethelred.core.exceptions import AethelredError
        err = AethelredError("test msg")
        r = repr(err)
        assert "AethelredError" in r
        assert "test msg" in r

    def test_aethelred_error_str_with_details_and_cause(self) -> None:
        from aethelred.core.exceptions import AethelredError
        cause = ValueError("root cause")
        err = AethelredError("test", details={"x": 1}, cause=cause)
        s = str(err)
        assert "Details" in s
        assert "Caused by" in s

    def test_proof_error_alias(self) -> None:
        from aethelred.core.exceptions import ProofError, VerificationError
        assert ProofError is VerificationError

    def test_error_code_values(self) -> None:
        from aethelred.core.exceptions import ErrorCode
        assert ErrorCode.UNKNOWN == 1000
        assert ErrorCode.RATE_LIMITED == 1300
        assert ErrorCode.VERIFICATION_FAILED == 1800


# ============================================================
# core/config.py - Config factory methods and properties
# ============================================================


class TestConfigFactoryMethods:
    """Test Config class methods and properties."""

    def test_from_network_mainnet(self) -> None:
        from aethelred.core.config import Config, Network
        cfg = Config.from_network(Network.MAINNET)
        assert "mainnet" in cfg.rpc_url

    def test_mainnet(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.mainnet()
        assert cfg.chain_id == "aethelred-1"

    def test_testnet(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.testnet()
        assert cfg.chain_id == "aethelred-testnet-1"

    def test_devnet(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.devnet()
        assert cfg.chain_id == "aethelred-devnet-1"

    def test_local(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.local()
        assert "127.0.0.1" in cfg.rpc_url

    def test_custom(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.custom("https://custom.rpc", "custom-chain-1")
        assert cfg.rpc_url == "https://custom.rpc"
        assert cfg.chain_id == "custom-chain-1"

    def test_ws_url_property(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.mainnet()
        assert cfg.ws_url is not None
        assert "ws" in cfg.ws_url

    def test_grpc_url_property(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.mainnet()
        assert cfg.grpc_url is not None

    def test_rest_url_property(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.mainnet()
        assert cfg.rest_url is not None

    def test_get_network_config(self) -> None:
        from aethelred.core.config import Config
        cfg = Config.mainnet()
        nc = cfg.get_network_config()
        assert nc is not None
        assert nc.chain_id == "aethelred-1"

    def test_endpoint_setter(self) -> None:
        from aethelred.core.config import Config
        cfg = Config()
        cfg.endpoint = "https://new.endpoint"
        assert cfg.rpc_url == "https://new.endpoint"
        assert cfg.endpoint == "https://new.endpoint"

    def test_config_with_api_key(self) -> None:
        from aethelred.core.config import Config, SecretStr
        cfg = Config(api_key=SecretStr("my-key"))
        assert cfg.api_key.get_secret_value() == "my-key"

    def test_network_enum(self) -> None:
        from aethelred.core.config import Network
        assert Network.MAINNET.value == "mainnet"
        assert Network.LOCAL.value == "local"

    def test_retry_config(self) -> None:
        from aethelred.core.config import RetryConfig
        rc = RetryConfig(max_retries=5, initial_delay=1.0)
        assert rc.max_retries == 5

    def test_timeout_config(self) -> None:
        from aethelred.core.config import TimeoutConfig
        tc = TimeoutConfig(connect_timeout=5.0)
        assert tc.connect_timeout == 5.0

    def test_network_config(self) -> None:
        from aethelred.core.config import NetworkConfig
        nc = NetworkConfig(rpc_url="http://test", chain_id="test-1")
        assert nc.rpc_url == "http://test"

    def test_secret_str_bool(self) -> None:
        from aethelred.core.config import SecretStr
        assert bool(SecretStr("non-empty"))
        assert not bool(SecretStr(""))

    def test_secret_str_equality(self) -> None:
        from aethelred.core.config import SecretStr
        a = SecretStr("x")
        b = SecretStr("x")
        c = SecretStr("y")
        assert a == b
        assert a != c
        assert a != "x"  # NotImplemented for non-SecretStr

    def test_secret_str_hash(self) -> None:
        from aethelred.core.config import SecretStr
        s = SecretStr("key")
        assert isinstance(hash(s), int)

    def test_config_cache_settings(self) -> None:
        from aethelred.core.config import Config
        cfg = Config(cache_enabled=False, cache_ttl=120.0, cache_max_size=500)
        assert not cfg.cache_enabled
        assert cfg.cache_ttl == 120.0
        assert cfg.cache_max_size == 500


# ============================================================
# core/types.py - Additional type coverage
# ============================================================


class TestAdditionalTypes:
    """Test additional types for coverage."""

    def test_circuit_creation(self) -> None:
        from aethelred.core.types import Circuit, CircuitMetrics
        c = Circuit(circuit_id="c1", model_hash="abc")
        assert c.circuit_id == "c1"
        assert isinstance(c.metrics, CircuitMetrics)

    def test_circuit_to_dict(self) -> None:
        from aethelred.core.types import Circuit
        c = Circuit(circuit_id="c1", model_hash="abc", input_shape=(1, 28, 28))
        d = c.to_dict()
        assert d["circuit_id"] == "c1"
        assert d["input_shape"] == [1, 28, 28]

    def test_circuit_metrics(self) -> None:
        from aethelred.core.types import CircuitMetrics
        m = CircuitMetrics(constraints=1000, gates=500)
        assert m.constraints == 1000

    def test_zk_proof(self) -> None:
        from aethelred.core.types import ZKProof
        p = ZKProof(proof_type="groth16", proof_bytes=b"data")
        assert p.proof_type == "groth16"

    def test_proof(self) -> None:
        from aethelred.core.types import Proof, ProofType
        p = Proof(proof_id="p1", proof_type=ProofType.ZKML, job_id="j1")
        assert p.proof_id == "p1"

    def test_model_info(self) -> None:
        from aethelred.core.types import ModelInfo
        m = ModelInfo(model_hash="abc", name="test-model", owner="aeth1owner")
        assert m.name == "test-model"

    def test_data_source_type(self) -> None:
        from aethelred.core.types import DataSourceType
        assert DataSourceType.MARKET_DATA.value == "market_data"
        assert DataSourceType.ORACLE_DID.value == "oracle_did"

    def test_data_provenance_new_fields(self) -> None:
        from aethelred.core.types import DataProvenance
        dp = DataProvenance(
            oracle_node_id="node_1",
            attestation_hash="abc",
            data_hash="def",
            verification_method="tee",
        )
        assert dp.oracle_node_id == "node_1"
        assert dp.attestation_hash == "abc"

    def test_job_input_new_fields(self) -> None:
        from aethelred.core.types import JobInput, DataSourceType
        ji = JobInput(
            source_type=DataSourceType.ORACLE_DID,
            payload="did:aethelred:oracle:btc_usd",
            did="did:aethelred:oracle:btc_usd",
            encrypted=True,
        )
        assert ji.source_type == DataSourceType.ORACLE_DID
        assert ji.encrypted is True

    def test_tee_attestation_new_fields(self) -> None:
        from aethelred.core.types import TEEAttestation
        att = TEEAttestation(
            measurement="abc123",
            report_data=b"output",
            snp_report=b"snp_data",
        )
        assert att.measurement == "abc123"
        assert att.report_data == b"output"

    def test_sla(self) -> None:
        from aethelred.core.types import SLA, PrivacyLevel
        sla = SLA(privacy_level=PrivacyLevel.TEE, min_validators=3)
        assert sla.min_validators == 3

    def test_hardware_target(self) -> None:
        from aethelred.core.types import HardwareTarget
        assert HardwareTarget.AUTO.value == "auto"
        assert HardwareTarget.NVIDIA_A100.value == "nvidia_a100"

    def test_hardware_requirements(self) -> None:
        from aethelred.core.types import HardwareRequirements, HardwareTarget
        hr = HardwareRequirements(min_memory_gb=16, target=HardwareTarget.CPU)
        assert hr.min_memory_gb == 16

    def test_job_result(self) -> None:
        from aethelred.core.types import JobResult
        jr = JobResult(job_id="j1", output_hash=b"\x00" * 32)
        assert jr.verified is False

    def test_page_response(self) -> None:
        from aethelred.core.types import PageResponse
        pr = PageResponse(total=42)
        assert pr.total == 42

    def test_epoch_stats(self) -> None:
        from aethelred.core.types import EpochStats
        es = EpochStats(epoch=1, start_block=0, end_block=100)
        assert es.epoch == 1

    def test_node_info(self) -> None:
        from aethelred.core.types import NodeInfo
        ni = NodeInfo(moniker="test-node")
        assert ni.moniker == "test-node"

    def test_block_and_block_header(self) -> None:
        from aethelred.core.types import Block, BlockHeader
        bh = BlockHeader(height=100, time=datetime.now(timezone.utc), chain_id="test-1")
        b = Block(header=bh)
        assert b.header.height == 100

    def test_useful_work_stats(self) -> None:
        from aethelred.core.types import UsefulWorkStats
        uws = UsefulWorkStats(total_jobs=100, completed_jobs=90)
        assert uws.completed_jobs == 90

    def test_hardware_capability(self) -> None:
        from aethelred.core.types import HardwareCapability, TEEPlatform
        hc = HardwareCapability(
            tee_platforms=[TEEPlatform.INTEL_SGX],
            zkml_supported=True,
            gpu_memory_gb=80,
        )
        assert TEEPlatform.INTEL_SGX in hc.tee_platforms

    def test_verification_result(self) -> None:
        from aethelred.core.types import VerificationResult
        vr = VerificationResult(valid=True, output_hash=b"\x00" * 32)
        assert vr.valid is True

    def test_create_seal_request(self) -> None:
        from aethelred.core.types import CreateSealRequest
        csr = CreateSealRequest(job_id="j1")
        assert csr.job_id == "j1"

    def test_create_seal_response(self) -> None:
        from aethelred.core.types import CreateSealResponse
        csr = CreateSealResponse(seal_id="s1", tx_hash="0xabc")
        assert csr.seal_id == "s1"

    def test_verify_seal_response(self) -> None:
        from aethelred.core.types import VerifySealResponse
        vsr = VerifySealResponse(valid=True)
        assert vsr.valid is True

    def test_register_model_request(self) -> None:
        from aethelred.core.types import RegisterModelRequest
        rmr = RegisterModelRequest(model_hash=b"\x00" * 32, name="m1")
        assert rmr.name == "m1"

    def test_register_model_response(self) -> None:
        from aethelred.core.types import RegisterModelResponse
        rmr = RegisterModelResponse(model_hash="abc", tx_hash="0xdef")
        assert rmr.model_hash == "abc"

    def test_job_seal_aliases(self) -> None:
        from aethelred.core.types import Job, Seal, ComputeJob, DigitalSeal
        assert Job is ComputeJob
        assert Seal is DigitalSeal

    def test_compliance_framework(self) -> None:
        from aethelred.core.types import ComplianceFramework
        assert ComplianceFramework.HIPAA.value == "HIPAA"
        assert ComplianceFramework.GDPR.value == "GDPR"

    def test_utility_category(self) -> None:
        from aethelred.core.types import UtilityCategory
        assert UtilityCategory.MEDICAL.value == "UTILITY_CATEGORY_MEDICAL"

    def test_privacy_level(self) -> None:
        from aethelred.core.types import PrivacyLevel
        assert PrivacyLevel.ZK.value == "zk"

    def test_validator_attestation(self) -> None:
        from aethelred.core.types import ValidatorAttestation
        va = ValidatorAttestation(
            validator_address="aeth1val",
            signature=b"sig",
            timestamp=datetime.now(timezone.utc),
        )
        assert va.validator_address == "aeth1val"

    def test_zkml_proof(self) -> None:
        from aethelred.core.types import ZKMLProof, ProofSystem
        zp = ZKMLProof(proof_system=ProofSystem.GROTH16, proof=b"proof_data")
        assert zp.proof_system == ProofSystem.GROTH16


# ============================================================
# utils - Additional utility coverage
# ============================================================


class TestAdditionalUtils:
    """Test additional utility functions."""

    def test_sha256_alias(self) -> None:
        from aethelred.utils import sha256
        result = sha256(b"hello")
        assert len(result) == 32
        assert isinstance(result, bytes)

    def test_keccak256(self) -> None:
        from aethelred.utils import keccak256
        result = keccak256(b"hello")
        assert len(result) == 32
        assert isinstance(result, bytes)

    def test_keccak256_string(self) -> None:
        from aethelred.utils import keccak256
        result = keccak256("hello")
        assert len(result) == 32

    def test_to_uaeth(self) -> None:
        from aethelred.utils import to_uaeth
        assert to_uaeth(1) == 1_000_000
        assert to_uaeth(0.5) == 500_000

    def test_from_uaeth(self) -> None:
        from aethelred.utils import from_uaeth
        assert from_uaeth(1_000_000) == 1.0
        assert from_uaeth(500_000) == 0.5

    def test_format_aeth(self) -> None:
        from aethelred.utils import format_aeth
        assert "AETH" in format_aeth(1_000_000)
        assert "1.000000" in format_aeth(1_000_000)

    def test_is_valid_address(self) -> None:
        from aethelred.utils import is_valid_address
        assert is_valid_address("aeth1qpzry9x8gf2tvdw")
        assert not is_valid_address("invalid")
        assert not is_valid_address("eth1short")
        assert not is_valid_address("")
        assert not is_valid_address(123)  # type: ignore

    def test_encode_decode_base64(self) -> None:
        from aethelred.utils import encode_base64, decode_base64
        data = b"hello world"
        encoded = encode_base64(data)
        assert isinstance(encoded, str)
        decoded = decode_base64(encoded)
        assert decoded == data

    def test_format_size_tib(self) -> None:
        from aethelred.utils import format_size
        result = format_size(1024 ** 4)
        assert "TiB" in result

    def test_sha256_hex_string_input(self) -> None:
        from aethelred.utils import sha256_hex
        h = sha256_hex("test string")
        assert len(h) == 64

    def test_sha256_bytes_string_input(self) -> None:
        from aethelred.utils import sha256_bytes
        d = sha256_bytes("test string")
        assert len(d) == 32


# ============================================================
# crypto/fallback.py - BackendInfo, protocol checks
# ============================================================


class TestBackendInfoDetails:
    """Test BackendInfo and protocol details."""

    def test_backend_info_str_fips(self) -> None:
        from aethelred.crypto.fallback import BackendInfo, BackendVariant
        bi = BackendInfo(
            name="test", variant=BackendVariant.PURE_PYTHON,
            version="1.0", fips_compliant=True, constant_time=True,
        )
        s = str(bi)
        assert "FIPS" in s

    def test_backend_info_str_non_fips(self) -> None:
        from aethelred.crypto.fallback import BackendInfo, BackendVariant
        bi = BackendInfo(
            name="test", variant=BackendVariant.PURE_PYTHON,
            version="1.0", fips_compliant=False, constant_time=False,
        )
        s = str(bi)
        assert "non-FIPS" in s

    def test_backend_variant_values(self) -> None:
        from aethelred.crypto.fallback import BackendVariant
        assert BackendVariant.NATIVE_LIBOQS.value == "native-liboqs"
        assert BackendVariant.PURE_PYTHON.value == "pure-python"

    def test_signer_protocol_check(self) -> None:
        from aethelred.crypto.fallback import SignerProtocol, HybridSigner
        signer = HybridSigner()
        assert isinstance(signer._impl, SignerProtocol)

    def test_hybrid_signer_sign_and_verify_roundtrip(self) -> None:
        from aethelred.crypto.fallback import HybridSigner
        signer = HybridSigner()
        msg = b"test roundtrip message"
        sig = signer.sign(msg)
        assert signer.verify(msg, sig)

    def test_hybrid_signer_fingerprint_consistency(self) -> None:
        from aethelred.crypto.fallback import HybridSigner
        signer = HybridSigner()
        fp1 = signer.fingerprint
        fp2 = signer.fingerprint
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_hybrid_verifier_roundtrip(self) -> None:
        from aethelred.crypto.fallback import HybridSigner, HybridVerifier
        signer = HybridSigner()
        msg = b"verify this"
        sig = signer.sign(msg)
        verifier = HybridVerifier(signer.public_key_bytes())
        assert verifier.verify(msg, sig)

    def test_tampered_sig_in_signer(self) -> None:
        from aethelred.crypto.fallback import HybridSigner
        signer = HybridSigner()
        sig = bytearray(signer.sign(b"data"))
        sig[-1] ^= 0xFF
        assert not signer.verify(b"data", bytes(sig))


# ============================================================
# crypto/pqc/kyber.py - KyberKeyPair, KyberCiphertext, utils
# ============================================================


class TestKyberExtended:
    """Extended tests for Kyber module."""

    def test_kyber_keypair_methods(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem = KyberKEM()
        kp = kem.keypair()
        assert len(kp.public_key_hex()) > 0
        assert len(kp.secret_key_hex()) > 0
        assert len(kp.fingerprint()) == 16

    def test_kyber_ciphertext_methods(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem = KyberKEM()
        ct, ss = kem.encapsulate()
        assert len(ct.hex()) > 0
        assert ct.to_bytes() == ct.ciphertext
        assert len(ct) > 0

    def test_kyber_encapsulate_to(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem = KyberKEM()
        ct, ss = KyberKEM.encapsulate_to(kem.public_key_bytes())
        derived = kem.decapsulate(ct)
        assert ss == derived

    def test_kyber_from_keypair(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem1 = KyberKEM()
        kp = kem1.keypair()
        kem2 = KyberKEM.from_keypair(kp)
        assert kem2.public_key_bytes() == kem1.public_key_bytes()

    def test_kyber_from_hex(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem1 = KyberKEM()
        kem2 = KyberKEM.from_hex(
            kem1.secret_key_bytes().hex(),
            kem1.public_key_bytes().hex(),
        )
        assert kem2.public_key_bytes() == kem1.public_key_bytes()

    def test_kyber_key_sizes(self) -> None:
        from aethelred.crypto.pqc.kyber import kyber_key_sizes, KyberSecurityLevel
        sizes = kyber_key_sizes(KyberSecurityLevel.LEVEL3)
        assert sizes["public_key"] == 1184
        assert sizes["shared_secret"] == 32

    def test_is_valid_kyber_public_key(self) -> None:
        from aethelred.crypto.pqc.kyber import is_valid_kyber_public_key, KyberKEM
        kem = KyberKEM()
        assert is_valid_kyber_public_key(kem.public_key_bytes())
        assert not is_valid_kyber_public_key(b"short")

    def test_is_valid_kyber_ciphertext(self) -> None:
        from aethelred.crypto.pqc.kyber import is_valid_kyber_ciphertext, KyberKEM
        kem = KyberKEM()
        ct, _ = kem.encapsulate()
        assert is_valid_kyber_ciphertext(ct.ciphertext)
        assert not is_valid_kyber_ciphertext(b"short")

    def test_hybrid_key_exchange(self) -> None:
        from aethelred.crypto.pqc.kyber import hybrid_key_exchange
        classical = b"\x01" * 32
        pqc = b"\x02" * 32
        combined = hybrid_key_exchange(classical, pqc)
        assert len(combined) == 32
        assert combined != classical
        assert combined != pqc

    def test_kyber_invalid_public_key_size(self) -> None:
        from aethelred.crypto.pqc.kyber import encapsulate_kyber, KyberSecurityLevel
        with pytest.raises(ValueError, match="Invalid public key size"):
            encapsulate_kyber(b"short", KyberSecurityLevel.LEVEL3)

    def test_kyber_invalid_ciphertext_size(self) -> None:
        from aethelred.crypto.pqc.kyber import decapsulate_kyber, KyberKEM, KyberSecurityLevel
        kem = KyberKEM()
        with pytest.raises(ValueError, match="Invalid ciphertext size"):
            decapsulate_kyber(b"short", kem.secret_key_bytes(), KyberSecurityLevel.LEVEL3)

    def test_kyber_invalid_secret_key_size(self) -> None:
        from aethelred.crypto.pqc.kyber import decapsulate_kyber, KyberSecurityLevel, KYBER_SIZES
        ct_size = KYBER_SIZES[KyberSecurityLevel.LEVEL3]["ciphertext"]
        with pytest.raises(ValueError, match="Invalid secret key size"):
            decapsulate_kyber(b"\x00" * ct_size, b"short", KyberSecurityLevel.LEVEL3)

    def test_kyber_all_levels(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM, KyberSecurityLevel
        for level in KyberSecurityLevel:
            kem = KyberKEM(level=level)
            ct, ss_enc = kem.encapsulate()
            ss_dec = kem.decapsulate(ct)
            assert ss_enc == ss_dec

    def test_kyber_decapsulate_raw_bytes(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem = KyberKEM()
        ct, ss_enc = kem.encapsulate()
        ss_dec = kem.decapsulate(ct.ciphertext)
        assert ss_enc == ss_dec

    def test_kyber_fingerprint(self) -> None:
        from aethelred.crypto.pqc.kyber import KyberKEM
        kem = KyberKEM()
        fp = kem.fingerprint
        assert len(fp) == 16
        assert isinstance(fp, str)


# ============================================================
# crypto/pqc/dilithium.py - Extended tests
# ============================================================


class TestDilithiumExtended:
    """Extended tests for Dilithium module."""

    def test_dilithium_keypair_methods(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        kp = signer.keypair()
        assert len(kp.public_key_hex()) > 0
        assert len(kp.secret_key_hex()) > 0
        assert len(kp.fingerprint()) == 16

    def test_dilithium_signature_methods(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        sig = signer.sign(b"hello")
        assert len(sig.hex()) > 0
        assert sig.to_bytes() == sig.signature
        assert len(sig) > 0

    def test_dilithium_from_hex(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer1 = DilithiumSigner()
        signer2 = DilithiumSigner.from_hex(
            signer1.secret_key_bytes().hex(),
            signer1.public_key_bytes().hex(),
        )
        assert signer2.public_key_bytes() == signer1.public_key_bytes()

    def test_dilithium_key_sizes(self) -> None:
        from aethelred.crypto.pqc.dilithium import dilithium_key_sizes, DilithiumSecurityLevel
        sizes = dilithium_key_sizes(DilithiumSecurityLevel.LEVEL3)
        assert sizes["public_key"] == 1952
        assert sizes["signature"] == 3309

    def test_is_valid_dilithium_public_key(self) -> None:
        from aethelred.crypto.pqc.dilithium import is_valid_dilithium_public_key, DilithiumSigner
        signer = DilithiumSigner()
        assert is_valid_dilithium_public_key(signer.public_key_bytes())
        assert not is_valid_dilithium_public_key(b"short")

    def test_is_valid_dilithium_signature(self) -> None:
        from aethelred.crypto.pqc.dilithium import is_valid_dilithium_signature, DilithiumSigner
        signer = DilithiumSigner()
        sig = signer.sign(b"data")
        assert is_valid_dilithium_signature(sig.signature)
        assert not is_valid_dilithium_signature(b"short")

    def test_dilithium_sign_string(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        sig = signer.sign("string message")
        assert signer.verify("string message", sig)

    def test_dilithium_verify_with_public_key(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        msg = b"test verify with pk"
        sig = signer.sign(msg)
        assert DilithiumSigner.verify_with_public_key(
            msg, sig, signer.public_key_bytes()
        )

    def test_dilithium_verify_wrong_signature_size(self) -> None:
        from aethelred.crypto.pqc.dilithium import verify_dilithium, DilithiumSecurityLevel
        signer_module = __import__("aethelred.crypto.pqc.dilithium", fromlist=["DilithiumSigner"])
        signer = signer_module.DilithiumSigner()
        assert not signer_module.verify_dilithium(
            b"msg", b"short_sig", signer.public_key_bytes(), DilithiumSecurityLevel.LEVEL3
        )

    def test_dilithium_verify_wrong_pk_size(self) -> None:
        from aethelred.crypto.pqc.dilithium import verify_dilithium, DilithiumSecurityLevel, DILITHIUM_SIZES
        sizes = DILITHIUM_SIZES[DilithiumSecurityLevel.LEVEL3]
        assert not verify_dilithium(
            b"msg", b"\x00" * sizes["signature"], b"short_pk", DilithiumSecurityLevel.LEVEL3
        )

    def test_dilithium_fingerprint(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        fp = signer.fingerprint
        assert len(fp) == 16
        assert isinstance(fp, str)

    def test_dilithium_secret_key_bytes(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        sk = signer.secret_key_bytes()
        assert isinstance(sk, bytes)
        assert len(sk) > 0

    def test_dilithium_signer_fingerprint_on_signature(self) -> None:
        from aethelred.crypto.pqc.dilithium import DilithiumSigner
        signer = DilithiumSigner()
        sig = signer.sign(b"data")
        assert sig.signer_fingerprint == signer.fingerprint


# ============================================================
# proofs/verifier.py - TEEVerifier, ZKVerifier, ProofVerifier
# ============================================================


class TestTEEVerifier:
    """Test TEE attestation verification."""

    def test_verify_sgx(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            quote=b"quote",
            timestamp=datetime.now(timezone.utc),
        )
        result = verifier.verify(att)
        assert result.is_valid

    def test_verify_amd_sev(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.AMD_SEV,
            timestamp=datetime.now(timezone.utc),
        )
        result = verifier.verify(att)
        assert result.is_valid

    def test_verify_amd_sev_snp(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.AMD_SEV_SNP,
            timestamp=datetime.now(timezone.utc),
            snp_report=b"snp_report_data",
        )
        result = verifier.verify(att)
        assert result.is_valid

    def test_verify_intel_tdx(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_TDX,
            timestamp=datetime.now(timezone.utc),
        )
        result = verifier.verify(att)
        assert result.is_valid

    def test_verify_nitro_with_pcrs(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.AWS_NITRO,
            timestamp=datetime.now(timezone.utc),
            pcr_values={0: b"pcr0", 1: b"pcr1", 2: b"pcr2"},
        )
        result = verifier.verify(att)
        assert result.is_valid

    def test_verify_expired_attestation(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier(max_attestation_age_hours=1)
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        result = verifier.verify(att)
        assert not result.is_valid
        assert result.status.value == "expired"

    def test_verify_unsupported_platform(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.ARM_TRUSTZONE,
            timestamp=datetime.now(timezone.utc),
        )
        result = verifier.verify(att)
        assert not result.is_valid

    def test_verify_with_measurement(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            timestamp=datetime.now(timezone.utc),
            measurement="expected_hash",
        )
        result = verifier.verify(att, expected_measurement="expected_hash")
        assert result.is_valid

    def test_verify_measurement_mismatch(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import TEEVerifier
        verifier = TEEVerifier()
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            timestamp=datetime.now(timezone.utc),
            measurement="actual_hash",
        )
        result = verifier.verify(att, expected_measurement="different_hash")
        assert not result.is_valid


class TestZKVerifier:
    """Test ZK proof verification."""

    def test_verify_valid_proof(self) -> None:
        from aethelred.core.types import ZKProof
        from aethelred.proofs.verifier import ZKVerifier
        verifier = ZKVerifier()
        vk = b"verification_key_data"
        proof = ZKProof(
            proof_type="groth16",
            proof_bytes=b"proof_data",
            public_inputs=["input1"],
            verification_key_hash=hashlib.sha256(vk).hexdigest(),
        )
        result = verifier.verify(proof, vk)
        assert result.is_valid

    def test_verify_empty_proof(self) -> None:
        from aethelred.core.types import ZKProof
        from aethelred.proofs.verifier import ZKVerifier
        verifier = ZKVerifier()
        proof = ZKProof(proof_type="groth16", proof_bytes=b"")
        result = verifier.verify(proof, b"vk")
        assert not result.is_valid

    def test_verify_vk_mismatch(self) -> None:
        from aethelred.core.types import ZKProof
        from aethelred.proofs.verifier import ZKVerifier
        verifier = ZKVerifier()
        proof = ZKProof(
            proof_type="groth16",
            proof_bytes=b"proof",
            public_inputs=["i1"],
            verification_key_hash="wrong_hash",
        )
        result = verifier.verify(proof, b"vk")
        assert not result.is_valid


class TestProofVerifier:
    """Test unified ProofVerifier."""

    def test_verify_none_proof(self) -> None:
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        result = verifier.verify(None)
        assert not result.is_valid

    def test_verify_zkml_proof(self) -> None:
        from aethelred.core.types import Proof, ProofType, ZKProof, Circuit
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        vk = b"verification_key"
        circuit = Circuit(verification_key=vk)
        zk_proof = ZKProof(
            proof_type="groth16",
            proof_bytes=b"proof",
            public_inputs=["i1"],
            verification_key_hash=hashlib.sha256(vk).hexdigest(),
        )
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.ZKML,
            zk_proof=zk_proof,
        )
        result = verifier.verify(proof, circuit)
        assert result.is_valid

    def test_verify_tee_proof(self) -> None:
        from aethelred.core.types import Proof, ProofType, TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.TEE,
            tee_attestation=TEEAttestation(
                platform=TEEPlatform.INTEL_SGX,
                timestamp=datetime.now(timezone.utc),
            ),
        )
        result = verifier.verify(proof)
        assert result.is_valid

    def test_verify_tee_proof_missing_attestation(self) -> None:
        from aethelred.core.types import Proof, ProofType
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        proof = Proof(proof_id="p1", proof_type=ProofType.TEE)
        result = verifier.verify(proof)
        assert not result.is_valid

    def test_verify_zkml_missing_proof(self) -> None:
        from aethelred.core.types import Proof, ProofType, Circuit
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        circuit = Circuit(verification_key=b"vk")
        proof = Proof(proof_id="p1", proof_type=ProofType.ZKML)
        result = verifier.verify(proof, circuit)
        assert not result.is_valid

    def test_verify_zkml_missing_vk(self) -> None:
        from aethelred.core.types import Proof, ProofType, ZKProof
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.ZKML,
            zk_proof=ZKProof(proof_bytes=b"p", public_inputs=["i"]),
        )
        result = verifier.verify(proof)
        assert not result.is_valid

    def test_verify_hybrid_proof(self) -> None:
        from aethelred.core.types import Proof, ProofType, ZKProof, TEEAttestation, TEEPlatform, Circuit
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        vk = b"verification_key"
        circuit = Circuit(verification_key=vk)
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.HYBRID,
            zk_proof=ZKProof(
                proof_bytes=b"p",
                public_inputs=["i"],
                verification_key_hash=hashlib.sha256(vk).hexdigest(),
            ),
            tee_attestation=TEEAttestation(
                platform=TEEPlatform.INTEL_SGX,
                timestamp=datetime.now(timezone.utc),
            ),
        )
        result = verifier.verify(proof, circuit)
        assert result.is_valid

    def test_verify_hybrid_missing_tee(self) -> None:
        from aethelred.core.types import Proof, ProofType, ZKProof, Circuit
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        vk = b"vk"
        circuit = Circuit(verification_key=vk)
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.HYBRID,
            zk_proof=ZKProof(
                proof_bytes=b"p",
                public_inputs=["i"],
                verification_key_hash=hashlib.sha256(vk).hexdigest(),
            ),
        )
        result = verifier.verify(proof, circuit)
        assert not result.is_valid

    def test_verify_unspecified_proof_type(self) -> None:
        from aethelred.core.types import Proof, ProofType
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        proof = Proof(proof_id="p1", proof_type=ProofType.UNSPECIFIED)
        result = verifier.verify(proof)
        assert not result.is_valid

    def test_verify_with_hash_checks(self) -> None:
        from aethelred.core.types import Proof, ProofType, TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.TEE,
            input_hash="input_abc",
            output_hash="output_xyz",
            model_hash="model_123",
            tee_attestation=TEEAttestation(
                platform=TEEPlatform.INTEL_SGX,
                timestamp=datetime.now(timezone.utc),
                report_data=b"output_xyz",
            ),
        )
        result = verifier.verify(
            proof,
            expected_input_hash="input_abc",
            expected_output_hash="output_xyz",
            expected_model_hash="model_123",
        )
        assert result.is_valid

    def test_verify_input_hash_mismatch(self) -> None:
        from aethelred.core.types import Proof, ProofType
        from aethelred.proofs.verifier import ProofVerifier
        verifier = ProofVerifier()
        proof = Proof(proof_id="p1", proof_type=ProofType.TEE, input_hash="abc")
        result = verifier.verify(proof, expected_input_hash="xyz")
        assert not result.is_valid

    def test_verification_result_to_dict(self) -> None:
        from aethelred.proofs.verifier import VerificationResult, VerificationStatus
        vr = VerificationResult(
            is_valid=True,
            status=VerificationStatus.VALID,
            proof_id="p1",
        )
        d = vr.to_dict()
        assert d["is_valid"] is True
        assert d["status"] == "valid"

    def test_verify_proof_convenience(self) -> None:
        from aethelred.core.types import Proof, ProofType, TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import verify_proof
        proof = Proof(
            proof_id="p1",
            proof_type=ProofType.TEE,
            tee_attestation=TEEAttestation(
                platform=TEEPlatform.INTEL_SGX,
                timestamp=datetime.now(timezone.utc),
            ),
        )
        result = verify_proof(proof)
        assert result.is_valid

    def test_verify_tee_attestation_convenience(self) -> None:
        from aethelred.core.types import TEEAttestation, TEEPlatform
        from aethelred.proofs.verifier import verify_tee_attestation
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            timestamp=datetime.now(timezone.utc),
        )
        result = verify_tee_attestation(att)
        assert result.is_valid


# ============================================================
# proofs/generator.py - Extended ProofGenerator tests
# ============================================================


class TestProofGeneratorExtended:
    """Extended tests for ProofGenerator."""

    def test_generate_groth16(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator, ProofSystem
        gen = ProofGenerator(default_proof_system=ProofSystem.GROTH16)
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input_data")
        assert result.proof.proof_type.value == "PROOF_TYPE_ZKML"

    def test_generate_plonk(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator, ProofSystem
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input", proof_system=ProofSystem.PLONK)
        assert result is not None

    def test_generate_stark(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator, ProofSystem
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input", proof_system=ProofSystem.STARK)
        assert result is not None

    def test_generate_ezkl(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator, ProofSystem
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input", proof_system=ProofSystem.EZKL)
        assert result is not None

    def test_generate_with_string_proof_system(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input", proof_system="PROOF_SYSTEM_GROTH16")
        assert result is not None

    def test_generate_batch(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        results = gen.generate_batch(circuit, [b"input1", b"input2"])
        assert len(results) == 2

    def test_generate_hybrid(self) -> None:
        from aethelred.core.types import Circuit, TEEAttestation, TEEPlatform
        from aethelred.proofs.generator import ProofGenerator
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        tee = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX,
            timestamp=datetime.now(timezone.utc),
        )
        result = gen.generate_hybrid(circuit, b"input", tee)
        assert result.proof.proof_type.value == "PROOF_TYPE_HYBRID"

    def test_estimate_proving_time(self) -> None:
        from aethelred.core.types import Circuit, CircuitMetrics
        from aethelred.proofs.generator import ProofGenerator, ProofSystem
        gen = ProofGenerator()
        circuit = Circuit(
            circuit_id="c1",
            metrics=CircuitMetrics(constraints=10000),
        )
        estimates = gen.estimate_proving_time(circuit, ProofSystem.GROTH16)
        assert "proving_time_ms" in estimates
        assert "proof_size_bytes" in estimates

    def test_proof_result_to_dict(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input")
        d = result.to_dict()
        assert "proof_id" in d
        assert "proving_time_ms" in d

    def test_create_proof_request(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import create_proof_request, ProofSystem
        circuit = Circuit(circuit_id="c1")
        req = create_proof_request(circuit, b"input", proof_system=ProofSystem.GROTH16)
        assert req.config.proof_system == ProofSystem.GROTH16

    def test_create_proof_request_with_string(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import create_proof_request
        circuit = Circuit(circuit_id="c1")
        req = create_proof_request(circuit, b"input", proof_system="GROTH16")
        assert req.circuit == circuit

    def test_proof_config_defaults(self) -> None:
        from aethelred.proofs.generator import ProofConfig
        pc = ProofConfig()
        assert pc.use_gpu is True
        assert pc.timeout_seconds == 300

    def test_proof_generator_with_config(self) -> None:
        from aethelred.proofs.generator import ProofGenerator, ProofConfig, ProofSystem
        config = ProofConfig(
            proof_system=ProofSystem.PLONK,
            use_gpu=False,
            num_threads=4,
        )
        gen = ProofGenerator(config=config)
        assert gen.default_proof_system == ProofSystem.PLONK

    def test_generate_with_output_data(self) -> None:
        from aethelred.core.types import Circuit
        from aethelred.proofs.generator import ProofGenerator
        gen = ProofGenerator()
        circuit = Circuit(circuit_id="c1")
        result = gen.generate(circuit, b"input", output_data=[0.9])
        assert result is not None


# ============================================================
# oracles/client.py - OracleClient and DataFeed
# ============================================================


class TestOracleClientExtended:
    """Extended tests for Oracle client module."""

    def test_data_feed_to_job_input(self) -> None:
        from aethelred.oracles.client import (
            DataFeed, FeedMetadata, AttestationRecord, AttestationMethod, FeedType,
        )
        from aethelred.core.types import DataProvenance, DataSourceType
        feed = DataFeed(
            did="did:aethelred:oracle:btc_usd",
            feed_id="btc_usd",
            metadata=FeedMetadata(
                feed_id="btc_usd", name="BTC/USD", description="",
                feed_type=FeedType.MARKET_DATA, value_type="number",
            ),
            value=42000.0,
            value_hash=hashlib.sha256(b'"42000.0"').hexdigest(),
            timestamp=datetime.now(timezone.utc),
            attestation=AttestationRecord(
                attestation_id="att1", feed_id="btc_usd",
                data_hash="abc", method=AttestationMethod.TEE,
                oracle_node_id="node1",
                timestamp=datetime.now(timezone.utc),
            ),
            provenance=DataProvenance(verification_method="tee"),
        )
        ji = feed.to_job_input()
        assert ji.did == "did:aethelred:oracle:btc_usd"
        assert ji.source_type == DataSourceType.ORACLE_DID

    def test_data_feed_verify_hash_bytes(self) -> None:
        from aethelred.oracles.client import (
            DataFeed, FeedMetadata, AttestationRecord, AttestationMethod, FeedType,
        )
        from aethelred.core.types import DataProvenance
        data = b"raw bytes"
        feed = DataFeed(
            did="did:aethelred:oracle:test",
            feed_id="test",
            metadata=FeedMetadata(
                feed_id="test", name="Test", description="",
                feed_type=FeedType.CUSTOM, value_type="binary",
            ),
            value=data,
            value_hash=hashlib.sha256(data).hexdigest(),
            timestamp=datetime.now(timezone.utc),
            attestation=AttestationRecord(
                attestation_id="att1", feed_id="test",
                data_hash="", method=AttestationMethod.TEE,
                oracle_node_id="node1",
                timestamp=datetime.now(timezone.utc),
            ),
            provenance=DataProvenance(),
        )
        assert feed.verify_hash()

    def test_feed_type_enum(self) -> None:
        from aethelred.oracles.client import FeedType
        assert FeedType.MARKET_DATA.value == "market_data"
        assert FeedType.WEATHER.value == "weather"

    def test_attestation_method_enum(self) -> None:
        from aethelred.oracles.client import AttestationMethod
        assert AttestationMethod.TEE.value == "tee"
        assert AttestationMethod.ZK.value == "zk"

    def test_feed_subscription_init(self) -> None:
        from aethelred.oracles.client import FeedSubscription, OracleClient
        mock_client = MagicMock()
        oracle = OracleClient(mock_client)
        sub = FeedSubscription(
            oracle_client=oracle,
            feed_id="test_feed",
            callback=lambda f: None,
        )
        assert sub.feed_id == "test_feed"
        assert not sub.is_active

    def test_oracle_client_list_feeds(self) -> None:
        from aethelred.oracles.client import OracleClient
        mock_client = MagicMock()
        mock_client._request.return_value = {"feeds": []}
        oracle = OracleClient(mock_client)
        feeds = oracle.list_feeds()
        assert feeds == []

    def test_oracle_client_get_oracle_nodes(self) -> None:
        from aethelred.oracles.client import OracleClient
        mock_client = MagicMock()
        mock_client._request.return_value = {
            "nodes": [{
                "node_id": "n1", "name": "Node1",
                "operator": "op1", "endpoint": "http://node1",
                "supported_feeds": ["btc"],
                "tee_platform": "TEE_PLATFORM_INTEL_SGX",
                "stake": 100, "uptime": 0.99,
                "is_active": True,
            }]
        }
        oracle = OracleClient(mock_client)
        nodes = oracle.get_oracle_nodes()
        assert len(nodes) == 1
        assert nodes[0].node_id == "n1"

    def test_oracle_client_get_historical_values(self) -> None:
        from aethelred.oracles.client import OracleClient
        mock_client = MagicMock()
        mock_client._request.return_value = {"values": [{"value": 42}]}
        oracle = OracleClient(mock_client)
        values = oracle.get_historical_values(
            "btc_usd",
            start_time=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert len(values) == 1

    def test_oracle_client_verify_attestation(self) -> None:
        from aethelred.oracles.client import (
            OracleClient, DataFeed, FeedMetadata, AttestationRecord,
            AttestationMethod, FeedType,
        )
        from aethelred.core.types import DataProvenance
        mock_client = MagicMock()
        mock_client._request.return_value = {"is_valid": True}
        oracle = OracleClient(mock_client)
        feed = DataFeed(
            did="did:aethelred:oracle:test",
            feed_id="test",
            metadata=FeedMetadata(
                feed_id="test", name="Test", description="",
                feed_type=FeedType.CUSTOM, value_type="binary",
            ),
            value=b"data",
            value_hash=hashlib.sha256(b"data").hexdigest(),
            timestamp=datetime.now(timezone.utc),
            attestation=AttestationRecord(
                attestation_id="att1", feed_id="test",
                data_hash="", method=AttestationMethod.TEE,
                oracle_node_id="node1",
                timestamp=datetime.now(timezone.utc),
            ),
            provenance=DataProvenance(),
        )
        result = oracle.verify_attestation(feed)
        assert result is True

    def test_async_oracle_client_create_trusted_input_from_did(self) -> None:
        from aethelred.oracles.client import AsyncOracleClient
        mock_client = MagicMock()
        oracle = AsyncOracleClient(mock_client)
        ji = oracle.create_trusted_input("did:aethelred:oracle:btc")
        assert ji.did == "did:aethelred:oracle:btc"


# ============================================================
# models/registry.py - ModelStatus, RegisteredModel
# ============================================================


class TestModelRegistry:
    """Test model registry types."""

    def test_model_status_enum(self) -> None:
        from aethelred.models.registry import ModelStatus
        assert ModelStatus.ACTIVE.value == "active"
        assert ModelStatus.DEPRECATED.value == "deprecated"

    def test_registered_model(self) -> None:
        from aethelred.models.registry import RegisteredModel
        rm = RegisteredModel(
            model_id="m1",
            name="test-model",
            version="1.0.0",
            owner="aeth1owner",
            model_hash="abc",
            circuit_hash="def",
            verification_key_hash="ghi",
        )
        assert rm.model_id == "m1"
        assert rm.circuit is None


# ============================================================
# Wallet additional coverage
# ============================================================


class TestWalletAdditional:
    """Additional wallet tests for coverage."""

    def test_wallet_get_public_keys(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        pks = wallet.get_public_keys()
        assert "classical" in pks
        assert "quantum" in pks
        assert "kem" in pks

    def test_wallet_get_fingerprints(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        fps = wallet.get_fingerprints()
        assert len(fps["classical"]) == 16
        assert len(fps["quantum"]) == 16
        assert len(fps["kem"]) == 16

    def test_wallet_sign_message_string(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        sig = wallet.sign_message("hello world")
        assert sig is not None
        assert sig.signer_address == wallet.address

    def test_wallet_sign_message_bytes(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        sig = wallet.sign_message(b"hello bytes")
        assert sig is not None

    def test_composite_signature_to_dict(self) -> None:
        from aethelred.core.wallet import CompositeSignature
        cs = CompositeSignature(classical_sig=b"ecdsa", pqc_sig=b"dil")
        d = cs.to_dict()
        assert "scheme" in d
        assert "classical_sig" in d

    def test_composite_signature_len(self) -> None:
        from aethelred.core.wallet import CompositeSignature
        cs = CompositeSignature(classical_sig=b"ecdsa", pqc_sig=b"dil")
        assert len(cs) == len(b"ecdsa") + len(b"dil")

    def test_composite_signature_from_bytes_too_short(self) -> None:
        from aethelred.core.wallet import CompositeSignature
        with pytest.raises(ValueError, match="too short"):
            CompositeSignature.from_bytes(b"\x02\x00")

    def test_composite_signature_from_bytes_bad_marker(self) -> None:
        from aethelred.core.wallet import CompositeSignature
        with pytest.raises(ValueError, match="marker"):
            CompositeSignature.from_bytes(b"\x01\x00\x00\x00\x00\x00")

    def test_ecdsa_signer_verify_wrong_sig(self) -> None:
        from aethelred.core.wallet import ECDSASigner
        signer = ECDSASigner()
        sig = signer.sign(b"data")
        assert not signer.verify(b"data", b"wrong_sig_" + b"\x00" * 54)

    def test_ecdsa_signer_invalid_key_size(self) -> None:
        from aethelred.core.wallet import ECDSASigner
        with pytest.raises(ValueError, match="32 bytes"):
            ECDSASigner(private_key=b"short")

    def test_bech32_encode(self) -> None:
        from aethelred.core.wallet import bech32_encode
        result = bech32_encode("aeth", b"\x00" * 20)
        assert result.startswith("aeth1")

    def test_wallet_close(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        wallet.close()
        assert wallet._closed is True
        # Double close is safe
        wallet.close()

    def test_wallet_export_password_too_short(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        with pytest.raises(ValueError, match="at least"):
            wallet.export_keys("short")

    def test_wallet_from_export_missing_password(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        with pytest.raises(ValueError, match="Password is required"):
            DualKeyWallet.from_export({"salt": "aa" * 16, "ciphertext": "x"}, "")

    def test_signature_scheme_enum(self) -> None:
        from aethelred.core.wallet import SignatureScheme
        assert SignatureScheme.COMPOSITE.value == "composite"

    def test_ecdsa_keypair(self) -> None:
        from aethelred.core.wallet import ECDSAKeyPair
        kp = ECDSAKeyPair(public_key=b"\x02" + b"\x00" * 32, private_key=b"\x01" * 32)
        assert len(kp.public_key_hex()) > 0
        assert len(kp.private_key_hex()) > 0

    def test_verify_with_public_keys_static(self) -> None:
        from aethelred.core.wallet import DualKeyWallet
        wallet = DualKeyWallet()
        msg = b"test message"
        sig = wallet.sign_transaction(msg)
        pks = wallet.get_public_keys()
        is_valid = DualKeyWallet.verify_with_public_keys(
            msg, sig, pks["classical"], pks["quantum"]
        )
        assert is_valid

    def test_create_wallet_function(self) -> None:
        from aethelred.core.wallet import create_wallet
        wallet = create_wallet()
        assert wallet.address.startswith("aeth1")

    def test_address_from_public_keys(self) -> None:
        from aethelred.core.wallet import DualKeyWallet, address_from_public_keys
        wallet = DualKeyWallet()
        pks = wallet.get_public_keys()
        addr = address_from_public_keys(pks["classical"], pks["quantum"])
        assert addr == wallet.address

    def test_verify_composite_signature_function(self) -> None:
        from aethelred.core.wallet import DualKeyWallet, verify_composite_signature
        wallet = DualKeyWallet()
        msg = b"verify func test"
        sig = wallet.sign_transaction(msg)
        pks = wallet.get_public_keys()
        assert verify_composite_signature(msg, sig, pks["classical"], pks["quantum"])
