"""
Core types for Aethelred SDK.

This module defines all the data models used across the SDK,
including jobs, seals, models, validators, and verification types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ============ Basic Types ============

Address = str  # Aethelred address (aethel1...)
Hash = bytes  # 32-byte hash
TxHash = str  # Transaction hash


# ============ Enums ============


class JobStatus(str, Enum):
    """Status of a compute job."""
    UNSPECIFIED = "JOB_STATUS_UNSPECIFIED"
    PENDING = "JOB_STATUS_PENDING"
    ASSIGNED = "JOB_STATUS_ASSIGNED"
    COMPUTING = "JOB_STATUS_COMPUTING"
    VERIFYING = "JOB_STATUS_VERIFYING"
    COMPLETED = "JOB_STATUS_COMPLETED"
    FAILED = "JOB_STATUS_FAILED"
    CANCELLED = "JOB_STATUS_CANCELLED"
    EXPIRED = "JOB_STATUS_EXPIRED"


class SealStatus(str, Enum):
    """Status of a digital seal."""
    UNSPECIFIED = "SEAL_STATUS_UNSPECIFIED"
    ACTIVE = "SEAL_STATUS_ACTIVE"
    REVOKED = "SEAL_STATUS_REVOKED"
    EXPIRED = "SEAL_STATUS_EXPIRED"
    SUPERSEDED = "SEAL_STATUS_SUPERSEDED"


class ProofType(str, Enum):
    """Type of cryptographic proof."""
    UNSPECIFIED = "PROOF_TYPE_UNSPECIFIED"
    TEE = "PROOF_TYPE_TEE"
    ZKML = "PROOF_TYPE_ZKML"
    HYBRID = "PROOF_TYPE_HYBRID"
    OPTIMISTIC = "PROOF_TYPE_OPTIMISTIC"


class ProofSystem(str, Enum):
    """Zero-knowledge proof system."""
    UNSPECIFIED = "PROOF_SYSTEM_UNSPECIFIED"
    GROTH16 = "PROOF_SYSTEM_GROTH16"
    PLONK = "PROOF_SYSTEM_PLONK"
    STARK = "PROOF_SYSTEM_STARK"
    EZKL = "PROOF_SYSTEM_EZKL"


class TEEPlatform(str, Enum):
    """Trusted Execution Environment platform."""
    UNSPECIFIED = "TEE_PLATFORM_UNSPECIFIED"
    INTEL_SGX = "TEE_PLATFORM_INTEL_SGX"
    INTEL_TDX = "TEE_PLATFORM_INTEL_TDX"
    AMD_SEV = "TEE_PLATFORM_AMD_SEV"
    AMD_SEV_SNP = "TEE_PLATFORM_AMD_SEV_SNP"
    AWS_NITRO = "TEE_PLATFORM_AWS_NITRO"
    ARM_TRUSTZONE = "TEE_PLATFORM_ARM_TRUSTZONE"


class UtilityCategory(str, Enum):
    """Category of useful work for reward multipliers."""
    UNSPECIFIED = "UTILITY_CATEGORY_UNSPECIFIED"
    MEDICAL = "UTILITY_CATEGORY_MEDICAL"
    SCIENTIFIC = "UTILITY_CATEGORY_SCIENTIFIC"
    FINANCIAL = "UTILITY_CATEGORY_FINANCIAL"
    LEGAL = "UTILITY_CATEGORY_LEGAL"
    EDUCATIONAL = "UTILITY_CATEGORY_EDUCATIONAL"
    ENVIRONMENTAL = "UTILITY_CATEGORY_ENVIRONMENTAL"
    GENERAL = "UTILITY_CATEGORY_GENERAL"


class PrivacyLevel(str, Enum):
    """Privacy level for computation."""
    NONE = "none"
    TEE = "tee"
    ZK = "zk"
    HYBRID = "hybrid"


class ComplianceFramework(str, Enum):
    """Regulatory compliance framework."""
    HIPAA = "HIPAA"
    GDPR = "GDPR"
    SOC2 = "SOC2"
    CCPA = "CCPA"
    PCI_DSS = "PCI_DSS"
    FERPA = "FERPA"


# ============ Pagination ============


class PageRequest(BaseModel):
    """Pagination request parameters."""
    key: Optional[bytes] = None
    offset: int = 0
    limit: int = 100
    count_total: bool = False
    reverse: bool = False


class PageResponse(BaseModel):
    """Pagination response metadata."""
    next_key: Optional[bytes] = None
    total: int = 0


# ============ Job Types ============


class ComputeJob(BaseModel):
    """A compute job on the Aethelred network."""
    id: str = Field(..., description="Unique job identifier")
    creator: Address = Field(..., description="Address of job creator")
    model_hash: bytes = Field(..., description="SHA-256 hash of the model")
    input_hash: bytes = Field(..., description="SHA-256 hash of input data")
    output_hash: Optional[bytes] = Field(None, description="SHA-256 hash of output")
    status: JobStatus = Field(default=JobStatus.PENDING)
    proof_type: ProofType = Field(default=ProofType.TEE)
    priority: int = Field(default=1, ge=1, le=10)
    max_gas: int = Field(default=0)
    timeout_blocks: int = Field(default=100)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    validator_address: Optional[Address] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class JobResult(BaseModel):
    """Result of a completed job."""
    job_id: str
    output_hash: bytes
    output_data: Optional[bytes] = None
    verified: bool = False
    consensus_validators: int = 0
    total_voting_power: int = 0


class SubmitJobRequest(BaseModel):
    """Request to submit a new compute job."""
    model_hash: bytes
    input_hash: bytes
    proof_type: ProofType = ProofType.TEE
    priority: int = 1
    max_gas: int = 0
    timeout_blocks: int = 100
    callback_url: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class SubmitJobResponse(BaseModel):
    """Response from submitting a job."""
    job_id: str
    tx_hash: TxHash
    estimated_blocks: int = 0


# ============ Seal Types ============


class RegulatoryInfo(BaseModel):
    """Regulatory compliance information for a seal."""
    jurisdiction: str = ""
    compliance_frameworks: List[str] = Field(default_factory=list)
    data_classification: str = ""
    retention_period: str = ""
    audit_trail_hash: Optional[bytes] = None


class ValidatorAttestation(BaseModel):
    """Attestation from a validator."""
    validator_address: Address
    signature: bytes
    timestamp: datetime
    voting_power: int = 0


class TEEAttestation(BaseModel):
    """TEE attestation data."""
    platform: TEEPlatform = TEEPlatform.UNSPECIFIED
    quote: bytes = b""
    enclave_hash: bytes = b""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    pcr_values: Dict[Any, bytes] = Field(default_factory=dict)
    nonce: Optional[bytes] = None
    # Additional fields used by verifier
    measurement: Optional[str] = None
    report_data: Optional[bytes] = None
    snp_report: Optional[bytes] = None


class ZKMLProof(BaseModel):
    """Zero-knowledge ML proof."""
    proof_system: ProofSystem = ProofSystem.UNSPECIFIED
    proof: bytes = b""
    public_inputs: List[bytes] = Field(default_factory=list)
    verifying_key_hash: bytes = b""


class DigitalSeal(BaseModel):
    """A digital seal representing verified AI computation."""
    id: str = Field(..., description="Unique seal identifier")
    job_id: str = Field(..., description="Associated job ID")
    model_hash: bytes = Field(..., description="SHA-256 hash of the model")
    input_commitment: bytes = Field(default=b"")
    output_commitment: bytes = Field(default=b"")
    model_commitment: bytes = Field(default=b"")
    status: SealStatus = Field(default=SealStatus.ACTIVE)
    requester: Address = ""
    validators: List[ValidatorAttestation] = Field(default_factory=list)
    tee_attestation: Optional[TEEAttestation] = None
    zkml_proof: Optional[ZKMLProof] = None
    regulatory_info: Optional[RegulatoryInfo] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None


class CreateSealRequest(BaseModel):
    """Request to create a digital seal."""
    job_id: str
    regulatory_info: Optional[RegulatoryInfo] = None
    expires_in_blocks: int = 0
    metadata: Dict[str, str] = Field(default_factory=dict)


class CreateSealResponse(BaseModel):
    """Response from creating a seal."""
    seal_id: str
    tx_hash: TxHash


class VerifySealResponse(BaseModel):
    """Response from verifying a seal."""
    valid: bool
    seal: Optional[DigitalSeal] = None
    verification_details: Dict[str, bool] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


# ============ Model Types ============


class RegisteredModel(BaseModel):
    """A registered AI model on the network."""
    model_hash: bytes
    name: str
    owner: Address
    architecture: str = ""
    version: str = "1.0.0"
    category: UtilityCategory = UtilityCategory.GENERAL
    input_schema: str = ""
    output_schema: str = ""
    storage_uri: str = ""
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    verified: bool = False
    total_jobs: int = 0


class RegisterModelRequest(BaseModel):
    """Request to register a model."""
    model_hash: bytes
    name: str
    architecture: str = ""
    version: str = "1.0.0"
    category: UtilityCategory = UtilityCategory.GENERAL
    input_schema: str = ""
    output_schema: str = ""
    storage_uri: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)


class RegisterModelResponse(BaseModel):
    """Response from registering a model."""
    model_hash: str
    tx_hash: TxHash


# ============ Validator Types ============


class HardwareCapability(BaseModel):
    """Hardware capabilities of a validator."""
    tee_platforms: List[TEEPlatform] = Field(default_factory=list)
    zkml_supported: bool = False
    max_model_size_mb: int = 0
    gpu_memory_gb: int = 0
    cpu_cores: int = 0
    memory_gb: int = 0


class ValidatorStats(BaseModel):
    """Statistics for a validator."""
    address: Address
    jobs_completed: int = 0
    jobs_failed: int = 0
    average_latency_ms: int = 0
    uptime_percentage: float = 0.0
    reputation_score: float = 0.0
    total_rewards: str = "0"
    slashing_events: int = 0
    hardware_capabilities: Optional[HardwareCapability] = None


# ============ Verification Types ============


class VerificationResult(BaseModel):
    """Result of verifying a computation."""
    valid: bool
    output_hash: bytes
    proof_type: ProofType = ProofType.UNSPECIFIED
    tee_attestation: Optional[TEEAttestation] = None
    zkml_proof: Optional[ZKMLProof] = None
    consensus_validators: int = 0
    total_voting_power: int = 0


# ============ Network Types ============


class UsefulWorkStats(BaseModel):
    """Network-wide useful work statistics."""
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    total_useful_work_units: str = "0"
    jobs_by_category: Dict[str, int] = Field(default_factory=dict)
    active_validators: int = 0
    average_job_latency_ms: int = 0


class EpochStats(BaseModel):
    """Statistics for an epoch."""
    epoch: int
    start_block: int
    end_block: int
    total_rewards: str = "0"
    jobs_completed: int = 0
    validator_count: int = 0
    useful_work_units: str = "0"


class NodeInfo(BaseModel):
    """Information about a node."""
    default_node_id: str = ""
    listen_addr: str = ""
    network: str = ""
    version: str = ""
    moniker: str = ""


class BlockHeader(BaseModel):
    """Block header information."""
    height: int
    time: datetime
    chain_id: str


class Block(BaseModel):
    """Block information."""
    block_id: Dict[str, Any] = Field(default_factory=dict)
    header: Optional[BlockHeader] = None


# ============ SLA Types ============


class SLA(BaseModel):
    """Service Level Agreement for compute jobs."""
    privacy_level: PrivacyLevel = PrivacyLevel.NONE
    allowed_regions: List[str] = Field(default_factory=list)
    max_latency_ms: int = 0
    min_validators: int = 1
    compliance_frameworks: List[str] = Field(default_factory=list)


# ============ Circuit Types ============


class HardwareTarget(str, Enum):
    """Target hardware for compilation."""
    AUTO = "auto"
    NVIDIA_A100 = "nvidia_a100"
    NVIDIA_H100 = "nvidia_h100"
    NVIDIA_L40S = "nvidia_l40s"
    XILINX_U280 = "xilinx_u280"
    XILINX_U55C = "xilinx_u55c"
    INTEL_SGX = "intel_sgx"
    AMD_SEV_SNP = "amd_sev_snp"
    INTEL_TDX = "intel_tdx"
    AWS_NITRO = "aws_nitro"
    CPU = "cpu"


class CircuitMetrics(BaseModel):
    """Metrics for a compiled circuit."""
    constraints: int = 0
    public_inputs: int = 0
    private_inputs: int = 0
    gates: int = 0
    depth: int = 0
    memory_bytes: int = 0
    estimated_proving_time_ms: int = 0


class HardwareRequirements(BaseModel):
    """Hardware requirements for circuit execution."""
    min_memory_gb: int = 0
    min_gpu_memory_gb: int = 0
    min_cpu_cores: int = 1
    target: HardwareTarget = HardwareTarget.AUTO


class Circuit(BaseModel):
    """Compiled arithmetic circuit for zkML."""
    circuit_id: str = ""
    model_hash: str = ""
    version: str = "1.0.0"
    circuit_binary: bytes = b""
    verification_key: bytes = b""
    proving_key_hash: str = ""
    input_shape: tuple = ()
    output_shape: tuple = ()
    quantization_bits: int = 8
    optimization_level: int = 2
    metrics: CircuitMetrics = Field(default_factory=CircuitMetrics)
    framework: str = "pytorch"
    original_model_path: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "circuit_id": self.circuit_id,
            "model_hash": self.model_hash,
            "version": self.version,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "quantization_bits": self.quantization_bits,
            "optimization_level": self.optimization_level,
            "framework": self.framework,
            "metrics": self.metrics.model_dump() if self.metrics else {},
        }


# ============ Model Info ============


class ModelInfo(BaseModel):
    """Summary information about a registered model."""
    model_hash: str = ""
    name: str = ""
    version: str = "1.0.0"
    owner: str = ""
    framework: str = "pytorch"
    category: str = ""


# ============ Oracle Types ============


class DataSourceType(str, Enum):
    """Types of oracle data sources."""
    MARKET_DATA = "market_data"
    WEATHER = "weather"
    SPORTS = "sports"
    IOT = "iot"
    CUSTOM = "custom"
    API = "api"
    BLOCKCHAIN = "blockchain"
    ORACLE_DID = "oracle_did"


class DataProvenance(BaseModel):
    """Provenance tracking for oracle data."""
    source_id: str = ""
    verification_method: str = ""
    verified_at: Optional[datetime] = None
    chain_of_custody: List[str] = Field(default_factory=list)
    # Additional fields used by oracle client
    oracle_node_id: str = ""
    attestation_hash: str = ""
    attestation_timestamp: Optional[datetime] = None
    data_hash: str = ""


class JobInput(BaseModel):
    """Input data for a compute job from oracle."""
    did: str = ""
    payload: Any = None
    data_hash: str = ""
    attestation_id: str = ""
    # Additional fields used by oracle client
    source_type: Optional[Any] = None
    proof_of_provenance: Optional[Any] = None
    encrypted: bool = False


# ============ Proof Types ============


class ZKProof(BaseModel):
    """Zero-knowledge proof data."""
    proof_type: str = ""
    proof_bytes: bytes = b""
    public_inputs: List[str] = Field(default_factory=list)
    verification_key_hash: str = ""
    circuit_hash: str = ""
    proving_time_ms: int = 0


class Proof(BaseModel):
    """A cryptographic proof of computation."""
    proof_id: str = ""
    proof_type: ProofType = ProofType.UNSPECIFIED
    job_id: str = ""
    zk_proof: Optional[ZKProof] = None
    tee_attestation: Optional[TEEAttestation] = None
    input_hash: str = ""
    output_hash: str = ""
    model_hash: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ============ Aliases ============

# Backwards-compatible aliases
Job = ComputeJob
Seal = DigitalSeal
