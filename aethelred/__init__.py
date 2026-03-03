"""
Aethelred SDK - Enterprise AI Blockchain Toolkit

A high-performance SDK for the Aethelred AI Blockchain with explicit backend behavior:

CORE FEATURES:
    - Hardware Abstraction Layer (HAL) for CPUs, GPUs, TEEs
    - JIT compilation with automatic optimization
    - Memory pool management with zero-copy transfers
    - Async execution streams with dependency graphs
    - Comprehensive profiling and tracing

NEURAL NETWORK:
    - PyTorch-compatible nn.Module API
    - Automatic model verification
    - Transformer, attention, and modern architectures
    - Loss functions and activation layers

DISTRIBUTED COMPUTING:
    - Data parallelism (DDP)
    - Model parallelism (tensor, pipeline)
    - ZeRO optimizer (stages 1-3)
    - Gradient compression
    - Elastic training

QUANTIZATION:
    - Post-training quantization (PTQ)
    - Quantization-aware training (QAT)
    - INT8, INT4, FP16, BF16, FP8 support
    - Per-tensor, per-channel, per-group granularity

TENSOR OPERATIONS:
    - Lazy evaluation and operation fusion
    - Broadcasting and views
    - Automatic differentiation support
    - NumPy, PyTorch, TensorFlow interop

BLOCKCHAIN INTEGRATION:
    - AI compute job submission and tracking
    - Digital seal creation and verification
    - TEE attestation (Intel SGX, AMD SEV, AWS Nitro)
    - zkML proof verification (Groth16, PLONK, STARK, EZKL)
    - Post-quantum cryptography (Dilithium, Kyber)

Quick Start:
    >>> import aethelred as aethel
    >>>
    >>> # Initialize runtime
    >>> aethel.runtime.initialize(enable_profiling=True)
    >>>
    >>> # Create tensors on GPU
    >>> with aethel.Device.gpu():
    ...     x = aethel.Tensor.randn(1024, 1024)
    ...     y = aethel.Tensor.randn(1024, 1024)
    ...     z = x @ y  # Lazy evaluation
    ...     z.realize()  # Execute
    >>>
    >>> # Neural network
    >>> model = aethel.nn.Sequential(
    ...     aethel.nn.Linear(784, 256),
    ...     aethel.nn.ReLU(),
    ...     aethel.nn.Linear(256, 10)
    ... )
    >>>
    >>> # Distributed training
    >>> model = aethel.distributed.DistributedDataParallel(model)
    >>>
    >>> # Quantize for deployment
    >>> model_int8 = aethel.quantize.quantize_dynamic(model)
    >>>
    >>> # Submit to blockchain
    >>> client = aethel.AethelredClient("https://rpc.mainnet.aethelred.org")
    >>> job = client.jobs.submit(model=model, input=data)
    >>> seal = client.seals.create(job_id=job.job_id)

Documentation:
    https://docs.aethelred.org/sdk/python

GitHub:
    https://github.com/aethelred/sdk-python
"""

__version__ = "1.0.0"
__author__ = "Aethelred Team"
__license__ = "Apache-2.0"

# ============ Core Client ============

from aethelred.core.client import AethelredClient, AsyncAethelredClient
from aethelred.core.config import Config, Network, NetworkConfig

# ============ Core Types ============

from aethelred.core.types import (
    # Addresses and hashes
    Address,
    Hash,
    TxHash,
    # Enums
    JobStatus,
    SealStatus,
    ProofType,
    ProofSystem,
    TEEPlatform,
    UtilityCategory,
    PrivacyLevel,
    ComplianceFramework,
    # Job types
    ComputeJob,
    JobResult,
    SubmitJobRequest,
    SubmitJobResponse,
    # Seal types
    DigitalSeal,
    CreateSealRequest,
    CreateSealResponse,
    VerifySealResponse,
    ValidatorAttestation,
    RegulatoryInfo,
    # Model types
    RegisteredModel,
    RegisterModelRequest,
    RegisterModelResponse,
    # Validator types
    ValidatorStats,
    HardwareCapability,
    # Verification types
    VerificationResult,
    TEEAttestation,
    ZKMLProof,
    # Network types
    UsefulWorkStats,
    EpochStats,
    NodeInfo,
    Block,
    # Pagination
    PageRequest,
    PageResponse,
)

# ============ Exceptions ============

from aethelred.core.exceptions import (
    AethelredError,
    AuthenticationError,
    ConnectionError,
    JobError,
    ModelError,
    RateLimitError,
    SealError,
    TransactionError,
    ValidationError,
    VerificationError,
    TimeoutError,
    NetworkError,
    ErrorCode,
)

# ============ Modules ============

from aethelred.jobs import JobsModule
from aethelred.seals import SealsModule
from aethelred.models import ModelsModule
from aethelred.validators import ValidatorsModule
from aethelred.verification import VerificationModule

# ============ Post-Quantum Cryptography ============

try:
    from aethelred.crypto.pqc import (
        DilithiumSigner,
        DilithiumKeyPair,
        DilithiumSecurityLevel,
        KyberKEM,
        KyberKeyPair,
        KyberSecurityLevel,
    )
    from aethelred.crypto.wallet import (
        Wallet,
        DualKeyWallet,
        CompositeSignature,
    )
except Exception:
    pass

# ============ Utilities ============

from aethelred.utils import (
    sha256,
    sha256_hex,
    keccak256,
    to_uaethel,
    from_uaethel,
    format_aethel,
    is_valid_address,
    encode_base64,
    decode_base64,
)

# ============ Compliance ============

try:
    from aethelred.compliance import (
        PIIScrubber,
        PIIType,
        ComplianceChecker,
        ComplianceResult,
        DataClassification,
    )
except Exception:  # Optional/legacy surface may be unavailable in minimal installs
    pass

# ============ Runtime & Hardware ============

try:
    from aethelred.core.runtime import (
        Runtime,
        Device,
        DeviceType,
        DeviceCapabilities,
        Stream,
        Event,
        MemoryPool,
        MemoryBlock,
        MemoryType,
        Profiler,
        JITCompiler,
        jit,
        profile,
    )
except Exception:
    pass

# ============ Tensor ============

try:
    from aethelred.core.tensor import (
        Tensor,
        DType,
    )
except Exception:
    pass

# ============ Neural Network ============

try:
    from aethelred import nn
    from aethelred.nn import (
        Module,
        Parameter,
        Sequential,
        ModuleList,
        ModuleDict,
        Linear,
        Embedding,
        LayerNorm,
        BatchNorm1d,
        RMSNorm,
        ReLU,
        GELU,
        SiLU,
        Softmax,
        Tanh,
        Sigmoid,
        Dropout,
        MultiheadAttention,
        TransformerEncoderLayer,
        MSELoss,
        CrossEntropyLoss,
    )
except Exception:
    pass

# ============ Optimizers ============

try:
    from aethelred import optim
    from aethelred.optim import (
        Optimizer,
        SGD,
        Adam,
        AdamW,
        Lion,
        LAMB,
        LRScheduler,
        StepLR,
        CosineAnnealingLR,
        OneCycleLR,
        WarmupLR,
        clip_grad_norm_,
        clip_grad_value_,
    )
except Exception:
    pass

# ============ Distributed ============

try:
    from aethelred import distributed
    from aethelred.distributed import (
        ProcessGroup,
        Backend,
        DistributedDataParallel,
        ZeROOptimizer,
        ZeROStage,
        PipelineParallel,
        ColumnParallelLinear,
        RowParallelLinear,
    )
except Exception:
    pass

# ============ Quantization ============

try:
    from aethelred import quantize
    from aethelred.quantize import (
        QuantizationConfig,
        QuantizationType,
        QuantizationScheme,
        CalibrationMethod,
        QuantizationEngine,
        QATEngine,
        QuantizedLinear,
        quantize_dynamic,
        quantize_static,
        prepare_qat,
        convert_qat,
    )
except Exception:
    pass

# ============ Framework Integrations ============

try:
    from aethelred import integrations
except Exception:
    pass

# ============ All Exports ============

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Clients
    "AethelredClient",
    "AsyncAethelredClient",
    # Configuration
    "Config",
    "Network",
    "NetworkConfig",
    # Types - Basic
    "Address",
    "Hash",
    "TxHash",
    # Types - Enums
    "JobStatus",
    "SealStatus",
    "ProofType",
    "ProofSystem",
    "TEEPlatform",
    "UtilityCategory",
    "PrivacyLevel",
    "ComplianceFramework",
    # Types - Jobs
    "ComputeJob",
    "JobResult",
    "SubmitJobRequest",
    "SubmitJobResponse",
    # Types - Seals
    "DigitalSeal",
    "CreateSealRequest",
    "CreateSealResponse",
    "VerifySealResponse",
    "ValidatorAttestation",
    "RegulatoryInfo",
    # Types - Models
    "RegisteredModel",
    "RegisterModelRequest",
    "RegisterModelResponse",
    # Types - Validators
    "ValidatorStats",
    "HardwareCapability",
    # Types - Verification
    "VerificationResult",
    "TEEAttestation",
    "ZKMLProof",
    # Types - Network
    "UsefulWorkStats",
    "EpochStats",
    "NodeInfo",
    "Block",
    # Types - Pagination
    "PageRequest",
    "PageResponse",
    # Exceptions
    "AethelredError",
    "AuthenticationError",
    "ConnectionError",
    "JobError",
    "ModelError",
    "RateLimitError",
    "SealError",
    "TransactionError",
    "ValidationError",
    "VerificationError",
    "TimeoutError",
    "NetworkError",
    "ErrorCode",
    # Modules
    "JobsModule",
    "SealsModule",
    "ModelsModule",
    "ValidatorsModule",
    "VerificationModule",
    # PQC
    "DilithiumSigner",
    "DilithiumKeyPair",
    "DilithiumSecurityLevel",
    "KyberKEM",
    "KyberKeyPair",
    "KyberSecurityLevel",
    # Wallet
    "Wallet",
    "DualKeyWallet",
    "CompositeSignature",
    # Utilities
    "sha256",
    "sha256_hex",
    "keccak256",
    "to_uaethel",
    "from_uaethel",
    "format_aethel",
    "is_valid_address",
    "encode_base64",
    "decode_base64",
    # Compliance
    "PIIScrubber",
    "PIIType",
    "ComplianceChecker",
    "ComplianceResult",
    "DataClassification",
    # Runtime & Hardware
    "Runtime",
    "Device",
    "DeviceType",
    "DeviceCapabilities",
    "Stream",
    "Event",
    "MemoryPool",
    "MemoryBlock",
    "MemoryType",
    "Profiler",
    "JITCompiler",
    "jit",
    "profile",
    # Tensor
    "Tensor",
    "DType",
    # Neural Network
    "nn",
    "Module",
    "Parameter",
    "Sequential",
    "ModuleList",
    "ModuleDict",
    "Linear",
    "Embedding",
    "LayerNorm",
    "BatchNorm1d",
    "RMSNorm",
    "ReLU",
    "GELU",
    "SiLU",
    "Softmax",
    "Tanh",
    "Sigmoid",
    "Dropout",
    "MultiheadAttention",
    "TransformerEncoderLayer",
    "MSELoss",
    "CrossEntropyLoss",
    # Optimizers
    "optim",
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "Lion",
    "LAMB",
    "LRScheduler",
    "StepLR",
    "CosineAnnealingLR",
    "OneCycleLR",
    "WarmupLR",
    "clip_grad_norm_",
    "clip_grad_value_",
    # Distributed
    "distributed",
    "ProcessGroup",
    "Backend",
    "DistributedDataParallel",
    "ZeROOptimizer",
    "ZeROStage",
    "PipelineParallel",
    "ColumnParallelLinear",
    "RowParallelLinear",
    # Quantization
    "quantize",
    "QuantizationConfig",
    "QuantizationType",
    "QuantizationScheme",
    "CalibrationMethod",
    "QuantizationEngine",
    "QATEngine",
    "QuantizedLinear",
    "quantize_dynamic",
    "quantize_static",
    "prepare_qat",
    "convert_qat",
    # Framework Integrations
    "integrations",
]


def get_version() -> str:
    """Get the SDK version."""
    return __version__


def get_sdk_info() -> dict:
    """Get comprehensive SDK information."""
    return {
        "name": "aethelred-sdk",
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "description": "Enterprise AI blockchain SDK with explicit compute-backend guarantees",
        "features": {
            "core": [
                "Hardware Abstraction Layer (CPU, GPU, TEE)",
                "GPU acceleration depends on detected backend/runtime support",
                "No implied native CUDA dispatch without dedicated backend integration",
                "JIT compilation with automatic optimization",
                "Memory pool with zero-copy transfers",
                "Async execution streams with dependency graphs",
                "Comprehensive profiling and tracing (Chrome Trace export)",
            ],
            "tensor": [
                "Multi-dimensional array with lazy evaluation",
                "Operation fusion and kernel optimization",
                "Broadcasting and memory-efficient views",
                "NumPy, PyTorch, TensorFlow interoperability",
            ],
            "neural_network": [
                "PyTorch-compatible nn.Module API",
                "Transformer and attention layers",
                "Modern activations (GELU, SiLU, RMSNorm)",
                "Loss functions and optimizers",
            ],
            "distributed": [
                "Data parallelism (DistributedDataParallel)",
                "Model parallelism (Tensor, Pipeline)",
                "ZeRO optimizer (Stages 1-3)",
                "Gradient compression (TopK, PowerSGD)",
                "Elastic training with checkpoint recovery",
            ],
            "quantization": [
                "Post-training quantization (PTQ)",
                "Quantization-aware training (QAT)",
                "INT8, INT4, FP16, BF16, FP8 precision",
                "Per-tensor, per-channel, per-group granularity",
                "Calibration (MinMax, Percentile, Entropy)",
            ],
            "blockchain": [
                "AI compute job submission and tracking",
                "Digital seal creation and verification",
                "TEE attestation (Intel SGX, AMD SEV, AWS Nitro, ARM TrustZone)",
                "zkML proof verification (Groth16, PLONK, STARK, EZKL)",
                "Post-quantum cryptography (Dilithium, Kyber)",
                "Compliance framework (HIPAA, GDPR, SOC2, CCPA)",
            ],
        },
        "supported_networks": {
            "mainnet": "https://rpc.mainnet.aethelred.org",
            "testnet": "https://rpc.testnet.aethelred.org",
            "devnet": "https://rpc.devnet.aethelred.org",
            "local": "http://127.0.0.1:26657",
        },
        "supported_devices": [
            "CPU (x86, ARM)",
            "NVIDIA GPU (CUDA)",
            "AMD GPU (ROCm)",
            "Intel GPU (oneAPI)",
            "Intel SGX Enclave",
            "AMD SEV Enclave",
            "AWS Nitro Enclave",
            "ARM TrustZone",
        ],
        "python_versions": ["3.9", "3.10", "3.11", "3.12"],
    }


def initialize(
    devices: list = None,
    default_device: 'Device' = None,
    enable_profiling: bool = False,
    thread_pool_size: int = 8
) -> 'Runtime':
    """
    Initialize the Aethelred runtime.

    This should be called once at program start to:
    - Enumerate available devices (CPUs, GPUs, TEEs)
    - Initialize memory pools
    - Start thread pools
    - Enable profiling if requested

    Example:
        >>> import aethelred as aethel
        >>> runtime = aethel.initialize(enable_profiling=True)
        >>> print(f"Found {len(runtime.devices)} devices")
    """
    if "Runtime" not in globals():
        raise RuntimeError("Aethelred runtime module is unavailable in this installation")
    return Runtime.initialize(
        devices=devices,
        default_device=default_device,
        enable_profiling=enable_profiling,
        thread_pool_size=thread_pool_size
    )


# Runtime alias
runtime = globals().get("Runtime")
