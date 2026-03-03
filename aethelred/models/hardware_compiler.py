"""
Enterprise Hardware-Specific Model Compilation

Provides hardware-optimized compilation targeting specific accelerators:
- Xilinx Alveo U280 (Vitis AI for FPGA acceleration)
- NVIDIA A100/H100 (TensorRT for GPU acceleration)
- Intel SGX/AMD SEV (TEE-optimized execution)

This module routes models to the optimal compilation pipeline based on
target hardware, applying vendor-specific optimizations.

Example:
    >>> from aethelred.models.hardware_compiler import HardwareCompiler
    >>>
    >>> compiler = HardwareCompiler()
    >>>
    >>> # Compile for FPGA (lowest cost, best for latency)
    >>> fpga_circuit = await compiler.compile(
    ...     model_path="credit_model.onnx",
    ...     target=HardwareTarget.XILINX_U280,
    ...     optimization_profile=OptimizationProfile.LATENCY
    ... )
    >>>
    >>> # Compile for GPU (best for throughput)
    >>> gpu_circuit = await compiler.compile(
    ...     model_path="fraud_detection.pt",
    ...     target=HardwareTarget.NVIDIA_A100,
    ...     optimization_profile=OptimizationProfile.THROUGHPUT
    ... )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar, Union

from aethelred.core.exceptions import ValidationError
from aethelred.core.types import (
    Circuit,
    CircuitMetrics,
    HardwareTarget,
    HardwareRequirements,
    TEEPlatform,
)

logger = logging.getLogger(__name__)


# ============ Configuration Types ============


class OptimizationProfile(str, Enum):
    """Optimization profile for compilation."""

    LATENCY = "latency"  # Minimize inference latency
    THROUGHPUT = "throughput"  # Maximize batch throughput
    BALANCED = "balanced"  # Balance latency and throughput
    POWER_EFFICIENT = "power_efficient"  # Minimize power consumption
    COST_OPTIMIZED = "cost_optimized"  # Minimize verification cost


class PrecisionMode(str, Enum):
    """Numerical precision mode."""

    FP32 = "fp32"  # Full precision
    FP16 = "fp16"  # Half precision
    INT8 = "int8"  # 8-bit integer
    INT4 = "int4"  # 4-bit integer (FPGA only)
    MIXED = "mixed"  # Mixed precision


class CompilerBackend(str, Enum):
    """Compiler backend for circuit generation."""

    EZKL = "ezkl"  # EZKL for zkML
    CIRCOM = "circom"  # Circom for R1CS
    HALO2 = "halo2"  # Halo2 for PLONK
    VITIS_AI = "vitis_ai"  # Xilinx Vitis AI
    TENSORRT = "tensorrt"  # NVIDIA TensorRT
    OPENVINO = "openvino"  # Intel OpenVINO
    TFLITE = "tflite"  # TensorFlow Lite


@dataclass
class HardwareProfile:
    """Hardware profile with capabilities and constraints."""

    target: HardwareTarget
    name: str
    vendor: str

    # Compute capabilities
    tflops_fp32: float
    tflops_fp16: float
    tflops_int8: float

    # Memory
    memory_gb: int = 0
    memory_bandwidth_gb_s: float = 0.0

    # Specializations
    supports_sparse: bool = False
    supports_int4: bool = False
    optimal_batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])

    # TEE capabilities
    tee_platform: Optional[TEEPlatform] = None
    tee_max_memory_mb: Optional[int] = None

    # Recommended compiler
    recommended_backend: CompilerBackend = CompilerBackend.EZKL
    recommended_precision: PrecisionMode = PrecisionMode.INT8

    # Cost metrics (relative units)
    cost_per_hour: float = 1.0
    energy_per_inference: float = 1.0


# Pre-defined hardware profiles
HARDWARE_PROFILES: Dict[HardwareTarget, HardwareProfile] = {
    HardwareTarget.NVIDIA_A100: HardwareProfile(
        target=HardwareTarget.NVIDIA_A100,
        name="NVIDIA A100 80GB",
        vendor="NVIDIA",
        tflops_fp32=19.5,
        tflops_fp16=312.0,  # with sparsity
        tflops_int8=624.0,
        memory_gb=80,
        memory_bandwidth_gb_s=2039.0,
        supports_sparse=True,
        supports_int4=False,
        optimal_batch_sizes=[1, 8, 16, 32, 64],
        recommended_backend=CompilerBackend.TENSORRT,
        recommended_precision=PrecisionMode.INT8,
        cost_per_hour=4.0,
        energy_per_inference=0.8,
    ),
    HardwareTarget.NVIDIA_H100: HardwareProfile(
        target=HardwareTarget.NVIDIA_H100,
        name="NVIDIA H100 80GB",
        vendor="NVIDIA",
        tflops_fp32=51.0,
        tflops_fp16=989.0,  # with sparsity
        tflops_int8=1978.0,
        memory_gb=80,
        memory_bandwidth_gb_s=3350.0,
        supports_sparse=True,
        supports_int4=True,
        optimal_batch_sizes=[1, 16, 32, 64, 128],
        recommended_backend=CompilerBackend.TENSORRT,
        recommended_precision=PrecisionMode.INT8,
        cost_per_hour=8.0,
        energy_per_inference=1.0,
    ),
    HardwareTarget.XILINX_U280: HardwareProfile(
        target=HardwareTarget.XILINX_U280,
        name="Xilinx Alveo U280",
        vendor="AMD/Xilinx",
        tflops_fp32=0.6,  # Not optimized for FP
        tflops_fp16=4.5,
        tflops_int8=38.0,  # DSP-based INT8 inference
        memory_gb=8,  # HBM2
        memory_bandwidth_gb_s=460.0,
        supports_sparse=True,
        supports_int4=True,
        optimal_batch_sizes=[1, 4, 8],  # Lower latency for single samples
        recommended_backend=CompilerBackend.VITIS_AI,
        recommended_precision=PrecisionMode.INT8,
        cost_per_hour=2.0,
        energy_per_inference=0.3,  # Most power efficient
    ),
    HardwareTarget.XILINX_U55C: HardwareProfile(
        target=HardwareTarget.XILINX_U55C,
        name="Xilinx Alveo U55C",
        vendor="AMD/Xilinx",
        tflops_fp32=0.8,
        tflops_fp16=6.0,
        tflops_int8=48.0,
        memory_gb=16,
        memory_bandwidth_gb_s=460.0,
        supports_sparse=True,
        supports_int4=True,
        optimal_batch_sizes=[1, 4, 8, 16],
        recommended_backend=CompilerBackend.VITIS_AI,
        recommended_precision=PrecisionMode.INT8,
        cost_per_hour=2.5,
        energy_per_inference=0.35,
    ),
    HardwareTarget.INTEL_SGX: HardwareProfile(
        target=HardwareTarget.INTEL_SGX,
        name="Intel SGX Enclave",
        vendor="Intel",
        tflops_fp32=0.1,  # CPU-bound
        tflops_fp16=0.2,
        tflops_int8=0.4,
        memory_gb=128,  # EPC size limited
        memory_bandwidth_gb_s=50.0,
        supports_sparse=False,
        supports_int4=False,
        optimal_batch_sizes=[1, 4],
        tee_platform=TEEPlatform.INTEL_SGX,
        tee_max_memory_mb=256,  # EPC limit
        recommended_backend=CompilerBackend.EZKL,
        recommended_precision=PrecisionMode.FP32,
        cost_per_hour=0.5,
        energy_per_inference=0.5,
    ),
    HardwareTarget.AMD_SEV_SNP: HardwareProfile(
        target=HardwareTarget.AMD_SEV_SNP,
        name="AMD SEV-SNP",
        vendor="AMD",
        tflops_fp32=1.0,
        tflops_fp16=2.0,
        tflops_int8=4.0,
        memory_gb=512,
        memory_bandwidth_gb_s=200.0,
        supports_sparse=False,
        supports_int4=False,
        optimal_batch_sizes=[1, 8, 16],
        tee_platform=TEEPlatform.AMD_SEV_SNP,
        tee_max_memory_mb=None,  # No memory limit
        recommended_backend=CompilerBackend.EZKL,
        recommended_precision=PrecisionMode.INT8,
        cost_per_hour=1.0,
        energy_per_inference=0.6,
    ),
    HardwareTarget.AWS_NITRO: HardwareProfile(
        target=HardwareTarget.AWS_NITRO,
        name="AWS Nitro Enclave",
        vendor="AWS",
        tflops_fp32=0.5,
        tflops_fp16=1.0,
        tflops_int8=2.0,
        memory_gb=16,  # Enclave limit
        memory_bandwidth_gb_s=100.0,
        supports_sparse=False,
        supports_int4=False,
        optimal_batch_sizes=[1, 4, 8],
        tee_platform=TEEPlatform.AWS_NITRO,
        tee_max_memory_mb=8192,
        recommended_backend=CompilerBackend.EZKL,
        recommended_precision=PrecisionMode.INT8,
        cost_per_hour=0.8,
        energy_per_inference=0.4,
    ),
}


@dataclass
class CompilationConfig:
    """Complete configuration for hardware-specific compilation."""

    # Target
    target: HardwareTarget
    profile: OptimizationProfile = OptimizationProfile.BALANCED
    precision: Optional[PrecisionMode] = None  # Auto-select if None

    # Input/Output
    input_shape: tuple[int, ...] = ()
    output_shape: Optional[tuple[int, ...]] = None
    batch_size: int = 1

    # Quantization
    calibration_dataset: Optional[Any] = None  # For INT8 calibration
    calibration_samples: int = 1000

    # zkML specific
    enable_zkml: bool = True
    zkml_backend: CompilerBackend = CompilerBackend.EZKL
    proof_system: str = "halo2"  # "groth16", "plonk", "halo2"

    # FPGA specific (Vitis AI)
    vitis_fingerprint: Optional[str] = None  # Target FPGA fingerprint
    vitis_dpu_arch: Optional[str] = None  # DPU architecture

    # GPU specific (TensorRT)
    tensorrt_workspace_gb: float = 4.0
    tensorrt_max_batch_size: int = 32
    tensorrt_dla_enabled: bool = False  # Deep Learning Accelerator

    # TEE specific
    tee_attestation_enabled: bool = True
    tee_measurement_hash: Optional[str] = None

    # Output
    output_dir: Optional[str] = None
    save_intermediate: bool = False

    # Metadata
    name: Optional[str] = None
    version: str = "1.0.0"

    def validate(self) -> None:
        """Validate configuration."""
        if self.target not in HARDWARE_PROFILES:
            raise ValidationError(f"Unknown hardware target: {self.target}")

        profile = HARDWARE_PROFILES[self.target]

        # Check batch size compatibility
        if self.batch_size not in profile.optimal_batch_sizes:
            logger.warning(
                f"Batch size {self.batch_size} not optimal for {self.target.value}. "
                f"Recommended: {profile.optimal_batch_sizes}"
            )

        # Check precision compatibility
        if self.precision == PrecisionMode.INT4 and not profile.supports_int4:
            raise ValidationError(f"INT4 precision not supported on {self.target.value}")


@dataclass
class CompilationResult:
    """Result of hardware-specific compilation."""

    circuit: Circuit
    success: bool

    # Timing
    compilation_time_ms: int
    estimated_inference_time_ms: int
    estimated_proving_time_ms: int

    # Hardware artifacts
    hardware_target: HardwareTarget
    precision_used: PrecisionMode
    backend_used: CompilerBackend

    # FPGA-specific
    xmodel_path: Optional[str] = None  # Vitis AI compiled model
    bitstream_hash: Optional[str] = None

    # GPU-specific
    tensorrt_engine_path: Optional[str] = None

    # zkML
    circuit_path: Optional[str] = None
    verification_key_path: Optional[str] = None

    # Metrics
    model_size_bytes: int = 0
    compiled_size_bytes: int = 0
    compression_ratio: float = 1.0

    # Warnings
    warnings: List[str] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)


# ============ Hardware-Specific Compilers ============


class HardwareCompilerStrategy(ABC):
    """Abstract base for hardware-specific compilation strategies."""

    @abstractmethod
    async def compile(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> CompilationResult:
        """Compile model for target hardware."""
        pass

    @abstractmethod
    def supports_target(self, target: HardwareTarget) -> bool:
        """Check if this strategy supports the target."""
        pass

    @abstractmethod
    def estimate_resources(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Estimate compilation resources."""
        pass


class VitisAICompiler(HardwareCompilerStrategy):
    """
    Xilinx Vitis AI compiler for FPGA targets.

    Compiles ONNX models to xmodel format optimized for Xilinx Alveo FPGAs.
    Uses the Vitis AI quantization and compilation pipeline.
    """

    SUPPORTED_TARGETS = {
        HardwareTarget.XILINX_U280,
        HardwareTarget.XILINX_U55C,
    }

    # DPU architectures for different FPGAs
    DPU_ARCHITECTURES = {
        HardwareTarget.XILINX_U280: "DPUCAHX8H",  # High throughput DPU
        HardwareTarget.XILINX_U55C: "DPUCAHX8L",  # Low latency DPU
    }

    def __init__(self, vitis_ai_path: Optional[str] = None):
        """
        Initialize Vitis AI compiler.

        Args:
            vitis_ai_path: Path to Vitis AI installation (auto-detect if None)
        """
        self.vitis_ai_path = vitis_ai_path or os.environ.get("VITIS_AI_PATH", "/opt/vitis_ai")
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate Vitis AI installation."""
        # Check if vai_q_pytorch and vai_c_xir are available
        # In production, this would verify the actual installation
        logger.debug(f"Using Vitis AI at: {self.vitis_ai_path}")

    def supports_target(self, target: HardwareTarget) -> bool:
        return target in self.SUPPORTED_TARGETS

    async def compile(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> CompilationResult:
        """
        Compile model for Xilinx FPGA using Vitis AI.

        Pipeline:
        1. Load ONNX model
        2. Quantize to INT8 using calibration data
        3. Compile to xmodel for target DPU
        4. Generate zkML circuit if enabled
        """
        import time
        start_time = time.time()

        logger.info(f"Compiling for {config.target.value} using Vitis AI")

        # Get DPU architecture
        dpu_arch = config.vitis_dpu_arch or self.DPU_ARCHITECTURES.get(config.target)
        if not dpu_arch:
            raise ValidationError(f"No DPU architecture for {config.target.value}")

        # Step 1: Quantize model
        quantized_path = await self._quantize_model(
            model_path,
            config,
        )

        # Step 2: Compile to xmodel
        xmodel_path = await self._compile_xmodel(
            quantized_path,
            config,
            dpu_arch,
        )

        # Step 3: Generate zkML circuit if enabled
        circuit_data = None
        if config.enable_zkml:
            circuit_data = await self._generate_zkml_circuit(
                quantized_path,
                config,
            )

        # Build result
        compilation_time = int((time.time() - start_time) * 1000)

        profile = HARDWARE_PROFILES[config.target]
        model_hash = self._compute_hash(model_path)
        circuit_id = f"fpga_{uuid.uuid4().hex[:16]}"

        # Estimate inference time based on model complexity
        input_elements = 1
        for d in config.input_shape:
            input_elements *= d

        estimated_inference_ms = max(1, int(input_elements / 1000))  # ~1000 elem/ms for FPGA

        circuit = Circuit(
            circuit_id=circuit_id,
            model_hash=model_hash,
            version=config.version,
            circuit_binary=circuit_data["binary"] if circuit_data else b"",
            verification_key=circuit_data["vk"] if circuit_data else b"",
            proving_key_hash=circuit_data["pk_hash"] if circuit_data else "",
            input_shape=config.input_shape,
            output_shape=config.output_shape or (1,),
            quantization_bits=8,  # Vitis AI uses INT8
            optimization_level=2,
            metrics=CircuitMetrics(
                constraints=circuit_data["constraints"] if circuit_data else 0,
                public_inputs=2,
                private_inputs=input_elements,
                gates=circuit_data["gates"] if circuit_data else 0,
                depth=circuit_data["depth"] if circuit_data else 0,
                memory_bytes=circuit_data["memory"] if circuit_data else 0,
                estimated_proving_time_ms=circuit_data["proving_time"] if circuit_data else 0,
            ),
            framework="vitis_ai",
            original_model_path=model_path,
        )

        return CompilationResult(
            circuit=circuit,
            success=True,
            compilation_time_ms=compilation_time,
            estimated_inference_time_ms=estimated_inference_ms,
            estimated_proving_time_ms=circuit_data["proving_time"] if circuit_data else 0,
            hardware_target=config.target,
            precision_used=PrecisionMode.INT8,
            backend_used=CompilerBackend.VITIS_AI,
            xmodel_path=xmodel_path,
            optimization_suggestions=[
                f"Use batch size {profile.optimal_batch_sizes[0]} for lowest latency",
                "Enable input pipelining for continuous inference",
            ],
        )

    async def _quantize_model(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> str:
        """Quantize model using Vitis AI quantizer."""
        logger.debug("Quantizing model for FPGA")

        # In production, this would call vai_q_pytorch or vai_q_onnx
        # For simulation, return a placeholder path

        output_path = os.path.join(
            config.output_dir or tempfile.mkdtemp(),
            "quantized_model.onnx"
        )

        # Simulation: would run quantization calibration
        # vai_q_onnx quantize \
        #   --model model.onnx \
        #   --calib_data_path ./calib_data \
        #   --output_dir ./quantized

        return output_path

    async def _compile_xmodel(
        self,
        quantized_path: str,
        config: CompilationConfig,
        dpu_arch: str,
    ) -> str:
        """Compile quantized model to xmodel."""
        logger.debug(f"Compiling xmodel for DPU: {dpu_arch}")

        output_path = os.path.join(
            config.output_dir or tempfile.mkdtemp(),
            "model.xmodel"
        )

        # In production, this would call vai_c_xir
        # vai_c_xir \
        #   --xmodel quantized_model.xmodel \
        #   --arch /opt/vitis_ai/arch/DPUCAHX8H.json \
        #   --output_dir ./compiled

        return output_path

    async def _generate_zkml_circuit(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Generate zkML circuit from quantized model."""
        logger.debug("Generating zkML circuit for FPGA model")

        # Estimate circuit metrics based on model
        input_size = 1
        for d in config.input_shape:
            input_size *= d

        # FPGA-optimized circuits have fewer constraints due to INT8
        constraints = int(input_size * 500)  # ~500 constraints per input element
        gates = constraints * 2
        depth = int(constraints ** 0.4)  # Shallower due to optimizations

        return {
            "binary": b"FPGA_CIRCUIT_BINARY",
            "vk": b"VERIFICATION_KEY",
            "pk_hash": hashlib.sha256(b"proving_key").hexdigest(),
            "constraints": constraints,
            "gates": gates,
            "depth": depth,
            "memory": constraints * 24,  # Less memory for INT8
            "proving_time": constraints // 20,  # Faster proving for FPGA-optimized
        }

    def estimate_resources(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Estimate FPGA compilation resources."""
        input_size = 1
        for d in config.input_shape:
            input_size *= d

        return {
            "estimated_luts": input_size * 100,
            "estimated_dsps": input_size // 10,
            "estimated_brams": max(1, input_size // 1000),
            "estimated_compilation_time_min": max(5, input_size // 10000),
            "estimated_power_watts": 30 + (input_size / 10000),
        }

    def _compute_hash(self, path: str) -> str:
        """Compute file hash."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class TensorRTCompiler(HardwareCompilerStrategy):
    """
    NVIDIA TensorRT compiler for GPU targets.

    Compiles models to TensorRT engines optimized for NVIDIA GPUs.
    Supports INT8 calibration, sparsity, and multi-instance execution.
    """

    SUPPORTED_TARGETS = {
        HardwareTarget.NVIDIA_A100,
        HardwareTarget.NVIDIA_H100,
        HardwareTarget.NVIDIA_L40S,
    }

    def __init__(self, tensorrt_path: Optional[str] = None):
        """
        Initialize TensorRT compiler.

        Args:
            tensorrt_path: Path to TensorRT installation (auto-detect if None)
        """
        self.tensorrt_path = tensorrt_path or os.environ.get("TENSORRT_PATH", "/usr/local/tensorrt")
        self._validate_installation()

    def _validate_installation(self) -> None:
        """Validate TensorRT installation."""
        logger.debug(f"Using TensorRT at: {self.tensorrt_path}")

    def supports_target(self, target: HardwareTarget) -> bool:
        return target in self.SUPPORTED_TARGETS

    async def compile(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> CompilationResult:
        """
        Compile model for NVIDIA GPU using TensorRT.

        Pipeline:
        1. Parse ONNX model
        2. Apply optimizations (layer fusion, kernel selection)
        3. Calibrate for INT8 if enabled
        4. Build TensorRT engine
        5. Generate zkML circuit if enabled
        """
        import time
        start_time = time.time()

        logger.info(f"Compiling for {config.target.value} using TensorRT")

        # Determine precision
        precision = config.precision or PrecisionMode.INT8
        profile = HARDWARE_PROFILES[config.target]

        # Step 1: Parse and optimize ONNX
        optimized_path = await self._optimize_onnx(model_path, config)

        # Step 2: Build TensorRT engine
        engine_path = await self._build_engine(
            optimized_path,
            config,
            precision,
        )

        # Step 3: Generate zkML circuit if enabled
        circuit_data = None
        if config.enable_zkml:
            circuit_data = await self._generate_zkml_circuit(
                optimized_path,
                config,
            )

        # Build result
        compilation_time = int((time.time() - start_time) * 1000)

        model_hash = self._compute_hash(model_path)
        circuit_id = f"gpu_{uuid.uuid4().hex[:16]}"

        # Estimate inference time
        input_elements = 1
        for d in config.input_shape:
            input_elements *= d

        # GPU inference is very fast
        estimated_inference_ms = max(1, int(input_elements / 50000))  # ~50k elem/ms for A100

        circuit = Circuit(
            circuit_id=circuit_id,
            model_hash=model_hash,
            version=config.version,
            circuit_binary=circuit_data["binary"] if circuit_data else b"",
            verification_key=circuit_data["vk"] if circuit_data else b"",
            proving_key_hash=circuit_data["pk_hash"] if circuit_data else "",
            input_shape=config.input_shape,
            output_shape=config.output_shape or (1,),
            quantization_bits=8 if precision == PrecisionMode.INT8 else 16,
            optimization_level=3,  # Aggressive for GPU
            metrics=CircuitMetrics(
                constraints=circuit_data["constraints"] if circuit_data else 0,
                public_inputs=2,
                private_inputs=input_elements,
                gates=circuit_data["gates"] if circuit_data else 0,
                depth=circuit_data["depth"] if circuit_data else 0,
                memory_bytes=circuit_data["memory"] if circuit_data else 0,
                estimated_proving_time_ms=circuit_data["proving_time"] if circuit_data else 0,
            ),
            framework="tensorrt",
            original_model_path=model_path,
        )

        return CompilationResult(
            circuit=circuit,
            success=True,
            compilation_time_ms=compilation_time,
            estimated_inference_time_ms=estimated_inference_ms,
            estimated_proving_time_ms=circuit_data["proving_time"] if circuit_data else 0,
            hardware_target=config.target,
            precision_used=precision,
            backend_used=CompilerBackend.TENSORRT,
            tensorrt_engine_path=engine_path,
            optimization_suggestions=[
                f"Use batch size {profile.optimal_batch_sizes[-1]} for maximum throughput",
                "Enable sparsity for 2x speedup on Ampere+ GPUs" if profile.supports_sparse else "",
            ],
        )

    async def _optimize_onnx(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> str:
        """Optimize ONNX model for TensorRT."""
        logger.debug("Optimizing ONNX for TensorRT")

        # In production: onnxruntime or onnx-simplifier optimizations
        output_path = os.path.join(
            config.output_dir or tempfile.mkdtemp(),
            "optimized_model.onnx"
        )

        return output_path

    async def _build_engine(
        self,
        model_path: str,
        config: CompilationConfig,
        precision: PrecisionMode,
    ) -> str:
        """Build TensorRT engine."""
        logger.debug(f"Building TensorRT engine with {precision.value} precision")

        output_path = os.path.join(
            config.output_dir or tempfile.mkdtemp(),
            "model.engine"
        )

        # In production, this would use tensorrt Python API or trtexec
        # trtexec --onnx=model.onnx \
        #   --saveEngine=model.engine \
        #   --int8 --calib=calibration.cache \
        #   --workspace=4096

        return output_path

    async def _generate_zkml_circuit(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Generate zkML circuit from optimized model."""
        logger.debug("Generating zkML circuit for GPU model")

        input_size = 1
        for d in config.input_shape:
            input_size *= d

        # GPU circuits can be larger due to more compute
        constraints = int(input_size * 800)
        gates = constraints * 2
        depth = int(constraints ** 0.45)

        return {
            "binary": b"GPU_CIRCUIT_BINARY",
            "vk": b"VERIFICATION_KEY",
            "pk_hash": hashlib.sha256(b"proving_key").hexdigest(),
            "constraints": constraints,
            "gates": gates,
            "depth": depth,
            "memory": constraints * 32,
            "proving_time": constraints // 15,  # GPU-accelerated proving
        }

    def estimate_resources(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Estimate GPU compilation resources."""
        input_size = 1
        for d in config.input_shape:
            input_size *= d

        profile = HARDWARE_PROFILES.get(config.target)
        vram = profile.memory_gb if profile else 40

        return {
            "estimated_vram_gb": min(vram, max(2, input_size // 500000)),
            "estimated_compilation_time_min": max(2, input_size // 100000),
            "estimated_throughput_samples_sec": input_size * 10,
            "estimated_power_watts": 250 + (input_size / 1000),
        }

    def _compute_hash(self, path: str) -> str:
        """Compute file hash."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class TEECompiler(HardwareCompilerStrategy):
    """
    TEE-optimized compiler for Intel SGX, AMD SEV-SNP, and AWS Nitro.

    Compiles models for secure execution within Trusted Execution Environments.
    Generates attestation-compatible circuits with measurement verification.
    """

    SUPPORTED_TARGETS = {
        HardwareTarget.INTEL_SGX,
        HardwareTarget.AMD_SEV_SNP,
        HardwareTarget.INTEL_TDX,
        HardwareTarget.AWS_NITRO,
    }

    def supports_target(self, target: HardwareTarget) -> bool:
        return target in self.SUPPORTED_TARGETS

    async def compile(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> CompilationResult:
        """Compile model for TEE execution."""
        import time
        start_time = time.time()

        logger.info(f"Compiling for {config.target.value} TEE")

        profile = HARDWARE_PROFILES.get(config.target)

        # Check memory limits
        if profile and profile.tee_max_memory_mb:
            model_size = os.path.getsize(model_path)
            if model_size > profile.tee_max_memory_mb * 1024 * 1024:
                raise ValidationError(
                    f"Model size {model_size} exceeds TEE memory limit "
                    f"{profile.tee_max_memory_mb}MB"
                )

        # TEE compilation focuses on attestation
        circuit_data = await self._generate_tee_circuit(model_path, config)

        compilation_time = int((time.time() - start_time) * 1000)
        model_hash = self._compute_hash(model_path)
        circuit_id = f"tee_{uuid.uuid4().hex[:16]}"

        input_elements = 1
        for d in config.input_shape:
            input_elements *= d

        # TEE inference is CPU-bound
        estimated_inference_ms = max(10, int(input_elements / 100))

        circuit = Circuit(
            circuit_id=circuit_id,
            model_hash=model_hash,
            version=config.version,
            circuit_binary=circuit_data["binary"],
            verification_key=circuit_data["vk"],
            proving_key_hash=circuit_data["pk_hash"],
            input_shape=config.input_shape,
            output_shape=config.output_shape or (1,),
            quantization_bits=32,  # FP32 for TEE
            optimization_level=1,  # Limited optimization in TEE
            metrics=CircuitMetrics(
                constraints=circuit_data["constraints"],
                public_inputs=2,
                private_inputs=input_elements,
                gates=circuit_data["gates"],
                depth=circuit_data["depth"],
                memory_bytes=circuit_data["memory"],
                estimated_proving_time_ms=circuit_data["proving_time"],
            ),
            framework="tee",
            original_model_path=model_path,
        )

        return CompilationResult(
            circuit=circuit,
            success=True,
            compilation_time_ms=compilation_time,
            estimated_inference_time_ms=estimated_inference_ms,
            estimated_proving_time_ms=circuit_data["proving_time"],
            hardware_target=config.target,
            precision_used=PrecisionMode.FP32,
            backend_used=CompilerBackend.EZKL,
            optimization_suggestions=[
                "Keep model size under enclave memory limit",
                "Use batch size 1 for lowest attestation overhead",
            ],
        )

    async def _generate_tee_circuit(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Generate circuit for TEE attestation."""
        input_size = 1
        for d in config.input_shape:
            input_size *= d

        # TEE circuits include attestation verification
        constraints = int(input_size * 1200)  # More constraints for attestation
        gates = constraints * 2
        depth = int(constraints ** 0.5)

        return {
            "binary": b"TEE_CIRCUIT_BINARY",
            "vk": b"VERIFICATION_KEY",
            "pk_hash": hashlib.sha256(b"proving_key").hexdigest(),
            "constraints": constraints,
            "gates": gates,
            "depth": depth,
            "memory": constraints * 40,  # More memory for attestation data
            "proving_time": constraints // 8,  # CPU-bound proving
        }

    def estimate_resources(
        self,
        model_path: str,
        config: CompilationConfig,
    ) -> Dict[str, Any]:
        """Estimate TEE compilation resources."""
        model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
        profile = HARDWARE_PROFILES.get(config.target)

        return {
            "model_size_mb": model_size / (1024 * 1024),
            "tee_memory_limit_mb": profile.tee_max_memory_mb if profile else None,
            "estimated_enclave_pages": max(256, model_size // 4096),
            "estimated_compilation_time_min": max(1, model_size // (10 * 1024 * 1024)),
        }

    def _compute_hash(self, path: str) -> str:
        """Compute file hash."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ============ Main Hardware Compiler ============


class HardwareCompiler:
    """
    Enterprise hardware compiler for multi-target model compilation.

    Automatically routes compilation to the optimal backend based on
    target hardware, providing unified API for FPGA, GPU, and TEE targets.

    Example:
        >>> compiler = HardwareCompiler()
        >>>
        >>> # Auto-select optimal hardware
        >>> circuit = await compiler.compile(
        ...     model_path="model.onnx",
        ...     input_shape=(1, 64),
        ...     target=HardwareTarget.AUTO,
        ... )
        >>>
        >>> # Target specific hardware
        >>> fpga_circuit = await compiler.compile(
        ...     model_path="model.onnx",
        ...     input_shape=(1, 64),
        ...     target=HardwareTarget.XILINX_U280,
        ...     profile=OptimizationProfile.LATENCY,
        ... )
    """

    def __init__(self):
        """Initialize hardware compiler with all backend strategies."""
        self._strategies: Dict[HardwareTarget, HardwareCompilerStrategy] = {}

        # Register compilers
        self._register_compiler(VitisAICompiler())
        self._register_compiler(TensorRTCompiler())
        self._register_compiler(TEECompiler())

    def _register_compiler(self, compiler: HardwareCompilerStrategy) -> None:
        """Register a compiler strategy for its supported targets."""
        for target in HardwareTarget:
            if compiler.supports_target(target):
                self._strategies[target] = compiler

    async def compile(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        target: HardwareTarget = HardwareTarget.AUTO,
        profile: OptimizationProfile = OptimizationProfile.BALANCED,
        precision: Optional[PrecisionMode] = None,
        output_dir: Optional[str] = None,
        enable_zkml: bool = True,
        calibration_data: Optional[Any] = None,
    ) -> CompilationResult:
        """
        Compile model for target hardware.

        Args:
            model_path: Path to model file (ONNX, PyTorch, etc.)
            input_shape: Model input shape
            target: Target hardware (AUTO for automatic selection)
            profile: Optimization profile
            precision: Numerical precision (auto-select if None)
            output_dir: Directory for compilation artifacts
            enable_zkml: Generate zkML circuit for verification
            calibration_data: Data for INT8 calibration

        Returns:
            CompilationResult with compiled circuit and artifacts

        Example:
            >>> result = await compiler.compile(
            ...     "credit_model.onnx",
            ...     input_shape=(1, 64),
            ...     target=HardwareTarget.XILINX_U280,
            ...     profile=OptimizationProfile.LATENCY,
            ... )
            >>> print(f"Inference time: {result.estimated_inference_time_ms}ms")
        """
        # Auto-select target if needed
        if target == HardwareTarget.AUTO:
            target = self._select_optimal_target(model_path, input_shape, profile)
            logger.info(f"Auto-selected target: {target.value}")

        # Get compiler strategy
        strategy = self._strategies.get(target)
        if not strategy:
            raise ValidationError(f"No compiler available for target: {target.value}")

        # Build configuration
        config = CompilationConfig(
            target=target,
            profile=profile,
            precision=precision,
            input_shape=input_shape,
            output_dir=output_dir,
            enable_zkml=enable_zkml,
            calibration_dataset=calibration_data,
        )
        config.validate()

        # Compile
        return await strategy.compile(model_path, config)

    def _select_optimal_target(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        profile: OptimizationProfile,
    ) -> HardwareTarget:
        """Select optimal hardware target based on model and profile."""
        input_size = 1
        for d in input_shape:
            input_size *= d

        # Selection heuristics
        if profile == OptimizationProfile.LATENCY:
            # FPGA for lowest latency
            return HardwareTarget.XILINX_U280

        elif profile == OptimizationProfile.THROUGHPUT:
            # GPU for highest throughput
            if input_size > 100000:
                return HardwareTarget.NVIDIA_H100
            return HardwareTarget.NVIDIA_A100

        elif profile == OptimizationProfile.POWER_EFFICIENT:
            # FPGA for best power efficiency
            return HardwareTarget.XILINX_U280

        elif profile == OptimizationProfile.COST_OPTIMIZED:
            # TEE for lowest cost
            return HardwareTarget.AWS_NITRO

        else:  # BALANCED
            # Choose based on model size
            if input_size < 10000:
                return HardwareTarget.XILINX_U280
            elif input_size < 100000:
                return HardwareTarget.NVIDIA_A100
            else:
                return HardwareTarget.NVIDIA_H100

    def estimate_cost(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        target: HardwareTarget,
        num_inferences: int = 1000,
    ) -> Dict[str, float]:
        """
        Estimate cost for running inferences on target hardware.

        Args:
            model_path: Path to model
            input_shape: Input shape
            target: Hardware target
            num_inferences: Number of inferences to estimate

        Returns:
            Cost estimates in various dimensions
        """
        profile = HARDWARE_PROFILES.get(target)
        if not profile:
            raise ValidationError(f"Unknown target: {target.value}")

        input_size = 1
        for d in input_shape:
            input_size *= d

        # Estimate inference time
        inference_time_ms = input_size / (profile.tflops_int8 * 1000)
        total_time_hours = (inference_time_ms * num_inferences) / (1000 * 3600)

        return {
            "compute_cost_usd": profile.cost_per_hour * total_time_hours,
            "energy_cost_kwh": profile.energy_per_inference * num_inferences / 1000,
            "estimated_time_hours": total_time_hours,
            "cost_per_inference_usd": (profile.cost_per_hour * total_time_hours) / num_inferences,
        }

    def get_hardware_profiles(self) -> Dict[HardwareTarget, HardwareProfile]:
        """Get all available hardware profiles."""
        return HARDWARE_PROFILES.copy()

    def supports_target(self, target: HardwareTarget) -> bool:
        """Check if target is supported."""
        return target in self._strategies


# ============ Exports ============

__all__ = [
    # Main class
    "HardwareCompiler",
    # Strategies
    "VitisAICompiler",
    "TensorRTCompiler",
    "TEECompiler",
    # Types
    "CompilationConfig",
    "CompilationResult",
    "HardwareProfile",
    "OptimizationProfile",
    "PrecisionMode",
    "CompilerBackend",
    # Constants
    "HARDWARE_PROFILES",
]
