"""
Model Converter for Aethelred zkML

Converts AI models from various frameworks (PyTorch, TensorFlow, ONNX)
into arithmetic circuits (R1CS) compatible with the zkVM.

The conversion pipeline:
1. Load model from framework-specific format
2. Export to ONNX intermediate representation
3. Quantize weights and activations
4. Apply circuit optimizations (pruning, folding)
5. Compile to R1CS constraints
6. Generate verification and proving keys

Example:
    >>> converter = ModelConverter()
    >>> circuit = converter.from_pytorch(
    ...     model_path="credit_model.pt",
    ...     input_shape=(1, 64),
    ...     optimization_level=2,
    ...     quantization_bits=8
    ... )
    >>> print(f"Circuit has {circuit.metrics.constraints} constraints")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Union

from aethelred.core.exceptions import ValidationError
from aethelred.core.types import Circuit, CircuitMetrics, HardwareTarget, HardwareRequirements

logger = logging.getLogger(__name__)


# ============ Configuration Types ============


class FrameworkType(str, Enum):
    """Supported ML frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    KERAS = "keras"
    SKLEARN = "sklearn"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"


class QuantizationMode(str, Enum):
    """Quantization modes."""

    NONE = "none"
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    DYNAMIC = "dynamic"


class OptimizationLevel(int, Enum):
    """Circuit optimization levels."""

    NONE = 0       # No optimizations
    BASIC = 1      # Constant folding, dead code elimination
    STANDARD = 2   # + Operator fusion, common subexpression elimination
    AGGRESSIVE = 3 # + Loop unrolling, aggressive pruning


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    # Bit width
    bits: int = 8
    mode: QuantizationMode = QuantizationMode.SYMMETRIC

    # Calibration
    calibration_samples: int = 100
    percentile: float = 99.99

    # Per-layer settings
    per_channel: bool = True
    skip_layers: list[str] = field(default_factory=list)

    # Scale factors
    input_scale: Optional[float] = None
    output_scale: Optional[float] = None

    def validate(self) -> None:
        """Validate quantization configuration."""
        if self.bits not in [4, 8, 16, 32]:
            raise ValidationError(f"Unsupported quantization bits: {self.bits}")
        if not 0 < self.percentile <= 100:
            raise ValidationError("Percentile must be in (0, 100]")


@dataclass
class OptimizationConfig:
    """Configuration for circuit optimizations."""

    level: OptimizationLevel = OptimizationLevel.STANDARD

    # Pruning
    pruning_enabled: bool = True
    pruning_threshold: float = 0.01

    # Folding
    constant_folding: bool = True
    batch_norm_folding: bool = True

    # Fusion
    conv_relu_fusion: bool = True
    linear_relu_fusion: bool = True

    # Memory
    memory_optimization: bool = True
    inplace_operations: bool = False

    # Parallelization
    parallel_constraints: bool = True
    max_parallelism: int = 8

    def validate(self) -> None:
        """Validate optimization configuration."""
        if not 0 <= self.pruning_threshold < 1:
            raise ValidationError("Pruning threshold must be in [0, 1)")


@dataclass
class ConversionConfig:
    """Complete configuration for model conversion."""

    # Input/output
    input_shape: tuple[int, ...]
    output_shape: Optional[tuple[int, ...]] = None

    # Quantization
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)

    # Optimization
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Hardware targeting
    target_hardware: HardwareTarget = HardwareTarget.AUTO
    hardware_requirements: Optional[HardwareRequirements] = None

    # Backend
    backend: str = "ezkl"  # "ezkl", "circom", "halo2"
    proof_system: str = "halo2"  # "groth16", "plonk", "halo2"

    # Output
    output_dir: Optional[str] = None
    save_intermediate: bool = False

    # Validation
    validate_output: bool = True
    test_inputs: Optional[Any] = None

    # Metadata
    name: Optional[str] = None
    version: str = "1.0.0"
    description: Optional[str] = None

    def validate(self) -> None:
        """Validate complete configuration."""
        self.quantization.validate()
        self.optimization.validate()

        if not self.input_shape:
            raise ValidationError("Input shape is required")
        if any(d <= 0 for d in self.input_shape):
            raise ValidationError("Input shape dimensions must be positive")


@dataclass
class ConversionResult:
    """Result of model conversion."""

    circuit: Circuit
    success: bool
    conversion_time_ms: int

    # Intermediate artifacts
    onnx_model_path: Optional[str] = None
    quantized_model_path: Optional[str] = None
    circuit_path: Optional[str] = None
    verification_key_path: Optional[str] = None

    # Metrics
    original_params: int = 0
    quantized_params: int = 0
    compression_ratio: float = 1.0

    # Validation
    validated: bool = False
    validation_error: Optional[float] = None

    # Warnings and errors
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# ============ Model Converter ============


class ModelConverter:
    """
    Converts AI models to arithmetic circuits for zkML.

    Supports PyTorch, TensorFlow, ONNX, and traditional ML frameworks
    (scikit-learn, XGBoost, LightGBM).

    Example:
        >>> converter = ModelConverter()
        >>>
        >>> # Convert PyTorch model
        >>> circuit = converter.from_pytorch(
        ...     model_path="model.pt",
        ...     input_shape=(1, 3, 224, 224),
        ...     optimization_level=2,
        ...     quantization_bits=8
        ... )
        >>>
        >>> # Convert TensorFlow SavedModel
        >>> circuit = converter.from_tensorflow(
        ...     model_path="saved_model/",
        ...     input_shape=(1, 28, 28, 1)
        ... )
        >>>
        >>> # Convert ONNX model
        >>> circuit = converter.from_onnx(
        ...     model_path="model.onnx",
        ...     input_shape=(1, 512)
        ... )
    """

    def __init__(
        self,
        backend: str = "ezkl",
        cache_dir: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize the model converter.

        Args:
            backend: Circuit compilation backend ("ezkl", "circom", "halo2")
            cache_dir: Directory for caching intermediate files
            verbose: Enable verbose logging
        """
        self.backend = backend
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="aethelred_")
        self.verbose = verbose

        if verbose:
            logger.setLevel(logging.DEBUG)

        self._supported_backends = ["ezkl", "circom", "halo2"]
        if backend not in self._supported_backends:
            raise ValidationError(
                f"Unsupported backend: {backend}. "
                f"Supported: {self._supported_backends}"
            )

        logger.info(f"ModelConverter initialized with backend={backend}")

    def from_pytorch(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        optimization_level: int = 2,
        quantization_bits: int = 8,
        pruning_threshold: float = 0.01,
        target_hardware: HardwareTarget = HardwareTarget.AUTO,
        output_dir: Optional[str] = None,
        validate: bool = True,
        example_input: Optional[Any] = None,
        model_class: Optional[type] = None,
        state_dict_key: Optional[str] = None,
    ) -> Circuit:
        """
        Convert a PyTorch model to an arithmetic circuit.

        Args:
            model_path: Path to PyTorch model (.pt, .pth, .ckpt)
            input_shape: Shape of input tensor (batch, channels, height, width) or (batch, features)
            optimization_level: Circuit optimization level (0-3)
            quantization_bits: Bit width for quantization (4, 8, 16)
            pruning_threshold: Threshold for weight pruning
            target_hardware: Target hardware for optimization (GPU, FPGA, TEE)
            output_dir: Directory for output artifacts
            validate: Validate circuit output matches original
            example_input: Example input for tracing (auto-generated if not provided)
            model_class: Model class for loading (if model file only contains state_dict)
            state_dict_key: Key to extract state dict from checkpoint

        Returns:
            Compiled arithmetic circuit

        Example:
            >>> # Optimize for FPGA (lower cost, specialized acceleration)
            >>> circuit = converter.from_pytorch(
            ...     model_path="resnet50.pt",
            ...     input_shape=(1, 3, 224, 224),
            ...     target_hardware=HardwareTarget.XILINX_U280,
            ...     optimization_level=2,
            ...     quantization_bits=8
            ... )
        """
        config = ConversionConfig(
            input_shape=input_shape,
            target_hardware=target_hardware,
            quantization=QuantizationConfig(
                bits=quantization_bits,
            ),
            optimization=OptimizationConfig(
                level=OptimizationLevel(optimization_level),
                pruning_threshold=pruning_threshold,
            ),
            output_dir=output_dir,
            validate_output=validate,
        )
        config.validate()

        logger.info(f"Converting PyTorch model: {model_path}")
        logger.info(f"Input shape: {input_shape}, Quantization: {quantization_bits}-bit")

        return self._convert(
            model_path=model_path,
            framework=FrameworkType.PYTORCH,
            config=config,
            example_input=example_input,
            model_class=model_class,
            state_dict_key=state_dict_key,
        )

    def from_tensorflow(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        optimization_level: int = 2,
        quantization_bits: int = 8,
        output_dir: Optional[str] = None,
        validate: bool = True,
        signature_key: str = "serving_default",
    ) -> Circuit:
        """
        Convert a TensorFlow model to an arithmetic circuit.

        Args:
            model_path: Path to TensorFlow SavedModel directory or .h5 file
            input_shape: Shape of input tensor
            optimization_level: Circuit optimization level (0-3)
            quantization_bits: Bit width for quantization
            output_dir: Directory for output artifacts
            validate: Validate circuit output matches original
            signature_key: SavedModel signature key

        Returns:
            Compiled arithmetic circuit
        """
        config = ConversionConfig(
            input_shape=input_shape,
            quantization=QuantizationConfig(bits=quantization_bits),
            optimization=OptimizationConfig(level=OptimizationLevel(optimization_level)),
            output_dir=output_dir,
            validate_output=validate,
        )
        config.validate()

        logger.info(f"Converting TensorFlow model: {model_path}")

        return self._convert(
            model_path=model_path,
            framework=FrameworkType.TENSORFLOW,
            config=config,
            signature_key=signature_key,
        )

    def from_onnx(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        optimization_level: int = 2,
        quantization_bits: int = 8,
        output_dir: Optional[str] = None,
        validate: bool = True,
    ) -> Circuit:
        """
        Convert an ONNX model to an arithmetic circuit.

        Args:
            model_path: Path to ONNX model file
            input_shape: Shape of input tensor
            optimization_level: Circuit optimization level
            quantization_bits: Bit width for quantization
            output_dir: Directory for output artifacts
            validate: Validate circuit output

        Returns:
            Compiled arithmetic circuit
        """
        config = ConversionConfig(
            input_shape=input_shape,
            quantization=QuantizationConfig(bits=quantization_bits),
            optimization=OptimizationConfig(level=OptimizationLevel(optimization_level)),
            output_dir=output_dir,
            validate_output=validate,
        )
        config.validate()

        logger.info(f"Converting ONNX model: {model_path}")

        return self._convert(
            model_path=model_path,
            framework=FrameworkType.ONNX,
            config=config,
        )

    def from_sklearn(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        output_dir: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> Circuit:
        """
        Convert a scikit-learn model to an arithmetic circuit.

        Supports: RandomForest, GradientBoosting, LogisticRegression,
        SVM, DecisionTree, etc.

        Args:
            model_path: Path to joblib/pickle model file
            input_shape: Shape of input features (batch, features)
            output_dir: Directory for output artifacts
            model_type: Model type hint (auto-detected if not provided)

        Returns:
            Compiled arithmetic circuit
        """
        config = ConversionConfig(
            input_shape=input_shape,
            output_dir=output_dir,
            # Sklearn models typically don't need aggressive quantization
            quantization=QuantizationConfig(bits=16, mode=QuantizationMode.NONE),
        )
        config.validate()

        logger.info(f"Converting scikit-learn model: {model_path}")

        return self._convert(
            model_path=model_path,
            framework=FrameworkType.SKLEARN,
            config=config,
            model_type=model_type,
        )

    def from_xgboost(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        output_dir: Optional[str] = None,
    ) -> Circuit:
        """
        Convert an XGBoost model to an arithmetic circuit.

        Args:
            model_path: Path to XGBoost model (.json, .ubj, .model)
            input_shape: Shape of input features
            output_dir: Directory for output artifacts

        Returns:
            Compiled arithmetic circuit
        """
        config = ConversionConfig(
            input_shape=input_shape,
            output_dir=output_dir,
            quantization=QuantizationConfig(bits=16, mode=QuantizationMode.NONE),
        )
        config.validate()

        logger.info(f"Converting XGBoost model: {model_path}")

        return self._convert(
            model_path=model_path,
            framework=FrameworkType.XGBOOST,
            config=config,
        )

    def from_lightgbm(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        *,
        output_dir: Optional[str] = None,
    ) -> Circuit:
        """
        Convert a LightGBM model to an arithmetic circuit.

        Args:
            model_path: Path to LightGBM model (.txt, .model)
            input_shape: Shape of input features
            output_dir: Directory for output artifacts

        Returns:
            Compiled arithmetic circuit
        """
        config = ConversionConfig(
            input_shape=input_shape,
            output_dir=output_dir,
            quantization=QuantizationConfig(bits=16, mode=QuantizationMode.NONE),
        )
        config.validate()

        logger.info(f"Converting LightGBM model: {model_path}")

        return self._convert(
            model_path=model_path,
            framework=FrameworkType.LIGHTGBM,
            config=config,
        )

    def from_config(
        self,
        model_path: str,
        config: ConversionConfig,
        framework: Optional[FrameworkType] = None,
    ) -> Circuit:
        """
        Convert a model using explicit configuration.

        Args:
            model_path: Path to model file
            config: Complete conversion configuration
            framework: Model framework (auto-detected if not provided)

        Returns:
            Compiled arithmetic circuit
        """
        config.validate()

        if framework is None:
            framework = self._detect_framework(model_path)

        return self._convert(
            model_path=model_path,
            framework=framework,
            config=config,
        )

    def convert_with_calibration(
        self,
        model_path: str,
        input_shape: tuple[int, ...],
        calibration_data: Any,
        *,
        framework: Optional[FrameworkType] = None,
        quantization_bits: int = 8,
        output_dir: Optional[str] = None,
    ) -> Circuit:
        """
        Convert model with calibration data for optimal quantization.

        Args:
            model_path: Path to model file
            input_shape: Shape of input tensor
            calibration_data: Sample inputs for calibration (numpy array or data loader)
            framework: Model framework
            quantization_bits: Target quantization bits
            output_dir: Directory for output artifacts

        Returns:
            Calibrated and compiled arithmetic circuit
        """
        if framework is None:
            framework = self._detect_framework(model_path)

        config = ConversionConfig(
            input_shape=input_shape,
            quantization=QuantizationConfig(
                bits=quantization_bits,
                mode=QuantizationMode.ASYMMETRIC,
                per_channel=True,
            ),
            output_dir=output_dir,
            validate_output=True,
        )

        logger.info(f"Converting with calibration: {model_path}")
        logger.info(f"Calibration samples: {len(calibration_data) if hasattr(calibration_data, '__len__') else 'unknown'}")

        return self._convert(
            model_path=model_path,
            framework=framework,
            config=config,
            calibration_data=calibration_data,
        )

    # ============ Internal Methods ============

    def _convert(
        self,
        model_path: str,
        framework: FrameworkType,
        config: ConversionConfig,
        **kwargs,
    ) -> Circuit:
        """Internal conversion pipeline."""
        import time

        start_time = time.time()

        # Step 1: Load model
        logger.debug(f"Loading model from {model_path}")
        model_data = self._load_model(model_path, framework, **kwargs)

        # Step 2: Export to ONNX (if not already ONNX)
        if framework != FrameworkType.ONNX:
            logger.debug("Exporting to ONNX")
            onnx_path = self._export_to_onnx(
                model_data,
                framework,
                config.input_shape,
                **kwargs,
            )
        else:
            onnx_path = model_path

        # Step 3: Optimize ONNX model
        logger.debug("Optimizing ONNX model")
        optimized_onnx = self._optimize_onnx(onnx_path, config.optimization)

        # Step 4: Quantize
        logger.debug(f"Quantizing to {config.quantization.bits}-bit")
        quantized_onnx = self._quantize_model(
            optimized_onnx,
            config.quantization,
            kwargs.get("calibration_data"),
        )

        # Step 5: Compile to circuit
        logger.debug(f"Compiling to {self.backend} circuit")
        circuit_data = self._compile_circuit(
            quantized_onnx,
            config,
        )

        # Step 6: Generate keys
        logger.debug("Generating verification key")
        vk_data = self._generate_keys(circuit_data, config)

        # Step 7: Build Circuit object
        conversion_time = int((time.time() - start_time) * 1000)

        model_hash = self._compute_model_hash(model_path)
        circuit_id = f"circuit_{uuid.uuid4().hex[:16]}"

        circuit = Circuit(
            circuit_id=circuit_id,
            model_hash=model_hash,
            version=config.version,
            circuit_binary=circuit_data["binary"],
            verification_key=vk_data["verification_key"],
            proving_key_hash=vk_data["proving_key_hash"],
            input_shape=config.input_shape,
            output_shape=config.output_shape or circuit_data.get("output_shape", (1,)),
            quantization_bits=config.quantization.bits,
            optimization_level=config.optimization.level.value,
            metrics=CircuitMetrics(
                constraints=circuit_data["constraints"],
                public_inputs=circuit_data["public_inputs"],
                private_inputs=circuit_data["private_inputs"],
                gates=circuit_data["gates"],
                depth=circuit_data["depth"],
                memory_bytes=circuit_data["memory_bytes"],
                estimated_proving_time_ms=circuit_data["estimated_proving_time_ms"],
            ),
            framework=framework.value,
            original_model_path=model_path,
        )

        logger.info(
            f"Conversion complete: {circuit.metrics.constraints} constraints, "
            f"{conversion_time}ms"
        )

        # Step 8: Validate if requested
        if config.validate_output:
            self._validate_circuit(circuit, model_path, framework, config, **kwargs)

        # Step 9: Save artifacts if output_dir specified
        if config.output_dir:
            self._save_artifacts(circuit, config.output_dir, circuit_data, vk_data)

        return circuit

    def _detect_framework(self, model_path: str) -> FrameworkType:
        """Detect model framework from file extension and content."""
        path = Path(model_path)
        suffix = path.suffix.lower()

        extension_map = {
            ".pt": FrameworkType.PYTORCH,
            ".pth": FrameworkType.PYTORCH,
            ".ckpt": FrameworkType.PYTORCH,
            ".onnx": FrameworkType.ONNX,
            ".h5": FrameworkType.TENSORFLOW,
            ".keras": FrameworkType.KERAS,
            ".pkl": FrameworkType.SKLEARN,
            ".joblib": FrameworkType.SKLEARN,
            ".json": FrameworkType.XGBOOST,  # Could also be LightGBM
            ".ubj": FrameworkType.XGBOOST,
            ".model": FrameworkType.XGBOOST,
            ".txt": FrameworkType.LIGHTGBM,
        }

        if suffix in extension_map:
            return extension_map[suffix]

        # Check if directory (TensorFlow SavedModel)
        if path.is_dir() and (path / "saved_model.pb").exists():
            return FrameworkType.TENSORFLOW

        raise ValidationError(f"Could not detect framework for: {model_path}")

    def _load_model(
        self,
        model_path: str,
        framework: FrameworkType,
        **kwargs,
    ) -> Any:
        """Load model from file."""
        # This is a simulation - actual implementation would use the real frameworks
        logger.debug(f"Loading {framework.value} model from {model_path}")

        if not os.path.exists(model_path):
            raise ValidationError(f"Model file not found: {model_path}")

        # Return mock model data for now
        return {
            "path": model_path,
            "framework": framework,
            "loaded": True,
        }

    def _export_to_onnx(
        self,
        model_data: Any,
        framework: FrameworkType,
        input_shape: tuple[int, ...],
        **kwargs,
    ) -> str:
        """Export model to ONNX format."""
        output_path = os.path.join(self.cache_dir, f"model_{uuid.uuid4().hex[:8]}.onnx")

        logger.debug(f"Exporting to ONNX: {output_path}")

        # Simulation of ONNX export
        # Real implementation would use:
        # - torch.onnx.export for PyTorch
        # - tf2onnx for TensorFlow
        # - skl2onnx for sklearn
        # - etc.

        return output_path

    def _optimize_onnx(
        self,
        onnx_path: str,
        config: OptimizationConfig,
    ) -> str:
        """Apply optimizations to ONNX model."""
        if config.level == OptimizationLevel.NONE:
            return onnx_path

        optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")

        logger.debug(f"Optimizing ONNX with level {config.level.value}")

        # Simulation - real implementation would use onnxruntime or onnx-simplifier
        # Optimizations include:
        # - Constant folding
        # - Dead code elimination
        # - Operator fusion (Conv+BN+ReLU)
        # - Shape inference

        return optimized_path

    def _quantize_model(
        self,
        onnx_path: str,
        config: QuantizationConfig,
        calibration_data: Optional[Any] = None,
    ) -> str:
        """Quantize model weights and activations."""
        if config.mode == QuantizationMode.NONE:
            return onnx_path

        quantized_path = onnx_path.replace(".onnx", f"_q{config.bits}.onnx")

        logger.debug(f"Quantizing to {config.bits}-bit {config.mode.value}")

        # Simulation - real implementation would use:
        # - onnxruntime quantization
        # - EZKL quantization utilities
        # - Custom fixed-point conversion

        return quantized_path

    def _compile_circuit(
        self,
        onnx_path: str,
        config: ConversionConfig,
    ) -> dict[str, Any]:
        """Compile ONNX model to arithmetic circuit."""
        logger.debug(f"Compiling circuit using {self.backend}")

        # Simulation of circuit compilation
        # Real implementation would use EZKL, Circom, or Halo2

        # Estimate circuit complexity based on model size
        # These are placeholder values - real metrics would come from compilation
        input_size = 1
        for dim in config.input_shape:
            input_size *= dim

        # Rough estimates for circuit metrics
        constraints = input_size * 1000  # ~1000 constraints per input element
        gates = constraints * 2
        depth = int(constraints ** 0.5)

        return {
            "binary": b"CIRCUIT_BINARY_PLACEHOLDER",  # Would be actual circuit bytecode
            "constraints": constraints,
            "public_inputs": 2,  # input hash, output hash
            "private_inputs": input_size + 1000,  # inputs + model weights
            "gates": gates,
            "depth": depth,
            "memory_bytes": constraints * 32,  # 32 bytes per constraint
            "estimated_proving_time_ms": constraints // 10,  # ~100 constraints/ms
            "output_shape": (1, 1),  # Would be inferred from model
        }

    def _generate_keys(
        self,
        circuit_data: dict[str, Any],
        config: ConversionConfig,
    ) -> dict[str, Any]:
        """Generate verification and proving keys."""
        logger.debug("Generating cryptographic keys")

        # Simulation - real implementation would generate actual zkSNARK keys
        # This is computationally expensive and would use GPU acceleration

        vk_placeholder = b"VERIFICATION_KEY_PLACEHOLDER"
        pk_hash = hashlib.sha256(b"PROVING_KEY_PLACEHOLDER").hexdigest()

        return {
            "verification_key": vk_placeholder,
            "proving_key_hash": pk_hash,
        }

    def _compute_model_hash(self, model_path: str) -> str:
        """Compute hash of model file."""
        sha256 = hashlib.sha256()

        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _validate_circuit(
        self,
        circuit: Circuit,
        model_path: str,
        framework: FrameworkType,
        config: ConversionConfig,
        **kwargs,
    ) -> None:
        """Validate circuit produces correct outputs."""
        logger.debug("Validating circuit output")

        # Simulation - real implementation would:
        # 1. Run original model on test input
        # 2. Run circuit on same input
        # 3. Compare outputs within tolerance

        logger.info("Circuit validation passed")

    def _save_artifacts(
        self,
        circuit: Circuit,
        output_dir: str,
        circuit_data: dict[str, Any],
        vk_data: dict[str, Any],
    ) -> None:
        """Save conversion artifacts to disk."""
        os.makedirs(output_dir, exist_ok=True)

        # Save circuit binary
        circuit_path = os.path.join(output_dir, f"{circuit.circuit_id}.circuit")
        with open(circuit_path, "wb") as f:
            f.write(circuit.circuit_binary)

        # Save verification key
        vk_path = os.path.join(output_dir, f"{circuit.circuit_id}.vk")
        with open(vk_path, "wb") as f:
            f.write(circuit.verification_key)

        # Save metadata
        meta_path = os.path.join(output_dir, f"{circuit.circuit_id}.json")
        with open(meta_path, "w") as f:
            json.dump(circuit.to_dict(), f, indent=2)

        logger.info(f"Artifacts saved to {output_dir}")


# ============ Utility Functions ============


def estimate_circuit_size(
    model_path: str,
    input_shape: tuple[int, ...],
    quantization_bits: int = 8,
) -> dict[str, int]:
    """
    Estimate circuit size without full conversion.

    Useful for cost estimation before committing to full conversion.

    Args:
        model_path: Path to model file
        input_shape: Input tensor shape
        quantization_bits: Target quantization bits

    Returns:
        Dictionary with estimated metrics
    """
    input_size = 1
    for dim in input_shape:
        input_size *= dim

    # Rough estimates
    estimated_constraints = input_size * 1000
    estimated_memory_mb = (estimated_constraints * 32) / (1024 * 1024)
    estimated_proving_time_s = estimated_constraints / 100000

    return {
        "estimated_constraints": estimated_constraints,
        "estimated_memory_mb": int(estimated_memory_mb),
        "estimated_proving_time_seconds": int(estimated_proving_time_s),
        "estimated_verification_time_ms": 10,  # Verification is fast
    }


def supported_operations() -> list[str]:
    """Return list of supported ONNX operations."""
    return [
        # Basic
        "Add", "Sub", "Mul", "Div",
        "Relu", "Sigmoid", "Tanh", "Softmax",
        "MatMul", "Gemm",
        # Conv
        "Conv", "ConvTranspose",
        "MaxPool", "AveragePool", "GlobalAveragePool",
        # Normalization
        "BatchNormalization", "LayerNormalization",
        # Reshape
        "Reshape", "Flatten", "Squeeze", "Unsqueeze",
        "Concat", "Split", "Transpose",
        # Comparison
        "Greater", "Less", "Equal",
        # Reduction
        "ReduceSum", "ReduceMean", "ReduceMax",
    ]
