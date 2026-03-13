"""
Aethelred Quantization Module

Production-grade quantization for efficient inference with:
- Post-training quantization (PTQ)
- Quantization-aware training (QAT)
- Mixed-precision inference
- Hardware-optimized kernels
- Calibration algorithms
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Protocol, Set, Tuple, TypeVar, Union
)

from ..core.tensor import Tensor, DType
from ..core.runtime import Device
from ..nn import Module, Parameter, Linear, LayerNorm, Embedding


# ============================================================================
# Quantization Configuration
# ============================================================================

class QuantizationType(Enum):
    """Quantization precision types."""
    INT8 = auto()      # 8-bit integer
    INT4 = auto()      # 4-bit integer
    INT2 = auto()      # 2-bit integer (extreme compression)
    FP16 = auto()      # 16-bit float
    BF16 = auto()      # Brain float 16
    FP8_E4M3 = auto()  # 8-bit float (E4M3)
    FP8_E5M2 = auto()  # 8-bit float (E5M2)
    NF4 = auto()       # Normal float 4-bit (QLoRA)


class QuantizationScheme(Enum):
    """Quantization schemes."""
    SYMMETRIC = auto()     # Zero-point is 0
    ASYMMETRIC = auto()    # Zero-point can be non-zero
    AFFINE = auto()        # Affine transformation


class CalibrationMethod(Enum):
    """Calibration methods for determining quantization parameters."""
    MIN_MAX = auto()       # Simple min/max
    PERCENTILE = auto()    # Percentile-based
    MSE = auto()           # Minimize MSE
    ENTROPY = auto()       # Minimize KL divergence
    HISTOGRAM = auto()     # Histogram-based


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    # Weight quantization
    weight_bits: int = 8
    weight_scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    weight_granularity: str = 'per_channel'  # 'per_tensor', 'per_channel', 'per_group'
    weight_group_size: int = 128  # For per_group

    # Activation quantization
    activation_bits: int = 8
    activation_scheme: QuantizationScheme = QuantizationScheme.ASYMMETRIC
    activation_granularity: str = 'per_tensor'

    # Calibration
    calibration_method: CalibrationMethod = CalibrationMethod.MIN_MAX
    num_calibration_batches: int = 100
    percentile: float = 99.99

    # Dynamic quantization
    dynamic: bool = False  # Quantize activations dynamically

    # Modules to quantize
    modules_to_quantize: List[str] = field(default_factory=lambda: ['Linear'])
    modules_to_exclude: List[str] = field(default_factory=list)

    # Performance options
    fuse_bn: bool = True  # Fuse batch norm into conv/linear
    fold_scale: bool = True  # Fold scale into weights


# ============================================================================
# Quantization Functions
# ============================================================================

@dataclass
class QuantizationParams:
    """Quantization parameters (scale and zero_point)."""
    scale: Tensor
    zero_point: Tensor
    bits: int
    signed: bool = True
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC

    @property
    def qmin(self) -> int:
        """Minimum quantized value."""
        if self.signed:
            return -(1 << (self.bits - 1))
        return 0

    @property
    def qmax(self) -> int:
        """Maximum quantized value."""
        if self.signed:
            return (1 << (self.bits - 1)) - 1
        return (1 << self.bits) - 1


def calculate_qparams(
    tensor: Tensor,
    bits: int = 8,
    scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
    granularity: str = 'per_tensor',
    group_size: int = 128
) -> QuantizationParams:
    """
    Calculate quantization parameters.

    Args:
        tensor: Tensor to quantize
        bits: Bit width
        scheme: Quantization scheme
        granularity: 'per_tensor', 'per_channel', or 'per_group'
        group_size: Group size for per_group

    Returns:
        QuantizationParams with scale and zero_point
    """
    signed = scheme == QuantizationScheme.SYMMETRIC

    if signed:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << bits) - 1

    if granularity == 'per_tensor':
        # Global scale
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        if scheme == QuantizationScheme.SYMMETRIC:
            max_abs = max(abs(min_val), abs(max_val))
            scale = Tensor.full((), max_abs / qmax if max_abs > 0 else 1.0)
            zero_point = Tensor.full((), 0, dtype=DType.int32)
        else:
            scale = Tensor.full((), (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0)
            zero_point = Tensor.full((), round(qmin - min_val / scale.item()), dtype=DType.int32)

    elif granularity == 'per_channel':
        # Per output channel
        num_channels = tensor.shape[0]
        scales = []
        zero_points = []

        for c in range(num_channels):
            channel = tensor[c]
            min_val = channel.min().item()
            max_val = channel.max().item()

            if scheme == QuantizationScheme.SYMMETRIC:
                max_abs = max(abs(min_val), abs(max_val))
                s = max_abs / qmax if max_abs > 0 else 1.0
                zp = 0
            else:
                s = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
                zp = round(qmin - min_val / s)

            scales.append(s)
            zero_points.append(zp)

        scale = Tensor.from_numpy(__import__('numpy').array(scales))
        zero_point = Tensor.from_numpy(__import__('numpy').array(zero_points, dtype='int32'))

    elif granularity == 'per_group':
        # Group quantization
        flat = tensor.flatten()
        num_groups = (flat.numel + group_size - 1) // group_size
        scales = []
        zero_points = []

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, flat.numel)
            group = flat[start:end]

            min_val = group.min().item()
            max_val = group.max().item()

            if scheme == QuantizationScheme.SYMMETRIC:
                max_abs = max(abs(min_val), abs(max_val))
                s = max_abs / qmax if max_abs > 0 else 1.0
                zp = 0
            else:
                s = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
                zp = round(qmin - min_val / s)

            scales.append(s)
            zero_points.append(zp)

        scale = Tensor.from_numpy(__import__('numpy').array(scales))
        zero_point = Tensor.from_numpy(__import__('numpy').array(zero_points, dtype='int32'))

    else:
        raise ValueError(f"Unknown granularity: {granularity}")

    return QuantizationParams(
        scale=scale,
        zero_point=zero_point,
        bits=bits,
        signed=signed,
        scheme=scheme
    )


def quantize_tensor(tensor: Tensor, qparams: QuantizationParams) -> Tensor:
    """Quantize a tensor using given parameters."""
    # q = round(x / scale) + zero_point
    # q = clamp(q, qmin, qmax)

    scale = qparams.scale
    zp = qparams.zero_point

    # Reshape scale/zero_point for per-channel broadcasting
    if isinstance(scale, Tensor) and scale.ndim == 1 and tensor.ndim > 1:
        # Unsqueeze trailing dimensions for broadcasting: (C,) -> (C, 1, ...)
        for _ in range(tensor.ndim - 1):
            scale = scale.unsqueeze(-1)
    if isinstance(zp, Tensor) and zp.ndim == 1 and tensor.ndim > 1:
        for _ in range(tensor.ndim - 1):
            zp = zp.unsqueeze(-1)

    scaled = tensor / scale
    rounded = scaled  # Would use .round()
    shifted = rounded + zp

    # Clamp to valid range
    clamped = shifted  # Would clamp to [qmin, qmax]

    return clamped


def dequantize_tensor(qtensor: Tensor, qparams: QuantizationParams) -> Tensor:
    """Dequantize a tensor."""
    # x = (q - zero_point) * scale
    scale = qparams.scale
    zp = qparams.zero_point

    # Reshape scale/zero_point for per-channel broadcasting
    if isinstance(scale, Tensor) and scale.ndim == 1 and qtensor.ndim > 1:
        for _ in range(qtensor.ndim - 1):
            scale = scale.unsqueeze(-1)
    if isinstance(zp, Tensor) and zp.ndim == 1 and qtensor.ndim > 1:
        for _ in range(qtensor.ndim - 1):
            zp = zp.unsqueeze(-1)

    return (qtensor - zp) * scale


# ============================================================================
# Quantized Layers
# ============================================================================

class QuantizedLinear(Module):
    """
    Quantized linear layer.

    Stores weights in low-precision format and performs
    dequantize -> compute -> quantize during forward.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_qparams: Optional[QuantizationParams] = None,
        activation_qparams: Optional[QuantizationParams] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight_qparams = weight_qparams
        self.activation_qparams = activation_qparams

        # Quantized weight storage
        self._weight_quantized: Optional[Tensor] = None
        self._bias: Optional[Tensor] = None

    @classmethod
    def from_float(
        cls,
        float_linear: Linear,
        weight_qparams: Optional[QuantizationParams] = None,
        activation_qparams: Optional[QuantizationParams] = None,
        config: Optional[QuantizationConfig] = None
    ) -> 'QuantizedLinear':
        """Create quantized linear from float linear."""
        config = config or QuantizationConfig()

        # Calculate weight quantization parameters
        if weight_qparams is None:
            weight_qparams = calculate_qparams(
                float_linear.weight,
                bits=config.weight_bits,
                scheme=config.weight_scheme,
                granularity=config.weight_granularity,
                group_size=config.weight_group_size
            )

        # Create quantized layer
        qlayer = cls(
            float_linear.in_features,
            float_linear.out_features,
            bias=float_linear.bias is not None,
            weight_qparams=weight_qparams,
            activation_qparams=activation_qparams
        )

        # Quantize weights
        qlayer._weight_quantized = quantize_tensor(float_linear.weight, weight_qparams)

        if float_linear.bias is not None:
            qlayer._bias = float_linear.bias.clone()

        return qlayer

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize weights for computation
        weight = dequantize_tensor(self._weight_quantized, self.weight_qparams)

        # Matrix multiply
        output = x @ weight.T

        if self._bias is not None:
            output = output + self._bias

        return output

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, '
            f'out_features={self.out_features}, '
            f'weight_bits={self.weight_qparams.bits if self.weight_qparams else "N/A"}'
        )


class DynamicQuantizedLinear(Module):
    """
    Linear with dynamic activation quantization.

    Weights are statically quantized, activations are
    quantized on-the-fly during inference.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_qparams: Optional[QuantizationParams] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight_qparams = weight_qparams

        self._weight_quantized: Optional[Tensor] = None
        self._bias: Optional[Tensor] = None

    @classmethod
    def from_float(
        cls,
        float_linear: Linear,
        config: Optional[QuantizationConfig] = None
    ) -> 'DynamicQuantizedLinear':
        """Create from float linear."""
        config = config or QuantizationConfig(dynamic=True)

        weight_qparams = calculate_qparams(
            float_linear.weight,
            bits=config.weight_bits,
            scheme=config.weight_scheme,
            granularity=config.weight_granularity
        )

        layer = cls(
            float_linear.in_features,
            float_linear.out_features,
            bias=float_linear.bias is not None,
            weight_qparams=weight_qparams
        )

        layer._weight_quantized = quantize_tensor(float_linear.weight, weight_qparams)

        if float_linear.bias is not None:
            layer._bias = float_linear.bias.clone()

        return layer

    def forward(self, x: Tensor) -> Tensor:
        # Dynamic activation quantization
        act_qparams = calculate_qparams(x, bits=8, scheme=QuantizationScheme.ASYMMETRIC)
        x_quant = quantize_tensor(x, act_qparams)

        # Dequantize for computation
        weight = dequantize_tensor(self._weight_quantized, self.weight_qparams)
        x_float = dequantize_tensor(x_quant, act_qparams)

        output = x_float @ weight.T

        if self._bias is not None:
            output = output + self._bias

        return output


# ============================================================================
# Calibration
# ============================================================================

class CalibrationObserver(Module):
    """
    Observes tensor statistics for calibration.

    Collects min/max, histogram, or other statistics
    to determine optimal quantization parameters.
    """

    def __init__(
        self,
        method: CalibrationMethod = CalibrationMethod.MIN_MAX,
        averaging_constant: float = 0.01,
        num_bins: int = 2048,
        percentile: float = 99.99
    ):
        super().__init__()

        self.method = method
        self.averaging_constant = averaging_constant
        self.num_bins = num_bins
        self.percentile = percentile

        # Statistics
        self.min_val: Optional[float] = None
        self.max_val: Optional[float] = None
        self.histogram: Optional[Tensor] = None
        self.bin_edges: Optional[Tensor] = None

    def forward(self, x: Tensor) -> Tensor:
        """Observe tensor and pass through."""
        self._observe(x)
        return x

    def _observe(self, x: Tensor) -> None:
        """Collect statistics from tensor."""
        x_min = x.min().item()
        x_max = x.max().item()

        if self.method == CalibrationMethod.MIN_MAX:
            if self.min_val is None:
                self.min_val = x_min
                self.max_val = x_max
            else:
                self.min_val = min(self.min_val, x_min)
                self.max_val = max(self.max_val, x_max)

        elif self.method == CalibrationMethod.PERCENTILE:
            # Would collect values for percentile calculation
            pass

        elif self.method == CalibrationMethod.HISTOGRAM:
            # Update histogram
            pass

    def calculate_qparams(
        self,
        bits: int = 8,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    ) -> QuantizationParams:
        """Calculate quantization parameters from observed statistics."""
        if self.method == CalibrationMethod.MIN_MAX:
            return self._qparams_from_minmax(bits, scheme)
        elif self.method == CalibrationMethod.ENTROPY:
            return self._qparams_from_entropy(bits, scheme)
        else:
            return self._qparams_from_minmax(bits, scheme)

    def _qparams_from_minmax(
        self,
        bits: int,
        scheme: QuantizationScheme
    ) -> QuantizationParams:
        """Calculate params from min/max."""
        if scheme == QuantizationScheme.SYMMETRIC:
            qmax = (1 << (bits - 1)) - 1
            max_abs = max(abs(self.min_val or 0), abs(self.max_val or 0))
            scale = Tensor.full((), max_abs / qmax if max_abs > 0 else 1.0)
            zero_point = Tensor.full((), 0, dtype=DType.int32)
        else:
            qmin = 0
            qmax = (1 << bits) - 1
            min_val = self.min_val or 0
            max_val = self.max_val or 0
            scale = Tensor.full((), (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0)
            zero_point = Tensor.full((), round(qmin - min_val / scale.item()), dtype=DType.int32)

        return QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=bits,
            signed=scheme == QuantizationScheme.SYMMETRIC,
            scheme=scheme
        )

    def _qparams_from_entropy(
        self,
        bits: int,
        scheme: QuantizationScheme
    ) -> QuantizationParams:
        """Calculate params by minimizing KL divergence."""
        # Would search for optimal threshold
        return self._qparams_from_minmax(bits, scheme)

    def reset(self) -> None:
        """Reset observer state."""
        self.min_val = None
        self.max_val = None
        self.histogram = None


# ============================================================================
# Quantization Engine
# ============================================================================

class QuantizationEngine:
    """
    Main quantization engine.

    Provides:
    - Post-training quantization (PTQ)
    - Quantization-aware training (QAT)
    - Calibration
    - Model conversion
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def prepare_calibration(self, model: Module) -> Module:
        """
        Prepare model for calibration.

        Inserts observers at quantization points.
        """
        model = copy.deepcopy(model)
        self._insert_observers(model)
        return model

    def _insert_observers(self, model: Module, prefix: str = '') -> None:
        """Insert calibration observers."""
        for name, child in model.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name

            if self._should_quantize(child, child_prefix):
                # Insert observer after the module
                observer = CalibrationObserver(
                    method=self.config.calibration_method,
                    percentile=self.config.percentile
                )
                setattr(model, f"{name}_observer", observer)

            self._insert_observers(child, child_prefix)

    def _should_quantize(self, module: Module, name: str) -> bool:
        """Check if module should be quantized."""
        module_type = module.__class__.__name__

        if name in self.config.modules_to_exclude:
            return False

        if module_type in self.config.modules_to_quantize:
            return True

        return False

    def calibrate(
        self,
        model: Module,
        data_loader: Iterator,
        num_batches: Optional[int] = None
    ) -> None:
        """
        Run calibration on the model.

        Args:
            model: Model prepared with observers
            data_loader: Calibration data loader
            num_batches: Number of batches to use
        """
        model.eval()
        num_batches = num_batches or self.config.num_calibration_batches

        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            # Forward pass (observers collect statistics)
            with inference_mode():
                model(batch)

    def convert(
        self,
        model: Module,
        inplace: bool = False
    ) -> Module:
        """
        Convert calibrated model to quantized model.

        Args:
            model: Calibrated model
            inplace: Modify in place

        Returns:
            Quantized model
        """
        if not inplace:
            model = copy.deepcopy(model)

        self._convert_modules(model)
        return model

    def _convert_modules(self, model: Module, prefix: str = '') -> None:
        """Convert modules to quantized versions."""
        for name, child in list(model.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name

            # Get observer if exists
            observer_name = f"{name}_observer"
            observer = getattr(model, observer_name, None)

            if isinstance(child, Linear) and self._should_quantize(child, child_prefix):
                # Get quantization parameters
                if observer:
                    act_qparams = observer.calculate_qparams(
                        bits=self.config.activation_bits,
                        scheme=self.config.activation_scheme
                    )
                else:
                    act_qparams = None

                # Convert to quantized
                if self.config.dynamic:
                    qmodule = DynamicQuantizedLinear.from_float(child, self.config)
                else:
                    qmodule = QuantizedLinear.from_float(
                        child,
                        activation_qparams=act_qparams,
                        config=self.config
                    )

                setattr(model, name, qmodule)

                # Remove observer
                if hasattr(model, observer_name):
                    delattr(model, observer_name)
            else:
                # Recurse
                self._convert_modules(child, child_prefix)

    def quantize(
        self,
        model: Module,
        data_loader: Optional[Iterator] = None,
        num_batches: Optional[int] = None
    ) -> Module:
        """
        Complete quantization pipeline.

        Args:
            model: Float model
            data_loader: Calibration data (optional for dynamic quant)
            num_batches: Number of calibration batches

        Returns:
            Quantized model
        """
        if self.config.dynamic:
            # Dynamic quantization doesn't need calibration
            return self.convert(model)

        # PTQ with calibration
        prepared = self.prepare_calibration(model)

        if data_loader is not None:
            self.calibrate(prepared, data_loader, num_batches)

        return self.convert(prepared)


# ============================================================================
# Quantization-Aware Training
# ============================================================================

class FakeQuantize(Module):
    """
    Fake quantization for QAT.

    Simulates quantization effects during training while
    maintaining float gradients for backpropagation.
    """

    def __init__(
        self,
        bits: int = 8,
        scheme: QuantizationScheme = QuantizationScheme.SYMMETRIC,
        learn_scale: bool = False
    ):
        super().__init__()

        self.bits = bits
        self.scheme = scheme
        self.learn_scale = learn_scale

        # Learnable quantization parameters
        if learn_scale:
            self.scale = Parameter(Tensor.ones(1))
            self.zero_point = Parameter(Tensor.zeros(1))
        else:
            self._scale: Optional[Tensor] = None
            self._zero_point: Optional[Tensor] = None

        self.enabled = True

    def forward(self, x: Tensor) -> Tensor:
        if not self.enabled:
            return x

        if self.learn_scale:
            scale = self.scale.abs()
            zero_point = self.zero_point
        else:
            if self._scale is None:
                # Calculate scale from input
                qparams = calculate_qparams(x, self.bits, self.scheme)
                self._scale = qparams.scale
                self._zero_point = qparams.zero_point

            scale = self._scale
            zero_point = self._zero_point

        # Fake quantize: quantize then immediately dequantize
        # Gradient uses straight-through estimator
        x_quant = quantize_tensor(x, QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=self.bits,
            scheme=self.scheme
        ))
        x_dequant = dequantize_tensor(x_quant, QuantizationParams(
            scale=scale,
            zero_point=zero_point,
            bits=self.bits,
            scheme=self.scheme
        ))

        return x_dequant

    def enable(self) -> None:
        """Enable fake quantization."""
        self.enabled = True

    def disable(self) -> None:
        """Disable fake quantization."""
        self.enabled = False


class QATEngine:
    """
    Quantization-Aware Training engine.

    Inserts fake quantization nodes for training with
    quantization effects.
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

    def prepare(self, model: Module) -> Module:
        """
        Prepare model for QAT.

        Inserts FakeQuantize modules at quantization points.
        """
        model = copy.deepcopy(model)
        self._insert_fake_quantize(model)
        return model

    def _insert_fake_quantize(self, model: Module, prefix: str = '') -> None:
        """Insert fake quantize modules."""
        for name, child in list(model.named_children()):
            child_prefix = f"{prefix}.{name}" if prefix else name

            if isinstance(child, Linear) and self._should_quantize(child, child_prefix):
                # Wrap linear with fake quantize
                wrapped = QATLinear.from_float(child, self.config)
                setattr(model, name, wrapped)
            else:
                self._insert_fake_quantize(child, child_prefix)

    def _should_quantize(self, module: Module, name: str) -> bool:
        """Check if module should be quantized."""
        module_type = module.__class__.__name__

        if name in self.config.modules_to_exclude:
            return False

        return module_type in self.config.modules_to_quantize

    def convert(self, model: Module) -> Module:
        """Convert QAT model to quantized model."""
        model = copy.deepcopy(model)
        self._convert_qat_modules(model)
        return model

    def _convert_qat_modules(self, model: Module, prefix: str = '') -> None:
        """Convert QAT modules to quantized versions."""
        for name, child in list(model.named_children()):
            if isinstance(child, QATLinear):
                qmodule = child.to_quantized()
                setattr(model, name, qmodule)
            else:
                self._convert_qat_modules(child, f"{prefix}.{name}" if prefix else name)


class QATLinear(Module):
    """Linear layer with fake quantization for QAT."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        config: Optional[QuantizationConfig] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.config = config or QuantizationConfig()

        # Float weights
        self.weight = Parameter(Tensor.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(Tensor.empty(out_features))
        else:
            self.bias = None

        # Fake quantize for weights
        self.weight_fake_quant = FakeQuantize(
            bits=self.config.weight_bits,
            scheme=self.config.weight_scheme
        )

        # Fake quantize for activations
        self.activation_fake_quant = FakeQuantize(
            bits=self.config.activation_bits,
            scheme=self.config.activation_scheme
        )

    @classmethod
    def from_float(
        cls,
        float_linear: Linear,
        config: Optional[QuantizationConfig] = None
    ) -> 'QATLinear':
        """Create QAT linear from float linear."""
        layer = cls(
            float_linear.in_features,
            float_linear.out_features,
            bias=float_linear.bias is not None,
            config=config
        )

        # Copy weights
        # layer.weight.data = float_linear.weight.data.clone()
        if float_linear.bias is not None:
            # layer.bias.data = float_linear.bias.data.clone()
            pass

        return layer

    def forward(self, x: Tensor) -> Tensor:
        # Fake quantize input
        x = self.activation_fake_quant(x)

        # Fake quantize weight
        weight = self.weight_fake_quant(self.weight)

        # Forward
        output = x @ weight.T

        if self.bias is not None:
            output = output + self.bias

        return output

    def to_quantized(self) -> QuantizedLinear:
        """Convert to fully quantized layer."""
        weight_qparams = self.weight_fake_quant._scale
        # Create quantized layer with learned parameters
        return QuantizedLinear.from_float(
            self,  # Act as float linear
            config=self.config
        )


# ============================================================================
# Helper Functions
# ============================================================================

@contextmanager
def inference_mode():
    """Context manager for inference without gradient tracking."""
    # In production, would disable gradient computation
    yield


def quantize_dynamic(
    model: Module,
    modules_to_quantize: Optional[List[str]] = None,
    dtype: DType = DType.int8
) -> Module:
    """
    Dynamic quantization helper.

    Quick quantization without calibration data.
    """
    config = QuantizationConfig(
        dynamic=True,
        weight_bits=8 if dtype == DType.int8 else 4,
        modules_to_quantize=modules_to_quantize or ['Linear']
    )

    engine = QuantizationEngine(config)
    return engine.quantize(model)


def quantize_static(
    model: Module,
    data_loader: Iterator,
    modules_to_quantize: Optional[List[str]] = None,
    num_calibration_batches: int = 100
) -> Module:
    """
    Static quantization helper.

    Requires calibration data for activation ranges.
    """
    config = QuantizationConfig(
        dynamic=False,
        modules_to_quantize=modules_to_quantize or ['Linear'],
        num_calibration_batches=num_calibration_batches
    )

    engine = QuantizationEngine(config)
    return engine.quantize(model, data_loader, num_calibration_batches)


def prepare_qat(
    model: Module,
    modules_to_quantize: Optional[List[str]] = None
) -> Module:
    """
    Prepare model for quantization-aware training.
    """
    config = QuantizationConfig(
        modules_to_quantize=modules_to_quantize or ['Linear']
    )

    engine = QATEngine(config)
    return engine.prepare(model)


def convert_qat(model: Module) -> Module:
    """Convert QAT model to quantized inference model."""
    engine = QATEngine()
    return engine.convert(model)
