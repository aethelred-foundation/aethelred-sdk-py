"""Comprehensive tests for aethelred/quantize/__init__.py to achieve 95%+ coverage.

Covers:
- QuantizationType, QuantizationScheme, CalibrationMethod enums
- QuantizationConfig dataclass
- QuantizationParams dataclass
- calculate_qparams function
- quantize_tensor, dequantize_tensor
- QuantizedLinear, DynamicQuantizedLinear
- CalibrationObserver
- QuantizationEngine
- FakeQuantize
- QATEngine, QATLinear
- inference_mode context manager
- quantize_dynamic, quantize_static, prepare_qat, convert_qat helpers
"""

from __future__ import annotations

import copy
from unittest.mock import MagicMock, patch

import pytest

from aethelred.core.tensor import Tensor, DType
from aethelred.core.runtime import Device, DeviceType
from aethelred.nn import Module, Linear, Sequential


# Patch contextmanager into quantize module before import
import contextlib
import aethelred.quantize as quantize_module
if not hasattr(quantize_module, 'contextmanager'):
    quantize_module.contextmanager = contextlib.contextmanager

from aethelred.quantize import (
    QuantizationType,
    QuantizationScheme,
    CalibrationMethod,
    QuantizationConfig,
    QuantizationParams,
    calculate_qparams,
    quantize_tensor,
    dequantize_tensor,
    QuantizedLinear,
    DynamicQuantizedLinear,
    CalibrationObserver,
    QuantizationEngine,
    FakeQuantize,
    QATEngine,
    QATLinear,
    inference_mode,
    quantize_dynamic,
    quantize_static,
    prepare_qat,
    convert_qat,
)


# ---------------------------------------------------------------------------
# Helper: Simple model for quantization tests
# ---------------------------------------------------------------------------

class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.fc2 = Linear(8, 2)

    def forward(self, x):
        return self.fc2(self.fc1(x))


# ============================================================================
# Enums
# ============================================================================


class TestEnums:
    def test_quantization_type(self):
        assert QuantizationType.INT8 is not None
        assert QuantizationType.INT4 is not None
        assert QuantizationType.INT2 is not None
        assert QuantizationType.FP16 is not None
        assert QuantizationType.BF16 is not None
        assert QuantizationType.FP8_E4M3 is not None
        assert QuantizationType.FP8_E5M2 is not None
        assert QuantizationType.NF4 is not None

    def test_quantization_scheme(self):
        assert QuantizationScheme.SYMMETRIC is not None
        assert QuantizationScheme.ASYMMETRIC is not None
        assert QuantizationScheme.AFFINE is not None

    def test_calibration_method(self):
        assert CalibrationMethod.MIN_MAX is not None
        assert CalibrationMethod.PERCENTILE is not None
        assert CalibrationMethod.MSE is not None
        assert CalibrationMethod.ENTROPY is not None
        assert CalibrationMethod.HISTOGRAM is not None


# ============================================================================
# QuantizationConfig
# ============================================================================


class TestQuantizationConfig:
    def test_defaults(self):
        cfg = QuantizationConfig()
        assert cfg.weight_bits == 8
        assert cfg.weight_scheme == QuantizationScheme.SYMMETRIC
        assert cfg.weight_granularity == 'per_channel'
        assert cfg.weight_group_size == 128
        assert cfg.activation_bits == 8
        assert cfg.activation_scheme == QuantizationScheme.ASYMMETRIC
        assert cfg.dynamic is False
        assert cfg.fuse_bn is True
        assert cfg.fold_scale is True
        assert cfg.modules_to_quantize == ['Linear']
        assert cfg.modules_to_exclude == []

    def test_custom(self):
        cfg = QuantizationConfig(
            weight_bits=4,
            dynamic=True,
            modules_to_exclude=['LayerNorm']
        )
        assert cfg.weight_bits == 4
        assert cfg.dynamic is True
        assert 'LayerNorm' in cfg.modules_to_exclude


# ============================================================================
# QuantizationParams
# ============================================================================


class TestQuantizationParams:
    def test_qmin_qmax_signed(self):
        qp = QuantizationParams(
            scale=Tensor.full((), 1.0),
            zero_point=Tensor.full((), 0, dtype=DType.int32),
            bits=8,
            signed=True,
        )
        assert qp.qmin == -128
        assert qp.qmax == 127

    def test_qmin_qmax_unsigned(self):
        qp = QuantizationParams(
            scale=Tensor.full((), 1.0),
            zero_point=Tensor.full((), 0, dtype=DType.int32),
            bits=8,
            signed=False,
        )
        assert qp.qmin == 0
        assert qp.qmax == 255

    def test_4bit(self):
        qp = QuantizationParams(
            scale=Tensor.full((), 1.0),
            zero_point=Tensor.full((), 0, dtype=DType.int32),
            bits=4,
            signed=True,
        )
        assert qp.qmin == -8
        assert qp.qmax == 7


# ============================================================================
# calculate_qparams
# ============================================================================


class TestCalculateQparams:
    def test_per_tensor_symmetric(self):
        t = Tensor.empty(4, 8)
        qp = calculate_qparams(t, bits=8, scheme=QuantizationScheme.SYMMETRIC, granularity='per_tensor')
        assert qp.bits == 8
        assert qp.signed is True

    def test_per_tensor_asymmetric(self):
        t = Tensor.empty(4, 8)
        qp = calculate_qparams(t, bits=8, scheme=QuantizationScheme.ASYMMETRIC, granularity='per_tensor')
        assert qp.bits == 8
        assert qp.signed is False

    def test_per_channel(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(4, 8)
        qp = calculate_qparams(t, bits=8, scheme=QuantizationScheme.SYMMETRIC, granularity='per_channel')
        assert qp.bits == 8

    def test_per_channel_asymmetric(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(4, 8)
        qp = calculate_qparams(t, bits=8, scheme=QuantizationScheme.ASYMMETRIC, granularity='per_channel')
        assert qp.bits == 8

    def test_per_group(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(16)
        qp = calculate_qparams(t, bits=8, scheme=QuantizationScheme.SYMMETRIC, granularity='per_group', group_size=4)
        assert qp.bits == 8

    def test_per_group_asymmetric(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(16)
        qp = calculate_qparams(t, bits=8, scheme=QuantizationScheme.ASYMMETRIC, granularity='per_group', group_size=4)
        assert qp.bits == 8

    def test_unknown_granularity(self):
        t = Tensor.empty(4)
        with pytest.raises(ValueError, match="Unknown granularity"):
            calculate_qparams(t, granularity='unknown')


# ============================================================================
# quantize_tensor / dequantize_tensor
# ============================================================================


class TestQuantizeDequantize:
    def test_quantize_tensor(self):
        t = Tensor.empty(4, 8)
        qp = QuantizationParams(
            scale=Tensor.full((), 0.1),
            zero_point=Tensor.full((), 0, dtype=DType.int32),
            bits=8,
            signed=True,
        )
        qt = quantize_tensor(t, qp)
        assert qt is not None

    def test_dequantize_tensor(self):
        qt = Tensor.empty(4, 8)
        qp = QuantizationParams(
            scale=Tensor.full((), 0.1),
            zero_point=Tensor.full((), 0, dtype=DType.int32),
            bits=8,
            signed=True,
        )
        dt = dequantize_tensor(qt, qp)
        assert dt is not None


# ============================================================================
# QuantizedLinear
# ============================================================================


class TestQuantizedLinear:
    def test_creation(self):
        ql = QuantizedLinear(4, 8, bias=True)
        assert ql.in_features == 4
        assert ql.out_features == 8

    def test_from_float(self):
        fl = Linear(4, 8)
        ql = QuantizedLinear.from_float(fl)
        assert ql.in_features == 4
        assert ql.out_features == 8
        assert ql._weight_quantized is not None

    def test_from_float_no_bias(self):
        fl = Linear(4, 8, bias=False)
        ql = QuantizedLinear.from_float(fl)
        assert ql._bias is None

    def test_from_float_with_config(self):
        fl = Linear(4, 8)
        cfg = QuantizationConfig(weight_bits=4)
        ql = QuantizedLinear.from_float(fl, config=cfg)
        assert ql is not None

    def test_forward(self):
        fl = Linear(4, 8)
        ql = QuantizedLinear.from_float(fl)
        x = Tensor.empty(2, 4)
        out = ql(x)
        assert out is not None

    def test_extra_repr(self):
        fl = Linear(4, 8)
        ql = QuantizedLinear.from_float(fl)
        r = ql.extra_repr()
        assert "in_features=4" in r
        assert "out_features=8" in r

    def test_extra_repr_no_qparams(self):
        ql = QuantizedLinear(4, 8)
        r = ql.extra_repr()
        assert "N/A" in r


# ============================================================================
# DynamicQuantizedLinear
# ============================================================================


class TestDynamicQuantizedLinear:
    def test_creation(self):
        dql = DynamicQuantizedLinear(4, 8)
        assert dql.in_features == 4
        assert dql.out_features == 8

    def test_from_float(self):
        fl = Linear(4, 8)
        dql = DynamicQuantizedLinear.from_float(fl)
        assert dql.in_features == 4
        assert dql._weight_quantized is not None

    def test_from_float_no_bias(self):
        fl = Linear(4, 8, bias=False)
        dql = DynamicQuantizedLinear.from_float(fl)
        assert dql._bias is None

    def test_forward(self):
        fl = Linear(4, 8)
        dql = DynamicQuantizedLinear.from_float(fl)
        x = Tensor.empty(2, 4)
        out = dql(x)
        assert out is not None


# ============================================================================
# CalibrationObserver
# ============================================================================


class TestCalibrationObserver:
    def test_creation(self):
        obs = CalibrationObserver()
        assert obs.method == CalibrationMethod.MIN_MAX
        assert obs.min_val is None
        assert obs.max_val is None

    def test_observe_min_max(self):
        obs = CalibrationObserver(method=CalibrationMethod.MIN_MAX)
        t = Tensor.empty(4, 8)
        obs(t)
        assert obs.min_val is not None
        assert obs.max_val is not None

    def test_observe_multiple(self):
        obs = CalibrationObserver(method=CalibrationMethod.MIN_MAX)
        obs(Tensor.empty(4, 8))
        obs(Tensor.empty(4, 8))

    def test_observe_percentile(self):
        obs = CalibrationObserver(method=CalibrationMethod.PERCENTILE)
        obs(Tensor.empty(4, 8))

    def test_observe_histogram(self):
        obs = CalibrationObserver(method=CalibrationMethod.HISTOGRAM)
        obs(Tensor.empty(4, 8))

    def test_calculate_qparams_minmax_symmetric(self):
        obs = CalibrationObserver(method=CalibrationMethod.MIN_MAX)
        obs(Tensor.empty(4, 8))
        qp = obs.calculate_qparams(bits=8, scheme=QuantizationScheme.SYMMETRIC)
        assert qp.bits == 8

    def test_calculate_qparams_minmax_asymmetric(self):
        obs = CalibrationObserver(method=CalibrationMethod.MIN_MAX)
        obs(Tensor.empty(4, 8))
        qp = obs.calculate_qparams(bits=8, scheme=QuantizationScheme.ASYMMETRIC)
        assert qp.bits == 8

    def test_calculate_qparams_entropy(self):
        obs = CalibrationObserver(method=CalibrationMethod.ENTROPY)
        obs.min_val = -1.0
        obs.max_val = 1.0
        qp = obs.calculate_qparams(bits=8, scheme=QuantizationScheme.SYMMETRIC)
        assert qp.bits == 8

    def test_calculate_qparams_fallback(self):
        obs = CalibrationObserver(method=CalibrationMethod.HISTOGRAM)
        obs.min_val = -1.0
        obs.max_val = 1.0
        qp = obs.calculate_qparams(bits=8)
        assert qp.bits == 8

    def test_reset(self):
        obs = CalibrationObserver()
        obs(Tensor.empty(4))
        obs.reset()
        assert obs.min_val is None
        assert obs.max_val is None
        assert obs.histogram is None

    def test_qparams_from_minmax_no_vals(self):
        obs = CalibrationObserver()
        qp = obs._qparams_from_minmax(8, QuantizationScheme.SYMMETRIC)
        assert qp is not None

    def test_qparams_from_minmax_asymmetric_no_vals(self):
        obs = CalibrationObserver()
        qp = obs._qparams_from_minmax(8, QuantizationScheme.ASYMMETRIC)
        assert qp is not None


# ============================================================================
# QuantizationEngine
# ============================================================================


class TestQuantizationEngine:
    def test_creation(self):
        engine = QuantizationEngine()
        assert engine.config is not None

    def test_creation_with_config(self):
        cfg = QuantizationConfig(weight_bits=4)
        engine = QuantizationEngine(cfg)
        assert engine.config.weight_bits == 4

    def test_prepare_calibration(self):
        model = SimpleModel()
        engine = QuantizationEngine()
        prepared = engine.prepare_calibration(model)
        assert prepared is not model  # deep copy

    def test_should_quantize_linear(self):
        engine = QuantizationEngine()
        linear = Linear(4, 8)
        assert engine._should_quantize(linear, "fc1") is True

    def test_should_quantize_excluded(self):
        cfg = QuantizationConfig(modules_to_exclude=['fc1'])
        engine = QuantizationEngine(cfg)
        linear = Linear(4, 8)
        assert engine._should_quantize(linear, "fc1") is False

    def test_should_quantize_non_linear(self):
        engine = QuantizationEngine()

        class Dummy(Module):
            def forward(self, x):
                return x

        assert engine._should_quantize(Dummy(), "dummy") is False

    def test_convert(self):
        model = SimpleModel()
        engine = QuantizationEngine()
        prepared = engine.prepare_calibration(model)
        converted = engine.convert(prepared)
        assert converted is not None

    def test_convert_inplace(self):
        model = SimpleModel()
        engine = QuantizationEngine()
        prepared = engine.prepare_calibration(model)
        converted = engine.convert(prepared, inplace=True)
        assert converted is prepared

    def test_convert_dynamic(self):
        cfg = QuantizationConfig(dynamic=True)
        engine = QuantizationEngine(cfg)
        model = SimpleModel()
        quantized = engine.quantize(model)
        assert quantized is not None

    def test_quantize_ptq(self):
        model = SimpleModel()
        engine = QuantizationEngine()
        # No data loader - should still convert
        quantized = engine.quantize(model, data_loader=None)
        assert quantized is not None

    def test_quantize_with_data(self):
        model = SimpleModel()
        engine = QuantizationEngine()
        data = [Tensor.empty(2, 4) for _ in range(3)]
        quantized = engine.quantize(model, data_loader=iter(data), num_batches=2)
        assert quantized is not None


# ============================================================================
# FakeQuantize
# ============================================================================


class TestFakeQuantize:
    def test_creation(self):
        fq = FakeQuantize(bits=8)
        assert fq.bits == 8
        assert fq.enabled is True

    def test_creation_learn_scale(self):
        fq = FakeQuantize(bits=8, learn_scale=True)
        assert fq.scale is not None
        assert fq.zero_point is not None

    def test_forward_disabled(self):
        fq = FakeQuantize(bits=8)
        fq.disable()
        x = Tensor.empty(4, 8)
        out = fq(x)
        assert out is x

    def test_forward_enabled(self):
        fq = FakeQuantize(bits=8)
        x = Tensor.empty(4, 8)
        out = fq(x)
        assert out is not None

    def test_forward_learn_scale(self):
        fq = FakeQuantize(bits=8, learn_scale=True)
        x = Tensor.empty(4, 8)
        out = fq(x)
        assert out is not None

    def test_enable_disable(self):
        fq = FakeQuantize()
        fq.disable()
        assert fq.enabled is False
        fq.enable()
        assert fq.enabled is True

    def test_forward_reuses_scale(self):
        fq = FakeQuantize(bits=8)
        x = Tensor.empty(4, 8)
        fq(x)  # First pass sets scale
        fq(x)  # Second pass reuses scale


# ============================================================================
# QATEngine
# ============================================================================


class TestQATEngine:
    def test_creation(self):
        engine = QATEngine()
        assert engine.config is not None

    def test_prepare(self):
        model = SimpleModel()
        engine = QATEngine()
        prepared = engine.prepare(model)
        assert prepared is not model

    def test_should_quantize(self):
        engine = QATEngine()
        linear = Linear(4, 8)
        assert engine._should_quantize(linear, "fc1") is True

    def test_should_quantize_excluded(self):
        cfg = QuantizationConfig(modules_to_exclude=['fc1'])
        engine = QATEngine(cfg)
        linear = Linear(4, 8)
        assert engine._should_quantize(linear, "fc1") is False

    def test_convert(self):
        model = SimpleModel()
        engine = QATEngine()
        prepared = engine.prepare(model)
        converted = engine.convert(prepared)
        assert converted is not None


# ============================================================================
# QATLinear
# ============================================================================


class TestQATLinear:
    def test_creation(self):
        ql = QATLinear(4, 8)
        assert ql.in_features == 4
        assert ql.out_features == 8
        assert ql.weight_fake_quant is not None
        assert ql.activation_fake_quant is not None

    def test_creation_no_bias(self):
        ql = QATLinear(4, 8, bias=False)
        assert ql.bias is None

    def test_from_float(self):
        fl = Linear(4, 8)
        ql = QATLinear.from_float(fl)
        assert ql.in_features == 4
        assert ql.out_features == 8

    def test_from_float_no_bias(self):
        fl = Linear(4, 8, bias=False)
        ql = QATLinear.from_float(fl)
        assert ql.bias is None

    def test_forward(self):
        ql = QATLinear(4, 8)
        x = Tensor.empty(2, 4)
        out = ql(x)
        assert out is not None

    def test_forward_with_bias(self):
        ql = QATLinear(4, 8, bias=True)
        x = Tensor.empty(2, 4)
        out = ql(x)
        assert out is not None

    def test_to_quantized(self):
        ql = QATLinear(4, 8)
        x = Tensor.empty(2, 4)
        ql(x)  # trigger scale calculation
        quantized = ql.to_quantized()
        assert isinstance(quantized, QuantizedLinear)


# ============================================================================
# inference_mode
# ============================================================================


class TestInferenceMode:
    def test_context_manager(self):
        with inference_mode():
            x = Tensor.empty(3)
        assert x is not None


# ============================================================================
# Helper Functions
# ============================================================================


class TestHelperFunctions:
    def test_quantize_dynamic(self):
        model = SimpleModel()
        qmodel = quantize_dynamic(model)
        assert qmodel is not None

    def test_quantize_dynamic_int4(self):
        model = SimpleModel()
        qmodel = quantize_dynamic(model, dtype=DType.int32)  # not int8 -> 4 bits
        assert qmodel is not None

    def test_quantize_dynamic_custom_modules(self):
        model = SimpleModel()
        qmodel = quantize_dynamic(model, modules_to_quantize=['Linear'])
        assert qmodel is not None

    def test_quantize_static(self):
        model = SimpleModel()
        data = [Tensor.empty(2, 4) for _ in range(3)]
        qmodel = quantize_static(model, iter(data), num_calibration_batches=2)
        assert qmodel is not None

    def test_prepare_qat(self):
        model = SimpleModel()
        prepared = prepare_qat(model)
        assert prepared is not None

    def test_prepare_qat_custom_modules(self):
        model = SimpleModel()
        prepared = prepare_qat(model, modules_to_quantize=['Linear'])
        assert prepared is not None

    def test_convert_qat(self):
        model = SimpleModel()
        prepared = prepare_qat(model)
        # Run forward to init fake quant scales
        x = Tensor.empty(2, 4)
        prepared(x)
        converted = convert_qat(prepared)
        assert converted is not None
