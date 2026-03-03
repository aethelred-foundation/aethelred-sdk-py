"""Tests for models.converter — config types, enums, conversion pipeline."""

from __future__ import annotations

import pytest

from aethelred.models.converter import (
    FrameworkType,
    QuantizationMode,
    OptimizationLevel,
    QuantizationConfig,
    OptimizationConfig,
    ConversionConfig,
    ConversionResult,
    ModelConverter,
)


class TestFrameworkType:
    """Test supported framework enum."""

    def test_all_frameworks(self) -> None:
        assert len(FrameworkType) >= 5
        assert FrameworkType.PYTORCH.value == "pytorch"
        assert FrameworkType.TENSORFLOW.value == "tensorflow"
        assert FrameworkType.ONNX.value == "onnx"


class TestQuantizationConfig:
    """Test quantization configuration."""

    def test_defaults(self) -> None:
        config = QuantizationConfig()
        assert config.bits == 8
        assert config.mode == QuantizationMode.SYMMETRIC
        assert config.per_channel is True

    def test_custom(self) -> None:
        config = QuantizationConfig(bits=4, mode=QuantizationMode.DYNAMIC)
        assert config.bits == 4

    def test_validate_valid(self) -> None:
        config = QuantizationConfig()
        config.validate()  # Should not raise

    def test_validate_invalid_bits(self) -> None:
        config = QuantizationConfig(bits=0)
        with pytest.raises(Exception):
            config.validate()


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_defaults(self) -> None:
        config = OptimizationConfig()
        assert config.level == OptimizationLevel.STANDARD
        assert config.pruning_enabled is True
        assert config.constant_folding is True

    def test_validate_valid(self) -> None:
        config = OptimizationConfig()
        config.validate()  # Should not raise


class TestConversionConfig:
    """Test conversion configuration."""

    def test_minimal(self) -> None:
        config = ConversionConfig(input_shape=(1, 28, 28))
        assert config.input_shape == (1, 28, 28)
        assert config.version == "1.0.0"

    def test_validate(self) -> None:
        config = ConversionConfig(input_shape=(1, 3, 224, 224))
        config.validate()  # Should not raise

    def test_with_custom_quantization(self) -> None:
        config = ConversionConfig(
            input_shape=(1, 784),
            quantization=QuantizationConfig(bits=4),
        )
        assert config.quantization.bits == 4


class TestModelConverter:
    """Test model converter initialization."""

    def test_init_default(self) -> None:
        converter = ModelConverter()
        assert converter is not None

    def test_init_with_backend(self) -> None:
        converter = ModelConverter(backend="circom")
        assert converter is not None

    def test_init_verbose(self) -> None:
        converter = ModelConverter(verbose=True)
        assert converter is not None
