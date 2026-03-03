"""Tests for models.hardware_compiler — compilation config, profiles, backends."""

from __future__ import annotations

import pytest

from aethelred.models.hardware_compiler import (
    OptimizationProfile,
    PrecisionMode,
    CompilerBackend,
    HardwareProfile,
    CompilationConfig,
    CompilationResult,
)
from aethelred.core.types import HardwareTarget


class TestOptimizationProfile:
    """Test optimization profile enum."""

    def test_all_profiles(self) -> None:
        assert OptimizationProfile.LATENCY.value == "latency"
        assert OptimizationProfile.THROUGHPUT.value == "throughput"
        assert OptimizationProfile.BALANCED.value == "balanced"


class TestPrecisionMode:
    """Test precision mode enum."""

    def test_all_modes(self) -> None:
        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.INT8.value == "int8"
        assert PrecisionMode.MIXED.value == "mixed"


class TestCompilerBackend:
    """Test compiler backend enum."""

    def test_all_backends(self) -> None:
        assert CompilerBackend.EZKL.value == "ezkl"
        assert CompilerBackend.CIRCOM.value == "circom"
        assert CompilerBackend.TENSORRT.value == "tensorrt"


class TestHardwareProfile:
    """Test hardware profile dataclass."""

    def test_create(self) -> None:
        profile = HardwareProfile(
            target=HardwareTarget.NVIDIA_A100,
            name="NVIDIA A100",
            vendor="NVIDIA",
            tflops_fp32=19.5,
            tflops_fp16=312.0,
            tflops_int8=624.0,
            memory_gb=80,
        )
        assert profile.name == "NVIDIA A100"
        assert profile.memory_gb == 80


class TestCompilationConfig:
    """Test compilation configuration."""

    def test_minimal(self) -> None:
        config = CompilationConfig(target=HardwareTarget.NVIDIA_A100)
        assert config.profile == OptimizationProfile.BALANCED

    def test_custom(self) -> None:
        config = CompilationConfig(
            target=HardwareTarget.NVIDIA_A100,
            profile=OptimizationProfile.LATENCY,
            precision=PrecisionMode.FP16,
        )
        assert config.precision == PrecisionMode.FP16

    def test_validate(self) -> None:
        config = CompilationConfig(target=HardwareTarget.NVIDIA_A100)
        config.validate()  # Should not raise
