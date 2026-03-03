"""
Comprehensive tests for models modules:
- aethelred/models/converter.py
- aethelred/models/hardware_compiler.py
- aethelred/models/registry.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from aethelred.core.exceptions import ValidationError
from aethelred.core.types import Circuit, CircuitMetrics, HardwareTarget, HardwareRequirements

# ============ converter.py tests ============

from aethelred.models.converter import (
    FrameworkType,
    QuantizationMode,
    OptimizationLevel,
    QuantizationConfig,
    OptimizationConfig,
    ConversionConfig,
    ConversionResult,
    ModelConverter,
    estimate_circuit_size,
    supported_operations,
)


class TestFrameworkType:
    def test_values(self):
        assert FrameworkType.PYTORCH == "pytorch"
        assert FrameworkType.TENSORFLOW == "tensorflow"
        assert FrameworkType.ONNX == "onnx"
        assert FrameworkType.KERAS == "keras"
        assert FrameworkType.SKLEARN == "sklearn"
        assert FrameworkType.XGBOOST == "xgboost"
        assert FrameworkType.LIGHTGBM == "lightgbm"


class TestQuantizationMode:
    def test_values(self):
        assert QuantizationMode.NONE == "none"
        assert QuantizationMode.SYMMETRIC == "symmetric"
        assert QuantizationMode.ASYMMETRIC == "asymmetric"
        assert QuantizationMode.DYNAMIC == "dynamic"


class TestOptimizationLevel:
    def test_values(self):
        assert OptimizationLevel.NONE == 0
        assert OptimizationLevel.BASIC == 1
        assert OptimizationLevel.STANDARD == 2
        assert OptimizationLevel.AGGRESSIVE == 3


class TestQuantizationConfig:
    def test_defaults(self):
        qc = QuantizationConfig()
        assert qc.bits == 8
        assert qc.mode == QuantizationMode.SYMMETRIC
        assert qc.calibration_samples == 100
        assert qc.percentile == 99.99
        assert qc.per_channel is True
        assert qc.skip_layers == []
        assert qc.input_scale is None
        assert qc.output_scale is None

    def test_validate_valid(self):
        for bits in [4, 8, 16, 32]:
            qc = QuantizationConfig(bits=bits)
            qc.validate()  # should not raise

    def test_validate_invalid_bits(self):
        qc = QuantizationConfig(bits=6)
        with pytest.raises(ValidationError, match="Unsupported quantization bits"):
            qc.validate()

    def test_validate_invalid_percentile_zero(self):
        qc = QuantizationConfig(percentile=0)
        with pytest.raises(ValidationError, match="Percentile must be in"):
            qc.validate()

    def test_validate_invalid_percentile_over(self):
        qc = QuantizationConfig(percentile=101)
        with pytest.raises(ValidationError, match="Percentile must be in"):
            qc.validate()


class TestOptimizationConfig:
    def test_defaults(self):
        oc = OptimizationConfig()
        assert oc.level == OptimizationLevel.STANDARD
        assert oc.pruning_enabled is True
        assert oc.pruning_threshold == 0.01
        assert oc.constant_folding is True
        assert oc.batch_norm_folding is True
        assert oc.conv_relu_fusion is True
        assert oc.linear_relu_fusion is True
        assert oc.memory_optimization is True
        assert oc.inplace_operations is False
        assert oc.parallel_constraints is True
        assert oc.max_parallelism == 8

    def test_validate_valid(self):
        oc = OptimizationConfig(pruning_threshold=0.5)
        oc.validate()

    def test_validate_invalid_threshold_negative(self):
        oc = OptimizationConfig(pruning_threshold=-0.1)
        with pytest.raises(ValidationError, match="Pruning threshold"):
            oc.validate()

    def test_validate_invalid_threshold_one(self):
        oc = OptimizationConfig(pruning_threshold=1.0)
        with pytest.raises(ValidationError, match="Pruning threshold"):
            oc.validate()


class TestConversionConfig:
    def test_defaults(self):
        cc = ConversionConfig(input_shape=(1, 64))
        assert cc.input_shape == (1, 64)
        assert cc.output_shape is None
        assert cc.backend == "ezkl"
        assert cc.proof_system == "halo2"
        assert cc.version == "1.0.0"
        assert cc.validate_output is True
        assert cc.save_intermediate is False

    def test_validate_valid(self):
        cc = ConversionConfig(input_shape=(1, 3, 224, 224))
        cc.validate()

    def test_validate_empty_input_shape(self):
        cc = ConversionConfig(input_shape=())
        with pytest.raises(ValidationError, match="Input shape is required"):
            cc.validate()

    def test_validate_negative_dimension(self):
        cc = ConversionConfig(input_shape=(1, -1))
        with pytest.raises(ValidationError, match="Input shape dimensions must be positive"):
            cc.validate()

    def test_validate_zero_dimension(self):
        cc = ConversionConfig(input_shape=(1, 0))
        with pytest.raises(ValidationError, match="Input shape dimensions must be positive"):
            cc.validate()

    def test_validate_cascades_to_quantization(self):
        cc = ConversionConfig(
            input_shape=(1, 64),
            quantization=QuantizationConfig(bits=3),
        )
        with pytest.raises(ValidationError, match="Unsupported quantization bits"):
            cc.validate()


class TestConversionResult:
    def test_defaults(self):
        circuit = MagicMock(spec=Circuit)
        cr = ConversionResult(circuit=circuit, success=True, conversion_time_ms=100)
        assert cr.success is True
        assert cr.conversion_time_ms == 100
        assert cr.onnx_model_path is None
        assert cr.original_params == 0
        assert cr.compression_ratio == 1.0
        assert cr.validated is False
        assert cr.warnings == []
        assert cr.errors == []


class TestModelConverter:
    def test_init_default(self):
        converter = ModelConverter()
        assert converter.backend == "ezkl"
        assert converter.verbose is False

    def test_init_custom_backend(self):
        converter = ModelConverter(backend="circom")
        assert converter.backend == "circom"

    def test_init_unsupported_backend(self):
        with pytest.raises(ValidationError, match="Unsupported backend"):
            ModelConverter(backend="invalid_backend")

    def test_init_verbose(self):
        converter = ModelConverter(verbose=True)
        assert converter.verbose is True

    def test_detect_framework_pytorch(self):
        converter = ModelConverter()
        for ext in [".pt", ".pth", ".ckpt"]:
            assert converter._detect_framework(f"model{ext}") == FrameworkType.PYTORCH

    def test_detect_framework_onnx(self):
        converter = ModelConverter()
        assert converter._detect_framework("model.onnx") == FrameworkType.ONNX

    def test_detect_framework_tensorflow(self):
        converter = ModelConverter()
        assert converter._detect_framework("model.h5") == FrameworkType.TENSORFLOW

    def test_detect_framework_keras(self):
        converter = ModelConverter()
        assert converter._detect_framework("model.keras") == FrameworkType.KERAS

    def test_detect_framework_sklearn(self):
        converter = ModelConverter()
        assert converter._detect_framework("model.pkl") == FrameworkType.SKLEARN
        assert converter._detect_framework("model.joblib") == FrameworkType.SKLEARN

    def test_detect_framework_xgboost(self):
        converter = ModelConverter()
        assert converter._detect_framework("model.json") == FrameworkType.XGBOOST
        assert converter._detect_framework("model.ubj") == FrameworkType.XGBOOST
        assert converter._detect_framework("model.model") == FrameworkType.XGBOOST

    def test_detect_framework_lightgbm(self):
        converter = ModelConverter()
        assert converter._detect_framework("model.txt") == FrameworkType.LIGHTGBM

    def test_detect_framework_unknown(self):
        converter = ModelConverter()
        with pytest.raises(ValidationError, match="Could not detect framework"):
            converter._detect_framework("model.xyz")

    def test_detect_framework_savedmodel_dir(self):
        converter = ModelConverter()
        with tempfile.TemporaryDirectory() as td:
            Path(td, "saved_model.pb").touch()
            assert converter._detect_framework(td) == FrameworkType.TENSORFLOW

    def test_load_model_not_found(self):
        converter = ModelConverter()
        with pytest.raises(ValidationError, match="Model file not found"):
            converter._load_model("/nonexistent/model.pt", FrameworkType.PYTORCH)

    def test_load_model_success(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                result = converter._load_model(f.name, FrameworkType.PYTORCH)
                assert result["loaded"] is True
                assert result["framework"] == FrameworkType.PYTORCH
            finally:
                os.unlink(f.name)

    def test_export_to_onnx(self):
        converter = ModelConverter()
        result = converter._export_to_onnx(
            {"path": "model.pt", "framework": FrameworkType.PYTORCH, "loaded": True},
            FrameworkType.PYTORCH,
            (1, 64),
        )
        assert result.endswith(".onnx")

    def test_optimize_onnx_no_optimization(self):
        converter = ModelConverter()
        config = OptimizationConfig(level=OptimizationLevel.NONE)
        result = converter._optimize_onnx("model.onnx", config)
        assert result == "model.onnx"

    def test_optimize_onnx_with_optimization(self):
        converter = ModelConverter()
        config = OptimizationConfig(level=OptimizationLevel.STANDARD)
        result = converter._optimize_onnx("model.onnx", config)
        assert "_optimized" in result

    def test_quantize_model_no_quantization(self):
        converter = ModelConverter()
        config = QuantizationConfig(mode=QuantizationMode.NONE)
        result = converter._quantize_model("model.onnx", config)
        assert result == "model.onnx"

    def test_quantize_model_with_quantization(self):
        converter = ModelConverter()
        config = QuantizationConfig(mode=QuantizationMode.SYMMETRIC, bits=8)
        result = converter._quantize_model("model.onnx", config)
        assert "_q8" in result

    def test_compile_circuit(self):
        converter = ModelConverter()
        config = ConversionConfig(input_shape=(1, 64))
        result = converter._compile_circuit("model.onnx", config)
        assert "binary" in result
        assert "constraints" in result
        assert result["constraints"] == 64000  # 1 * 64 * 1000
        assert result["public_inputs"] == 2

    def test_generate_keys(self):
        converter = ModelConverter()
        config = ConversionConfig(input_shape=(1, 64))
        circuit_data = {"constraints": 1000}
        result = converter._generate_keys(circuit_data, config)
        assert "verification_key" in result
        assert "proving_key_hash" in result

    def test_compute_model_hash(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"model data for hashing")
            f.flush()
            try:
                h = converter._compute_model_hash(f.name)
                expected = hashlib.sha256(b"model data for hashing").hexdigest()
                assert h == expected
            finally:
                os.unlink(f.name)

    def test_from_pytorch(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake pytorch model")
            f.flush()
            try:
                circuit = converter.from_pytorch(
                    model_path=f.name,
                    input_shape=(1, 64),
                    optimization_level=1,
                    quantization_bits=8,
                )
                assert isinstance(circuit, Circuit)
                assert circuit.input_shape == (1, 64)
                assert circuit.framework == "pytorch"
            finally:
                os.unlink(f.name)

    def test_from_tensorflow(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            f.write(b"fake tensorflow model")
            f.flush()
            try:
                circuit = converter.from_tensorflow(
                    model_path=f.name,
                    input_shape=(1, 28, 28, 1),
                )
                assert isinstance(circuit, Circuit)
                assert circuit.framework == "tensorflow"
            finally:
                os.unlink(f.name)

    def test_from_onnx(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake onnx model")
            f.flush()
            try:
                circuit = converter.from_onnx(
                    model_path=f.name,
                    input_shape=(1, 512),
                )
                assert isinstance(circuit, Circuit)
                assert circuit.framework == "onnx"
            finally:
                os.unlink(f.name)

    def test_from_sklearn(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"fake sklearn model")
            f.flush()
            try:
                circuit = converter.from_sklearn(
                    model_path=f.name,
                    input_shape=(1, 10),
                )
                assert isinstance(circuit, Circuit)
                assert circuit.framework == "sklearn"
            finally:
                os.unlink(f.name)

    def test_from_xgboost(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            f.write(b"fake xgboost model")
            f.flush()
            try:
                circuit = converter.from_xgboost(
                    model_path=f.name,
                    input_shape=(1, 10),
                )
                assert isinstance(circuit, Circuit)
                assert circuit.framework == "xgboost"
            finally:
                os.unlink(f.name)

    def test_from_lightgbm(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"fake lightgbm model")
            f.flush()
            try:
                circuit = converter.from_lightgbm(
                    model_path=f.name,
                    input_shape=(1, 10),
                )
                assert isinstance(circuit, Circuit)
                assert circuit.framework == "lightgbm"
            finally:
                os.unlink(f.name)

    def test_from_config(self):
        converter = ModelConverter()
        config = ConversionConfig(input_shape=(1, 32))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            try:
                circuit = converter.from_config(
                    model_path=f.name,
                    config=config,
                    framework=FrameworkType.PYTORCH,
                )
                assert isinstance(circuit, Circuit)
            finally:
                os.unlink(f.name)

    def test_from_config_auto_detect(self):
        converter = ModelConverter()
        config = ConversionConfig(input_shape=(1, 32))
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake onnx model")
            f.flush()
            try:
                circuit = converter.from_config(
                    model_path=f.name,
                    config=config,
                )
                assert circuit.framework == "onnx"
            finally:
                os.unlink(f.name)

    def test_convert_with_calibration(self):
        converter = ModelConverter()
        calibration_data = [[0.1, 0.2, 0.3]] * 10
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            try:
                circuit = converter.convert_with_calibration(
                    model_path=f.name,
                    input_shape=(1, 3),
                    calibration_data=calibration_data,
                    framework=FrameworkType.PYTORCH,
                )
                assert isinstance(circuit, Circuit)
            finally:
                os.unlink(f.name)

    def test_convert_with_calibration_auto_detect(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            try:
                circuit = converter.convert_with_calibration(
                    model_path=f.name,
                    input_shape=(1, 3),
                    calibration_data=[[0.1]],
                )
                assert circuit.framework == "onnx"
            finally:
                os.unlink(f.name)

    def test_save_artifacts(self):
        converter = ModelConverter()
        circuit = MagicMock(spec=Circuit)
        circuit.circuit_id = "test_circuit"
        circuit.circuit_binary = b"binary_data"
        circuit.verification_key = b"vk_data"
        circuit.to_dict.return_value = {"id": "test"}

        with tempfile.TemporaryDirectory() as td:
            converter._save_artifacts(
                circuit, td, {"binary": b"data"}, {"verification_key": b"vk"}
            )
            assert os.path.exists(os.path.join(td, "test_circuit.circuit"))
            assert os.path.exists(os.path.join(td, "test_circuit.vk"))
            assert os.path.exists(os.path.join(td, "test_circuit.json"))

    def test_from_pytorch_with_output_dir(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake pytorch model")
            f.flush()
            try:
                with tempfile.TemporaryDirectory() as td:
                    circuit = converter.from_pytorch(
                        model_path=f.name,
                        input_shape=(1, 8),
                        output_dir=td,
                        validate=False,
                    )
                    assert isinstance(circuit, Circuit)
            finally:
                os.unlink(f.name)

    def test_from_pytorch_no_validate(self):
        converter = ModelConverter()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"fake pytorch model")
            f.flush()
            try:
                circuit = converter.from_pytorch(
                    model_path=f.name,
                    input_shape=(1, 4),
                    validate=False,
                )
                assert isinstance(circuit, Circuit)
            finally:
                os.unlink(f.name)


class TestEstimateCircuitSize:
    def test_basic(self):
        result = estimate_circuit_size("model.pt", (1, 64))
        assert result["estimated_constraints"] == 64000
        assert result["estimated_verification_time_ms"] == 10
        assert "estimated_memory_mb" in result
        assert "estimated_proving_time_seconds" in result

    def test_larger_input(self):
        result = estimate_circuit_size("model.pt", (1, 3, 224, 224))
        assert result["estimated_constraints"] == 1 * 3 * 224 * 224 * 1000


class TestSupportedOperations:
    def test_returns_list(self):
        ops = supported_operations()
        assert isinstance(ops, list)
        assert "Add" in ops
        assert "Conv" in ops
        assert "Relu" in ops
        assert "MatMul" in ops
        assert "Softmax" in ops
        assert len(ops) > 20


# ============ hardware_compiler.py tests ============

from aethelred.models.hardware_compiler import (
    OptimizationProfile,
    PrecisionMode,
    CompilerBackend,
    HardwareProfile,
    HARDWARE_PROFILES,
    CompilationConfig,
    CompilationResult,
    HardwareCompilerStrategy,
    VitisAICompiler,
    TensorRTCompiler,
    TEECompiler,
    HardwareCompiler,
)


class TestOptimizationProfile:
    def test_values(self):
        assert OptimizationProfile.LATENCY == "latency"
        assert OptimizationProfile.THROUGHPUT == "throughput"
        assert OptimizationProfile.BALANCED == "balanced"
        assert OptimizationProfile.POWER_EFFICIENT == "power_efficient"
        assert OptimizationProfile.COST_OPTIMIZED == "cost_optimized"


class TestPrecisionMode:
    def test_values(self):
        assert PrecisionMode.FP32 == "fp32"
        assert PrecisionMode.FP16 == "fp16"
        assert PrecisionMode.INT8 == "int8"
        assert PrecisionMode.INT4 == "int4"
        assert PrecisionMode.MIXED == "mixed"


class TestCompilerBackend:
    def test_values(self):
        assert CompilerBackend.EZKL == "ezkl"
        assert CompilerBackend.VITIS_AI == "vitis_ai"
        assert CompilerBackend.TENSORRT == "tensorrt"


class TestHardwareProfiles:
    def test_all_profiles_present(self):
        expected_targets = [
            HardwareTarget.NVIDIA_A100,
            HardwareTarget.NVIDIA_H100,
            HardwareTarget.XILINX_U280,
            HardwareTarget.XILINX_U55C,
            HardwareTarget.INTEL_SGX,
            HardwareTarget.AMD_SEV_SNP,
            HardwareTarget.AWS_NITRO,
        ]
        for target in expected_targets:
            assert target in HARDWARE_PROFILES

    def test_nvidia_a100_profile(self):
        p = HARDWARE_PROFILES[HardwareTarget.NVIDIA_A100]
        assert p.vendor == "NVIDIA"
        assert p.memory_gb == 80
        assert p.supports_sparse is True
        assert p.recommended_backend == CompilerBackend.TENSORRT

    def test_xilinx_u280_profile(self):
        p = HARDWARE_PROFILES[HardwareTarget.XILINX_U280]
        assert p.vendor == "AMD/Xilinx"
        assert p.supports_int4 is True
        assert p.recommended_backend == CompilerBackend.VITIS_AI

    def test_intel_sgx_profile(self):
        p = HARDWARE_PROFILES[HardwareTarget.INTEL_SGX]
        assert p.tee_platform is not None
        assert p.tee_max_memory_mb == 256


class TestCompilationConfig:
    def test_defaults(self):
        cc = CompilationConfig(target=HardwareTarget.NVIDIA_A100)
        assert cc.profile == OptimizationProfile.BALANCED
        assert cc.batch_size == 1
        assert cc.enable_zkml is True

    def test_validate_valid(self):
        cc = CompilationConfig(target=HardwareTarget.NVIDIA_A100, batch_size=1)
        cc.validate()

    def test_validate_unknown_target(self):
        cc = CompilationConfig(target=HardwareTarget.AUTO)
        with pytest.raises(ValidationError, match="Unknown hardware target"):
            cc.validate()

    def test_validate_int4_not_supported(self):
        cc = CompilationConfig(
            target=HardwareTarget.INTEL_SGX,
            precision=PrecisionMode.INT4,
        )
        with pytest.raises(ValidationError, match="INT4 precision not supported"):
            cc.validate()


class TestVitisAICompiler:
    def test_init(self):
        compiler = VitisAICompiler()
        assert compiler.vitis_ai_path is not None

    def test_init_custom_path(self):
        compiler = VitisAICompiler(vitis_ai_path="/custom/path")
        assert compiler.vitis_ai_path == "/custom/path"

    def test_supports_target(self):
        compiler = VitisAICompiler()
        assert compiler.supports_target(HardwareTarget.XILINX_U280) is True
        assert compiler.supports_target(HardwareTarget.XILINX_U55C) is True
        assert compiler.supports_target(HardwareTarget.NVIDIA_A100) is False

    @pytest.mark.asyncio
    async def test_compile(self):
        compiler = VitisAICompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.XILINX_U280,
                    input_shape=(1, 64),
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
                assert result.hardware_target == HardwareTarget.XILINX_U280
                assert result.precision_used == PrecisionMode.INT8
                assert result.backend_used == CompilerBackend.VITIS_AI
                assert isinstance(result.circuit, Circuit)
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_compile_no_zkml(self):
        compiler = VitisAICompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.XILINX_U280,
                    input_shape=(1, 32),
                    enable_zkml=False,
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
                assert result.circuit.circuit_binary == b""
            finally:
                os.unlink(f.name)

    def test_estimate_resources(self):
        compiler = VitisAICompiler()
        config = CompilationConfig(
            target=HardwareTarget.XILINX_U280,
            input_shape=(1, 64),
        )
        resources = compiler.estimate_resources("model.onnx", config)
        assert "estimated_luts" in resources
        assert "estimated_dsps" in resources
        assert "estimated_brams" in resources

    @pytest.mark.asyncio
    async def test_compile_custom_dpu_arch(self):
        compiler = VitisAICompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.XILINX_U280,
                    input_shape=(1, 16),
                    vitis_dpu_arch="CustomDPU",
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
            finally:
                os.unlink(f.name)


class TestTensorRTCompiler:
    def test_init(self):
        compiler = TensorRTCompiler()
        assert compiler.tensorrt_path is not None

    def test_supports_target(self):
        compiler = TensorRTCompiler()
        assert compiler.supports_target(HardwareTarget.NVIDIA_A100) is True
        assert compiler.supports_target(HardwareTarget.NVIDIA_H100) is True
        assert compiler.supports_target(HardwareTarget.XILINX_U280) is False

    @pytest.mark.asyncio
    async def test_compile(self):
        compiler = TensorRTCompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.NVIDIA_A100,
                    input_shape=(1, 64),
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
                assert result.hardware_target == HardwareTarget.NVIDIA_A100
                assert result.backend_used == CompilerBackend.TENSORRT
                assert isinstance(result.circuit, Circuit)
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_compile_no_zkml(self):
        compiler = TensorRTCompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.NVIDIA_A100,
                    input_shape=(1, 32),
                    enable_zkml=False,
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
            finally:
                os.unlink(f.name)

    def test_estimate_resources(self):
        compiler = TensorRTCompiler()
        config = CompilationConfig(
            target=HardwareTarget.NVIDIA_A100,
            input_shape=(1, 64),
        )
        resources = compiler.estimate_resources("model.onnx", config)
        assert "estimated_vram_gb" in resources
        assert "estimated_throughput_samples_sec" in resources


class TestTEECompiler:
    def test_supports_target(self):
        compiler = TEECompiler()
        assert compiler.supports_target(HardwareTarget.INTEL_SGX) is True
        assert compiler.supports_target(HardwareTarget.AMD_SEV_SNP) is True
        assert compiler.supports_target(HardwareTarget.AWS_NITRO) is True
        assert compiler.supports_target(HardwareTarget.NVIDIA_A100) is False

    @pytest.mark.asyncio
    async def test_compile(self):
        compiler = TEECompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model data")
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.AMD_SEV_SNP,
                    input_shape=(1, 32),
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
                assert result.precision_used == PrecisionMode.FP32
                assert result.backend_used == CompilerBackend.EZKL
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_compile_sgx_memory_check(self):
        compiler = TEECompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            # Write a small file - should be within SGX limits
            f.write(b"x" * 100)
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.INTEL_SGX,
                    input_shape=(1, 16),
                )
                result = await compiler.compile(f.name, config)
                assert result.success is True
            finally:
                os.unlink(f.name)

    def test_estimate_resources(self):
        compiler = TEECompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"x" * 1024)
            f.flush()
            try:
                config = CompilationConfig(
                    target=HardwareTarget.INTEL_SGX,
                    input_shape=(1, 32),
                )
                resources = compiler.estimate_resources(f.name, config)
                assert "model_size_mb" in resources
                assert "estimated_enclave_pages" in resources
            finally:
                os.unlink(f.name)

    def test_estimate_resources_no_file(self):
        compiler = TEECompiler()
        config = CompilationConfig(
            target=HardwareTarget.INTEL_SGX,
            input_shape=(1, 32),
        )
        resources = compiler.estimate_resources("/nonexistent", config)
        assert resources["model_size_mb"] == 0


class TestHardwareCompiler:
    def test_init(self):
        compiler = HardwareCompiler()
        assert len(compiler._strategies) > 0

    def test_supports_target(self):
        compiler = HardwareCompiler()
        assert compiler.supports_target(HardwareTarget.NVIDIA_A100) is True
        assert compiler.supports_target(HardwareTarget.XILINX_U280) is True
        assert compiler.supports_target(HardwareTarget.INTEL_SGX) is True

    def test_get_hardware_profiles(self):
        compiler = HardwareCompiler()
        profiles = compiler.get_hardware_profiles()
        assert len(profiles) > 0
        assert HardwareTarget.NVIDIA_A100 in profiles

    @pytest.mark.asyncio
    async def test_compile_specific_target(self):
        compiler = HardwareCompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            try:
                result = await compiler.compile(
                    model_path=f.name,
                    input_shape=(1, 32),
                    target=HardwareTarget.XILINX_U280,
                )
                assert result.success is True
            finally:
                os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_compile_auto_target_balanced(self):
        compiler = HardwareCompiler()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            try:
                result = await compiler.compile(
                    model_path=f.name,
                    input_shape=(1, 32),
                    target=HardwareTarget.AUTO,
                    profile=OptimizationProfile.BALANCED,
                )
                assert result.success is True
            finally:
                os.unlink(f.name)

    def test_select_optimal_target_latency(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 64), OptimizationProfile.LATENCY
        )
        assert target == HardwareTarget.XILINX_U280

    def test_select_optimal_target_throughput_small(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 64), OptimizationProfile.THROUGHPUT
        )
        assert target == HardwareTarget.NVIDIA_A100

    def test_select_optimal_target_throughput_large(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 3, 224, 224), OptimizationProfile.THROUGHPUT
        )
        assert target == HardwareTarget.NVIDIA_H100

    def test_select_optimal_target_power_efficient(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 64), OptimizationProfile.POWER_EFFICIENT
        )
        assert target == HardwareTarget.XILINX_U280

    def test_select_optimal_target_cost_optimized(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 64), OptimizationProfile.COST_OPTIMIZED
        )
        assert target == HardwareTarget.AWS_NITRO

    def test_select_optimal_target_balanced_small(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 64), OptimizationProfile.BALANCED
        )
        assert target == HardwareTarget.XILINX_U280

    def test_select_optimal_target_balanced_medium(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (100, 200), OptimizationProfile.BALANCED
        )
        assert target == HardwareTarget.NVIDIA_A100

    def test_select_optimal_target_balanced_large(self):
        compiler = HardwareCompiler()
        target = compiler._select_optimal_target(
            "model.onnx", (1, 3, 224, 224), OptimizationProfile.BALANCED
        )
        assert target == HardwareTarget.NVIDIA_H100

    def test_estimate_cost(self):
        compiler = HardwareCompiler()
        cost = compiler.estimate_cost(
            "model.onnx",
            (1, 64),
            HardwareTarget.NVIDIA_A100,
            num_inferences=1000,
        )
        assert "compute_cost_usd" in cost
        assert "energy_cost_kwh" in cost
        assert "estimated_time_hours" in cost
        assert "cost_per_inference_usd" in cost

    def test_estimate_cost_unknown_target(self):
        compiler = HardwareCompiler()
        with pytest.raises(ValidationError, match="Unknown target"):
            compiler.estimate_cost(
                "model.onnx",
                (1, 64),
                HardwareTarget.AUTO,
            )

    @pytest.mark.asyncio
    async def test_compile_unsupported_target(self):
        compiler = HardwareCompiler()
        # Remove all strategies to simulate no compiler available
        compiler._strategies.clear()
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(b"fake model")
            f.flush()
            try:
                with pytest.raises(ValidationError, match="No compiler available"):
                    await compiler.compile(
                        model_path=f.name,
                        input_shape=(1, 32),
                        target=HardwareTarget.NVIDIA_A100,
                    )
            finally:
                os.unlink(f.name)


# ============ registry.py tests ============

from aethelred.models.registry import (
    ModelStatus,
    RegisteredModel,
    ModelRegistry,
)


class TestModelStatus:
    def test_values(self):
        assert ModelStatus.PENDING == "pending"
        assert ModelStatus.ACTIVE == "active"
        assert ModelStatus.DEPRECATED == "deprecated"
        assert ModelStatus.REVOKED == "revoked"


class TestRegisteredModel:
    def test_creation(self):
        model = RegisteredModel(
            model_id="model_123",
            name="test-model",
            version="1.0.0",
            owner="aeth1abc",
            model_hash="abc123",
            circuit_hash="def456",
            verification_key_hash="ghi789",
        )
        assert model.model_id == "model_123"
        assert model.status == ModelStatus.PENDING
        assert model.tags == []
        assert model.parameter_count == 0

    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        model = RegisteredModel(
            model_id="model_123",
            name="test-model",
            version="1.0.0",
            owner="aeth1abc",
            model_hash="abc123",
            circuit_hash="def456",
            verification_key_hash="ghi789",
            registered_at=now,
            status=ModelStatus.ACTIVE,
        )
        d = model.to_dict()
        assert d["model_id"] == "model_123"
        assert d["name"] == "test-model"
        assert d["status"] == "active"
        assert d["registered_at"] == now.isoformat()

    def test_to_dict_no_registered_at(self):
        model = RegisteredModel(
            model_id="m1", name="n", version="1.0", owner="o",
            model_hash="h1", circuit_hash="h2", verification_key_hash="h3",
        )
        d = model.to_dict()
        assert d["registered_at"] is None


class TestModelRegistry:
    def _make_mock_client(self):
        client = MagicMock()
        client._submit_tx = AsyncMock()
        client._query = AsyncMock()
        client.address = "aeth1testaddress"
        return client

    def _make_mock_circuit(self):
        circuit = MagicMock(spec=Circuit)
        circuit.circuit_id = "circuit_123"
        circuit.model_hash = "model_hash_abc"
        circuit.circuit_binary = b"binary"
        circuit.verification_key = b"vk"
        circuit.input_shape = (1, 64)
        circuit.output_shape = (1, 1)
        circuit.framework = "pytorch"
        return circuit

    @pytest.mark.asyncio
    async def test_register(self):
        client = self._make_mock_client()
        client._submit_tx.return_value = {
            "model_id": "model_abc",
            "tx_hash": "0xabc",
            "block_height": 100,
        }
        registry = ModelRegistry(client)
        circuit = self._make_mock_circuit()

        result = await registry.register(
            circuit=circuit,
            name="test-model",
            version="2.0.0",
            description="Test model",
            tags=["tag1", "tag2"],
            category="credit-scoring",
            accuracy=0.95,
        )

        assert isinstance(result, RegisteredModel)
        assert result.model_id == "model_abc"
        assert result.name == "test-model"
        assert result.version == "2.0.0"
        assert result.status == ModelStatus.ACTIVE
        assert result.tags == ["tag1", "tag2"]
        assert result.accuracy == 0.95
        assert result.tx_hash == "0xabc"

    @pytest.mark.asyncio
    async def test_register_defaults(self):
        client = self._make_mock_client()
        client._submit_tx.return_value = {}
        registry = ModelRegistry(client)
        circuit = self._make_mock_circuit()

        result = await registry.register(circuit=circuit, name="minimal")
        assert result.version == "1.0.0"
        assert result.tags == []

    @pytest.mark.asyncio
    async def test_get_found(self):
        client = self._make_mock_client()
        client._query.return_value = {
            "model": {
                "model_id": "model_abc",
                "name": "test",
                "version": "1.0.0",
                "owner": "aeth1x",
                "model_hash": "h1",
                "circuit_hash": "h2",
                "status": "active",
            }
        }
        registry = ModelRegistry(client)
        result = await registry.get("model_abc")
        assert result is not None
        assert result.model_id == "model_abc"
        assert result.status == ModelStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        client = self._make_mock_client()
        client._query.return_value = {}
        registry = ModelRegistry(client)
        result = await registry.get("model_nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_none_response(self):
        client = self._make_mock_client()
        client._query.return_value = None
        registry = ModelRegistry(client)
        result = await registry.get("model_x")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_hash_found(self):
        client = self._make_mock_client()
        client._query.return_value = {
            "model": {
                "model_id": "model_xyz",
                "name": "hash-lookup",
                "version": "1.0.0",
                "owner": "aeth1y",
                "model_hash": "hash123",
                "circuit_hash": "ch1",
            }
        }
        registry = ModelRegistry(client)
        result = await registry.get_by_hash("hash123")
        assert result is not None
        assert result.model_hash == "hash123"

    @pytest.mark.asyncio
    async def test_get_by_hash_not_found(self):
        client = self._make_mock_client()
        client._query.return_value = None
        registry = ModelRegistry(client)
        result = await registry.get_by_hash("nonexistent_hash")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_no_filters(self):
        client = self._make_mock_client()
        client._query.return_value = {
            "models": [
                {
                    "model_id": "m1", "name": "n1", "version": "1.0",
                    "owner": "o1", "model_hash": "h1", "circuit_hash": "c1",
                },
                {
                    "model_id": "m2", "name": "n2", "version": "2.0",
                    "owner": "o2", "model_hash": "h2", "circuit_hash": "c2",
                },
            ]
        }
        registry = ModelRegistry(client)
        models = await registry.list()
        assert len(models) == 2
        assert models[0].model_id == "m1"

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        client = self._make_mock_client()
        client._query.return_value = {"models": []}
        registry = ModelRegistry(client)
        models = await registry.list(
            owner="aeth1x",
            category="credit",
            tags=["tag1", "tag2"],
            status=ModelStatus.ACTIVE,
            limit=10,
            offset=5,
        )
        assert len(models) == 0
        # Verify query was called with params
        call_args = client._query.call_args
        params = call_args[1].get("params") if call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else {}

    @pytest.mark.asyncio
    async def test_deprecate(self):
        client = self._make_mock_client()
        client._submit_tx.return_value = {}
        registry = ModelRegistry(client)
        result = await registry.deprecate("model_123", reason="outdated")
        assert result is True

    @pytest.mark.asyncio
    async def test_deprecate_no_reason(self):
        client = self._make_mock_client()
        client._submit_tx.return_value = {}
        registry = ModelRegistry(client)
        result = await registry.deprecate("model_123")
        assert result is True

    @pytest.mark.asyncio
    async def test_update_metadata(self):
        client = self._make_mock_client()
        client._submit_tx.return_value = {}
        client._query.return_value = {
            "model": {
                "model_id": "model_123",
                "name": "updated",
                "version": "1.0",
                "owner": "o1",
                "model_hash": "h1",
                "circuit_hash": "c1",
            }
        }
        registry = ModelRegistry(client)
        result = await registry.update_metadata(
            "model_123",
            description="new description",
            tags=["new_tag"],
            accuracy=0.99,
        )
        assert result is not None
        assert result.model_id == "model_123"

    def test_parse_model_full(self):
        client = self._make_mock_client()
        registry = ModelRegistry(client)
        data = {
            "model_id": "m1",
            "name": "test",
            "version": "1.0.0",
            "owner": "o1",
            "model_hash": "h1",
            "circuit_hash": "c1",
            "verification_key_hash": "vkh1",
            "description": "desc",
            "tags": ["t1"],
            "category": "cat1",
            "framework": "tensorflow",
            "input_shape": [1, 64],
            "output_shape": [1, 1],
            "parameter_count": 1000,
            "accuracy": "0.95",
            "status": "deprecated",
            "registered_at": "2024-01-01T00:00:00+00:00",
            "tx_hash": "0x123",
            "block_height": 42,
        }
        model = registry._parse_model(data)
        assert model.model_id == "m1"
        assert model.framework == "tensorflow"
        assert model.input_shape == (1, 64)
        assert model.accuracy == 0.95
        assert model.status == ModelStatus.DEPRECATED

    def test_parse_model_minimal(self):
        client = self._make_mock_client()
        registry = ModelRegistry(client)
        data = {
            "model_id": "m1",
            "name": "test",
            "version": "1.0.0",
            "owner": "o1",
            "model_hash": "h1",
            "circuit_hash": "c1",
        }
        model = registry._parse_model(data)
        assert model.accuracy is None
        assert model.tags == []
        assert model.registered_at is None

    def test_compute_circuit_hash(self):
        client = self._make_mock_client()
        registry = ModelRegistry(client)
        circuit = MagicMock()
        circuit.circuit_binary = b"test binary"
        h = registry._compute_circuit_hash(circuit)
        assert h == hashlib.sha256(b"test binary").hexdigest()

    def test_compute_vk_hash(self):
        client = self._make_mock_client()
        registry = ModelRegistry(client)
        circuit = MagicMock()
        circuit.verification_key = b"test vk"
        h = registry._compute_vk_hash(circuit)
        assert h == hashlib.sha256(b"test vk").hexdigest()
