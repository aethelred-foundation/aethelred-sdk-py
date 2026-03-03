#!/usr/bin/env python3
"""
Model Conversion Example

This example demonstrates how to convert various AI models
to zkML-compatible arithmetic circuits using the Aethelred SDK.

Supported frameworks:
- PyTorch (.pt, .pth)
- TensorFlow/Keras (.h5, SavedModel)
- ONNX (.onnx)
- scikit-learn (.pkl, .joblib)
- XGBoost (.json, .model)
- LightGBM (.txt, .model)

Prerequisites:
    pip install -e /Users/rameshtamilselvan/Downloads/AethelredMVP/sdk/python

Usage:
    python model_conversion.py
"""

import os
import tempfile
from pathlib import Path

from aethelred import (
    ModelConverter,
    ConversionConfig,
    QuantizationConfig,
    OptimizationConfig,
    QuantizationMode,
    OptimizationLevel,
    setup_logging,
)
from aethelred.models.converter import estimate_circuit_size


def pytorch_conversion_example():
    """Convert a PyTorch model to zkML circuit."""
    print("\n" + "=" * 50)
    print("PyTorch Model Conversion")
    print("=" * 50)

    converter = ModelConverter()

    # Basic conversion
    print("\nBasic conversion:")
    print("  converter.from_pytorch(")
    print("      model_path='resnet50.pt',")
    print("      input_shape=(1, 3, 224, 224)")
    print("  )")

    # Advanced conversion with all options
    print("\nAdvanced conversion with options:")
    print("""
    circuit = converter.from_pytorch(
        model_path='resnet50.pt',
        input_shape=(1, 3, 224, 224),
        optimization_level=3,      # Aggressive optimization
        quantization_bits=8,       # 8-bit quantization
        pruning_threshold=0.01,    # Prune weights < 1%
        output_dir='./circuits',   # Save artifacts
        validate=True              # Validate output
    )
    """)

    # Estimate circuit size before conversion
    print("Estimating circuit size for (1, 3, 224, 224) input:")
    estimates = estimate_circuit_size(
        model_path="dummy.pt",  # Would be real model path
        input_shape=(1, 3, 224, 224),
        quantization_bits=8
    )
    print(f"  - Estimated constraints: {estimates['estimated_constraints']:,}")
    print(f"  - Estimated memory: {estimates['estimated_memory_mb']} MB")
    print(f"  - Estimated proving time: {estimates['estimated_proving_time_seconds']}s")


def tensorflow_conversion_example():
    """Convert a TensorFlow/Keras model to zkML circuit."""
    print("\n" + "=" * 50)
    print("TensorFlow/Keras Model Conversion")
    print("=" * 50)

    print("\nTensorFlow SavedModel:")
    print("""
    circuit = converter.from_tensorflow(
        model_path='saved_model/',   # SavedModel directory
        input_shape=(1, 28, 28, 1),  # MNIST-like input
        optimization_level=2,
        quantization_bits=8
    )
    """)

    print("Keras H5 file:")
    print("""
    circuit = converter.from_tensorflow(
        model_path='model.h5',
        input_shape=(1, 128),
        signature_key='serving_default'
    )
    """)


def onnx_conversion_example():
    """Convert an ONNX model to zkML circuit."""
    print("\n" + "=" * 50)
    print("ONNX Model Conversion")
    print("=" * 50)

    print("\nONNX conversion:")
    print("""
    circuit = converter.from_onnx(
        model_path='model.onnx',
        input_shape=(1, 512),
        optimization_level=2,
        quantization_bits=8
    )
    """)

    print("ONNX is the intermediate format used internally.")
    print("Converting directly from ONNX skips the export step.")


def tree_model_conversion_example():
    """Convert tree-based models to zkML circuits."""
    print("\n" + "=" * 50)
    print("Tree-Based Model Conversion")
    print("=" * 50)

    print("\nXGBoost:")
    print("""
    circuit = converter.from_xgboost(
        model_path='xgb_model.json',
        input_shape=(1, 64)
    )
    """)

    print("LightGBM:")
    print("""
    circuit = converter.from_lightgbm(
        model_path='lgb_model.txt',
        input_shape=(1, 64)
    )
    """)

    print("scikit-learn:")
    print("""
    circuit = converter.from_sklearn(
        model_path='rf_model.pkl',
        input_shape=(1, 32),
        model_type='RandomForestClassifier'
    )
    """)


def advanced_configuration_example():
    """Demonstrate advanced conversion configuration."""
    print("\n" + "=" * 50)
    print("Advanced Configuration")
    print("=" * 50)

    print("\nQuantization Configuration:")
    print("""
    quant_config = QuantizationConfig(
        bits=8,                        # 8-bit quantization
        mode=QuantizationMode.SYMMETRIC,  # or ASYMMETRIC, DYNAMIC
        calibration_samples=100,       # Samples for calibration
        percentile=99.99,              # Outlier handling
        per_channel=True,              # Per-channel quantization
        skip_layers=['output']         # Skip certain layers
    )
    """)

    print("Optimization Configuration:")
    print("""
    opt_config = OptimizationConfig(
        level=OptimizationLevel.AGGRESSIVE,  # 0-3
        pruning_enabled=True,
        pruning_threshold=0.01,        # Prune weights < 1%
        constant_folding=True,         # Fold constants
        batch_norm_folding=True,       # Fold batch norm into conv
        conv_relu_fusion=True,         # Fuse Conv+ReLU
        memory_optimization=True,
        parallel_constraints=True,
        max_parallelism=8
    )
    """)

    print("Complete Configuration:")
    print("""
    config = ConversionConfig(
        input_shape=(1, 3, 224, 224),
        output_shape=(1, 1000),
        quantization=quant_config,
        optimization=opt_config,
        backend='ezkl',               # 'ezkl', 'circom', 'halo2'
        proof_system='halo2',         # 'groth16', 'plonk', 'halo2'
        output_dir='./circuits',
        save_intermediate=True,
        validate_output=True,
        name='resnet50-imagenet',
        version='1.0.0'
    )

    circuit = converter.from_config(
        model_path='resnet50.pt',
        config=config,
        framework=FrameworkType.PYTORCH
    )
    """)


def calibrated_conversion_example():
    """Demonstrate calibration-based conversion."""
    print("\n" + "=" * 50)
    print("Calibration-Based Conversion")
    print("=" * 50)

    print("""
    import numpy as np

    # Create calibration dataset
    calibration_data = np.random.randn(100, 1, 64).astype(np.float32)

    # Convert with calibration
    circuit = converter.convert_with_calibration(
        model_path='credit_model.pt',
        input_shape=(1, 64),
        calibration_data=calibration_data,
        quantization_bits=8
    )

    # Calibration improves quantization accuracy by
    # analyzing the distribution of activations
    """)


def circuit_inspection_example():
    """Show how to inspect a converted circuit."""
    print("\n" + "=" * 50)
    print("Circuit Inspection")
    print("=" * 50)

    print("""
    # After conversion
    circuit = converter.from_pytorch(...)

    # Inspect circuit properties
    print(f"Circuit ID: {circuit.circuit_id}")
    print(f"Model Hash: {circuit.model_hash}")
    print(f"Input Shape: {circuit.input_shape}")
    print(f"Output Shape: {circuit.output_shape}")
    print(f"Quantization: {circuit.quantization_bits}-bit")
    print(f"Optimization Level: {circuit.optimization_level}")

    # Inspect metrics
    metrics = circuit.metrics
    print(f"Constraints: {metrics.constraints:,}")
    print(f"Public Inputs: {metrics.public_inputs}")
    print(f"Private Inputs: {metrics.private_inputs}")
    print(f"Gates: {metrics.gates:,}")
    print(f"Depth: {metrics.depth}")
    print(f"Memory: {metrics.memory_bytes / 1024 / 1024:.1f} MB")
    print(f"Est. Proving Time: {metrics.estimated_proving_time_ms}ms")

    # Export metadata
    metadata = circuit.to_dict()
    import json
    print(json.dumps(metadata, indent=2))
    """)


def supported_operations_example():
    """List supported ONNX operations."""
    print("\n" + "=" * 50)
    print("Supported ONNX Operations")
    print("=" * 50)

    from aethelred.models.converter import supported_operations

    ops = supported_operations()
    print("\nCurrently supported operations:")
    for i, op in enumerate(ops, 1):
        print(f"  {i:2d}. {op}")


def main():
    """Run all conversion examples."""
    setup_logging(level="INFO")

    print("=" * 60)
    print("Aethelred Model Conversion Examples")
    print("=" * 60)

    pytorch_conversion_example()
    tensorflow_conversion_example()
    onnx_conversion_example()
    tree_model_conversion_example()
    advanced_configuration_example()
    calibrated_conversion_example()
    circuit_inspection_example()
    supported_operations_example()

    print("\n" + "=" * 60)
    print("Examples Complete")
    print("=" * 60)
    print("\nFor more information, see:")
    print("  - Documentation: https://docs.aethelred.io/sdk/python/models")
    print("  - API Reference: https://docs.aethelred.io/api/model-converter")


if __name__ == "__main__":
    main()
