"""Native framework integrations for Aethelred verification capture."""

from ._common import VerificationEnvelope, VerificationRecorder, canonical_json, hash_payload
from .fastapi import AethelredVerificationMiddleware
from .huggingface import VerifiedTransformersPipeline, wrap_transformers_pipeline
from .langchain import VerifiedLangChainRunnable, wrap_langchain_runnable
from .pytorch import VerifiedPyTorchModule, wrap_pytorch_model
from .tensorflow import AethelredKerasCallback, create_keras_callback

__all__ = [
    "AethelredKerasCallback",
    "AethelredVerificationMiddleware",
    "VerificationEnvelope",
    "VerificationRecorder",
    "VerifiedLangChainRunnable",
    "VerifiedPyTorchModule",
    "VerifiedTransformersPipeline",
    "canonical_json",
    "create_keras_callback",
    "hash_payload",
    "wrap_langchain_runnable",
    "wrap_pytorch_model",
    "wrap_transformers_pipeline",
]
