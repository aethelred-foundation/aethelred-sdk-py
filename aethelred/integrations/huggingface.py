"""Hugging Face Transformers pipeline integration wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ._common import VerificationEnvelope, VerificationRecorder


class VerifiedTransformersPipeline:
    """Wrap a transformers pipeline-like callable and emit verification envelopes."""

    def __init__(
        self,
        pipeline: Any,
        *,
        recorder: Optional[VerificationRecorder] = None,
        component_name: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._pipeline = pipeline
        self._recorder = recorder or VerificationRecorder()
        self._component_name = component_name or getattr(pipeline, "task", None) or type(pipeline).__name__
        self._extra_metadata = dict(extra_metadata or {})
        self.last_verification: Optional[VerificationEnvelope] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        metadata = {
            "component": self._component_name,
            "task": getattr(self._pipeline, "task", None),
            **self._extra_metadata,
        }
        output = self._pipeline(*args, **kwargs)
        self.last_verification = self._recorder.record(
            framework="huggingface.transformers",
            operation="pipeline",
            input_data={"args": args, "kwargs": kwargs},
            output_data=output,
            metadata=metadata,
        )
        return output


def wrap_transformers_pipeline(
    pipeline: Any,
    *,
    recorder: Optional[VerificationRecorder] = None,
    component_name: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> VerifiedTransformersPipeline:
    """Convenience wrapper for transformers.pipeline outputs."""
    return VerifiedTransformersPipeline(
        pipeline,
        recorder=recorder,
        component_name=component_name,
        extra_metadata=extra_metadata,
    )


__all__ = ["VerifiedTransformersPipeline", "wrap_transformers_pipeline"]
