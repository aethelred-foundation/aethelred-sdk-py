"""PyTorch-native verification wrappers (optional dependency via duck typing)."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from ._common import VerificationEnvelope, VerificationRecorder


class VerifiedPyTorchModule:
    """Wrap a PyTorch-style module and emit verification envelopes on forward calls."""

    def __init__(
        self,
        model: Any,
        *,
        recorder: Optional[VerificationRecorder] = None,
        component_name: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._model = model
        self._recorder = recorder or VerificationRecorder()
        self._component_name = component_name or getattr(model, "__class__", type(model)).__name__
        self._extra_metadata = dict(extra_metadata or {})
        self.last_verification: Optional[VerificationEnvelope] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        metadata = {
            "component": self._component_name,
            "arg_count": len(args),
            "kwarg_keys": sorted(kwargs.keys()),
            **self._extra_metadata,
        }
        try:
            output = self._model(*args, **kwargs)
        except Exception as exc:
            self.last_verification = self._recorder.record(
                framework="pytorch",
                operation="forward.error",
                input_data={"args": args, "kwargs": kwargs},
                output_data={"error": str(exc), "error_type": type(exc).__name__},
                metadata=metadata,
            )
            raise

        self.last_verification = self._recorder.record(
            framework="pytorch",
            operation="forward",
            input_data={"args": args, "kwargs": kwargs},
            output_data=output,
            metadata=metadata,
        )
        return output

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.__call__(*args, **kwargs)


def wrap_pytorch_model(
    model: Any,
    *,
    recorder: Optional[VerificationRecorder] = None,
    component_name: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> VerifiedPyTorchModule:
    """Convenience helper for wrapping a torch.nn.Module-like object."""
    return VerifiedPyTorchModule(
        model,
        recorder=recorder,
        component_name=component_name,
        extra_metadata=extra_metadata,
    )


__all__ = ["VerifiedPyTorchModule", "wrap_pytorch_model"]
