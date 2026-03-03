"""TensorFlow / Keras callback integration for Aethelred verification."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from ._common import VerificationEnvelope, VerificationRecorder


class AethelredKerasCallback:
    """Framework-agnostic callback core with Keras-compatible hook methods."""

    def __init__(
        self,
        *,
        recorder: Optional[VerificationRecorder] = None,
        capture_batch_events: bool = False,
        include_log_keys: Optional[Iterable[str]] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._recorder = recorder or VerificationRecorder()
        self._capture_batch_events = capture_batch_events
        self._include_log_keys = set(include_log_keys or [])
        self._extra_metadata = dict(extra_metadata or {})
        self._model: Any = None
        self.last_verification: Optional[VerificationEnvelope] = None

    def set_model(self, model: Any) -> None:
        self._model = model

    def _filtered_logs(self, logs: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
        if not logs:
            return {}
        if not self._include_log_keys:
            return dict(logs)
        return {k: v for k, v in logs.items() if k in self._include_log_keys}

    def _record(self, operation: str, *, payload: Mapping[str, Any]) -> None:
        model_name = getattr(getattr(self._model, "__class__", None), "__name__", None)
        self.last_verification = self._recorder.record(
            framework="tensorflow.keras",
            operation=operation,
            input_data=payload,
            output_data={"status": "ok"},
            metadata={
                "model": model_name,
                **self._extra_metadata,
            },
        )

    def on_train_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
        if not self._capture_batch_events:
            return
        self._record("train_batch_end", payload={"batch": batch, "logs": self._filtered_logs(logs)})

    def on_test_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
        if not self._capture_batch_events:
            return
        self._record("test_batch_end", payload={"batch": batch, "logs": self._filtered_logs(logs)})

    def on_predict_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
        if not self._capture_batch_events:
            return
        self._record("predict_batch_end", payload={"batch": batch, "logs": self._filtered_logs(logs)})

    def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
        self._record("epoch_end", payload={"epoch": epoch, "logs": self._filtered_logs(logs)})


def create_keras_callback(**kwargs: Any) -> Any:
    """Create a TensorFlow-native Callback when tensorflow is installed, else fallback core."""
    core = AethelredKerasCallback(**kwargs)
    try:
        import tensorflow as tf  # type: ignore
    except Exception:
        return core

    class _TensorFlowAethelredCallback(tf.keras.callbacks.Callback):  # type: ignore[misc]
        def __init__(self, inner: AethelredKerasCallback) -> None:
            super().__init__()
            self._inner = inner

        def set_model(self, model: Any) -> None:
            super().set_model(model)
            self._inner.set_model(model)

        def on_train_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._inner.on_train_batch_end(batch, logs)

        def on_test_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._inner.on_test_batch_end(batch, logs)

        def on_predict_batch_end(self, batch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._inner.on_predict_batch_end(batch, logs)

        def on_epoch_end(self, epoch: int, logs: Optional[Mapping[str, Any]] = None) -> None:
            self._inner.on_epoch_end(epoch, logs)

        @property
        def last_verification(self) -> Optional[VerificationEnvelope]:
            return self._inner.last_verification

    return _TensorFlowAethelredCallback(core)


__all__ = ["AethelredKerasCallback", "create_keras_callback"]
