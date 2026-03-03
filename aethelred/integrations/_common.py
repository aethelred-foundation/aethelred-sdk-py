"""Shared verification envelope and recorder utilities for framework integrations."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import time
import uuid
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional


VerificationSink = Callable[["VerificationEnvelope"], Any]


def _normalize_for_hash(value: Any) -> Any:
    """Convert framework objects into stable, JSON-hashable structures."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    if isinstance(value, bytearray):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {str(k): _normalize_for_hash(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(v) for v in value]
    if isinstance(value, set):
        return sorted((_normalize_for_hash(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True, default=str))
    if is_dataclass(value):
        return _normalize_for_hash(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _normalize_for_hash(value.model_dump(mode="json"))  # pydantic v2
        except TypeError:
            return _normalize_for_hash(value.model_dump())
    if hasattr(value, "dict"):
        try:
            return _normalize_for_hash(value.dict())
        except TypeError:
            pass
    # Tensor-like normalization (torch/tf/numpy compatible via duck typing)
    for attr in ("detach", "cpu", "numpy"):
        if hasattr(value, attr):
            try:
                value = getattr(value, attr)()
            except Exception:
                break
    if hasattr(value, "tolist"):
        try:
            return _normalize_for_hash(value.tolist())
        except Exception:
            pass
    if hasattr(value, "__dict__") and value.__dict__:
        # Avoid deep object graphs and secrets by snapshotting shallow public attrs only.
        public = {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        if public:
            return _normalize_for_hash(public)
    return {"__repr__": repr(value)}


def canonical_json(value: Any) -> str:
    """Serialize value deterministically for hashing."""
    normalized = _normalize_for_hash(value)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_payload(value: Any) -> str:
    """Hash arbitrary framework input/output payloads using stable canonicalization."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class VerificationEnvelope:
    """Portable verification envelope emitted by framework adapters."""

    trace_id: str
    framework: str
    operation: str
    input_hash: str
    output_hash: str
    timestamp_ms: int
    metadata: Dict[str, Any]

    def to_headers(self, header_prefix: str = "x-aethelred") -> Dict[str, str]:
        prefix = header_prefix.lower().rstrip("-")
        return {
            f"{prefix}-trace-id": self.trace_id,
            f"{prefix}-framework": self.framework,
            f"{prefix}-operation": self.operation,
            f"{prefix}-input-hash": self.input_hash,
            f"{prefix}-output-hash": self.output_hash,
            f"{prefix}-ts-ms": str(self.timestamp_ms),
        }


class VerificationRecorder:
    """Records framework invocations and emits verification envelopes."""

    def __init__(
        self,
        sink: Optional[VerificationSink] = None,
        *,
        default_metadata: Optional[Mapping[str, Any]] = None,
        header_prefix: str = "x-aethelred",
    ) -> None:
        self._sink = sink
        self._default_metadata = dict(default_metadata or {})
        self.header_prefix = header_prefix

    def _build_envelope(
        self,
        *,
        framework: str,
        operation: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VerificationEnvelope:
        merged_metadata: Dict[str, Any] = dict(self._default_metadata)
        if metadata:
            merged_metadata.update(dict(metadata))
        return VerificationEnvelope(
            trace_id=uuid.uuid4().hex,
            framework=framework,
            operation=operation,
            input_hash=hash_payload(input_data),
            output_hash=hash_payload(output_data),
            timestamp_ms=int(time.time() * 1000),
            metadata=dict(_normalize_for_hash(merged_metadata)),
        )

    def record(
        self,
        *,
        framework: str,
        operation: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VerificationEnvelope:
        envelope = self._build_envelope(
            framework=framework,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )
        if self._sink is not None:
            result = self._sink(envelope)
            if inspect.isawaitable(result):
                raise TypeError("VerificationRecorder.record() received async sink; use arecord()")
        return envelope

    async def arecord(
        self,
        *,
        framework: str,
        operation: str,
        input_data: Any,
        output_data: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> VerificationEnvelope:
        envelope = self._build_envelope(
            framework=framework,
            operation=operation,
            input_data=input_data,
            output_data=output_data,
            metadata=metadata,
        )
        if self._sink is not None:
            result = self._sink(envelope)
            if inspect.isawaitable(result):
                await result  # type: ignore[misc]
        return envelope


__all__ = [
    "VerificationEnvelope",
    "VerificationRecorder",
    "canonical_json",
    "hash_payload",
]
