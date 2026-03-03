"""Comprehensive tests for integration adapters — targeting 95%+ coverage.

Covers:
- aethelred/integrations/_common.py:
  _normalize_for_hash (all branches: None, bool, int, float, str, bytes, bytearray,
   Mapping, list, tuple, set, dataclass, pydantic v2/v1, tensor-like, tolist,
   __dict__, fallback __repr__),
  canonical_json, hash_payload,
  VerificationEnvelope (to_headers),
  VerificationRecorder (record with sync sink, record with async sink raises,
   arecord with async sink, arecord with sync sink, no sink)

- aethelred/integrations/tensorflow.py:
  AethelredKerasCallback (on_train/test/predict_batch_end with capture on/off,
   on_epoch_end, _filtered_logs with/without include_log_keys,
   _record with model class name),
  create_keras_callback (without tensorflow installed)

- aethelred/integrations/langchain.py:
  VerifiedLangChainRunnable (invoke with/without .invoke, ainvoke with/without .ainvoke,
   batch with/without .batch, abatch all 3 branches, __call__ with args,
   __getattr__, _metadata),
  wrap_langchain_runnable

- aethelred/integrations/fastapi.py:
  AethelredVerificationMiddleware (_should_process all branches,
   non-http scope passthrough, excluded path passthrough,
   included path processing, streamed large body, final flush fallback,
   _merge_headers existing/new keys, _normalize_header_prefix,
   _headers_to_dict)
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
from unittest.mock import MagicMock, patch

import pytest

from aethelred.integrations._common import (
    VerificationEnvelope,
    VerificationRecorder,
    _normalize_for_hash,
    canonical_json,
    hash_payload,
)
from aethelred.integrations.fastapi import (
    AethelredVerificationMiddleware,
    _headers_to_dict,
    _merge_headers,
    _normalize_header_prefix,
)
from aethelred.integrations.langchain import (
    VerifiedLangChainRunnable,
    wrap_langchain_runnable,
)
from aethelred.integrations.tensorflow import (
    AethelredKerasCallback,
    create_keras_callback,
)


# ===========================================================================
# _common.py
# ===========================================================================


class TestNormalizeForHash:
    """Test all branches of _normalize_for_hash."""

    def test_none(self) -> None:
        assert _normalize_for_hash(None) is None

    def test_bool(self) -> None:
        assert _normalize_for_hash(True) is True
        assert _normalize_for_hash(False) is False

    def test_int(self) -> None:
        assert _normalize_for_hash(42) == 42

    def test_float(self) -> None:
        assert _normalize_for_hash(3.14) == 3.14

    def test_str(self) -> None:
        assert _normalize_for_hash("hello") == "hello"

    def test_bytes(self) -> None:
        result = _normalize_for_hash(b"\x01\x02")
        assert result == {"__bytes__": "AQI="}

    def test_bytearray(self) -> None:
        result = _normalize_for_hash(bytearray(b"\x03\x04"))
        assert result == {"__bytes__": "AwQ="}

    def test_mapping(self) -> None:
        result = _normalize_for_hash({"b": 2, "a": 1})
        assert result == {"a": 1, "b": 2}  # sorted keys

    def test_list(self) -> None:
        result = _normalize_for_hash([1, "two", 3])
        assert result == [1, "two", 3]

    def test_tuple(self) -> None:
        result = _normalize_for_hash((1, 2, 3))
        assert result == [1, 2, 3]

    def test_set(self) -> None:
        result = _normalize_for_hash({1, 2, 3})
        # Sets are sorted by JSON representation
        assert isinstance(result, list)
        assert len(result) == 3

    def test_dataclass(self) -> None:
        @dataclass
        class Point:
            x: int
            y: int

        result = _normalize_for_hash(Point(1, 2))
        assert result == {"x": 1, "y": 2}

    def test_pydantic_v2_model_dump(self) -> None:
        class FakeModel:
            def model_dump(self, mode: str = "python") -> dict:
                return {"field": "value"}

        result = _normalize_for_hash(FakeModel())
        assert result == {"field": "value"}

    def test_pydantic_v2_model_dump_type_error_fallback(self) -> None:
        """When model_dump(mode="json") raises TypeError, try without mode."""

        class FakeModel:
            def model_dump(self, **kwargs: Any) -> dict:
                if "mode" in kwargs:
                    raise TypeError("no mode arg")
                return {"fallback": True}

        result = _normalize_for_hash(FakeModel())
        assert result == {"fallback": True}

    def test_pydantic_v1_dict(self) -> None:
        class FakeModel:
            def dict(self) -> dict:
                return {"v1_field": "v1_value"}

        result = _normalize_for_hash(FakeModel())
        assert result == {"v1_field": "v1_value"}

    def test_pydantic_v1_dict_type_error(self) -> None:
        """When dict() raises TypeError, fall through."""

        class FakeModel:
            def dict(self) -> dict:
                raise TypeError("no args")

        result = _normalize_for_hash(FakeModel())
        assert "__repr__" in result

    def test_tensor_like_detach_cpu_numpy_tolist(self) -> None:
        class FakeTensor:
            def detach(self) -> "FakeTensor":
                return self

            def cpu(self) -> "FakeTensor":
                return self

            def numpy(self) -> "FakeNDArray":
                return FakeNDArray()

        class FakeNDArray:
            def tolist(self) -> list:
                return [1.0, 2.0, 3.0]

        result = _normalize_for_hash(FakeTensor())
        assert result == [1.0, 2.0, 3.0]

    def test_tensor_detach_exception_breaks(self) -> None:
        """When detach() raises, we break the chain and try tolist or __dict__."""

        class FakeTensor:
            def detach(self) -> Any:
                raise RuntimeError("no detach")

            def tolist(self) -> list:
                return [4.0, 5.0]

        result = _normalize_for_hash(FakeTensor())
        assert result == [4.0, 5.0]

    def test_tolist_exception_falls_through(self) -> None:
        class FakeObj:
            def tolist(self) -> Any:
                raise RuntimeError("no tolist")

            def __init__(self) -> None:
                self.x = 10

        # Has __dict__ with public attrs (instance variable, not class annotation)
        result = _normalize_for_hash(FakeObj())
        assert result == {"x": 10}

    def test_object_with_dict_public_attrs(self) -> None:
        class MyObj:
            def __init__(self) -> None:
                self.public = "visible"
                self._private = "hidden"

        result = _normalize_for_hash(MyObj())
        assert result == {"public": "visible"}

    def test_object_with_empty_public_dict(self) -> None:
        class MyObj:
            def __init__(self) -> None:
                self._private = "only_private"

        result = _normalize_for_hash(MyObj())
        assert "__repr__" in result

    def test_fallback_repr(self) -> None:
        class Opaque:
            pass

        # Opaque has empty __dict__ after removing _-prefixed
        obj = Opaque()
        result = _normalize_for_hash(obj)
        assert "__repr__" in result


class TestCanonicalJson:
    """Test canonical_json produces deterministic output."""

    def test_deterministic(self) -> None:
        a = canonical_json({"z": 1, "a": 2})
        b = canonical_json({"a": 2, "z": 1})
        assert a == b

    def test_compact_format(self) -> None:
        result = canonical_json({"key": "value"})
        assert " " not in result


class TestHashPayload:
    """Test hash_payload returns consistent SHA-256 hex."""

    def test_same_input_same_hash(self) -> None:
        h1 = hash_payload({"a": 1, "b": 2})
        h2 = hash_payload({"b": 2, "a": 1})
        assert h1 == h2

    def test_different_input_different_hash(self) -> None:
        h1 = hash_payload({"a": 1})
        h2 = hash_payload({"a": 2})
        assert h1 != h2

    def test_returns_hex_string(self) -> None:
        h = hash_payload("test")
        assert len(h) == 64
        int(h, 16)  # valid hex


class TestVerificationEnvelope:
    """Test VerificationEnvelope construction and to_headers."""

    def test_construction(self) -> None:
        env = VerificationEnvelope(
            trace_id="abc123",
            framework="pytorch",
            operation="forward",
            input_hash="aaa",
            output_hash="bbb",
            timestamp_ms=1000,
            metadata={"key": "val"},
        )
        assert env.framework == "pytorch"
        assert env.metadata["key"] == "val"

    def test_to_headers_default_prefix(self) -> None:
        env = VerificationEnvelope(
            trace_id="abc",
            framework="tf",
            operation="predict",
            input_hash="in",
            output_hash="out",
            timestamp_ms=2000,
            metadata={},
        )
        headers = env.to_headers()
        assert headers["x-aethelred-trace-id"] == "abc"
        assert headers["x-aethelred-framework"] == "tf"
        assert headers["x-aethelred-operation"] == "predict"
        assert headers["x-aethelred-input-hash"] == "in"
        assert headers["x-aethelred-output-hash"] == "out"
        assert headers["x-aethelred-ts-ms"] == "2000"

    def test_to_headers_custom_prefix(self) -> None:
        env = VerificationEnvelope(
            trace_id="abc",
            framework="tf",
            operation="predict",
            input_hash="in",
            output_hash="out",
            timestamp_ms=2000,
            metadata={},
        )
        headers = env.to_headers(header_prefix="X-MyApp-")
        assert "x-myapp-trace-id" in headers


class TestVerificationRecorder:
    """Test VerificationRecorder sync/async methods."""

    def test_record_without_sink(self) -> None:
        recorder = VerificationRecorder()
        env = recorder.record(
            framework="test",
            operation="op",
            input_data="in",
            output_data="out",
        )
        assert env.framework == "test"
        assert env.operation == "op"

    def test_record_with_sync_sink(self) -> None:
        results: list = []
        recorder = VerificationRecorder(sink=results.append)
        env = recorder.record(
            framework="test",
            operation="op",
            input_data="in",
            output_data="out",
        )
        assert len(results) == 1
        assert results[0] is env

    def test_record_with_async_sink_raises(self) -> None:
        async def async_sink(env: VerificationEnvelope) -> None:
            pass

        recorder = VerificationRecorder(sink=async_sink)
        with pytest.raises(TypeError, match="async sink"):
            recorder.record(
                framework="test",
                operation="op",
                input_data="in",
                output_data="out",
            )

    @pytest.mark.asyncio
    async def test_arecord_without_sink(self) -> None:
        recorder = VerificationRecorder()
        env = await recorder.arecord(
            framework="test",
            operation="op",
            input_data="in",
            output_data="out",
        )
        assert env.framework == "test"

    @pytest.mark.asyncio
    async def test_arecord_with_async_sink(self) -> None:
        results: list = []

        async def async_sink(env: VerificationEnvelope) -> None:
            results.append(env)

        recorder = VerificationRecorder(sink=async_sink)
        env = await recorder.arecord(
            framework="test",
            operation="op",
            input_data="in",
            output_data="out",
        )
        assert len(results) == 1
        assert results[0] is env

    @pytest.mark.asyncio
    async def test_arecord_with_sync_sink(self) -> None:
        results: list = []
        recorder = VerificationRecorder(sink=results.append)
        env = await recorder.arecord(
            framework="test",
            operation="op",
            input_data="in",
            output_data="out",
        )
        assert len(results) == 1

    def test_record_with_default_metadata(self) -> None:
        recorder = VerificationRecorder(
            default_metadata={"env": "prod"},
        )
        env = recorder.record(
            framework="test",
            operation="op",
            input_data="in",
            output_data="out",
            metadata={"extra": "val"},
        )
        assert env.metadata["env"] == "prod"
        assert env.metadata["extra"] == "val"

    def test_record_with_custom_header_prefix(self) -> None:
        recorder = VerificationRecorder(header_prefix="x-custom-")
        assert recorder.header_prefix == "x-custom-"


# ===========================================================================
# tensorflow.py
# ===========================================================================


class TestAethelredKerasCallback:
    """Test AethelredKerasCallback."""

    def test_init_defaults(self) -> None:
        cb = AethelredKerasCallback()
        assert cb._capture_batch_events is False
        assert cb.last_verification is None

    def test_set_model(self) -> None:
        cb = AethelredKerasCallback()

        class FakeModel:
            pass

        cb.set_model(FakeModel())
        assert cb._model is not None

    def test_filtered_logs_empty(self) -> None:
        cb = AethelredKerasCallback()
        assert cb._filtered_logs(None) == {}
        assert cb._filtered_logs({}) == {}

    def test_filtered_logs_no_filter(self) -> None:
        cb = AethelredKerasCallback()
        logs = {"loss": 0.1, "acc": 0.99}
        assert cb._filtered_logs(logs) == logs

    def test_filtered_logs_with_filter(self) -> None:
        cb = AethelredKerasCallback(include_log_keys=("loss",))
        logs = {"loss": 0.1, "acc": 0.99}
        assert cb._filtered_logs(logs) == {"loss": 0.1}

    def test_on_train_batch_end_capture_off(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=False,
        )
        cb.on_train_batch_end(0, {"loss": 0.5})
        assert len(envelopes) == 0

    def test_on_train_batch_end_capture_on(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=True,
        )
        cb.on_train_batch_end(0, {"loss": 0.5})
        assert len(envelopes) == 1
        assert envelopes[0].operation == "train_batch_end"

    def test_on_test_batch_end_capture_off(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=False,
        )
        cb.on_test_batch_end(0, {"loss": 0.5})
        assert len(envelopes) == 0

    def test_on_test_batch_end_capture_on(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=True,
        )
        cb.on_test_batch_end(0, {"loss": 0.5})
        assert len(envelopes) == 1
        assert envelopes[0].operation == "test_batch_end"

    def test_on_predict_batch_end_capture_off(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=False,
        )
        cb.on_predict_batch_end(0)
        assert len(envelopes) == 0

    def test_on_predict_batch_end_capture_on(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=True,
        )
        cb.on_predict_batch_end(0, {"output": [1, 2, 3]})
        assert len(envelopes) == 1
        assert envelopes[0].operation == "predict_batch_end"

    def test_on_epoch_end(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        cb.on_epoch_end(1, {"loss": 0.1, "val_loss": 0.2})
        assert len(envelopes) == 1
        assert envelopes[0].operation == "epoch_end"

    def test_record_with_model_classname(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
            extra_metadata={"version": "2.0"},
        )

        class MyNeuralNet:
            pass

        cb.set_model(MyNeuralNet())
        cb.on_epoch_end(0, {})
        assert envelopes[0].metadata["model"] == "MyNeuralNet"
        assert envelopes[0].metadata["version"] == "2.0"

    def test_record_without_model(self) -> None:
        envelopes: list = []
        cb = AethelredKerasCallback(
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        cb.on_epoch_end(0, {})
        # When model is None, getattr(None.__class__, "__name__") returns "NoneType"
        assert envelopes[0].metadata["model"] == "NoneType"


class TestCreateKerasCallback:
    """Test create_keras_callback factory."""

    def test_returns_usable_callback(self) -> None:
        cb = create_keras_callback()
        assert hasattr(cb, "on_epoch_end")
        cb.on_epoch_end(0, {"loss": 1.0})
        assert cb.last_verification is not None

    def test_with_recorder(self) -> None:
        envelopes: list = []
        cb = create_keras_callback(
            recorder=VerificationRecorder(sink=envelopes.append),
            capture_batch_events=True,
        )
        cb.on_train_batch_end(0, {})
        cb.on_epoch_end(0, {})
        assert len(envelopes) == 2

    def test_with_extra_metadata(self) -> None:
        envelopes: list = []
        cb = create_keras_callback(
            recorder=VerificationRecorder(sink=envelopes.append),
            extra_metadata={"experiment": "test_run"},
        )
        cb.on_epoch_end(0, {})
        assert envelopes[0].metadata["experiment"] == "test_run"


# ===========================================================================
# langchain.py
# ===========================================================================


class TestVerifiedLangChainRunnable:
    """Test VerifiedLangChainRunnable."""

    def test_invoke_with_invoke_method(self) -> None:
        envelopes: list = []

        class FakeRunnable:
            def invoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
                return {"result": input}

        wrapped = VerifiedLangChainRunnable(
            FakeRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = wrapped.invoke({"q": "test"})
        assert result == {"result": {"q": "test"}}
        assert envelopes[0].operation == "invoke"

    def test_invoke_without_invoke_method(self) -> None:
        """Falls back to calling runnable directly."""
        envelopes: list = []

        class FakeCallable:
            def __call__(self, input: Any, **kw: Any) -> Any:
                return {"fallback": input}

        wrapped = VerifiedLangChainRunnable(
            FakeCallable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = wrapped.invoke({"q": "test"})
        assert result == {"fallback": {"q": "test"}}

    @pytest.mark.asyncio
    async def test_ainvoke_with_ainvoke_method(self) -> None:
        envelopes: list = []

        class FakeAsyncRunnable:
            async def ainvoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
                return {"async_result": input}

        wrapped = VerifiedLangChainRunnable(
            FakeAsyncRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = await wrapped.ainvoke({"q": "async_test"})
        assert result == {"async_result": {"q": "async_test"}}
        assert envelopes[0].operation == "ainvoke"

    @pytest.mark.asyncio
    async def test_ainvoke_without_ainvoke_method(self) -> None:
        """Falls back to calling runnable directly."""
        envelopes: list = []

        class FakeCallable:
            def __call__(self, input: Any, **kw: Any) -> Any:
                return {"sync_fallback": input}

        wrapped = VerifiedLangChainRunnable(
            FakeCallable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = await wrapped.ainvoke({"q": "test"})
        assert result == {"sync_fallback": {"q": "test"}}

    def test_batch_with_batch_method(self) -> None:
        envelopes: list = []

        class FakeRunnable:
            def batch(self, inputs: Sequence[Any], config: Any = None, **kw: Any) -> Any:
                return [{"batched": i} for i in inputs]

        wrapped = VerifiedLangChainRunnable(
            FakeRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = wrapped.batch([1, 2, 3])
        assert len(result) == 3
        assert envelopes[0].operation == "batch"
        assert envelopes[0].metadata["batch_size"] == 3

    def test_batch_without_batch_method(self) -> None:
        """Falls back to calling runnable per item."""
        envelopes: list = []

        class FakeCallable:
            def __call__(self, input: Any, **kw: Any) -> Any:
                return {"single": input}

        wrapped = VerifiedLangChainRunnable(
            FakeCallable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = wrapped.batch([1, 2])
        assert len(result) == 2
        assert result[0] == {"single": 1}

    @pytest.mark.asyncio
    async def test_abatch_with_abatch_method(self) -> None:
        envelopes: list = []

        class FakeRunnable:
            async def abatch(self, inputs: Sequence[Any], config: Any = None, **kw: Any) -> Any:
                return [{"async_batched": i} for i in inputs]

        wrapped = VerifiedLangChainRunnable(
            FakeRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = await wrapped.abatch([1, 2])
        assert len(result) == 2
        assert envelopes[0].operation == "abatch"

    @pytest.mark.asyncio
    async def test_abatch_with_ainvoke_fallback(self) -> None:
        """When abatch not available but ainvoke is, use ainvoke per item."""
        envelopes: list = []

        class FakeRunnable:
            async def ainvoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
                return {"ainvoked": input}

        wrapped = VerifiedLangChainRunnable(
            FakeRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = await wrapped.abatch([1, 2])
        assert len(result) == 2
        assert result[0] == {"ainvoked": 1}

    @pytest.mark.asyncio
    async def test_abatch_with_callable_fallback(self) -> None:
        """When neither abatch nor ainvoke, use __call__ per item."""
        envelopes: list = []

        class FakeCallable:
            def __call__(self, input: Any, **kw: Any) -> Any:
                return {"called": input}

        wrapped = VerifiedLangChainRunnable(
            FakeCallable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = await wrapped.abatch([1, 2])
        assert len(result) == 2
        assert result[0] == {"called": 1}

    def test_call_delegates_to_invoke(self) -> None:
        envelopes: list = []

        class FakeRunnable:
            def invoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
                return {"invoked": input, "kw": kw}

        wrapped = VerifiedLangChainRunnable(
            FakeRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = wrapped({"q": "test"})
        assert result["invoked"] == {"q": "test"}
        assert envelopes[0].operation == "invoke"

    def test_call_with_positional_args(self) -> None:
        """When called with extra positional args, they go into kwargs."""
        envelopes: list = []

        class FakeRunnable:
            def invoke(self, input: Any, config: Any = None, **kw: Any) -> Any:
                return {"input": input, "kw": kw}

        wrapped = VerifiedLangChainRunnable(
            FakeRunnable(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = wrapped({"q": "test"}, "extra_arg1", "extra_arg2")
        assert "_args" in result["kw"]
        assert result["kw"]["_args"] == ("extra_arg1", "extra_arg2")

    def test_getattr_delegates(self) -> None:
        class FakeRunnable:
            some_attr = "hello"

        wrapped = VerifiedLangChainRunnable(FakeRunnable())
        assert wrapped.some_attr == "hello"

    def test_component_name_default(self) -> None:
        class MyChain:
            pass

        wrapped = VerifiedLangChainRunnable(MyChain())
        assert wrapped._component_name == "MyChain"

    def test_component_name_custom(self) -> None:
        wrapped = VerifiedLangChainRunnable(
            MagicMock(),
            component_name="CustomName",
        )
        assert wrapped._component_name == "CustomName"


class TestWrapLangchainRunnable:
    """Test the wrap_langchain_runnable factory."""

    def test_wraps_with_defaults(self) -> None:
        mock = MagicMock()
        wrapped = wrap_langchain_runnable(mock)
        assert isinstance(wrapped, VerifiedLangChainRunnable)

    def test_wraps_with_options(self) -> None:
        mock = MagicMock()
        recorder = VerificationRecorder()
        wrapped = wrap_langchain_runnable(
            mock,
            recorder=recorder,
            component_name="TestComponent",
            extra_metadata={"key": "val"},
        )
        assert wrapped._component_name == "TestComponent"
        assert wrapped._extra_metadata["key"] == "val"


# ===========================================================================
# fastapi.py
# ===========================================================================


class TestFastapiHelpers:
    """Test fastapi module helper functions."""

    def test_normalize_header_prefix(self) -> None:
        assert _normalize_header_prefix("X-Custom-") == "x-custom"
        assert _normalize_header_prefix("x-aethelred") == "x-aethelred"
        assert _normalize_header_prefix("X-FOO---") == "x-foo"

    def test_headers_to_dict(self) -> None:
        headers = [
            (b"Content-Type", b"application/json"),
            (b"X-Custom", b"value"),
        ]
        result = _headers_to_dict(headers)
        assert result["content-type"] == "application/json"
        assert result["x-custom"] == "value"

    def test_merge_headers_new_key(self) -> None:
        headers: list = [(b"existing", b"val")]
        result = _merge_headers(headers, {"new-key": "new-val"})
        assert len(result) == 2

    def test_merge_headers_update_existing(self) -> None:
        headers: list = [(b"existing", b"old")]
        result = _merge_headers(headers, {"existing": "new"})
        assert len(result) == 1
        assert result[0] == (b"existing", b"new")


class TestAethelredVerificationMiddleware:
    """Test ASGI middleware."""

    def test_should_process_no_filters(self) -> None:
        async def app(scope, receive, send):
            pass

        mw = AethelredVerificationMiddleware(app)
        assert mw._should_process("/any/path") is True

    def test_should_process_include_paths_match(self) -> None:
        async def app(scope, receive, send):
            pass

        mw = AethelredVerificationMiddleware(app, include_paths=["/api/"])
        assert mw._should_process("/api/test") is True
        assert mw._should_process("/health") is False

    def test_should_process_exclude_paths_match(self) -> None:
        async def app(scope, receive, send):
            pass

        mw = AethelredVerificationMiddleware(app, exclude_paths=["/health"])
        assert mw._should_process("/health") is False
        assert mw._should_process("/api/test") is True

    def test_should_process_include_and_exclude(self) -> None:
        async def app(scope, receive, send):
            pass

        mw = AethelredVerificationMiddleware(
            app, include_paths=["/api/"], exclude_paths=["/api/internal"]
        )
        assert mw._should_process("/api/public") is True
        assert mw._should_process("/api/internal/secret") is False
        assert mw._should_process("/other") is False

    @pytest.mark.asyncio
    async def test_non_http_scope_passthrough(self) -> None:
        calls: list = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            calls.append("app_called")

        mw = AethelredVerificationMiddleware(app)
        await mw({"type": "websocket"}, None, None)
        assert calls == ["app_called"]

    @pytest.mark.asyncio
    async def test_excluded_path_passthrough(self) -> None:
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder, exclude_paths=["/skip"])
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/skip/this", "method": "GET", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert envelopes == []
        assert sent[0]["status"] == 200

    @pytest.mark.asyncio
    async def test_normal_request_adds_headers(self) -> None:
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"hello", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder)
        messages = [{"type": "http.request", "body": b'{"test":1}', "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/test", "method": "POST", "query_string": b"q=1", "headers": [(b"host", b"localhost")]},
            receive,
            send,
        )
        assert len(envelopes) == 1
        assert envelopes[0].framework == "fastapi"
        # Check headers were added to response start
        start_headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in sent[0]["headers"]}
        assert "x-aethelred-trace-id" in start_headers

    @pytest.mark.asyncio
    async def test_streamed_large_body(self) -> None:
        """When response exceeds max_capture_bytes, switches to streamed passthrough."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            # First chunk fits
            await send({"type": "http.response.body", "body": b"A" * 50, "more_body": True})
            # Second chunk pushes over the limit
            await send({"type": "http.response.body", "body": b"B" * 200, "more_body": True})
            # Third chunk is pass-through
            await send({"type": "http.response.body", "body": b"C" * 10, "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder, max_capture_bytes=100)
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/big", "method": "GET", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_request_body_truncation(self) -> None:
        """Request body exceeding max_capture_bytes is truncated."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            # Read all request body
            while True:
                msg = await receive()
                if not msg.get("more_body", False):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder, max_capture_bytes=10)
        messages = [
            {"type": "http.request", "body": b"A" * 20, "more_body": True},
            {"type": "http.request", "body": b"B" * 20, "more_body": False},
        ]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/upload", "method": "POST", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_app_completes_without_body(self) -> None:
        """When app returns response.start but no body, middleware flushes."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 204, "headers": []})
            # No body sent

        mw = AethelredVerificationMiddleware(app, recorder=recorder)
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/noop", "method": "DELETE", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_custom_header_prefix(self) -> None:
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append, header_prefix="x-myprefix")

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = AethelredVerificationMiddleware(
            app, recorder=recorder, header_prefix="x-myprefix"
        )
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/test", "method": "GET", "query_string": b"", "headers": []},
            receive,
            send,
        )
        start_headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in sent[0]["headers"]}
        assert "x-myprefix-trace-id" in start_headers

    @pytest.mark.asyncio
    async def test_non_body_message_passthrough(self) -> None:
        """Messages that are not http.response.start or http.response.body are passed through."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            # Send some weird message type before response
            await send({"type": "http.disconnect"})
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder)
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/test", "method": "GET", "query_string": b"", "headers": []},
            receive,
            send,
        )
        # http.disconnect should be passed through
        assert sent[0]["type"] == "http.disconnect"

    @pytest.mark.asyncio
    async def test_multi_chunk_body_within_limit(self) -> None:
        """Multiple body chunks that stay within max_capture_bytes get buffered."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"chunk1", "more_body": True})
            await send({"type": "http.response.body", "body": b"chunk2", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder, max_capture_bytes=1000)
        messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/test", "method": "GET", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert len(envelopes) == 1
        # Body should be combined
        body_msg = [m for m in sent if m.get("type") == "http.response.body"]
        assert body_msg[-1]["body"] == b"chunk1chunk2"

    @pytest.mark.asyncio
    async def test_receive_non_request_message(self) -> None:
        """Non http.request messages from receive are passed through."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            # First receive returns a non-request message
            msg1 = await receive()
            # Second is the actual request
            msg2 = await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder)
        messages = [
            {"type": "http.disconnect"},
            {"type": "http.request", "body": b"hello", "more_body": False},
        ]
        sent: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent.append(msg)

        await mw(
            {"type": "http", "path": "/api/test", "method": "POST", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_empty_request_body(self) -> None:
        """Handles empty or None body in request message."""
        envelopes: list = []
        recorder = VerificationRecorder(sink=envelopes.append)

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await receive()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok", "more_body": False})

        mw = AethelredVerificationMiddleware(app, recorder=recorder)
        messages = [{"type": "http.request", "body": None, "more_body": False}]
        sent_msgs: list = []

        async def receive() -> dict:
            return messages.pop(0)

        async def send(msg: dict) -> None:
            sent_msgs.append(msg)

        await mw(
            {"type": "http", "path": "/api/test", "method": "GET", "query_string": b"", "headers": []},
            receive,
            send,
        )
        assert len(envelopes) == 1


# ---------------------------------------------------------------------------
# PyTorch Integration Tests
# ---------------------------------------------------------------------------

from aethelred.integrations.pytorch import VerifiedPyTorchModule, wrap_pytorch_model
from aethelred.integrations.huggingface import VerifiedTransformersPipeline, wrap_transformers_pipeline


class TestVerifiedPyTorchModule:
    """Tests for VerifiedPyTorchModule from pytorch.py."""

    def test_init_defaults(self) -> None:
        class FakeModel:
            pass

        module = VerifiedPyTorchModule(FakeModel())
        assert module._component_name == "FakeModel"
        assert module._extra_metadata == {}
        assert module.last_verification is None

    def test_init_with_options(self) -> None:
        model = lambda x: x  # noqa: E731
        recorder = VerificationRecorder()
        module = VerifiedPyTorchModule(
            model,
            recorder=recorder,
            component_name="custom",
            extra_metadata={"key": "val"},
        )
        assert module._component_name == "custom"
        assert module._extra_metadata == {"key": "val"}

    def test_call_success(self) -> None:
        envelopes: list = []

        class FakeModel:
            def __call__(self, x: Any) -> Any:
                return x * 2

        module = VerifiedPyTorchModule(
            FakeModel(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = module(5)
        assert result == 10
        assert len(envelopes) == 1
        assert envelopes[0].operation == "forward"
        assert module.last_verification is envelopes[0]

    def test_call_with_kwargs(self) -> None:
        envelopes: list = []

        class FakeModel:
            def __call__(self, x: Any, scale: int = 1) -> Any:
                return x * scale

        module = VerifiedPyTorchModule(
            FakeModel(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = module(3, scale=4)
        assert result == 12
        assert envelopes[0].metadata["kwarg_keys"] == ["scale"]

    def test_call_error(self) -> None:
        envelopes: list = []

        class FakeModel:
            def __call__(self, x: Any) -> Any:
                raise ValueError("bad input")

        module = VerifiedPyTorchModule(
            FakeModel(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        with pytest.raises(ValueError, match="bad input"):
            module(None)
        assert len(envelopes) == 1
        assert envelopes[0].operation == "forward.error"
        assert module.last_verification is envelopes[0]

    def test_forward_delegates_to_call(self) -> None:
        class FakeModel:
            def __call__(self, x: Any) -> Any:
                return x + 1

        module = VerifiedPyTorchModule(FakeModel())
        assert module.forward(5) == 6

    def test_getattr_delegates(self) -> None:
        class FakeModel:
            custom_prop = "hello"

        module = VerifiedPyTorchModule(FakeModel())
        assert module.custom_prop == "hello"


class TestWrapPytorchModel:
    """Tests for wrap_pytorch_model helper."""

    def test_wraps_model(self) -> None:
        class FakeModel:
            def __call__(self, x: Any) -> Any:
                return x

        wrapped = wrap_pytorch_model(FakeModel())
        assert isinstance(wrapped, VerifiedPyTorchModule)

    def test_wraps_model_with_options(self) -> None:
        recorder = VerificationRecorder()
        wrapped = wrap_pytorch_model(
            lambda x: x,
            recorder=recorder,
            component_name="test",
            extra_metadata={"a": 1},
        )
        assert wrapped._component_name == "test"


# ---------------------------------------------------------------------------
# HuggingFace Integration Tests
# ---------------------------------------------------------------------------


class TestVerifiedTransformersPipeline:
    """Tests for VerifiedTransformersPipeline from huggingface.py."""

    def test_init_defaults_with_task(self) -> None:
        class FakePipeline:
            task = "text-classification"

            def __call__(self, text: str) -> Any:
                return [{"label": "POSITIVE", "score": 0.99}]

        pipe = VerifiedTransformersPipeline(FakePipeline())
        assert pipe._component_name == "text-classification"

    def test_init_defaults_without_task(self) -> None:
        class FakePipeline:
            def __call__(self, text: str) -> Any:
                return text

        pipe = VerifiedTransformersPipeline(FakePipeline())
        assert pipe._component_name == "FakePipeline"

    def test_init_with_options(self) -> None:
        recorder = VerificationRecorder()
        pipe = VerifiedTransformersPipeline(
            lambda x: x,
            recorder=recorder,
            component_name="my-pipe",
            extra_metadata={"key": "value"},
        )
        assert pipe._component_name == "my-pipe"
        assert pipe._extra_metadata == {"key": "value"}

    def test_call_records_envelope(self) -> None:
        envelopes: list = []

        class FakePipeline:
            task = "sentiment-analysis"

            def __call__(self, text: str) -> Any:
                return [{"label": "POSITIVE", "score": 0.95}]

        pipe = VerifiedTransformersPipeline(
            FakePipeline(),
            recorder=VerificationRecorder(sink=envelopes.append),
        )
        result = pipe("hello world")
        assert result == [{"label": "POSITIVE", "score": 0.95}]
        assert len(envelopes) == 1
        assert envelopes[0].framework == "huggingface.transformers"
        assert envelopes[0].operation == "pipeline"
        assert envelopes[0].metadata["task"] == "sentiment-analysis"
        assert pipe.last_verification is envelopes[0]

    def test_getattr_delegates(self) -> None:
        class FakePipeline:
            task = "ner"
            model_name = "bert-base"

        pipe = VerifiedTransformersPipeline(FakePipeline())
        assert pipe.model_name == "bert-base"


class TestWrapTransformersPipeline:
    """Tests for wrap_transformers_pipeline helper."""

    def test_wraps_pipeline(self) -> None:
        class FakePipeline:
            task = "qa"

            def __call__(self, *args: Any) -> Any:
                return "answer"

        wrapped = wrap_transformers_pipeline(FakePipeline())
        assert isinstance(wrapped, VerifiedTransformersPipeline)

    def test_wraps_pipeline_with_options(self) -> None:
        recorder = VerificationRecorder()
        wrapped = wrap_transformers_pipeline(
            lambda x: x,
            recorder=recorder,
            component_name="test-pipe",
            extra_metadata={"model": "gpt2"},
        )
        assert wrapped._component_name == "test-pipe"


# ---------------------------------------------------------------------------
# TensorFlow Native Callback Tests (mock-based, covers tensorflow.py lines 78-103)
# ---------------------------------------------------------------------------


class TestTensorFlowNativeCallback:
    """Test the _TensorFlowAethelredCallback class created when TensorFlow is available.

    Uses a mock tensorflow module injected into sys.modules to trigger
    the TF-native callback path (lines 78-103 of tensorflow.py).
    """

    @staticmethod
    def _make_mock_tf() -> MagicMock:
        """Create a mock tensorflow module with a real Callback base class."""

        class MockCallback:
            """Stand-in for tf.keras.callbacks.Callback."""

            def __init__(self) -> None:
                pass

            def set_model(self, model: Any) -> None:
                self._model_from_super = model

        mock_tf = MagicMock()
        mock_tf.keras.callbacks.Callback = MockCallback
        return mock_tf

    def test_creates_tf_native_callback(self) -> None:
        """With TF available, create_keras_callback returns a TF-native instance."""
        mock_tf = self._make_mock_tf()
        # Remove any cached tensorflow module to force re-import inside the function
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback()
            # Should NOT be a plain AethelredKerasCallback
            assert not type(cb).__name__ == "AethelredKerasCallback"
            # Should have the _inner attribute from _TensorFlowAethelredCallback
            assert hasattr(cb, "_inner")
            assert isinstance(cb._inner, AethelredKerasCallback)
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_set_model_delegates(self) -> None:
        """_TensorFlowAethelredCallback.set_model calls both super and inner."""
        mock_tf = self._make_mock_tf()
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback()

            class FakeModel:
                pass

            model = FakeModel()
            cb.set_model(model)
            # Inner should have the model
            assert cb._inner._model is model
            # Super (MockCallback) should also have been called
            assert hasattr(cb, "_model_from_super")
            assert cb._model_from_super is model
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_on_train_batch_end_delegates(self) -> None:
        """on_train_batch_end delegates to inner."""
        mock_tf = self._make_mock_tf()
        envelopes: list = []
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback(
                    recorder=VerificationRecorder(sink=envelopes.append),
                    capture_batch_events=True,
                )
            cb.on_train_batch_end(0, {"loss": 0.5})
            assert len(envelopes) == 1
            assert envelopes[0].operation == "train_batch_end"
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_on_test_batch_end_delegates(self) -> None:
        """on_test_batch_end delegates to inner."""
        mock_tf = self._make_mock_tf()
        envelopes: list = []
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback(
                    recorder=VerificationRecorder(sink=envelopes.append),
                    capture_batch_events=True,
                )
            cb.on_test_batch_end(0, {"loss": 0.3})
            assert len(envelopes) == 1
            assert envelopes[0].operation == "test_batch_end"
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_on_predict_batch_end_delegates(self) -> None:
        """on_predict_batch_end delegates to inner."""
        mock_tf = self._make_mock_tf()
        envelopes: list = []
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback(
                    recorder=VerificationRecorder(sink=envelopes.append),
                    capture_batch_events=True,
                )
            cb.on_predict_batch_end(0, {"output": [1, 2]})
            assert len(envelopes) == 1
            assert envelopes[0].operation == "predict_batch_end"
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_on_epoch_end_delegates(self) -> None:
        """on_epoch_end delegates to inner."""
        mock_tf = self._make_mock_tf()
        envelopes: list = []
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback(
                    recorder=VerificationRecorder(sink=envelopes.append),
                )
            cb.on_epoch_end(1, {"loss": 0.1})
            assert len(envelopes) == 1
            assert envelopes[0].operation == "epoch_end"
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved

    def test_last_verification_property(self) -> None:
        """last_verification property returns inner's value."""
        mock_tf = self._make_mock_tf()
        envelopes: list = []
        saved = sys.modules.pop("tensorflow", None)
        try:
            with patch.dict(sys.modules, {"tensorflow": mock_tf}):
                cb = create_keras_callback(
                    recorder=VerificationRecorder(sink=envelopes.append),
                )
            # Initially None
            assert cb.last_verification is None
            # After an event, should reflect the inner's value
            cb.on_epoch_end(0, {})
            assert cb.last_verification is not None
            assert cb.last_verification is cb._inner.last_verification
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved
