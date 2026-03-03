"""Tests for native framework integration adapters."""

from __future__ import annotations

import asyncio
from typing import Any, List

from aethelred.integrations import (
    AethelredKerasCallback,
    AethelredVerificationMiddleware,
    VerificationRecorder,
    hash_payload,
    wrap_langchain_runnable,
    wrap_pytorch_model,
    wrap_transformers_pipeline,
)
from aethelred.integrations.tensorflow import create_keras_callback


def test_hash_payload_is_stable_across_dict_order() -> None:
    left = {"b": 2, "a": [3, 4]}
    right = {"a": [3, 4], "b": 2}
    assert hash_payload(left) == hash_payload(right)


def test_wrap_pytorch_model_records_forward() -> None:
    envelopes: List[Any] = []
    recorder = VerificationRecorder(sink=envelopes.append)

    class FakeModel:
        def __call__(self, x: int, scale: int = 1) -> int:
            return x * scale

    wrapped = wrap_pytorch_model(FakeModel(), recorder=recorder, component_name="FraudNet")
    result = wrapped(7, scale=3)

    assert result == 21
    assert wrapped.last_verification is not None
    assert wrapped.last_verification.framework == "pytorch"
    assert wrapped.last_verification.operation == "forward"
    assert envelopes[0].metadata["component"] == "FraudNet"


def test_wrap_pytorch_model_records_errors() -> None:
    envelopes: List[Any] = []
    recorder = VerificationRecorder(sink=envelopes.append)

    class FailingModel:
        def __call__(self, x: int) -> int:
            raise ValueError(f"bad:{x}")

    wrapped = wrap_pytorch_model(FailingModel(), recorder=recorder)
    try:
        wrapped(5)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")

    assert wrapped.last_verification is not None
    assert wrapped.last_verification.operation == "forward.error"
    assert envelopes and envelopes[0].framework == "pytorch"


def test_keras_callback_records_epoch_and_optionally_batch() -> None:
    envelopes: List[Any] = []
    callback = AethelredKerasCallback(
        recorder=VerificationRecorder(sink=envelopes.append),
        capture_batch_events=True,
        include_log_keys=("loss",),
    )

    class FakeKerasModel:
        pass

    callback.set_model(FakeKerasModel())
    callback.on_train_batch_end(3, {"loss": 0.4, "acc": 0.99})
    callback.on_epoch_end(1, {"loss": 0.2, "val_loss": 0.3})

    assert len(envelopes) == 2
    assert envelopes[0].operation == "train_batch_end"
    assert envelopes[1].operation == "epoch_end"


def test_create_keras_callback_returns_usable_callback_without_tensorflow_dependency() -> None:
    callback = create_keras_callback()
    assert hasattr(callback, "on_epoch_end")
    callback.on_epoch_end(0, {"loss": 1.0})
    assert getattr(callback, "last_verification", None) is not None


def test_huggingface_pipeline_wrapper_records_call() -> None:
    envelopes: List[Any] = []

    class FakePipeline:
        task = "text-classification"

        def __call__(self, text: str) -> list[dict[str, Any]]:
            return [{"label": "SAFE", "score": 0.98, "text": text}]

    wrapped = wrap_transformers_pipeline(FakePipeline(), recorder=VerificationRecorder(sink=envelopes.append))
    result = wrapped("approve invoice")

    assert result[0]["label"] == "SAFE"
    assert wrapped.last_verification is not None
    assert wrapped.last_verification.framework == "huggingface.transformers"
    assert envelopes[0].metadata["task"] == "text-classification"


def test_langchain_wrapper_invoke_and_batch_record() -> None:
    envelopes: List[Any] = []

    class FakeRunnable:
        def invoke(self, payload: dict[str, Any], config: Any = None) -> dict[str, Any]:
            return {"answer": payload["question"].upper(), "config": config}

        def batch(self, items: list[dict[str, Any]], config: Any = None) -> list[dict[str, Any]]:
            return [self.invoke(item, config=config) for item in items]

    wrapped = wrap_langchain_runnable(FakeRunnable(), recorder=VerificationRecorder(sink=envelopes.append))
    single = wrapped.invoke({"question": "kyc status"}, config={"tenant": "fab"})
    batch = wrapped.batch([{"question": "a"}, {"question": "b"}], config={"tenant": "fab"})

    assert single["answer"] == "KYC STATUS"
    assert [x["answer"] for x in batch] == ["A", "B"]
    assert [e.operation for e in envelopes] == ["invoke", "batch"]


def test_langchain_wrapper_ainvoke_records_async() -> None:
    envelopes: List[Any] = []

    class FakeAsyncRunnable:
        async def ainvoke(self, payload: dict[str, Any], config: Any = None) -> dict[str, Any]:
            await asyncio.sleep(0)
            return {"answer": payload["question"], "config": config}

    wrapped = wrap_langchain_runnable(FakeAsyncRunnable(), recorder=VerificationRecorder(sink=envelopes.append))
    result = asyncio.run(wrapped.ainvoke({"question": "hello"}, config={"trace": 1}))

    assert result["answer"] == "hello"
    assert wrapped.last_verification is not None
    assert wrapped.last_verification.operation == "ainvoke"
    assert envelopes[0].framework == "langchain"


def test_fastapi_middleware_adds_verification_headers_and_records() -> None:
    envelopes: List[Any] = []
    recorder = VerificationRecorder(sink=envelopes.append)

    async def app(scope: Any, receive: Any, send: Any) -> None:
        body = b""
        while True:
            msg = await receive()
            if msg["type"] != "http.request":
                continue
            body += msg.get("body", b"") or b""
            if not msg.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{\"ok\":true}', "more_body": False})

    middleware = AethelredVerificationMiddleware(app, recorder=recorder)

    async def run() -> list[dict[str, Any]]:
        request_messages = [
            {"type": "http.request", "body": b'{\"input\":\"x\"}', "more_body": False},
        ]
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return request_messages.pop(0)

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/score",
                "query_string": b"",
                "headers": [(b"host", b"example.com")],
            },
            receive,
            send,
        )
        return sent

    sent = asyncio.run(run())
    assert envelopes and envelopes[0].framework == "fastapi"
    start = sent[0]
    headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in start["headers"]}
    assert "x-aethelred-trace-id" in headers
    assert "x-aethelred-input-hash" in headers
    assert "x-aethelred-output-hash" in headers


def test_fastapi_middleware_respects_excluded_paths() -> None:
    envelopes: List[Any] = []
    recorder = VerificationRecorder(sink=envelopes.append)

    async def app(scope: Any, receive: Any, send: Any) -> None:
        await receive()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    middleware = AethelredVerificationMiddleware(app, recorder=recorder, exclude_paths=["/health"])

    async def run() -> list[dict[str, Any]]:
        request_messages = [{"type": "http.request", "body": b"", "more_body": False}]
        sent: list[dict[str, Any]] = []

        async def receive() -> dict[str, Any]:
            return request_messages.pop(0)

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await middleware(
            {"type": "http", "method": "GET", "path": "/healthz", "query_string": b"", "headers": []},
            receive,
            send,
        )
        return sent

    sent = asyncio.run(run())
    assert sent[0]["status"] == 204
    assert envelopes == []
