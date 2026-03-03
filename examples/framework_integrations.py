"""Framework integration examples for Aethelred Python SDK.

These examples intentionally use light wrappers/stubs so they can be read
without requiring all framework dependencies at import time.
"""

from __future__ import annotations

from aethelred.integrations import (
    AethelredKerasCallback,
    AethelredVerificationMiddleware,
    VerificationRecorder,
    wrap_langchain_runnable,
    wrap_pytorch_model,
    wrap_transformers_pipeline,
)


def _print_record(envelope) -> None:
    print(
        f"[{envelope.framework}] {envelope.operation} "
        f"trace={envelope.trace_id} input={envelope.input_hash[:12]}.. "
        f"output={envelope.output_hash[:12]}.."
    )


def pytorch_example() -> None:
    class FraudModel:
        def __call__(self, features, threshold: float = 0.5):
            score = sum(features) / max(len(features), 1)
            return {"score": score, "fraud": score >= threshold}

    recorder = VerificationRecorder(sink=_print_record, default_metadata={"service": "fraud-api"})
    model = wrap_pytorch_model(FraudModel(), recorder=recorder, component_name="FraudScorerV1")
    print(model([0.2, 0.8, 0.9], threshold=0.6))


def keras_callback_example() -> None:
    callback = AethelredKerasCallback(
        recorder=VerificationRecorder(sink=_print_record),
        capture_batch_events=True,
        include_log_keys=("loss", "val_loss"),
    )
    callback.on_train_batch_end(0, {"loss": 0.42, "acc": 0.98})
    callback.on_epoch_end(1, {"loss": 0.12, "val_loss": 0.19, "val_acc": 0.95})


def huggingface_example() -> None:
    class FakePipeline:
        task = "text-classification"

        def __call__(self, text: str):
            return [{"label": "APPROVE", "score": 0.97, "text": text}]

    pipeline = wrap_transformers_pipeline(FakePipeline(), recorder=VerificationRecorder(sink=_print_record))
    print(pipeline("Approve payment for vendor INV-2381"))


def langchain_example() -> None:
    class Chain:
        def invoke(self, payload, config=None):
            return {"answer": payload["question"].upper(), "trace": config}

    chain = wrap_langchain_runnable(Chain(), recorder=VerificationRecorder(sink=_print_record))
    print(chain.invoke({"question": "aml screening status?"}, config={"tenant": "fab"}))


def fastapi_example() -> None:
    # To use in a real app:
    #
    # from fastapi import FastAPI
    # app = FastAPI()
    # app.add_middleware(
    #     AethelredVerificationMiddleware,
    #     recorder=VerificationRecorder(sink=my_record_sink),
    # )
    #
    print("FastAPI middleware available:", AethelredVerificationMiddleware.__name__)


if __name__ == "__main__":
    pytorch_example()
    keras_callback_example()
    huggingface_example()
    langchain_example()
    fastapi_example()
