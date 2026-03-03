"""LangChain chain/runnable wrappers for Aethelred verification envelopes."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ._common import VerificationEnvelope, VerificationRecorder


class VerifiedLangChainRunnable:
    """Wraps a LangChain Runnable/Chain and records invoke/batch operations."""

    def __init__(
        self,
        runnable: Any,
        *,
        recorder: Optional[VerificationRecorder] = None,
        component_name: Optional[str] = None,
        extra_metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._runnable = runnable
        self._recorder = recorder or VerificationRecorder()
        self._component_name = component_name or type(runnable).__name__
        self._extra_metadata = dict(extra_metadata or {})
        self.last_verification: Optional[VerificationEnvelope] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runnable, name)

    def _metadata(self, operation: str) -> Mapping[str, Any]:
        return {
            "component": self._component_name,
            "operation_kind": operation,
            **self._extra_metadata,
        }

    def invoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        if hasattr(self._runnable, "invoke"):
            output = self._runnable.invoke(input, config=config, **kwargs)
        else:
            output = self._runnable(input, **kwargs)
        self.last_verification = self._recorder.record(
            framework="langchain",
            operation="invoke",
            input_data={"input": input, "config": config, "kwargs": kwargs},
            output_data=output,
            metadata=self._metadata("invoke"),
        )
        return output

    async def ainvoke(self, input: Any, config: Optional[Any] = None, **kwargs: Any) -> Any:
        if hasattr(self._runnable, "ainvoke"):
            output = await self._runnable.ainvoke(input, config=config, **kwargs)
        else:
            output = self._runnable(input, **kwargs)
        self.last_verification = await self._recorder.arecord(
            framework="langchain",
            operation="ainvoke",
            input_data={"input": input, "config": config, "kwargs": kwargs},
            output_data=output,
            metadata=self._metadata("ainvoke"),
        )
        return output

    def batch(self, inputs: Sequence[Any], config: Optional[Any] = None, **kwargs: Any) -> Any:
        if hasattr(self._runnable, "batch"):
            output = self._runnable.batch(inputs, config=config, **kwargs)
        else:
            output = [self._runnable(item, **kwargs) for item in inputs]
        self.last_verification = self._recorder.record(
            framework="langchain",
            operation="batch",
            input_data={"inputs": list(inputs), "config": config, "kwargs": kwargs},
            output_data=output,
            metadata={**self._metadata("batch"), "batch_size": len(inputs)},
        )
        return output

    async def abatch(self, inputs: Sequence[Any], config: Optional[Any] = None, **kwargs: Any) -> Any:
        if hasattr(self._runnable, "abatch"):
            output = await self._runnable.abatch(inputs, config=config, **kwargs)
        elif hasattr(self._runnable, "ainvoke"):
            output = [await self._runnable.ainvoke(item, config=config, **kwargs) for item in inputs]
        else:
            output = [self._runnable(item, **kwargs) for item in inputs]
        self.last_verification = await self._recorder.arecord(
            framework="langchain",
            operation="abatch",
            input_data={"inputs": list(inputs), "config": config, "kwargs": kwargs},
            output_data=output,
            metadata={**self._metadata("abatch"), "batch_size": len(inputs)},
        )
        return output

    def __call__(self, input: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            kwargs = {"_args": args, **kwargs}
        return self.invoke(input, **kwargs)


def wrap_langchain_runnable(
    runnable: Any,
    *,
    recorder: Optional[VerificationRecorder] = None,
    component_name: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> VerifiedLangChainRunnable:
    return VerifiedLangChainRunnable(
        runnable,
        recorder=recorder,
        component_name=component_name,
        extra_metadata=extra_metadata,
    )


__all__ = ["VerifiedLangChainRunnable", "wrap_langchain_runnable"]
