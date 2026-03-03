"""FastAPI/ASGI middleware for emitting Aethelred verification headers and records."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, MutableSequence, Optional, Sequence, Tuple

from ._common import VerificationEnvelope, VerificationRecorder


HeaderList = MutableSequence[Tuple[bytes, bytes]]


def _normalize_header_prefix(prefix: str) -> str:
    return prefix.lower().rstrip("-")


def _headers_to_dict(headers: Sequence[Tuple[bytes, bytes]]) -> dict[str, str]:
    return {
        k.decode("latin1").lower(): v.decode("latin1")
        for k, v in headers
    }


def _merge_headers(headers: HeaderList, additions: Mapping[str, str]) -> HeaderList:
    existing = {k.decode("latin1").lower(): idx for idx, (k, _) in enumerate(headers)}
    for key, value in additions.items():
        key_bytes = key.encode("latin1")
        value_bytes = value.encode("latin1")
        lower_key = key.lower()
        if lower_key in existing:
            headers[existing[lower_key]] = (key_bytes, value_bytes)
        else:
            headers.append((key_bytes, value_bytes))
    return headers


class AethelredVerificationMiddleware:
    """ASGI middleware that emits verification headers for HTTP responses."""

    def __init__(
        self,
        app: Any,
        *,
        recorder: Optional[VerificationRecorder] = None,
        include_paths: Optional[Iterable[str]] = None,
        exclude_paths: Optional[Iterable[str]] = None,
        max_capture_bytes: int = 262_144,
        header_prefix: str = "x-aethelred",
    ) -> None:
        self.app = app
        self.recorder = recorder or VerificationRecorder(header_prefix=header_prefix)
        self.include_paths = tuple(include_paths or ())
        self.exclude_paths = tuple(exclude_paths or ())
        self.max_capture_bytes = max_capture_bytes
        self.header_prefix = _normalize_header_prefix(header_prefix)

    def _should_process(self, path: str) -> bool:
        if self.include_paths and not any(path.startswith(p) for p in self.include_paths):
            return False
        if self.exclude_paths and any(path.startswith(p) for p in self.exclude_paths):
            return False
        return True

    async def __call__(self, scope: Mapping[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", "/"))
        if not self._should_process(path):
            await self.app(scope, receive, send)
            return

        request_chunks: List[bytes] = []
        request_truncated = False
        request_size = 0

        async def wrapped_receive() -> Mapping[str, Any]:
            nonlocal request_truncated, request_size
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"") or b""
                if body and not request_truncated:
                    remaining = self.max_capture_bytes - request_size
                    if remaining > 0:
                        request_chunks.append(body[:remaining])
                        request_size += min(len(body), remaining)
                    if len(body) > remaining:
                        request_truncated = True
            return message

        response_start: Optional[dict[str, Any]] = None
        buffered_body: List[bytes] = []
        buffered_size = 0
        streamed_passthrough = False
        record_emitted = False

        async def emit_record(output_payload: Any, *, status_code: int, streamed: bool) -> VerificationEnvelope:
            request_headers = _headers_to_dict(scope.get("headers", []) or [])
            request_payload = {
                "method": scope.get("method"),
                "path": path,
                "query_string": (scope.get("query_string") or b"").decode("latin1"),
                "headers": request_headers,
                "body": b"".join(request_chunks),
                "request_truncated": request_truncated,
            }
            return await self.recorder.arecord(
                framework="fastapi",
                operation="http.request",
                input_data=request_payload,
                output_data=output_payload,
                metadata={
                    "status_code": status_code,
                    "streamed": streamed,
                    "response_truncated": streamed_passthrough,
                },
            )

        async def flush_start_with_headers(envelope: Optional[VerificationEnvelope]) -> None:
            nonlocal response_start
            if response_start is None:
                return
            headers = list(response_start.get("headers", []))
            if envelope is not None:
                headers = _merge_headers(headers, envelope.to_headers(self.header_prefix))
            response_start["headers"] = headers
            await send(response_start)
            response_start = None

        async def wrapped_send(message: Mapping[str, Any]) -> None:
            nonlocal response_start, buffered_size, streamed_passthrough, record_emitted

            msg_type = message.get("type")
            if msg_type == "http.response.start":
                response_start = dict(message)
                response_start["headers"] = list(message.get("headers", []))
                return

            if msg_type != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"") or b""
            more_body = bool(message.get("more_body", False))
            status_code = int(response_start.get("status", 200) if response_start else 200)

            if streamed_passthrough:
                await send(message)
                return

            next_size = buffered_size + len(body)
            if next_size <= self.max_capture_bytes:
                buffered_body.append(body)
                buffered_size = next_size
            else:
                streamed_passthrough = True

            if streamed_passthrough:
                if not record_emitted:
                    envelope = await emit_record(
                        {
                            "body": b"".join(buffered_body),
                            "truncated": True,
                        },
                        status_code=status_code,
                        streamed=True,
                    )
                    record_emitted = True
                    await flush_start_with_headers(envelope)
                for chunk in buffered_body:
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                buffered_body.clear()
                await send(message)
                return

            if more_body:
                return

            envelope = None
            if not record_emitted:
                envelope = await emit_record(
                    {"body": b"".join(buffered_body), "truncated": False},
                    status_code=status_code,
                    streamed=False,
                )
                record_emitted = True

            await flush_start_with_headers(envelope)
            await send(
                {
                    "type": "http.response.body",
                    "body": b"".join(buffered_body),
                    "more_body": False,
                }
            )
            buffered_body.clear()

        await self.app(scope, wrapped_receive, wrapped_send)

        # Some handlers can complete without body. Flush stored start if needed.
        if response_start is not None:
            envelope = None
            if not record_emitted:
                envelope = await emit_record(
                    {"body": b"", "truncated": False},
                    status_code=int(response_start.get("status", 200)),
                    streamed=False,
                )
            await flush_start_with_headers(envelope)
            await send({"type": "http.response.body", "body": b"", "more_body": False})


__all__ = ["AethelredVerificationMiddleware"]
