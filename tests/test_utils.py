"""Tests for utils module — hashing, hex conversion, formatting, retry."""

from __future__ import annotations

import pytest

from aethelred.utils import (
    sha256_hex,
    sha256_bytes,
    bytes_to_hex,
    hex_to_bytes,
    format_size,
    retry,
)


class TestSHA256:
    """Test SHA-256 helper functions."""

    def test_sha256_hex_bytes(self) -> None:
        h = sha256_hex(b"hello")
        assert len(h) == 64
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_sha256_hex_string(self) -> None:
        h = sha256_hex("hello")
        assert h == sha256_hex(b"hello")

    def test_sha256_bytes_length(self) -> None:
        digest = sha256_bytes(b"data")
        assert len(digest) == 32
        assert isinstance(digest, bytes)

    def test_sha256_deterministic(self) -> None:
        assert sha256_hex(b"x") == sha256_hex(b"x")

    def test_sha256_different_input(self) -> None:
        assert sha256_hex(b"a") != sha256_hex(b"b")

    def test_sha256_empty(self) -> None:
        h = sha256_hex(b"")
        assert len(h) == 64


class TestHexConversion:
    """Test hex conversion utilities."""

    def test_bytes_to_hex(self) -> None:
        assert bytes_to_hex(b"\xde\xad") == "dead"

    def test_hex_to_bytes(self) -> None:
        assert hex_to_bytes("dead") == b"\xde\xad"

    def test_roundtrip(self) -> None:
        data = b"\x00\x01\xff\xab"
        assert hex_to_bytes(bytes_to_hex(data)) == data

    def test_invalid_hex_raises(self) -> None:
        with pytest.raises(ValueError):
            hex_to_bytes("zzzz")


class TestFormatSize:
    """Test human-readable size formatting."""

    def test_zero(self) -> None:
        assert format_size(0) == "0 B"

    def test_bytes(self) -> None:
        assert format_size(500) == "500 B"

    def test_kib(self) -> None:
        assert format_size(1024) == "1.0 KiB"

    def test_mib(self) -> None:
        assert format_size(1048576) == "1.0 MiB"

    def test_gib(self) -> None:
        assert format_size(1073741824) == "1.0 GiB"


class TestRetry:
    """Test retry decorator."""

    def test_succeeds_first_try(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert succeed() == "ok"
        assert call_count == 1

    def test_retries_on_failure(self) -> None:
        call_count = 0

        @retry(max_attempts=3, backoff_base=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "done"

        assert fail_twice() == "done"
        assert call_count == 3

    def test_raises_after_max_attempts(self) -> None:
        @retry(max_attempts=2, backoff_base=0.01)
        def always_fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            always_fail()

    def test_only_catches_specified_exceptions(self) -> None:
        @retry(max_attempts=3, backoff_base=0.01, exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("wrong type")

        with pytest.raises(TypeError):
            raise_type_error()
