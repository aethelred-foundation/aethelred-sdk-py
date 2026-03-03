"""Comprehensive tests for aethelred/core/runtime.py to achieve 95%+ coverage.

Covers:
- DeviceType, MemoryType enums
- DeviceCapabilities dataclass and properties
- Device class: factory methods, context management, initialization
- MemoryBlock dataclass
- MemoryPool: allocation, free, defragment, trim, stats
- Event, Stream, StreamPriority, StreamCommand
- JITCache, JITCompiler, jit decorator
- CompilationOptions, OptimizationLevel
- ProfileEvent, Profiler, profile decorator
- Runtime singleton
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aethelred.core.runtime import (
    DeviceType,
    MemoryType,
    DeviceCapabilities,
    Device,
    MemoryBlock,
    MemoryPool,
    Event,
    Stream,
    StreamPriority,
    StreamCommand,
    JITCache,
    JITCompiler,
    CompilationOptions,
    OptimizationLevel,
    ProfileEvent,
    Profiler,
    Runtime,
    jit,
    profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cpu_caps(**overrides) -> DeviceCapabilities:
    """Create a DeviceCapabilities for testing."""
    defaults = dict(
        device_type=DeviceType.CPU,
        device_id=0,
        name="Test CPU",
        compute_capability=(1, 0),
        total_memory=8 * 1024**3,
        memory_bandwidth=50.0,
        l1_cache_size=32 * 1024,
        l2_cache_size=256 * 1024,
        shared_memory_per_block=0,
        multiprocessors=1,
        cores_per_multiprocessor=4,
        max_threads_per_block=1,
        max_threads_per_multiprocessor=4,
        warp_size=1,
        clock_rate=3000,
    )
    defaults.update(overrides)
    return DeviceCapabilities(**defaults)


@pytest.fixture(autouse=True)
def _reset_device_state():
    """Reset Device class state between tests."""
    old_instances = Device._instances.copy()
    old_current = Device._current
    old_runtime_init = Runtime._initialized
    old_runtime_inst = Runtime._instance
    yield
    Device._instances = old_instances
    Device._current = old_current
    Runtime._initialized = old_runtime_init
    Runtime._instance = old_runtime_inst


# ============================================================================
# DeviceType & MemoryType Enums
# ============================================================================


class TestDeviceType:
    def test_all_device_types_exist(self):
        assert DeviceType.CPU is not None
        assert DeviceType.GPU_NVIDIA is not None
        assert DeviceType.GPU_AMD is not None
        assert DeviceType.GPU_INTEL is not None
        assert DeviceType.TPU is not None
        assert DeviceType.NPU is not None
        assert DeviceType.TEE_SGX is not None
        assert DeviceType.TEE_SEV is not None
        assert DeviceType.TEE_NITRO is not None
        assert DeviceType.TEE_TRUSTZONE is not None
        assert DeviceType.FPGA is not None
        assert DeviceType.REMOTE is not None


class TestMemoryType:
    def test_all_memory_types_exist(self):
        assert MemoryType.HOST is not None
        assert MemoryType.DEVICE is not None
        assert MemoryType.UNIFIED is not None
        assert MemoryType.PINNED is not None
        assert MemoryType.MAPPED is not None
        assert MemoryType.SHARED is not None
        assert MemoryType.ENCRYPTED is not None


# ============================================================================
# DeviceCapabilities
# ============================================================================


class TestDeviceCapabilities:
    def test_total_cores(self):
        caps = _make_cpu_caps(multiprocessors=2, cores_per_multiprocessor=8)
        assert caps.total_cores == 16

    def test_theoretical_flops(self):
        caps = _make_cpu_caps(multiprocessors=1, cores_per_multiprocessor=4, clock_rate=3000)
        expected = 4 * 3000 * 1e6 * 2
        assert caps.theoretical_flops == expected

    def test_repr(self):
        caps = _make_cpu_caps(name="MyDevice")
        r = repr(caps)
        assert "MyDevice" in r
        assert "cores=" in r
        assert "memory=" in r
        assert "compute=" in r

    def test_default_features(self):
        caps = _make_cpu_caps()
        assert caps.supports_fp16 is True
        assert caps.supports_bf16 is False
        assert caps.supports_fp64 is True
        assert caps.supports_int8 is True
        assert caps.supports_int4 is False
        assert caps.supports_tensor_cores is False
        assert caps.supports_async_copy is True
        assert caps.supports_cooperative_groups is False
        assert caps.tee_attestation_supported is False
        assert caps.tee_max_enclave_size == 0
        assert caps.tee_epc_size == 0


# ============================================================================
# Device
# ============================================================================


class TestDevice:
    def test_cpu_factory(self):
        Device._instances = {}
        dev = Device.cpu(0)
        assert dev.device_type == DeviceType.CPU
        assert dev.device_id == 0

    def test_cpu_factory_cached(self):
        Device._instances = {}
        d1 = Device.cpu(0)
        d2 = Device.cpu(0)
        assert d1 is d2

    def test_get_default_returns_cpu_when_no_gpu(self):
        Device._current = None
        Device._instances = {}
        dev = Device.get_default()
        assert dev.device_type == DeviceType.CPU

    def test_get_default_returns_current(self):
        cpu = Device(DeviceType.CPU, 0, _make_cpu_caps())
        Device._current = cpu
        assert Device.get_default() is cpu

    def test_gpu_not_found_raises(self):
        with pytest.raises(RuntimeError, match="GPU"):
            Device.gpu(99)

    def test_tee_unknown_platform(self):
        with pytest.raises(ValueError, match="Unknown TEE platform"):
            Device.tee(platform="nonexistent")

    def test_tee_auto_no_device(self):
        # enumerate_devices returns only CPU by default
        with pytest.raises(RuntimeError, match="No TEE device"):
            Device.tee(platform="auto")

    def test_tee_sgx_factory(self):
        Device._instances = {}
        dev = Device.tee(platform="sgx")
        assert dev.device_type == DeviceType.TEE_SGX

    def test_tee_sev_factory(self):
        Device._instances = {}
        dev = Device.tee(platform="sev")
        assert dev.device_type == DeviceType.TEE_SEV

    def test_tee_nitro_factory(self):
        Device._instances = {}
        dev = Device.tee(platform="nitro")
        assert dev.device_type == DeviceType.TEE_NITRO

    def test_tee_trustzone_factory(self):
        Device._instances = {}
        dev = Device.tee(platform="trustzone")
        assert dev.device_type == DeviceType.TEE_TRUSTZONE

    def test_enumerate_devices_includes_cpu(self):
        devs = Device.enumerate_devices()
        assert len(devs) >= 1
        assert devs[0].device_type == DeviceType.CPU

    def test_detect_cpu_capabilities(self):
        caps = Device._detect_cpu_capabilities()
        assert caps.device_type == DeviceType.CPU
        assert caps.total_memory > 0

    def test_detect_nvidia_gpus(self):
        result = Device._detect_nvidia_gpus()
        assert isinstance(result, list)

    def test_detect_amd_gpus(self):
        result = Device._detect_amd_gpus()
        assert isinstance(result, list)

    def test_detect_tee_environments(self):
        result = Device._detect_tee_environments()
        assert isinstance(result, list)

    def test_capabilities_property(self):
        dev = Device(DeviceType.CPU, 0)
        caps = dev.capabilities
        assert caps is not None
        assert caps.device_type == DeviceType.CPU

    def test_capabilities_with_provided(self):
        caps = _make_cpu_caps(name="ProvCPU")
        dev = Device(DeviceType.CPU, 0, caps)
        assert dev.capabilities.name == "ProvCPU"

    def test_initialize(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        assert dev._initialized is True
        assert dev._memory_pool is not None
        assert dev._default_stream is not None
        assert len(dev._stream_pool) >= 4

    def test_initialize_idempotent(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        pool1 = dev._memory_pool
        dev.initialize()
        assert dev._memory_pool is pool1

    def test_synchronize(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        dev.synchronize()  # Should not raise

    def test_synchronize_no_stream(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.synchronize()  # No default stream, should be fine

    def test_memory_pool_property(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        pool = dev.memory_pool
        assert pool is not None

    def test_default_stream_property(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        stream = dev.default_stream
        assert stream is not None
        assert stream.is_default is True

    def test_get_stream(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = dev.get_stream()
        assert s._in_use is True

    def test_get_stream_creates_new_when_all_in_use(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        # Mark all existing as in use
        for s in dev._stream_pool:
            s._in_use = True
        initial_count = len(dev._stream_pool)
        new_s = dev.get_stream()
        assert new_s._in_use is True
        assert len(dev._stream_pool) == initial_count + 1

    def test_return_stream(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = dev.get_stream()
        assert s._in_use is True
        dev.return_stream(s)
        assert s._in_use is False

    def test_context_manager(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        old = Device._current
        with dev:
            assert Device._current is dev
        assert Device._current is old

    def test_repr(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        r = repr(dev)
        assert "CPU" in r
        assert "id=0" in r


# ============================================================================
# MemoryBlock
# ============================================================================


class TestMemoryBlock:
    def test_touch(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        blk = MemoryBlock(ptr=0, size=64, memory_type=MemoryType.HOST, device=dev)
        old_time = blk.last_accessed
        time.sleep(0.01)
        blk.touch()
        assert blk.last_accessed >= old_time

    def test_add_ref(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        blk = MemoryBlock(ptr=0, size=64, memory_type=MemoryType.HOST, device=dev)
        assert blk.ref_count == 1
        blk.add_ref()
        assert blk.ref_count == 2

    def test_release(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        blk = MemoryBlock(ptr=0, size=64, memory_type=MemoryType.HOST, device=dev)
        freed = blk.release()
        assert freed is True
        assert blk.is_free is True

    def test_release_not_freed(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        blk = MemoryBlock(ptr=0, size=64, memory_type=MemoryType.HOST, device=dev, ref_count=2)
        freed = blk.release()
        assert freed is False
        assert blk.is_free is False


# ============================================================================
# MemoryPool
# ============================================================================


class TestMemoryPool:
    def _make_pool(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        return dev.memory_pool

    def test_allocate(self):
        pool = self._make_pool()
        blk = pool.allocate(128, MemoryType.DEVICE)
        assert blk is not None
        assert blk.size >= 128

    def test_allocate_host(self):
        pool = self._make_pool()
        blk = pool.allocate(128, MemoryType.HOST)
        assert blk is not None

    def test_allocate_pinned(self):
        pool = self._make_pool()
        blk = pool.allocate(128, MemoryType.PINNED)
        assert blk is not None

    def test_allocate_zero_fill(self):
        pool = self._make_pool()
        blk = pool.allocate(128, MemoryType.DEVICE, zero_fill=True)
        assert blk is not None

    def test_free_and_reuse(self):
        pool = self._make_pool()
        blk = pool.allocate(64, MemoryType.DEVICE)
        pool.free(blk)
        # Next alloc should hit cache
        blk2 = pool.allocate(64, MemoryType.DEVICE)
        assert blk2 is not None

    def test_current_usage(self):
        pool = self._make_pool()
        initial = pool.current_usage
        blk = pool.allocate(256, MemoryType.DEVICE)
        assert pool.current_usage > initial

    def test_get_stats(self):
        pool = self._make_pool()
        pool.allocate(128, MemoryType.DEVICE)
        stats = pool.get_stats()
        assert "total_allocated" in stats
        assert "total_freed" in stats
        assert "current_usage" in stats
        assert "peak_usage" in stats
        assert "allocation_count" in stats
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "free_blocks" in stats
        assert "large_free_blocks" in stats

    def test_defragment(self):
        pool = self._make_pool()
        b1 = pool.allocate(64, MemoryType.DEVICE)
        b2 = pool.allocate(64, MemoryType.DEVICE)
        pool.free(b1)
        pool.free(b2)
        reclaimed = pool.defragment()
        assert isinstance(reclaimed, int)

    def test_trim(self):
        pool = self._make_pool()
        b1 = pool.allocate(64, MemoryType.DEVICE)
        pool.free(b1)
        released = pool.trim()
        assert isinstance(released, int)

    def test_trim_with_target(self):
        pool = self._make_pool()
        b1 = pool.allocate(64, MemoryType.DEVICE)
        pool.free(b1)
        released = pool.trim(target_size=0)
        assert isinstance(released, int)

    def test_return_to_pool_large_block(self):
        pool = self._make_pool()
        # Allocate something larger than any SIZE_CLASS
        big = pool.allocate(2 * 1024 * 1024, MemoryType.DEVICE)
        pool.free(big)
        assert len(pool._large_free_blocks) >= 1

    def test_find_free_block_from_large(self):
        pool = self._make_pool()
        big = pool.allocate(2 * 1024 * 1024, MemoryType.DEVICE)
        pool.free(big)
        # Allocate same size should hit large block cache
        big2 = pool.allocate(2 * 1024 * 1024, MemoryType.DEVICE)
        assert big2 is not None


# ============================================================================
# Event
# ============================================================================


class TestEvent:
    def test_record(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        ev = Event(dev)
        ev.record(dev.default_stream)
        assert ev._recorded is True
        assert ev._timestamp is not None

    def test_record_default_stream(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        ev = Event(dev)
        ev.record()
        assert ev._recorded is True

    def test_query_not_completed(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        ev = Event(dev)
        assert ev.query() is False

    def test_query_completed(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        ev = Event(dev)
        ev._completed.set()
        assert ev.query() is True

    def test_elapsed_time(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        e1 = Event(dev)
        e2 = Event(dev)
        e1.record()
        time.sleep(0.01)
        e2.record()
        elapsed = e1.elapsed_time(e2)
        assert elapsed > 0

    def test_elapsed_time_not_recorded(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        e1 = Event(dev)
        e2 = Event(dev)
        with pytest.raises(RuntimeError, match="recorded"):
            e1.elapsed_time(e2)


# ============================================================================
# Stream
# ============================================================================


class TestStream:
    def test_creation(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = Stream(dev, priority=StreamPriority.HIGH)
        assert s.device is dev
        assert s.priority == StreamPriority.HIGH
        s._running = False

    def test_enqueue(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = Stream(dev)
        future = s.enqueue(lambda: 42)
        result = future.result(timeout=5)
        assert result == 42
        s._running = False

    def test_synchronize(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = Stream(dev)
        s.enqueue(lambda: None)
        s.synchronize()
        s._running = False

    def test_record_event(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = Stream(dev)
        ev = s.record_event()
        assert ev._recorded is True
        s._running = False

    def test_record_event_with_existing(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = Stream(dev)
        ev = Event(dev)
        returned = s.record_event(ev)
        assert returned is ev
        assert ev._recorded is True
        s._running = False

    def test_context_manager(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = Stream(dev)
        with s:
            pass
        s._running = False

    def test_context_manager_non_default_returns_to_pool(self):
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        dev.initialize()
        s = dev.get_stream()
        assert s._in_use is True
        with s:
            pass
        assert s._in_use is False
        s._running = False


class TestStreamPriority:
    def test_values(self):
        assert StreamPriority.LOW.value == 0
        assert StreamPriority.NORMAL.value == 1
        assert StreamPriority.HIGH.value == 2


class TestStreamCommand:
    def test_creation(self):
        cmd = StreamCommand(func=lambda: None, args=(), kwargs={})
        assert cmd.event is None
        assert cmd.priority == 0
        assert cmd.dependencies == []


# ============================================================================
# OptimizationLevel & CompilationOptions
# ============================================================================


class TestOptimizationLevel:
    def test_levels(self):
        assert OptimizationLevel.O0.value == 0
        assert OptimizationLevel.O1.value == 1
        assert OptimizationLevel.O2.value == 2
        assert OptimizationLevel.O3.value == 3
        assert OptimizationLevel.Os.value == 4
        assert OptimizationLevel.Ofast.value == 5


class TestCompilationOptions:
    def test_defaults(self):
        opts = CompilationOptions()
        assert opts.optimization_level == OptimizationLevel.O2
        assert opts.debug_info is False
        assert opts.fast_math is False
        assert opts.fuse_operations is True
        assert opts.vectorize is True
        assert opts.parallelize is True
        assert opts.target_device is None
        assert opts.cache_compiled is True
        assert opts.max_registers is None
        assert opts.preferred_shared_memory is None


# ============================================================================
# JITCache
# ============================================================================


class TestJITCache:
    def test_compute_key(self):
        with tempfile.TemporaryDirectory() as td:
            cache = JITCache(cache_dir=Path(td), max_size=10)
            opts = CompilationOptions()

            def my_func(x):
                return x + 1

            key = cache._compute_key(my_func, opts, (int,))
            assert isinstance(key, str)
            assert len(key) == 16

    def test_get_miss(self):
        with tempfile.TemporaryDirectory() as td:
            cache = JITCache(cache_dir=Path(td), max_size=10)
            opts = CompilationOptions()

            def f(x):
                return x

            result = cache.get(f, opts, (int,))
            assert result is None

    def test_put_and_get(self):
        with tempfile.TemporaryDirectory() as td:
            cache = JITCache(cache_dir=Path(td), max_size=10)
            opts = CompilationOptions()

            def f(x):
                return x * 2

            cache.put(f, opts, (int,), f)
            result = cache.get(f, opts, (int,))
            assert result is f

    def test_eviction(self):
        with tempfile.TemporaryDirectory() as td:
            cache = JITCache(cache_dir=Path(td), max_size=2)
            opts = CompilationOptions()

            def f1(x):
                return x + 1

            def f2(x):
                return x + 2

            def f3(x):
                return x + 3

            cache.put(f1, opts, (int,), f1)
            cache.put(f2, opts, (int,), f2)
            cache.put(f3, opts, (int,), f3)
            # f1 should have been evicted
            assert len(cache._memory_cache) <= 2

    def test_clear(self):
        with tempfile.TemporaryDirectory() as td:
            cache = JITCache(cache_dir=Path(td), max_size=10)
            opts = CompilationOptions()

            def f(x):
                return x

            cache.put(f, opts, (int,), f)
            cache.clear()
            assert len(cache._memory_cache) == 0
            assert len(cache._access_order) == 0

    def test_disk_cache_read(self):
        with tempfile.TemporaryDirectory() as td:
            cache = JITCache(cache_dir=Path(td), max_size=10)
            opts = CompilationOptions()

            def f(x):
                return x

            # Use a picklable compiled object (dict) so disk caching works
            compiled_obj = {"compiled": True, "version": 1}
            cache.put(f, opts, (int,), compiled_obj)
            # Clear memory cache to force disk read
            cache._memory_cache.clear()
            cache._access_order.clear()
            result = cache.get(f, opts, (int,))
            assert result is not None
            assert result["compiled"] is True


# ============================================================================
# JITCompiler
# ============================================================================


class TestJITCompiler:
    def test_get_instance(self):
        old = JITCompiler._instance
        JITCompiler._instance = None
        c = JITCompiler.get_instance()
        assert c is not None
        c2 = JITCompiler.get_instance()
        assert c is c2
        JITCompiler._instance = old

    def test_compile(self):
        compiler = JITCompiler()

        def add(a, b):
            return a + b

        compiled = compiler.compile(add)
        result = compiled(1, 2)
        assert result == 3


# ============================================================================
# jit decorator
# ============================================================================


class TestJitDecorator:
    def test_jit_bare(self):
        @jit
        def my_func(x, y):
            return x + y

        assert my_func(2, 3) == 5

    def test_jit_with_args(self):
        @jit(optimization=OptimizationLevel.O3, debug=True, cache=False)
        def my_func(x):
            return x * 2

        assert my_func(5) == 10


# ============================================================================
# ProfileEvent
# ============================================================================


class TestProfileEvent:
    def test_creation(self):
        ev = ProfileEvent(
            name="test",
            category="compute",
            start_time=0.0,
            end_time=1.0,
            duration_ms=1000.0,
            device="CPU",
            stream="default",
        )
        assert ev.name == "test"
        assert ev.category == "compute"
        assert ev.flops == 0
        assert ev.memory_throughput == 0.0


# ============================================================================
# Profiler
# ============================================================================


class TestProfiler:
    def test_context_manager(self):
        profiler = Profiler(enabled=True)
        assert Profiler.get_current() is None
        with profiler:
            assert Profiler.get_current() is profiler
        assert Profiler.get_current() is None

    def test_trace(self):
        profiler = Profiler(enabled=True)
        with profiler:
            with profiler.trace("op1", "compute"):
                pass
        assert len(profiler._events) == 1
        assert profiler._events[0].name == "op1"

    def test_trace_disabled(self):
        profiler = Profiler(enabled=False)
        with profiler:
            with profiler.trace("op1", "compute"):
                pass
        assert len(profiler._events) == 0

    def test_summary_no_events(self):
        profiler = Profiler()
        assert profiler.summary() == "No events recorded"

    def test_summary_with_events(self):
        profiler = Profiler()
        with profiler:
            with profiler.trace("op1", "compute"):
                time.sleep(0.001)
            with profiler.trace("op2", "io"):
                time.sleep(0.001)
        s = profiler.summary()
        assert "AETHELRED PROFILER SUMMARY" in s
        assert "COMPUTE" in s
        assert "IO" in s

    def test_export_chrome_trace(self):
        profiler = Profiler()
        with profiler:
            with profiler.trace("test_op"):
                pass

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            profiler.export_chrome_trace(path)
            with open(path) as f:
                data = json.load(f)
            assert "traceEvents" in data
            assert len(data["traceEvents"]) == 1
        finally:
            os.unlink(path)

    def test_export_json(self):
        profiler = Profiler()
        with profiler:
            with profiler.trace("test_op"):
                pass

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            profiler.export_json(path)
            with open(path) as f:
                data = json.load(f)
            assert "events" in data
            assert "summary" in data
        finally:
            os.unlink(path)

    def test_get_current_memory(self):
        profiler = Profiler()
        mem = profiler._get_current_memory()
        assert isinstance(mem, int)


# ============================================================================
# profile decorator
# ============================================================================


class TestProfileDecorator:
    def test_profile_without_profiler(self):
        @profile("my_op", category="compute")
        def compute(x):
            return x + 1

        result = compute(5)
        assert result == 6

    def test_profile_with_profiler(self):
        profiler = Profiler()

        @profile("my_op", category="compute")
        def compute(x):
            return x + 1

        with profiler:
            result = compute(5)

        assert result == 6
        assert len(profiler._events) == 1
        assert profiler._events[0].name == "my_op"

    def test_profile_uses_func_name(self):
        @profile(category="compute")
        def my_special_func(x):
            return x

        profiler = Profiler()
        with profiler:
            my_special_func(1)
        assert profiler._events[0].name == "my_special_func"


# ============================================================================
# Runtime
# ============================================================================


class TestRuntime:
    def test_get_instance(self):
        Runtime._instance = None
        rt = Runtime.get_instance()
        assert rt is not None
        rt2 = Runtime.get_instance()
        assert rt is rt2

    def test_initialize(self):
        Runtime._instance = None
        Runtime._initialized = False
        rt = Runtime.initialize()
        assert Runtime._initialized is True
        assert len(rt.devices) >= 1

    def test_initialize_idempotent(self):
        Runtime._instance = None
        Runtime._initialized = False
        rt1 = Runtime.initialize()
        rt2 = Runtime.initialize()
        assert rt1 is rt2

    def test_initialize_with_custom_device(self):
        Runtime._instance = None
        Runtime._initialized = False
        dev = Device(DeviceType.CPU, 0, _make_cpu_caps())
        rt = Runtime.initialize(devices=[dev], default_device=dev)
        assert rt.default_device is dev
        assert rt.devices == [dev]

    def test_initialize_with_profiling(self):
        Runtime._instance = None
        Runtime._initialized = False
        rt = Runtime.initialize(enable_profiling=True)
        assert rt._profiler is not None

    def test_shutdown(self):
        Runtime._instance = None
        Runtime._initialized = False
        rt = Runtime.initialize()
        rt.shutdown()
        assert Runtime._initialized is False

    def test_devices_property(self):
        Runtime._instance = None
        Runtime._initialized = False
        rt = Runtime.initialize()
        assert isinstance(rt.devices, list)
