"""
Aethelred Runtime Engine - The Heart of the SDK

This module provides GPU-aware runtime capabilities including:
- Just-In-Time (JIT) compilation for model optimization
- Automatic hardware detection and optimization
- Memory pool management with zero-copy transfers
- Async execution streams with dependency graphs
- Profiling and tracing infrastructure
- Multi-device orchestration
"""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import mmap
import os
import platform
import struct
import sys
import threading
import time
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache, wraps
from pathlib import Path
from queue import PriorityQueue
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Protocol, Set, Tuple, TypeVar, Union, overload, runtime_checkable
)
import struct
import weakref

# Type variables for generic programming
T = TypeVar('T')
R = TypeVar('R')


# ============================================================================
# Hardware Abstraction Layer (HAL)
# ============================================================================

class DeviceType(Enum):
    """Supported compute device types."""
    CPU = auto()
    GPU_NVIDIA = auto()
    GPU_AMD = auto()
    GPU_INTEL = auto()
    TPU = auto()
    NPU = auto()
    TEE_SGX = auto()
    TEE_SEV = auto()
    TEE_NITRO = auto()
    TEE_TRUSTZONE = auto()
    FPGA = auto()
    REMOTE = auto()


class MemoryType(Enum):
    """Memory allocation types."""
    HOST = auto()          # Regular CPU memory
    DEVICE = auto()        # Device memory (GPU VRAM, etc.)
    UNIFIED = auto()       # Unified memory (accessible from both)
    PINNED = auto()        # Pinned/page-locked host memory
    MAPPED = auto()        # Memory-mapped files
    SHARED = auto()        # Shared memory between processes
    ENCRYPTED = auto()     # TEE encrypted memory


@dataclass(frozen=True)
class DeviceCapabilities:
    """Hardware capabilities of a compute device."""
    device_type: DeviceType
    device_id: int
    name: str
    compute_capability: Tuple[int, int]  # Major, minor version

    # Memory
    total_memory: int  # bytes
    memory_bandwidth: float  # GB/s
    l1_cache_size: int
    l2_cache_size: int
    shared_memory_per_block: int

    # Compute
    multiprocessors: int
    cores_per_multiprocessor: int
    max_threads_per_block: int
    max_threads_per_multiprocessor: int
    warp_size: int
    clock_rate: int  # MHz

    # Features
    supports_fp16: bool = True
    supports_bf16: bool = False
    supports_fp64: bool = True
    supports_int8: bool = True
    supports_int4: bool = False
    supports_tensor_cores: bool = False
    supports_async_copy: bool = True
    supports_cooperative_groups: bool = False

    # TEE specific
    tee_attestation_supported: bool = False
    tee_max_enclave_size: int = 0
    tee_epc_size: int = 0

    @property
    def total_cores(self) -> int:
        return self.multiprocessors * self.cores_per_multiprocessor

    @property
    def theoretical_flops(self) -> float:
        """Theoretical peak FLOPS."""
        return self.total_cores * self.clock_rate * 1e6 * 2  # FMA = 2 ops

    def __repr__(self) -> str:
        return (
            f"Device({self.name}, "
            f"cores={self.total_cores}, "
            f"memory={self.total_memory / 1e9:.1f}GB, "
            f"compute={self.compute_capability})"
        )


class Device:
    """
    Represents a compute device with explicit backend semantics.

    Example:
        device = Device.get_default()
        with device:
            # All operations happen on this device
            result = compute(data)
    """

    _instances: Dict[Tuple[DeviceType, int], 'Device'] = {}
    _current: Optional['Device'] = None
    _lock = threading.Lock()

    def __init__(
        self,
        device_type: DeviceType,
        device_id: int = 0,
        capabilities: Optional[DeviceCapabilities] = None
    ):
        self.device_type = device_type
        self.device_id = device_id
        self._capabilities = capabilities
        self._memory_pool: Optional['MemoryPool'] = None
        self._stream_pool: List['Stream'] = []
        self._default_stream: Optional['Stream'] = None
        self._initialized = False
        self._context_stack: List['Device'] = []

    @classmethod
    def get_default(cls) -> 'Device':
        """Get the default compute device."""
        if cls._current is not None:
            return cls._current

        # Auto-detect best available device
        devices = cls.enumerate_devices()
        if not devices:
            # Fallback to CPU
            return cls.cpu()

        # Prefer GPU > TEE > CPU
        for dtype in [DeviceType.GPU_NVIDIA, DeviceType.GPU_AMD,
                      DeviceType.TEE_NITRO, DeviceType.TEE_SGX]:
            for dev in devices:
                if dev.device_type == dtype:
                    cls._current = dev
                    return dev

        return devices[0]

    @classmethod
    def cpu(cls, device_id: int = 0) -> 'Device':
        """Get a CPU device."""
        key = (DeviceType.CPU, device_id)
        if key not in cls._instances:
            cls._instances[key] = cls(DeviceType.CPU, device_id)
        return cls._instances[key]

    @classmethod
    def gpu(cls, device_id: int = 0) -> 'Device':
        """Get a GPU device."""
        devices = cls.enumerate_devices()
        gpus = [d for d in devices if 'GPU' in d.device_type.name]
        if device_id >= len(gpus):
            raise RuntimeError(f"GPU {device_id} not found. Available: {len(gpus)}")
        return gpus[device_id]

    @classmethod
    def tee(cls, platform: str = 'auto', device_id: int = 0) -> 'Device':
        """Get a TEE device."""
        platform_map = {
            'sgx': DeviceType.TEE_SGX,
            'sev': DeviceType.TEE_SEV,
            'nitro': DeviceType.TEE_NITRO,
            'trustzone': DeviceType.TEE_TRUSTZONE,
        }

        if platform == 'auto':
            # Auto-detect available TEE
            devices = cls.enumerate_devices()
            for d in devices:
                if 'TEE' in d.device_type.name:
                    return d
            raise RuntimeError("No TEE device available")

        dtype = platform_map.get(platform.lower())
        if dtype is None:
            raise ValueError(f"Unknown TEE platform: {platform}")

        key = (dtype, device_id)
        if key not in cls._instances:
            cls._instances[key] = cls(dtype, device_id)
        return cls._instances[key]

    @classmethod
    def enumerate_devices(cls) -> List['Device']:
        """Enumerate all available compute devices."""
        devices = []

        # Detect CPUs
        cpu_caps = cls._detect_cpu_capabilities()
        devices.append(cls(DeviceType.CPU, 0, cpu_caps))

        # Detect NVIDIA GPUs
        try:
            nvidia_devices = cls._detect_nvidia_gpus()
            devices.extend(nvidia_devices)
        except Exception:
            pass

        # Detect AMD GPUs
        try:
            amd_devices = cls._detect_amd_gpus()
            devices.extend(amd_devices)
        except Exception:
            pass

        # Detect TEE environments
        try:
            tee_devices = cls._detect_tee_environments()
            devices.extend(tee_devices)
        except Exception:
            pass

        return devices

    @staticmethod
    def _detect_cpu_capabilities() -> DeviceCapabilities:
        """Detect CPU capabilities."""
        import multiprocessing

        return DeviceCapabilities(
            device_type=DeviceType.CPU,
            device_id=0,
            name=platform.processor() or "CPU",
            compute_capability=(1, 0),
            total_memory=os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                if hasattr(os, 'sysconf') else 8 * 1024**3,
            memory_bandwidth=50.0,  # Approximate
            l1_cache_size=32 * 1024,
            l2_cache_size=256 * 1024,
            shared_memory_per_block=0,
            multiprocessors=1,
            cores_per_multiprocessor=multiprocessing.cpu_count(),
            max_threads_per_block=1,
            max_threads_per_multiprocessor=multiprocessing.cpu_count(),
            warp_size=1,
            clock_rate=3000,  # 3 GHz approximate
            supports_fp16=True,
            supports_bf16=True,
            supports_fp64=True,
            supports_int8=True,
            supports_tensor_cores=False,
        )

    @staticmethod
    def _detect_nvidia_gpus() -> List['Device']:
        """Detect NVIDIA GPUs using NVML."""
        devices = []
        # In production, this would use pynvml
        # For now, return empty list
        return devices

    @staticmethod
    def _detect_amd_gpus() -> List['Device']:
        """Detect AMD GPUs using ROCm."""
        devices = []
        # In production, this would use rocm-smi
        return devices

    @staticmethod
    def _detect_tee_environments() -> List['Device']:
        """Detect available TEE environments."""
        devices = []

        # Check for AWS Nitro Enclaves
        if os.path.exists('/dev/nitro_enclaves'):
            caps = DeviceCapabilities(
                device_type=DeviceType.TEE_NITRO,
                device_id=0,
                name="AWS Nitro Enclave",
                compute_capability=(1, 0),
                total_memory=8 * 1024**3,  # Configurable
                memory_bandwidth=10.0,
                l1_cache_size=0,
                l2_cache_size=0,
                shared_memory_per_block=0,
                multiprocessors=1,
                cores_per_multiprocessor=4,
                max_threads_per_block=1,
                max_threads_per_multiprocessor=4,
                warp_size=1,
                clock_rate=2500,
                tee_attestation_supported=True,
                tee_max_enclave_size=8 * 1024**3,
            )
            devices.append(Device(DeviceType.TEE_NITRO, 0, caps))

        # Check for Intel SGX
        if os.path.exists('/dev/sgx_enclave') or os.path.exists('/dev/isgx'):
            caps = DeviceCapabilities(
                device_type=DeviceType.TEE_SGX,
                device_id=0,
                name="Intel SGX Enclave",
                compute_capability=(2, 0),
                total_memory=128 * 1024**2,  # EPC size
                memory_bandwidth=5.0,
                l1_cache_size=0,
                l2_cache_size=0,
                shared_memory_per_block=0,
                multiprocessors=1,
                cores_per_multiprocessor=1,
                max_threads_per_block=1,
                max_threads_per_multiprocessor=1,
                warp_size=1,
                clock_rate=3000,
                tee_attestation_supported=True,
                tee_max_enclave_size=128 * 1024**2,
                tee_epc_size=128 * 1024**2,
            )
            devices.append(Device(DeviceType.TEE_SGX, 0, caps))

        return devices

    @property
    def capabilities(self) -> DeviceCapabilities:
        """Get device capabilities."""
        if self._capabilities is None:
            self._capabilities = self._detect_cpu_capabilities()
        return self._capabilities

    def initialize(self) -> None:
        """Initialize the device context."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            # Create memory pool
            self._memory_pool = MemoryPool(self)

            # Create default stream
            self._default_stream = Stream(self, is_default=True)

            # Pre-allocate stream pool
            for _ in range(4):
                self._stream_pool.append(Stream(self))

            self._initialized = True

    def synchronize(self) -> None:
        """Synchronize all streams on this device."""
        if self._default_stream:
            self._default_stream.synchronize()
        for stream in self._stream_pool:
            stream.synchronize()

    @property
    def memory_pool(self) -> 'MemoryPool':
        """Get the device's memory pool."""
        if not self._initialized:
            self.initialize()
        return self._memory_pool

    @property
    def default_stream(self) -> 'Stream':
        """Get the default execution stream."""
        if not self._initialized:
            self.initialize()
        return self._default_stream

    def get_stream(self) -> 'Stream':
        """Get a stream from the pool or create a new one."""
        if not self._initialized:
            self.initialize()

        # Try to get from pool
        for stream in self._stream_pool:
            if not stream._in_use:
                stream._in_use = True
                return stream

        # Create new stream
        stream = Stream(self)
        stream._in_use = True
        self._stream_pool.append(stream)
        return stream

    def return_stream(self, stream: 'Stream') -> None:
        """Return a stream to the pool."""
        stream._in_use = False

    def __enter__(self) -> 'Device':
        """Enter device context."""
        self._context_stack.append(Device._current)
        Device._current = self
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, *args) -> None:
        """Exit device context."""
        if self._context_stack:
            Device._current = self._context_stack.pop()

    def __repr__(self) -> str:
        return f"Device({self.device_type.name}, id={self.device_id})"


# ============================================================================
# Memory Management
# ============================================================================

@dataclass
class MemoryBlock:
    """A block of allocated memory."""
    ptr: int  # Memory address
    size: int  # Size in bytes
    memory_type: MemoryType
    device: Device
    is_free: bool = False
    ref_count: int = 1

    # Metadata
    alignment: int = 64  # Cache line alignment
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()

    def add_ref(self) -> None:
        """Increment reference count."""
        self.ref_count += 1

    def release(self) -> bool:
        """Decrement reference count. Returns True if freed."""
        self.ref_count -= 1
        if self.ref_count <= 0:
            self.is_free = True
            return True
        return False


class MemoryPool:
    """
    High-performance memory pool with sub-allocator.

    Features:
    - Coalescing free blocks
    - Size-class based allocation
    - Async memory operations
    - Memory defragmentation
    - Usage statistics and profiling
    """

    # Size classes for efficient allocation (powers of 2)
    SIZE_CLASSES = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                    32768, 65536, 131072, 262144, 524288, 1048576]

    def __init__(
        self,
        device: Device,
        initial_size: int = 256 * 1024 * 1024,  # 256 MB
        growth_factor: float = 1.5,
        max_size: Optional[int] = None
    ):
        self.device = device
        self.initial_size = initial_size
        self.growth_factor = growth_factor
        self.max_size = max_size or device.capabilities.total_memory

        self._lock = threading.RLock()
        self._blocks: Dict[int, MemoryBlock] = {}
        self._free_lists: Dict[int, List[MemoryBlock]] = {
            size: [] for size in self.SIZE_CLASSES
        }
        self._large_free_blocks: List[MemoryBlock] = []

        # Statistics
        self._total_allocated = 0
        self._total_freed = 0
        self._peak_usage = 0
        self._allocation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def allocate(
        self,
        size: int,
        memory_type: MemoryType = MemoryType.DEVICE,
        alignment: int = 64,
        zero_fill: bool = False
    ) -> MemoryBlock:
        """
        Allocate a memory block.

        Args:
            size: Size in bytes
            memory_type: Type of memory to allocate
            alignment: Memory alignment requirement
            zero_fill: Whether to zero-initialize the memory

        Returns:
            Allocated memory block
        """
        with self._lock:
            self._allocation_count += 1

            # Round up to alignment
            aligned_size = (size + alignment - 1) & ~(alignment - 1)

            # Try to find in free list
            block = self._find_free_block(aligned_size)

            if block is not None:
                self._cache_hits += 1
                block.is_free = False
                block.ref_count = 1
                block.touch()
            else:
                self._cache_misses += 1
                # Allocate new block
                block = self._allocate_new(aligned_size, memory_type, alignment)

            if zero_fill:
                self._zero_memory(block)

            self._total_allocated += block.size
            self._peak_usage = max(self._peak_usage, self.current_usage)

            return block

    def _find_free_block(self, size: int) -> Optional[MemoryBlock]:
        """Find a suitable free block."""
        # Check size-class free lists
        for size_class in self.SIZE_CLASSES:
            if size_class >= size:
                free_list = self._free_lists[size_class]
                if free_list:
                    return free_list.pop()

        # Check large blocks
        for i, block in enumerate(self._large_free_blocks):
            if block.size >= size:
                return self._large_free_blocks.pop(i)

        return None

    def _allocate_new(
        self,
        size: int,
        memory_type: MemoryType,
        alignment: int
    ) -> MemoryBlock:
        """Allocate a new memory block from the system."""
        # In production, this would use device-specific allocation
        # For now, use ctypes for demonstration

        if memory_type == MemoryType.HOST:
            # Aligned allocation
            buffer = (ctypes.c_char * (size + alignment))()
            ptr = ctypes.addressof(buffer)
            aligned_ptr = (ptr + alignment - 1) & ~(alignment - 1)
        elif memory_type == MemoryType.PINNED:
            # Page-locked memory (mlock)
            buffer = mmap.mmap(-1, size)
            ptr = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            aligned_ptr = ptr
        else:
            # Default allocation
            buffer = (ctypes.c_char * size)()
            ptr = ctypes.addressof(buffer)
            aligned_ptr = ptr

        block = MemoryBlock(
            ptr=aligned_ptr,
            size=size,
            memory_type=memory_type,
            device=self.device,
            alignment=alignment
        )

        self._blocks[aligned_ptr] = block
        return block

    def free(self, block: MemoryBlock) -> None:
        """Free a memory block (returns to pool)."""
        with self._lock:
            if block.release():
                self._total_freed += block.size
                self._return_to_pool(block)

    def _return_to_pool(self, block: MemoryBlock) -> None:
        """Return a block to the appropriate free list."""
        # Find appropriate size class
        for size_class in self.SIZE_CLASSES:
            if block.size <= size_class:
                self._free_lists[size_class].append(block)
                return

        # Large block
        self._large_free_blocks.append(block)

    def _zero_memory(self, block: MemoryBlock) -> None:
        """Zero-initialize memory."""
        ctypes.memset(block.ptr, 0, block.size)

    @property
    def current_usage(self) -> int:
        """Current memory usage in bytes."""
        return self._total_allocated - self._total_freed

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        return {
            'total_allocated': self._total_allocated,
            'total_freed': self._total_freed,
            'current_usage': self.current_usage,
            'peak_usage': self._peak_usage,
            'allocation_count': self._allocation_count,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': self._cache_hits / max(1, self._allocation_count),
            'free_blocks': sum(len(fl) for fl in self._free_lists.values()),
            'large_free_blocks': len(self._large_free_blocks),
        }

    def defragment(self) -> int:
        """
        Defragment the memory pool.

        Returns:
            Number of bytes reclaimed
        """
        with self._lock:
            reclaimed = 0

            # Coalesce adjacent free blocks
            # Sort by address
            all_free = []
            for free_list in self._free_lists.values():
                all_free.extend(free_list)
            all_free.extend(self._large_free_blocks)
            all_free.sort(key=lambda b: b.ptr)

            # Coalesce
            coalesced = []
            for block in all_free:
                if coalesced and coalesced[-1].ptr + coalesced[-1].size == block.ptr:
                    # Merge with previous
                    coalesced[-1] = MemoryBlock(
                        ptr=coalesced[-1].ptr,
                        size=coalesced[-1].size + block.size,
                        memory_type=block.memory_type,
                        device=block.device,
                        is_free=True
                    )
                    reclaimed += block.size  # Overhead reduction
                else:
                    coalesced.append(block)

            # Rebuild free lists
            for size_class in self.SIZE_CLASSES:
                self._free_lists[size_class] = []
            self._large_free_blocks = []

            for block in coalesced:
                self._return_to_pool(block)

            return reclaimed

    def trim(self, target_size: Optional[int] = None) -> int:
        """
        Release unused memory back to the system.

        Args:
            target_size: Target pool size (None = minimum)

        Returns:
            Bytes released
        """
        with self._lock:
            released = 0
            target = target_size or self.current_usage

            # Release from large blocks first
            while self._large_free_blocks and self.current_usage > target:
                block = self._large_free_blocks.pop()
                released += block.size
                del self._blocks[block.ptr]

            # Release from size-class pools
            for size_class in reversed(self.SIZE_CLASSES):
                while self._free_lists[size_class] and self.current_usage > target:
                    block = self._free_lists[size_class].pop()
                    released += block.size
                    del self._blocks[block.ptr]

            return released


# ============================================================================
# Execution Streams
# ============================================================================

class Event:
    """
    Synchronization event for streams.

    Like CUDA events, these can be used to:
    - Measure elapsed time between operations
    - Synchronize between streams
    - Create dependencies in the execution graph
    """

    def __init__(self, device: Optional[Device] = None):
        self.device = device or Device.get_default()
        self._timestamp: Optional[float] = None
        self._recorded = False
        self._completed = threading.Event()

    def record(self, stream: Optional['Stream'] = None) -> None:
        """Record the event in a stream."""
        stream = stream or self.device.default_stream
        self._timestamp = time.perf_counter()
        self._recorded = True
        # In async implementation, this would be added to stream's command queue

    def synchronize(self) -> None:
        """Wait for the event to complete."""
        self._completed.wait()

    def query(self) -> bool:
        """Check if the event has completed."""
        return self._completed.is_set()

    def elapsed_time(self, end_event: 'Event') -> float:
        """
        Calculate elapsed time in milliseconds.

        Args:
            end_event: The ending event

        Returns:
            Elapsed time in milliseconds
        """
        if not self._recorded or not end_event._recorded:
            raise RuntimeError("Events must be recorded first")

        return (end_event._timestamp - self._timestamp) * 1000


class StreamPriority(Enum):
    """Stream execution priority."""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class StreamCommand:
    """A command in the stream's execution queue."""
    func: Callable[[], Any]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    event: Optional[Event] = None
    priority: int = 0
    dependencies: List['StreamCommand'] = field(default_factory=list)
    result: Optional[Future] = None


class Stream:
    """
    Execution stream for async operations.

    Like CUDA streams, operations within a stream execute in order,
    while operations in different streams may execute concurrently.

    Example:
        stream = device.get_stream()
        with stream:
            result = compute_async(data)
        stream.synchronize()
    """

    def __init__(
        self,
        device: Optional[Device] = None,
        priority: StreamPriority = StreamPriority.NORMAL,
        is_default: bool = False
    ):
        self.device = device or Device.get_default()
        self.priority = priority
        self.is_default = is_default

        self._in_use = False
        self._command_queue: PriorityQueue[Tuple[int, StreamCommand]] = PriorityQueue()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._pending_futures: List[Future] = []
        self._event_counter = 0
        self._lock = threading.Lock()

        # Start worker thread
        self._running = True
        self._worker = threading.Thread(target=self._process_commands, daemon=True)
        self._worker.start()

    def _process_commands(self) -> None:
        """Process commands in the queue."""
        while self._running:
            try:
                _, cmd = self._command_queue.get(timeout=0.1)
            except:
                continue

            # Wait for dependencies
            for dep in cmd.dependencies:
                if dep.result:
                    dep.result.result()

            # Execute command
            try:
                result = cmd.func(*cmd.args, **cmd.kwargs)
                if cmd.result:
                    cmd.result.set_result(result)
            except Exception as e:
                if cmd.result:
                    cmd.result.set_exception(e)

            # Signal event if present
            if cmd.event:
                cmd.event._completed.set()

    def enqueue(
        self,
        func: Callable[..., R],
        *args,
        event: Optional[Event] = None,
        dependencies: Optional[List[StreamCommand]] = None,
        **kwargs
    ) -> Future[R]:
        """
        Enqueue a function for execution.

        Args:
            func: Function to execute
            *args: Function arguments
            event: Optional event to signal on completion
            dependencies: Commands that must complete first
            **kwargs: Function keyword arguments

        Returns:
            Future representing the pending result
        """
        future: Future[R] = Future()

        cmd = StreamCommand(
            func=func,
            args=args,
            kwargs=kwargs,
            event=event,
            priority=-self.priority.value,  # Negative for priority queue
            dependencies=dependencies or [],
            result=future
        )

        with self._lock:
            self._event_counter += 1
            self._command_queue.put((self._event_counter, cmd))
            self._pending_futures.append(future)

        return future

    def synchronize(self) -> None:
        """Wait for all pending operations to complete."""
        futures = list(self._pending_futures)
        for future in futures:
            try:
                future.result(timeout=60)
            except Exception:
                pass
        self._pending_futures.clear()

    def record_event(self, event: Optional[Event] = None) -> Event:
        """Record an event in the stream."""
        if event is None:
            event = Event(self.device)
        event.record(self)
        return event

    def wait_event(self, event: Event) -> None:
        """Make stream wait for an event."""
        # Enqueue a wait operation
        self.enqueue(event.synchronize)

    def __enter__(self) -> 'Stream':
        """Enter stream context."""
        return self

    def __exit__(self, *args) -> None:
        """Exit stream context."""
        if not self.is_default:
            self.device.return_stream(self)

    def __del__(self):
        """Cleanup stream resources."""
        self._running = False


# ============================================================================
# JIT Compilation
# ============================================================================

class OptimizationLevel(Enum):
    """Optimization levels for JIT compilation."""
    O0 = 0  # No optimization (debug)
    O1 = 1  # Basic optimizations
    O2 = 2  # Standard optimizations
    O3 = 3  # Aggressive optimizations
    Os = 4  # Size optimization
    Ofast = 5  # Maximum speed (may break IEEE compliance)


@dataclass
class CompilationOptions:
    """Options for JIT compilation."""
    optimization_level: OptimizationLevel = OptimizationLevel.O2
    debug_info: bool = False
    fast_math: bool = False
    fuse_operations: bool = True
    vectorize: bool = True
    parallelize: bool = True
    target_device: Optional[Device] = None
    cache_compiled: bool = True
    max_registers: Optional[int] = None
    preferred_shared_memory: Optional[int] = None


class JITCache:
    """
    Cache for JIT-compiled functions.

    Features:
    - Persistent disk cache
    - LRU eviction
    - Version-aware invalidation
    """

    def __init__(self, cache_dir: Optional[Path] = None, max_size: int = 1024):
        self.cache_dir = cache_dir or Path.home() / '.aethelred' / 'jit_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size

        self._memory_cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()

    def _compute_key(
        self,
        func: Callable,
        options: CompilationOptions,
        arg_types: Tuple[type, ...]
    ) -> str:
        """Compute cache key for a function."""
        # Hash function bytecode
        import dis
        func_code = func.__code__
        bytecode = dis.Bytecode(func_code).dis()

        # Include options and types
        key_parts = [
            bytecode,
            str(options),
            str(arg_types),
            str(options.target_device),
        ]

        return hashlib.sha256('|'.join(key_parts).encode()).hexdigest()[:16]

    def get(
        self,
        func: Callable,
        options: CompilationOptions,
        arg_types: Tuple[type, ...]
    ) -> Optional[Any]:
        """Get cached compiled function."""
        key = self._compute_key(func, options, arg_types)

        with self._lock:
            if key in self._memory_cache:
                # Move to end (LRU)
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            try:
                import pickle
                with open(cache_file, 'rb') as f:
                    compiled = pickle.load(f)

                with self._lock:
                    self._memory_cache[key] = compiled
                    self._access_order.append(key)
                    self._evict_if_needed()

                return compiled
            except Exception:
                cache_file.unlink()

        return None

    def put(
        self,
        func: Callable,
        options: CompilationOptions,
        arg_types: Tuple[type, ...],
        compiled: Any
    ) -> None:
        """Cache a compiled function."""
        key = self._compute_key(func, options, arg_types)

        with self._lock:
            self._memory_cache[key] = compiled
            self._access_order.append(key)
            self._evict_if_needed()

        # Write to disk
        if options.cache_compiled:
            import pickle
            cache_file = self.cache_dir / f"{key}.cache"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(compiled, f)
            except Exception:
                pass

    def _evict_if_needed(self) -> None:
        """Evict old entries if cache is full."""
        while len(self._memory_cache) > self.max_size:
            oldest = self._access_order.pop(0)
            del self._memory_cache[oldest]

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._memory_cache.clear()
            self._access_order.clear()

        # Clear disk cache
        for f in self.cache_dir.glob("*.cache"):
            f.unlink()


class JITCompiler:
    """
    Just-In-Time compiler for Aethelred operations.

    Features:
    - Automatic type specialization
    - Kernel fusion
    - Device-specific optimization
    - Caching
    """

    _instance: Optional['JITCompiler'] = None
    _cache = JITCache()

    @classmethod
    def get_instance(cls) -> 'JITCompiler':
        """Get the singleton compiler instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def compile(
        self,
        func: Callable[..., R],
        options: Optional[CompilationOptions] = None
    ) -> Callable[..., R]:
        """
        Compile a function for optimal execution.

        Args:
            func: Function to compile
            options: Compilation options

        Returns:
            Compiled function
        """
        options = options or CompilationOptions()

        @wraps(func)
        def compiled_wrapper(*args, **kwargs):
            # Get argument types
            arg_types = tuple(type(a) for a in args)

            # Check cache
            cached = self._cache.get(func, options, arg_types)
            if cached is not None:
                return cached(*args, **kwargs)

            # Compile
            compiled = self._compile_specialized(func, options, arg_types)

            # Cache
            self._cache.put(func, options, arg_types, compiled)

            return compiled(*args, **kwargs)

        return compiled_wrapper

    def _compile_specialized(
        self,
        func: Callable,
        options: CompilationOptions,
        arg_types: Tuple[type, ...]
    ) -> Callable:
        """Compile a type-specialized version of the function."""
        # In a real implementation, this would:
        # 1. Analyze the function's AST
        # 2. Apply type-specific optimizations
        # 3. Generate optimized code (possibly native)
        # 4. Return the optimized function

        # For now, return the original function with profiling wrapper
        @wraps(func)
        def optimized(*args, **kwargs):
            return func(*args, **kwargs)

        return optimized


def jit(
    func: Optional[Callable[..., R]] = None,
    *,
    optimization: OptimizationLevel = OptimizationLevel.O2,
    debug: bool = False,
    cache: bool = True,
    device: Optional[Device] = None
) -> Union[Callable[[Callable[..., R]], Callable[..., R]], Callable[..., R]]:
    """
    Decorator for JIT compilation.

    Example:
        @jit
        def compute(x, y):
            return x + y

        @jit(optimization=OptimizationLevel.O3, device=Device.gpu())
        def gpu_compute(x, y):
            return x * y
    """
    options = CompilationOptions(
        optimization_level=optimization,
        debug_info=debug,
        cache_compiled=cache,
        target_device=device
    )

    compiler = JITCompiler.get_instance()

    def decorator(f: Callable[..., R]) -> Callable[..., R]:
        return compiler.compile(f, options)

    if func is not None:
        return decorator(func)
    return decorator


# ============================================================================
# Profiling Infrastructure
# ============================================================================

@dataclass
class ProfileEvent:
    """A single profiling event."""
    name: str
    category: str
    start_time: float
    end_time: float
    duration_ms: float
    device: str
    stream: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Memory metrics
    memory_allocated: int = 0
    memory_freed: int = 0
    peak_memory: int = 0

    # Compute metrics
    flops: int = 0
    memory_throughput: float = 0.0


class Profiler:
    """
    Comprehensive profiling system.

    Features:
    - Timeline tracing
    - Memory profiling
    - Kernel profiling
    - Export to Chrome Trace format

    Example:
        with Profiler() as profiler:
            result = compute(data)

        profiler.export_chrome_trace('trace.json')
        print(profiler.summary())
    """

    _current: Optional['Profiler'] = None

    def __init__(
        self,
        enabled: bool = True,
        record_memory: bool = True,
        record_kernels: bool = True,
        record_api_calls: bool = True
    ):
        self.enabled = enabled
        self.record_memory = record_memory
        self.record_kernels = record_kernels
        self.record_api_calls = record_api_calls

        self._events: List[ProfileEvent] = []
        self._active_events: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._start_time = 0.0

    def __enter__(self) -> 'Profiler':
        """Enter profiling context."""
        Profiler._current = self
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Exit profiling context."""
        Profiler._current = None

    @classmethod
    def get_current(cls) -> Optional['Profiler']:
        """Get the current active profiler."""
        return cls._current

    @contextmanager
    def trace(
        self,
        name: str,
        category: str = "default",
        **metadata
    ) -> Iterator[None]:
        """
        Trace a code block.

        Example:
            with profiler.trace("forward_pass", category="model"):
                output = model(input)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        start_memory = self._get_current_memory()

        try:
            yield
        finally:
            end = time.perf_counter()
            end_memory = self._get_current_memory()

            event = ProfileEvent(
                name=name,
                category=category,
                start_time=start - self._start_time,
                end_time=end - self._start_time,
                duration_ms=(end - start) * 1000,
                device=str(Device.get_default()),
                stream="default",
                metadata=metadata,
                memory_allocated=max(0, end_memory - start_memory),
                memory_freed=max(0, start_memory - end_memory),
            )

            with self._lock:
                self._events.append(event)

    def _get_current_memory(self) -> int:
        """Get current memory usage."""
        try:
            device = Device.get_default()
            if device._memory_pool:
                return device._memory_pool.current_usage
        except Exception:
            pass
        return 0

    def summary(self) -> str:
        """Get a summary of profiling results."""
        if not self._events:
            return "No events recorded"

        lines = ["=" * 60]
        lines.append("AETHELRED PROFILER SUMMARY")
        lines.append("=" * 60)

        # Group by category
        by_category: Dict[str, List[ProfileEvent]] = {}
        for event in self._events:
            if event.category not in by_category:
                by_category[event.category] = []
            by_category[event.category].append(event)

        total_time = sum(e.duration_ms for e in self._events)

        for category, events in sorted(by_category.items()):
            lines.append(f"\n{category.upper()}")
            lines.append("-" * 40)

            # Aggregate by name
            by_name: Dict[str, List[float]] = {}
            for e in events:
                if e.name not in by_name:
                    by_name[e.name] = []
                by_name[e.name].append(e.duration_ms)

            for name, times in sorted(by_name.items(), key=lambda x: -sum(x[1])):
                total = sum(times)
                avg = total / len(times)
                pct = (total / total_time) * 100
                lines.append(
                    f"  {name:30s} "
                    f"calls={len(times):4d}  "
                    f"total={total:8.2f}ms  "
                    f"avg={avg:8.2f}ms  "
                    f"({pct:5.1f}%)"
                )

        lines.append("\n" + "=" * 60)
        lines.append(f"Total profiled time: {total_time:.2f}ms")
        lines.append(f"Total events: {len(self._events)}")

        return "\n".join(lines)

    def export_chrome_trace(self, filepath: str) -> None:
        """Export to Chrome Trace format for visualization."""
        import json

        trace_events = []

        for event in self._events:
            trace_events.append({
                "name": event.name,
                "cat": event.category,
                "ph": "X",  # Complete event
                "ts": event.start_time * 1e6,  # microseconds
                "dur": event.duration_ms * 1e3,  # microseconds
                "pid": 1,
                "tid": 1,
                "args": event.metadata
            })

        trace = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ms"
        }

        with open(filepath, 'w') as f:
            json.dump(trace, f, indent=2)

    def export_json(self, filepath: str) -> None:
        """Export events to JSON."""
        import json
        from dataclasses import asdict

        data = {
            "events": [asdict(e) for e in self._events],
            "summary": {
                "total_events": len(self._events),
                "total_time_ms": sum(e.duration_ms for e in self._events),
                "categories": list(set(e.category for e in self._events))
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def profile(name: Optional[str] = None, category: str = "default"):
    """
    Decorator for profiling functions.

    Example:
        @profile("matrix_multiply", category="compute")
        def matmul(a, b):
            return a @ b
    """
    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        func_name = name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = Profiler.get_current()
            if profiler is None:
                return func(*args, **kwargs)

            with profiler.trace(func_name, category):
                return func(*args, **kwargs)

        return wrapper
    return decorator


# ============================================================================
# Runtime Initialization
# ============================================================================

class Runtime:
    """
    Global runtime for Aethelred SDK.

    Manages:
    - Device initialization
    - Memory pools
    - Thread pools
    - Profiling
    - Logging
    """

    _instance: Optional['Runtime'] = None
    _initialized = False

    def __init__(self):
        self._devices: List[Device] = []
        self._default_device: Optional[Device] = None
        self._profiler: Optional[Profiler] = None
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    def get_instance(cls) -> 'Runtime':
        """Get the singleton runtime instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def initialize(
        cls,
        devices: Optional[List[Device]] = None,
        default_device: Optional[Device] = None,
        enable_profiling: bool = False,
        thread_pool_size: int = 8
    ) -> 'Runtime':
        """
        Initialize the runtime.

        Should be called once at program start.
        """
        runtime = cls.get_instance()

        if cls._initialized:
            return runtime

        # Enumerate devices
        if devices is None:
            runtime._devices = Device.enumerate_devices()
        else:
            runtime._devices = devices

        # Set default device
        if default_device is not None:
            runtime._default_device = default_device
            Device._current = default_device
        elif runtime._devices:
            runtime._default_device = runtime._devices[0]
            Device._current = runtime._devices[0]

        # Initialize thread pool
        runtime._thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)

        # Initialize profiler if enabled
        if enable_profiling:
            runtime._profiler = Profiler()

        cls._initialized = True
        return runtime

    @property
    def devices(self) -> List[Device]:
        """List of available devices."""
        return self._devices

    @property
    def default_device(self) -> Optional[Device]:
        """The default compute device."""
        return self._default_device

    def shutdown(self) -> None:
        """Shutdown the runtime."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)

        for device in self._devices:
            device.synchronize()

        Runtime._initialized = False


# Initialize on import
def _auto_init():
    """Auto-initialize runtime with sensible defaults."""
    if not Runtime._initialized:
        Runtime.initialize()

# Don't auto-init, let user control initialization
# _auto_init()
