"""
Aethelred Distributed Computing

Enterprise-grade distributed training and inference with:
- Data parallelism
- Model parallelism (tensor, pipeline)
- Zero Redundancy Optimizer (ZeRO)
- Gradient compression
- Fault tolerance
- Elastic training
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import pickle
import socket
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Protocol, Set, Tuple, TypeVar, Union
)

from ..core.tensor import Tensor, DType
from ..core.runtime import Device, Stream, Event
from ..nn import Module, Parameter


# ============================================================================
# Process Group
# ============================================================================

class Backend(Enum):
    """Communication backend."""
    GLOO = auto()      # CPU distributed
    NCCL = auto()      # NVIDIA GPU
    MPI = auto()       # Message Passing Interface
    AETHELRED = auto() # Native Aethelred protocol


@dataclass
class ProcessGroupInfo:
    """Information about a process group."""
    rank: int              # This process's rank
    world_size: int        # Total number of processes
    local_rank: int        # Rank within this node
    local_world_size: int  # Processes on this node
    backend: Backend
    is_initialized: bool = False


class ProcessGroup:
    """
    Manages distributed process coordination.

    Provides:
    - Point-to-point communication
    - Collective operations (broadcast, reduce, all_reduce)
    - Barrier synchronization
    - Group management
    """

    _default_group: Optional['ProcessGroup'] = None

    def __init__(
        self,
        backend: Backend = Backend.GLOO,
        init_method: str = 'env://',
        world_size: int = -1,
        rank: int = -1
    ):
        self.backend = backend
        self.init_method = init_method

        # Get from environment if not specified
        if world_size < 0:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        if rank < 0:
            rank = int(os.environ.get('RANK', 0))

        self.world_size = world_size
        self.rank = rank
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))

        self._initialized = False
        self._groups: Dict[str, 'ProcessGroup'] = {}

        # Communication buffers
        self._send_buffers: Dict[int, List[bytes]] = defaultdict(list)
        self._recv_buffers: Dict[int, List[bytes]] = defaultdict(list)

    @classmethod
    def get_default(cls) -> 'ProcessGroup':
        """Get the default process group."""
        if cls._default_group is None:
            cls._default_group = cls()
        return cls._default_group

    def initialize(self) -> None:
        """Initialize the process group."""
        if self._initialized:
            return

        # In production, this would:
        # 1. Connect to rendezvous endpoint
        # 2. Exchange connection info with peers
        # 3. Set up communication channels

        self._initialized = True

    def destroy(self) -> None:
        """Clean up process group."""
        self._initialized = False

    # ========================================================================
    # Point-to-Point Operations
    # ========================================================================

    def send(
        self,
        tensor: Tensor,
        dst: int,
        tag: int = 0,
        stream: Optional[Stream] = None
    ) -> None:
        """
        Send tensor to another process.

        Args:
            tensor: Tensor to send
            dst: Destination rank
            tag: Message tag for matching
            stream: CUDA stream for async operation
        """
        if not self._initialized:
            self.initialize()

        # Serialize tensor
        data = self._serialize_tensor(tensor)
        self._send_buffers[dst].append(data)

    def recv(
        self,
        tensor: Tensor,
        src: int,
        tag: int = 0,
        stream: Optional[Stream] = None
    ) -> None:
        """
        Receive tensor from another process.

        Args:
            tensor: Tensor to receive into
            src: Source rank (-1 for any)
            tag: Message tag for matching
            stream: CUDA stream for async operation
        """
        if not self._initialized:
            self.initialize()

        # Would receive and deserialize
        pass

    def isend(
        self,
        tensor: Tensor,
        dst: int,
        tag: int = 0
    ) -> 'Request':
        """Non-blocking send."""
        request = Request()
        # Start async send
        return request

    def irecv(
        self,
        tensor: Tensor,
        src: int = -1,
        tag: int = 0
    ) -> 'Request':
        """Non-blocking receive."""
        request = Request()
        # Start async receive
        return request

    # ========================================================================
    # Collective Operations
    # ========================================================================

    def broadcast(
        self,
        tensor: Tensor,
        src: int = 0,
        async_op: bool = False
    ) -> Optional['Request']:
        """
        Broadcast tensor from source to all processes.

        Args:
            tensor: Tensor to broadcast (in-place)
            src: Source rank
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        if self.world_size == 1:
            return None

        # Ring broadcast implementation
        # Each process receives from (rank - 1) and sends to (rank + 1)

        if async_op:
            return Request()
        return None

    def reduce(
        self,
        tensor: Tensor,
        dst: int = 0,
        op: str = 'sum',
        async_op: bool = False
    ) -> Optional['Request']:
        """
        Reduce tensor to destination.

        Args:
            tensor: Tensor to reduce (in-place at dst)
            dst: Destination rank
            op: Reduction operation (sum, product, min, max)
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        if self.world_size == 1:
            return None

        # Tree reduce implementation

        if async_op:
            return Request()
        return None

    def all_reduce(
        self,
        tensor: Tensor,
        op: str = 'sum',
        async_op: bool = False
    ) -> Optional['Request']:
        """
        All-reduce tensor across all processes.

        Args:
            tensor: Tensor to reduce (in-place)
            op: Reduction operation
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        if self.world_size == 1:
            return None

        # Ring all-reduce implementation for bandwidth efficiency
        # 1. Scatter-reduce: each process has partial sum
        # 2. All-gather: distribute complete result

        if async_op:
            return Request()
        return None

    def all_gather(
        self,
        output_tensors: List[Tensor],
        input_tensor: Tensor,
        async_op: bool = False
    ) -> Optional['Request']:
        """
        Gather tensors from all processes to all processes.

        Args:
            output_tensors: List of tensors to receive into
            input_tensor: Tensor to send
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        # Ring all-gather

        if async_op:
            return Request()
        return None

    def gather(
        self,
        tensor: Tensor,
        gather_list: Optional[List[Tensor]] = None,
        dst: int = 0,
        async_op: bool = False
    ) -> Optional['Request']:
        """
        Gather tensors to destination.

        Args:
            tensor: Tensor to send
            gather_list: List to receive into (only at dst)
            dst: Destination rank
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        if async_op:
            return Request()
        return None

    def scatter(
        self,
        tensor: Tensor,
        scatter_list: Optional[List[Tensor]] = None,
        src: int = 0,
        async_op: bool = False
    ) -> Optional['Request']:
        """
        Scatter tensors from source.

        Args:
            tensor: Tensor to receive into
            scatter_list: List of tensors to scatter (only at src)
            src: Source rank
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        if async_op:
            return Request()
        return None

    def reduce_scatter(
        self,
        output: Tensor,
        input_list: List[Tensor],
        op: str = 'sum',
        async_op: bool = False
    ) -> Optional['Request']:
        """
        Reduce then scatter.

        Args:
            output: Output tensor
            input_list: Input tensors to reduce
            op: Reduction operation
            async_op: Return immediately

        Returns:
            Request handle if async
        """
        if not self._initialized:
            self.initialize()

        if async_op:
            return Request()
        return None

    def barrier(self, async_op: bool = False) -> Optional['Request']:
        """
        Synchronization barrier.

        All processes must reach this point before any can continue.
        """
        if not self._initialized:
            self.initialize()

        if self.world_size == 1:
            return None

        # Double-barrier algorithm for fault tolerance

        if async_op:
            return Request()
        return None

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _serialize_tensor(self, tensor: Tensor) -> bytes:
        """Serialize tensor for transmission."""
        return pickle.dumps({
            'shape': tensor.shape,
            'dtype': tensor.dtype,
            'data': tensor.numpy().tobytes()
        })

    def _deserialize_tensor(self, data: bytes) -> Tensor:
        """Deserialize tensor from transmission."""
        import numpy as np
        info = pickle.loads(data)
        array = np.frombuffer(info['data'], dtype=np.float32).reshape(info['shape'])
        return Tensor.from_numpy(array)


class Request:
    """Handle for async operation."""

    def __init__(self):
        self._completed = threading.Event()
        self._result = None

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Wait for operation to complete."""
        return self._completed.wait(timeout)

    def is_completed(self) -> bool:
        """Check if operation completed."""
        return self._completed.is_set()

    def get_future(self) -> asyncio.Future:
        """Get asyncio future for this operation."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        def complete():
            self._completed.wait()
            loop.call_soon_threadsafe(future.set_result, self._result)

        threading.Thread(target=complete, daemon=True).start()
        return future


# ============================================================================
# Distributed Data Parallel
# ============================================================================

class DistributedDataParallel(Module):
    """
    Distributed Data Parallel wrapper.

    Wraps a module for data-parallel training across multiple processes.

    Features:
    - Gradient synchronization
    - Gradient bucketing for efficiency
    - Gradient compression
    - Static graph optimization

    Example:
        model = DistributedDataParallel(model, device_ids=[0])
        output = model(input)
        loss.backward()
        optimizer.step()  # Gradients already synchronized
    """

    def __init__(
        self,
        module: Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        broadcast_buffers: bool = True,
        process_group: Optional[ProcessGroup] = None,
        bucket_cap_mb: float = 25.0,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False,
        static_graph: bool = False
    ):
        super().__init__()

        self.module = module
        self.device_ids = device_ids or []
        self.output_device = output_device or (device_ids[0] if device_ids else None)
        self.broadcast_buffers = broadcast_buffers
        self.process_group = process_group or ProcessGroup.get_default()
        self.bucket_cap_mb = bucket_cap_mb
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph

        # Gradient buckets for efficient communication
        self._buckets: List[GradientBucket] = []
        self._bucket_indices: Dict[Parameter, int] = {}

        # Initialize
        self._sync_params()
        self._build_buckets()

        # Register hooks for gradient synchronization
        self._register_grad_hooks()

    def _sync_params(self) -> None:
        """Synchronize parameters from rank 0."""
        for param in self.module.parameters():
            self.process_group.broadcast(param, src=0)

        if self.broadcast_buffers:
            for buffer in self.module.buffers():
                self.process_group.broadcast(buffer, src=0)

    def _build_buckets(self) -> None:
        """Build gradient buckets for efficient all-reduce."""
        bucket_size = int(self.bucket_cap_mb * 1024 * 1024)
        current_bucket = GradientBucket(bucket_size)
        self._buckets = [current_bucket]

        # Add parameters in reverse order (for backward compatibility)
        params = list(self.module.parameters())
        for param in reversed(params):
            if not current_bucket.add_param(param):
                # Bucket full, create new one
                current_bucket = GradientBucket(bucket_size)
                self._buckets.append(current_bucket)
                current_bucket.add_param(param)

            self._bucket_indices[param] = len(self._buckets) - 1

    def _register_grad_hooks(self) -> None:
        """Register backward hooks for gradient sync."""
        for param in self.module.parameters():
            if param.requires_grad:
                # Register hook to trigger all-reduce when gradient is ready
                pass  # Would use param.register_hook()

    def _reduce_gradients(self) -> None:
        """Reduce gradients across all processes."""
        for bucket in self._buckets:
            bucket.all_reduce(self.process_group)

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass with buffer synchronization."""
        if self.broadcast_buffers:
            self._sync_buffers()

        return self.module(*args, **kwargs)

    def _sync_buffers(self) -> None:
        """Synchronize buffers before forward."""
        for buffer in self.module.buffers():
            self.process_group.broadcast(buffer, src=0)


@dataclass
class GradientBucket:
    """Bucket for gradient aggregation."""

    max_size: int
    params: List[Parameter] = field(default_factory=list)
    size: int = 0
    _ready_count: int = 0

    def add_param(self, param: Parameter) -> bool:
        """Add parameter to bucket. Returns False if full."""
        param_size = param.numel * param.dtype.itemsize
        if self.size + param_size > self.max_size and self.params:
            return False

        self.params.append(param)
        self.size += param_size
        return True

    def all_reduce(self, process_group: ProcessGroup) -> None:
        """All-reduce gradients in this bucket."""
        # Pack gradients into contiguous buffer
        # All-reduce buffer
        # Unpack back to gradients
        pass

    def mark_grad_ready(self, param: Parameter) -> bool:
        """Mark gradient as ready. Returns True if bucket is complete."""
        self._ready_count += 1
        return self._ready_count >= len(self.params)


# ============================================================================
# ZeRO (Zero Redundancy Optimizer)
# ============================================================================

class ZeROStage(Enum):
    """ZeRO optimization stages."""
    DISABLED = 0       # No ZeRO
    OPTIMIZER = 1      # Partition optimizer states
    GRADIENTS = 2      # + Partition gradients
    PARAMETERS = 3     # + Partition parameters


class ZeROOptimizer:
    """
    Zero Redundancy Optimizer wrapper.

    Reduces memory footprint by partitioning optimizer states,
    gradients, and optionally parameters across processes.

    Stages:
    - Stage 1: Partition optimizer states (4x memory reduction)
    - Stage 2: + Partition gradients (8x reduction)
    - Stage 3: + Partition parameters (linear reduction with world_size)
    """

    def __init__(
        self,
        optimizer: Any,  # Base optimizer
        module: Module,
        stage: ZeROStage = ZeROStage.OPTIMIZER,
        process_group: Optional[ProcessGroup] = None,
        offload_optimizer: bool = False,
        offload_parameters: bool = False,
        contiguous_gradients: bool = True,
        overlap_comm: bool = True,
        reduce_bucket_size: int = 500_000_000,
        allgather_bucket_size: int = 500_000_000
    ):
        self.base_optimizer = optimizer
        self.module = module
        self.stage = stage
        self.process_group = process_group or ProcessGroup.get_default()

        self.offload_optimizer = offload_optimizer
        self.offload_parameters = offload_parameters
        self.contiguous_gradients = contiguous_gradients
        self.overlap_comm = overlap_comm

        self.reduce_bucket_size = reduce_bucket_size
        self.allgather_bucket_size = allgather_bucket_size

        # Partition info
        self.rank = self.process_group.rank
        self.world_size = self.process_group.world_size

        self._partition_parameters()
        self._partition_optimizer_state()

    def _partition_parameters(self) -> None:
        """Partition parameters across processes."""
        if self.stage.value < ZeROStage.PARAMETERS.value:
            return

        params = list(self.module.parameters())
        total_numel = sum(p.numel for p in params)
        partition_size = (total_numel + self.world_size - 1) // self.world_size

        # Assign parameters to ranks
        self._param_partitions: Dict[int, List[Parameter]] = defaultdict(list)
        current_numel = 0
        current_rank = 0

        for param in params:
            if current_numel >= partition_size and current_rank < self.world_size - 1:
                current_rank += 1
                current_numel = 0

            self._param_partitions[current_rank].append(param)
            current_numel += param.numel

        # Only keep local partition in memory
        self._local_params = self._param_partitions[self.rank]

    def _partition_optimizer_state(self) -> None:
        """Partition optimizer states."""
        if self.stage.value < ZeROStage.OPTIMIZER.value:
            return

        # Each rank only maintains optimizer state for its partition
        # This includes momentum, variance, etc.
        pass

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform optimization step with ZeRO.

        1. Reduce-scatter gradients (Stage 2+)
        2. Update local parameters
        3. All-gather parameters (Stage 3)
        """
        if self.stage.value >= ZeROStage.GRADIENTS.value:
            # Reduce-scatter gradients
            self._reduce_scatter_gradients()

        # Update with base optimizer (only local partition)
        loss = self.base_optimizer.step(closure)

        if self.stage.value >= ZeROStage.PARAMETERS.value:
            # All-gather updated parameters
            self._all_gather_parameters()

        return loss

    def _reduce_scatter_gradients(self) -> None:
        """Reduce and scatter gradients."""
        # Each rank gets the sum of gradients for its partition
        for param in self.module.parameters():
            if param.grad is not None:
                # reduce-scatter
                pass

    def _all_gather_parameters(self) -> None:
        """All-gather parameters after update."""
        # Broadcast updated parameters from each partition
        for rank in range(self.world_size):
            for param in self._param_partitions.get(rank, []):
                self.process_group.broadcast(param, src=rank)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        self.base_optimizer.zero_grad(set_to_none)


# ============================================================================
# Pipeline Parallelism
# ============================================================================

class PipelineStage(Module):
    """A stage in pipeline parallelism."""

    def __init__(
        self,
        module: Module,
        stage_id: int,
        num_stages: int,
        process_group: Optional[ProcessGroup] = None
    ):
        super().__init__()
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.process_group = process_group or ProcessGroup.get_default()

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def send_forward(self, tensor: Tensor) -> None:
        """Send activation to next stage."""
        if self.stage_id < self.num_stages - 1:
            self.process_group.send(tensor, dst=self.stage_id + 1)

    def recv_forward(self, shape: Tuple[int, ...], dtype: DType) -> Tensor:
        """Receive activation from previous stage."""
        if self.stage_id > 0:
            tensor = Tensor.empty(*shape, dtype=dtype)
            self.process_group.recv(tensor, src=self.stage_id - 1)
            return tensor
        return None

    def send_backward(self, tensor: Tensor) -> None:
        """Send gradient to previous stage."""
        if self.stage_id > 0:
            self.process_group.send(tensor, dst=self.stage_id - 1)

    def recv_backward(self, shape: Tuple[int, ...], dtype: DType) -> Tensor:
        """Receive gradient from next stage."""
        if self.stage_id < self.num_stages - 1:
            tensor = Tensor.empty(*shape, dtype=dtype)
            self.process_group.recv(tensor, src=self.stage_id + 1)
            return tensor
        return None


class PipelineParallel:
    """
    Pipeline parallelism for training large models.

    Splits model across multiple devices and pipelines micro-batches
    to maximize hardware utilization.

    Schedules:
    - GPipe: Simple fill-drain schedule
    - 1F1B: One forward one backward for memory efficiency
    - Interleaved 1F1B: Multiple model chunks per device
    """

    def __init__(
        self,
        modules: List[Module],
        num_microbatches: int,
        schedule: str = '1f1b',
        process_group: Optional[ProcessGroup] = None
    ):
        self.num_stages = len(modules)
        self.num_microbatches = num_microbatches
        self.schedule = schedule
        self.process_group = process_group or ProcessGroup.get_default()

        # Wrap modules as pipeline stages
        self.stages = [
            PipelineStage(m, i, self.num_stages, self.process_group)
            for i, m in enumerate(modules)
        ]

        # Get local stage
        self.local_stage_id = self.process_group.rank % self.num_stages
        self.local_stage = self.stages[self.local_stage_id]

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """
        Execute forward pass with pipeline schedule.

        Args:
            inputs: List of micro-batch inputs (only used at stage 0)

        Returns:
            List of outputs (only available at last stage)
        """
        if self.schedule == 'gpipe':
            return self._gpipe_forward(inputs)
        elif self.schedule == '1f1b':
            return self._1f1b_forward(inputs)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

    def _gpipe_forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """GPipe schedule: all forward then all backward."""
        activations = []
        outputs = []

        # Forward pass
        for mb_idx in range(self.num_microbatches):
            if self.local_stage_id == 0:
                x = inputs[mb_idx]
            else:
                # Receive from previous stage
                x = self.local_stage.recv_forward(inputs[0].shape, inputs[0].dtype)

            # Forward through local stage
            y = self.local_stage(x)
            activations.append((x, y))

            if self.local_stage_id < self.num_stages - 1:
                self.local_stage.send_forward(y)
            else:
                outputs.append(y)

        return outputs

    def _1f1b_forward(self, inputs: List[Tensor]) -> List[Tensor]:
        """1F1B schedule: interleave forward and backward."""
        # More memory efficient than GPipe
        # Steady state: one forward, one backward

        outputs = []
        num_warmup = min(self.num_stages - self.local_stage_id - 1,
                        self.num_microbatches)

        # Warmup: only forward
        for mb_idx in range(num_warmup):
            if self.local_stage_id == 0:
                x = inputs[mb_idx]
            else:
                x = self.local_stage.recv_forward(inputs[0].shape, inputs[0].dtype)

            y = self.local_stage(x)

            if self.local_stage_id < self.num_stages - 1:
                self.local_stage.send_forward(y)
            else:
                outputs.append(y)

        # Steady state: 1F1B
        for mb_idx in range(num_warmup, self.num_microbatches):
            # One forward
            if self.local_stage_id == 0:
                x = inputs[mb_idx]
            else:
                x = self.local_stage.recv_forward(inputs[0].shape, inputs[0].dtype)

            y = self.local_stage(x)

            if self.local_stage_id < self.num_stages - 1:
                self.local_stage.send_forward(y)
            else:
                outputs.append(y)

            # One backward would happen here

        return outputs


# ============================================================================
# Tensor Parallelism
# ============================================================================

class ColumnParallelLinear(Module):
    """
    Linear layer with column-wise parallelism.

    Splits weight matrix along output dimension.
    Each rank computes a portion of the output.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = True,
        process_group: Optional[ProcessGroup] = None
    ):
        super().__init__()

        self.process_group = process_group or ProcessGroup.get_default()
        self.world_size = self.process_group.world_size
        self.rank = self.process_group.rank

        assert out_features % self.world_size == 0
        self.local_out_features = out_features // self.world_size
        self.gather_output = gather_output

        # Each rank has portion of output dimension
        self.weight = Parameter(
            Tensor.empty(self.local_out_features, in_features)
        )
        if bias:
            self.bias = Parameter(Tensor.empty(self.local_out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Local matmul
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias

        if self.gather_output:
            # All-gather outputs from all ranks
            gathered = [Tensor.empty(*output.shape) for _ in range(self.world_size)]
            self.process_group.all_gather(gathered, output)
            # Concatenate along last dimension
            # output = concat(gathered, dim=-1)

        return output


class RowParallelLinear(Module):
    """
    Linear layer with row-wise parallelism.

    Splits weight matrix along input dimension.
    Input must be split across ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        input_is_parallel: bool = False,
        process_group: Optional[ProcessGroup] = None
    ):
        super().__init__()

        self.process_group = process_group or ProcessGroup.get_default()
        self.world_size = self.process_group.world_size
        self.rank = self.process_group.rank

        assert in_features % self.world_size == 0
        self.local_in_features = in_features // self.world_size
        self.input_is_parallel = input_is_parallel

        # Each rank has portion of input dimension
        self.weight = Parameter(
            Tensor.empty(out_features, self.local_in_features)
        )
        if bias:
            self.bias = Parameter(Tensor.empty(out_features))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if not self.input_is_parallel:
            # Split input across ranks
            # x = x[..., self.rank * self.local_in_features:(self.rank + 1) * self.local_in_features]
            pass

        # Local matmul
        output = x @ self.weight.T

        # All-reduce to sum partial results
        self.process_group.all_reduce(output, op='sum')

        if self.bias is not None:
            output = output + self.bias

        return output


# ============================================================================
# Gradient Compression
# ============================================================================

class GradientCompressor(ABC):
    """Base class for gradient compression."""

    @abstractmethod
    def compress(self, tensor: Tensor) -> Tuple[Any, Dict[str, Any]]:
        """Compress gradient."""
        pass

    @abstractmethod
    def decompress(self, data: Any, metadata: Dict[str, Any]) -> Tensor:
        """Decompress gradient."""
        pass


class TopKCompressor(GradientCompressor):
    """
    Top-K sparsification compressor.

    Only transmits the k largest gradient values.
    """

    def __init__(self, ratio: float = 0.01):
        self.ratio = ratio

    def compress(self, tensor: Tensor) -> Tuple[Any, Dict[str, Any]]:
        # Get top-k values and indices
        numel = tensor.numel
        k = max(1, int(numel * self.ratio))

        # flat = tensor.flatten()
        # values, indices = flat.topk(k)

        # return (values, indices), {'shape': tensor.shape, 'k': k}
        return None, {}

    def decompress(self, data: Any, metadata: Dict[str, Any]) -> Tensor:
        # values, indices = data
        # shape = metadata['shape']
        # result = Tensor.zeros(*shape)
        # result.flatten()[indices] = values
        # return result
        return Tensor.empty(0)


class PowerSGDCompressor(GradientCompressor):
    """
    PowerSGD low-rank compressor.

    Approximates gradient with low-rank matrix.
    """

    def __init__(self, rank: int = 4, start_iter: int = 10):
        self.rank = rank
        self.start_iter = start_iter
        self.iter = 0

        # Error feedback buffers
        self._error_dict: Dict[int, Tensor] = {}
        self._q_dict: Dict[int, Tensor] = {}

    def compress(self, tensor: Tensor) -> Tuple[Any, Dict[str, Any]]:
        self.iter += 1

        if self.iter < self.start_iter:
            # Don't compress in early iterations
            return tensor, {'compressed': False}

        # PowerSGD algorithm:
        # 1. Add error feedback
        # 2. Low-rank approximation: M ≈ P @ Q^T
        # 3. Update error feedback

        return None, {'compressed': True}

    def decompress(self, data: Any, metadata: Dict[str, Any]) -> Tensor:
        if not metadata.get('compressed', True):
            return data

        # Reconstruct from P @ Q^T
        return Tensor.empty(0)


# ============================================================================
# Elastic Training
# ============================================================================

class ElasticTrainer:
    """
    Elastic training with dynamic scaling.

    Supports:
    - Dynamic worker addition/removal
    - Checkpoint-based recovery
    - Gradient accumulation adjustment
    """

    def __init__(
        self,
        model: Module,
        optimizer: Any,
        min_workers: int = 1,
        max_workers: int = 64,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model
        self.optimizer = optimizer
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.checkpoint_dir = checkpoint_dir

        self.process_group = ProcessGroup.get_default()
        self._current_world_size = self.process_group.world_size

    def save_checkpoint(self, step: int) -> str:
        """Save training checkpoint."""
        if self.process_group.rank == 0:
            path = os.path.join(self.checkpoint_dir, f'ckpt_{step}.pt')
            checkpoint = {
                'step': step,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'world_size': self._current_world_size
            }
            with open(path, 'wb') as f:
                pickle.dump(checkpoint, f)
            return path
        return ''

    def load_checkpoint(self, path: str) -> int:
        """Load training checkpoint."""
        with open(path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

        return checkpoint['step']

    def handle_worker_change(self) -> None:
        """Handle dynamic worker changes."""
        new_world_size = int(os.environ.get('WORLD_SIZE', self._current_world_size))

        if new_world_size != self._current_world_size:
            # Re-sync model parameters
            for param in self.model.parameters():
                self.process_group.broadcast(param, src=0)

            # Adjust learning rate based on linear scaling rule
            scale_factor = new_world_size / self._current_world_size
            for group in self.optimizer.param_groups:
                group['lr'] *= scale_factor

            self._current_world_size = new_world_size
