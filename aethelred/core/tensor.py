"""
Aethelred Tensor - High-Performance Multi-Dimensional Array

A backend-aware tensor implementation with:
- Automatic device placement
- Lazy evaluation and operation fusion
- Broadcasting and views
- Automatic differentiation support
- Memory-efficient operations
- Interoperability with NumPy, PyTorch, TensorFlow
"""

from __future__ import annotations

import ctypes
import math
import operator
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce, wraps
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Protocol, Sequence, Tuple, TypeVar, Union, overload
)

from .runtime import (
    Device, DeviceType, MemoryBlock, MemoryPool, MemoryType,
    Stream, Event, Profiler, profile, jit
)

# Type aliases
Shape = Tuple[int, ...]
Strides = Tuple[int, ...]
Index = Union[int, slice, 'Tensor', None, Tuple[Union[int, slice, 'Tensor', None], ...]]


class DType(Enum):
    """Data types for tensors."""
    # Floating point
    float16 = auto()
    float32 = auto()
    float64 = auto()
    bfloat16 = auto()

    # Integer
    int8 = auto()
    int16 = auto()
    int32 = auto()
    int64 = auto()
    uint8 = auto()
    uint16 = auto()
    uint32 = auto()
    uint64 = auto()

    # Boolean
    bool_ = auto()

    # Complex
    complex64 = auto()
    complex128 = auto()

    @property
    def itemsize(self) -> int:
        """Size of one element in bytes."""
        sizes = {
            DType.float16: 2, DType.bfloat16: 2,
            DType.float32: 4, DType.float64: 8,
            DType.int8: 1, DType.uint8: 1,
            DType.int16: 2, DType.uint16: 2,
            DType.int32: 4, DType.uint32: 4,
            DType.int64: 8, DType.uint64: 8,
            DType.bool_: 1,
            DType.complex64: 8, DType.complex128: 16,
        }
        return sizes[self]

    @property
    def is_floating_point(self) -> bool:
        return self in (DType.float16, DType.float32, DType.float64, DType.bfloat16)

    @property
    def is_integer(self) -> bool:
        return self in (DType.int8, DType.int16, DType.int32, DType.int64,
                       DType.uint8, DType.uint16, DType.uint32, DType.uint64)

    @property
    def is_complex(self) -> bool:
        return self in (DType.complex64, DType.complex128)

    @classmethod
    def from_numpy(cls, dtype) -> 'DType':
        """Convert numpy dtype to DType."""
        import numpy as np
        mapping = {
            np.float16: cls.float16,
            np.float32: cls.float32,
            np.float64: cls.float64,
            np.int8: cls.int8,
            np.int16: cls.int16,
            np.int32: cls.int32,
            np.int64: cls.int64,
            np.uint8: cls.uint8,
            np.uint16: cls.uint16,
            np.uint32: cls.uint32,
            np.uint64: cls.uint64,
            np.bool_: cls.bool_,
            np.complex64: cls.complex64,
            np.complex128: cls.complex128,
        }
        return mapping.get(dtype.type, cls.float32)


@dataclass
class TensorStorage:
    """
    Low-level storage for tensor data.

    Implements copy-on-write semantics and reference counting.
    """
    data: MemoryBlock
    offset: int = 0
    ref_count: int = 1

    # Metadata
    device: Device = field(default_factory=Device.get_default)
    dtype: DType = DType.float32

    def add_ref(self) -> None:
        """Increment reference count."""
        self.ref_count += 1

    def release(self) -> bool:
        """Decrement reference count. Returns True if freed."""
        self.ref_count -= 1
        return self.ref_count <= 0

    @property
    def data_ptr(self) -> int:
        """Get the data pointer."""
        return self.data.ptr + self.offset

    @property
    def nbytes(self) -> int:
        """Total bytes in storage."""
        return self.data.size - self.offset


class LazyOp(ABC):
    """Base class for lazy operations."""

    @abstractmethod
    def realize(self) -> 'Tensor':
        """Execute the operation and return result."""
        pass

    @abstractmethod
    def get_shape(self) -> Shape:
        """Get output shape without executing."""
        pass

    @abstractmethod
    def get_dtype(self) -> DType:
        """Get output dtype without executing."""
        pass


@dataclass
class BinaryOp(LazyOp):
    """Binary operation (add, mul, etc.)."""
    op: str
    left: 'Tensor'
    right: 'Tensor'

    def realize(self) -> 'Tensor':
        # Realize operands first
        left_data = self.left._realize()
        right_data = self.right._realize()

        # Compute output shape
        out_shape = self.get_shape()

        # Execute operation
        ops = {
            'add': operator.add,
            'sub': operator.sub,
            'mul': operator.mul,
            'div': operator.truediv,
            'pow': operator.pow,
            'mod': operator.mod,
        }

        # This would use optimized kernels in production
        result = Tensor.empty(out_shape, dtype=self.get_dtype(), device=self.left.device)
        return result

    def get_shape(self) -> Shape:
        if self.op == 'matmul':
            ls, rs = self.left.shape, self.right.shape
            if len(ls) == 1 and len(rs) == 1:
                return ()
            elif len(ls) == 1:
                return rs[:-2] + (rs[-1],)
            elif len(rs) == 1:
                return ls[:-1]
            else:
                batch = _broadcast_shapes(ls[:-2], rs[:-2]) if ls[:-2] or rs[:-2] else ()
                return batch + (ls[-2], rs[-1])
        return _broadcast_shapes(self.left.shape, self.right.shape)

    def get_dtype(self) -> DType:
        return _promote_dtypes(self.left.dtype, self.right.dtype)


@dataclass
class UnaryOp(LazyOp):
    """Unary operation (neg, exp, etc.)."""
    op: str
    input: 'Tensor'

    def realize(self) -> 'Tensor':
        input_data = self.input._realize()
        result = Tensor.empty(input_data.shape, dtype=input_data.dtype, device=input_data.device)
        return result

    def get_shape(self) -> Shape:
        return self.input.shape

    def get_dtype(self) -> DType:
        return self.input.dtype


@dataclass
class ReduceOp(LazyOp):
    """Reduction operation (sum, mean, etc.)."""
    op: str
    input: 'Tensor'
    dims: Optional[Tuple[int, ...]]
    keepdim: bool

    def realize(self) -> 'Tensor':
        input_data = self.input._realize()
        out_shape = self.get_shape()
        result = Tensor.empty(out_shape, dtype=self.get_dtype(), device=self.input.device)
        return result

    def get_shape(self) -> Shape:
        if self.dims is None:
            return () if not self.keepdim else (1,) * len(self.input.shape)

        shape = list(self.input.shape)
        for dim in sorted(self.dims, reverse=True):
            if self.keepdim:
                shape[dim] = 1
            else:
                shape.pop(dim)
        return tuple(shape)

    def get_dtype(self) -> DType:
        if self.op in ('sum', 'mean') and self.input.dtype.is_integer:
            return DType.int64
        return self.input.dtype


class Tensor:
    """
    Multi-dimensional array with automatic device management.

    Features:
    - Lazy evaluation for operation fusion
    - Automatic broadcasting
    - Views and slicing without copy
    - Gradient tracking for autodiff
    - Device-agnostic operations

    Example:
        # Create tensors
        x = Tensor.randn(3, 4, device=Device.gpu())
        y = Tensor.ones(4, 5, device=Device.gpu())

        # Operations are lazy
        z = x @ y + 1.0

        # Realize when needed
        result = z.numpy()  # Executes the graph
    """

    __slots__ = (
        '_storage', '_shape', '_strides', '_offset', '_dtype', '_device',
        '_lazy_op', '_requires_grad', '_grad', '_grad_fn', '_version',
        '__weakref__'
    )

    def __init__(
        self,
        storage: Optional[TensorStorage] = None,
        shape: Shape = (),
        strides: Optional[Strides] = None,
        offset: int = 0,
        dtype: DType = DType.float32,
        device: Optional[Device] = None,
        lazy_op: Optional[LazyOp] = None,
        requires_grad: bool = False
    ):
        self._storage = storage
        self._shape = shape
        self._strides = strides or self._compute_strides(shape)
        self._offset = offset
        self._dtype = dtype
        self._device = device or Device.get_default()
        self._lazy_op = lazy_op
        self._requires_grad = requires_grad
        self._grad: Optional[Tensor] = None
        self._grad_fn: Optional[Callable] = None
        self._version = 0

    @staticmethod
    def _compute_strides(shape: Shape) -> Strides:
        """Compute contiguous strides for a shape."""
        if not shape:
            return ()
        strides = [1]
        for dim in reversed(shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))

    # ========================================================================
    # Factory Methods
    # ========================================================================

    @classmethod
    def empty(
        cls,
        *shape: int,
        dtype: DType = DType.float32,
        device: Optional[Device] = None,
        requires_grad: bool = False
    ) -> 'Tensor':
        """Create an uninitialized tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        device = device or Device.get_default()
        size = reduce(operator.mul, shape, 1) * dtype.itemsize

        # Allocate from device memory pool
        block = device.memory_pool.allocate(size, MemoryType.DEVICE)
        storage = TensorStorage(data=block, device=device, dtype=dtype)

        return cls(
            storage=storage,
            shape=shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )

    @classmethod
    def zeros(
        cls,
        *shape: int,
        dtype: DType = DType.float32,
        device: Optional[Device] = None,
        requires_grad: bool = False
    ) -> 'Tensor':
        """Create a tensor filled with zeros."""
        tensor = cls.empty(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
        # Zero-fill would happen here
        return tensor

    @classmethod
    def ones(
        cls,
        *shape: int,
        dtype: DType = DType.float32,
        device: Optional[Device] = None,
        requires_grad: bool = False
    ) -> 'Tensor':
        """Create a tensor filled with ones."""
        tensor = cls.empty(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
        # Fill with ones would happen here
        return tensor

    @classmethod
    def full(
        cls,
        shape: Shape,
        fill_value: Union[int, float],
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False
    ) -> 'Tensor':
        """Create a tensor filled with a value."""
        if dtype is None:
            dtype = DType.float32 if isinstance(fill_value, float) else DType.int64
        tensor = cls.empty(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
        # Fill would happen here
        return tensor

    @classmethod
    def randn(
        cls,
        *shape: int,
        dtype: DType = DType.float32,
        device: Optional[Device] = None,
        requires_grad: bool = False,
        generator: Optional[Any] = None
    ) -> 'Tensor':
        """Create a tensor with random normal values."""
        tensor = cls.empty(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
        # Random normal fill would happen here
        return tensor

    @classmethod
    def rand(
        cls,
        *shape: int,
        dtype: DType = DType.float32,
        device: Optional[Device] = None,
        requires_grad: bool = False,
        generator: Optional[Any] = None
    ) -> 'Tensor':
        """Create a tensor with random uniform values in [0, 1)."""
        tensor = cls.empty(*shape, dtype=dtype, device=device, requires_grad=requires_grad)
        # Random uniform fill would happen here
        return tensor

    @classmethod
    def arange(
        cls,
        start: Union[int, float],
        end: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None
    ) -> 'Tensor':
        """Create a tensor with evenly spaced values."""
        if end is None:
            start, end = 0, start

        if dtype is None:
            dtype = DType.float32 if any(isinstance(x, float) for x in [start, end, step]) else DType.int64

        size = max(0, math.ceil((end - start) / step))
        tensor = cls.empty(size, dtype=dtype, device=device)
        # Fill with range would happen here
        return tensor

    @classmethod
    def linspace(
        cls,
        start: float,
        end: float,
        steps: int,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ) -> 'Tensor':
        """Create a tensor with evenly spaced values."""
        tensor = cls.empty(steps, dtype=dtype, device=device)
        # Fill with linspace would happen here
        return tensor

    @classmethod
    def eye(
        cls,
        n: int,
        m: Optional[int] = None,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ) -> 'Tensor':
        """Create an identity matrix."""
        m = m or n
        tensor = cls.zeros(n, m, dtype=dtype, device=device)
        # Set diagonal to 1 would happen here
        return tensor

    @classmethod
    def from_numpy(cls, array, device: Optional[Device] = None) -> 'Tensor':
        """Create tensor from numpy array."""
        import numpy as np

        if not isinstance(array, np.ndarray):
            array = np.asarray(array)

        dtype = DType.from_numpy(array.dtype)
        tensor = cls.empty(*array.shape, dtype=dtype, device=device)

        # Copy data would happen here
        return tensor

    @classmethod
    def from_torch(cls, tensor, device: Optional[Device] = None) -> 'Tensor':
        """Create tensor from PyTorch tensor."""
        return cls.from_numpy(tensor.detach().cpu().numpy(), device=device)

    # ========================================================================
    # Properties
    # ========================================================================

    @property
    def shape(self) -> Shape:
        """Get tensor shape."""
        if self._lazy_op:
            return self._lazy_op.get_shape()
        return self._shape

    @property
    def strides(self) -> Strides:
        """Get tensor strides."""
        self._realize()
        return self._strides

    @property
    def dtype(self) -> DType:
        """Get data type."""
        if self._lazy_op:
            return self._lazy_op.get_dtype()
        return self._dtype

    @property
    def device(self) -> Device:
        """Get device."""
        return self._device

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return reduce(operator.mul, self.shape, 1)

    @property
    def nbytes(self) -> int:
        """Total bytes."""
        return self.numel * self.dtype.itemsize

    @property
    def requires_grad(self) -> bool:
        """Whether gradient is tracked."""
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        self._requires_grad = value

    @property
    def grad(self) -> Optional['Tensor']:
        """Get gradient tensor."""
        return self._grad

    @property
    def is_contiguous(self) -> bool:
        """Check if memory is contiguous."""
        if self._lazy_op:
            return True
        expected = self._compute_strides(self.shape)
        return self._strides == expected

    @property
    def T(self) -> 'Tensor':
        """Transpose (last two dimensions)."""
        return self.transpose(-2, -1)

    # ========================================================================
    # Lazy Evaluation
    # ========================================================================

    def _realize(self) -> 'Tensor':
        """Execute any pending lazy operations."""
        if self._lazy_op is not None:
            with Profiler.get_current().trace("realize", "lazy") if Profiler.get_current() else _nullcontext():
                result = self._lazy_op.realize()
                self._storage = result._storage
                self._shape = result._shape
                self._strides = result._strides
                self._lazy_op = None
        return self

    def realize(self) -> 'Tensor':
        """Force execution of lazy operations."""
        return self._realize()

    # ========================================================================
    # Data Access
    # ========================================================================

    @property
    def data_ptr(self) -> int:
        """Get raw data pointer."""
        self._realize()
        if self._storage is None:
            raise RuntimeError("Tensor has no storage")
        return self._storage.data_ptr + self._offset * self.dtype.itemsize

    def numpy(self) -> Any:
        """Convert to numpy array."""
        import numpy as np

        self._realize()
        self.device.synchronize()

        # Create numpy array view of the data
        dtype_map = {
            DType.float16: np.float16,
            DType.float32: np.float32,
            DType.float64: np.float64,
            DType.int8: np.int8,
            DType.int16: np.int16,
            DType.int32: np.int32,
            DType.int64: np.int64,
            DType.uint8: np.uint8,
            DType.uint16: np.uint16,
            DType.uint32: np.uint32,
            DType.uint64: np.uint64,
            DType.bool_: np.bool_,
        }

        # For actual implementation, would copy from device memory
        return np.empty(self.shape, dtype=dtype_map.get(self.dtype, np.float32))

    def tolist(self) -> List[Any]:
        """Convert to Python list."""
        return self.numpy().tolist()

    def item(self) -> Union[int, float, bool]:
        """Get scalar value."""
        if self.numel != 1:
            raise ValueError(f"item() requires exactly 1 element, got {self.numel}")
        return self.numpy().item()

    # ========================================================================
    # Device Operations
    # ========================================================================

    def to(
        self,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None,
        non_blocking: bool = False
    ) -> 'Tensor':
        """Move tensor to device and/or convert dtype."""
        if device is None and dtype is None:
            return self

        if device == self.device and (dtype is None or dtype == self.dtype):
            return self

        self._realize()

        target_device = device or self.device
        target_dtype = dtype or self.dtype

        # Allocate on target device
        result = Tensor.empty(*self.shape, dtype=target_dtype, device=target_device)

        # Copy data (would be async if non_blocking)
        # In production, this would use device-to-device copy

        return result

    def cpu(self) -> 'Tensor':
        """Move to CPU."""
        return self.to(Device.cpu())

    def gpu(self, device_id: int = 0) -> 'Tensor':
        """Move to GPU."""
        return self.to(Device.gpu(device_id))

    def cuda(self, device_id: int = 0) -> 'Tensor':
        """Move to GPU (alias for gpu())."""
        return self.gpu(device_id)

    def contiguous(self) -> 'Tensor':
        """Return contiguous tensor."""
        if self.is_contiguous:
            return self
        return self.clone()

    def clone(self) -> 'Tensor':
        """Create a copy of the tensor."""
        self._realize()
        result = Tensor.empty(*self.shape, dtype=self.dtype, device=self.device)
        # Copy data would happen here
        return result

    def detach(self) -> 'Tensor':
        """Return tensor without gradient tracking."""
        result = Tensor(
            storage=self._storage,
            shape=self._shape,
            strides=self._strides,
            offset=self._offset,
            dtype=self._dtype,
            device=self._device,
            lazy_op=self._lazy_op,
            requires_grad=False
        )
        return result

    # ========================================================================
    # Shape Operations
    # ========================================================================

    def reshape(self, *shape: int) -> 'Tensor':
        """Reshape tensor."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        # Handle -1 dimension
        if -1 in shape:
            known_dims = [d for d in shape if d != -1]
            known_size = reduce(operator.mul, known_dims, 1)
            inferred = self.numel // known_size
            shape = tuple(inferred if d == -1 else d for d in shape)

        if reduce(operator.mul, shape, 1) != self.numel:
            raise ValueError(f"Cannot reshape {self.shape} to {shape}")

        self._realize()

        if self.is_contiguous:
            # View reshape
            return Tensor(
                storage=self._storage,
                shape=shape,
                strides=self._compute_strides(shape),
                offset=self._offset,
                dtype=self._dtype,
                device=self._device,
                requires_grad=self._requires_grad
            )
        else:
            # Copy reshape
            return self.contiguous().reshape(*shape)

    def view(self, *shape: int) -> 'Tensor':
        """Return view with different shape."""
        return self.reshape(*shape)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> 'Tensor':
        """Flatten dimensions."""
        if end_dim < 0:
            end_dim = self.ndim + end_dim

        new_shape = (
            self.shape[:start_dim] +
            (reduce(operator.mul, self.shape[start_dim:end_dim + 1], 1),) +
            self.shape[end_dim + 1:]
        )
        return self.reshape(*new_shape)

    def squeeze(self, dim: Optional[int] = None) -> 'Tensor':
        """Remove dimensions of size 1."""
        if dim is not None:
            if self.shape[dim] != 1:
                return self
            new_shape = self.shape[:dim] + self.shape[dim + 1:]
        else:
            new_shape = tuple(d for d in self.shape if d != 1)

        return self.view(*new_shape) if new_shape else self.view(1)

    def unsqueeze(self, dim: int) -> 'Tensor':
        """Add dimension of size 1."""
        if dim < 0:
            dim = self.ndim + dim + 1
        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        return self.view(*new_shape)

    def expand(self, *sizes: int) -> 'Tensor':
        """Expand tensor to larger size."""
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])

        # Broadcasting logic would go here
        return self  # Placeholder

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        """Swap two dimensions."""
        self._realize()

        ndim = self.ndim
        if dim0 < 0:
            dim0 = ndim + dim0
        if dim1 < 0:
            dim1 = ndim + dim1

        new_shape = list(self.shape)
        new_strides = list(self._strides)

        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        return Tensor(
            storage=self._storage,
            shape=tuple(new_shape),
            strides=tuple(new_strides),
            offset=self._offset,
            dtype=self._dtype,
            device=self._device,
            requires_grad=self._requires_grad
        )

    def permute(self, *dims: int) -> 'Tensor':
        """Permute dimensions."""
        self._realize()

        new_shape = tuple(self.shape[d] for d in dims)
        new_strides = tuple(self._strides[d] for d in dims)

        return Tensor(
            storage=self._storage,
            shape=new_shape,
            strides=new_strides,
            offset=self._offset,
            dtype=self._dtype,
            device=self._device,
            requires_grad=self._requires_grad
        )

    # ========================================================================
    # Indexing
    # ========================================================================

    def __getitem__(self, index: Index) -> 'Tensor':
        """Get subtensor by index."""
        self._realize()

        # Handle different index types
        if isinstance(index, int):
            # Single integer index
            if index < 0:
                index = self.shape[0] + index

            new_shape = self.shape[1:]
            new_offset = self._offset + index * self._strides[0]

            return Tensor(
                storage=self._storage,
                shape=new_shape,
                strides=self._strides[1:],
                offset=new_offset,
                dtype=self._dtype,
                device=self._device,
                requires_grad=self._requires_grad
            )

        elif isinstance(index, slice):
            # Slice index
            start, stop, step = index.indices(self.shape[0])
            new_size = max(0, (stop - start + step - 1) // step)

            new_shape = (new_size,) + self.shape[1:]
            new_strides = (self._strides[0] * step,) + self._strides[1:]
            new_offset = self._offset + start * self._strides[0]

            return Tensor(
                storage=self._storage,
                shape=new_shape,
                strides=new_strides,
                offset=new_offset,
                dtype=self._dtype,
                device=self._device,
                requires_grad=self._requires_grad
            )

        elif isinstance(index, tuple):
            # Multi-dimensional indexing
            result = self
            offset = 0
            for i, idx in enumerate(index):
                if isinstance(idx, int):
                    result = result[idx]
                elif isinstance(idx, slice):
                    # Would handle slice at dimension i
                    pass
                elif idx is None:
                    result = result.unsqueeze(i + offset)
                    offset += 1

            return result

        else:
            raise TypeError(f"Unsupported index type: {type(index)}")

    def __setitem__(self, index: Index, value: Union['Tensor', float, int]) -> None:
        """Set subtensor by index."""
        self._realize()
        self._version += 1
        # Implementation would go here

    # ========================================================================
    # Arithmetic Operations
    # ========================================================================

    def _binary_op(self, other: Union['Tensor', float, int], op: str) -> 'Tensor':
        """Create lazy binary operation."""
        if not isinstance(other, Tensor):
            other = Tensor.full((), other, dtype=self.dtype, device=self.device)

        return Tensor(
            shape=_broadcast_shapes(self.shape, other.shape),
            dtype=_promote_dtypes(self.dtype, other.dtype),
            device=self.device,
            lazy_op=BinaryOp(op, self, other),
            requires_grad=self._requires_grad or other._requires_grad
        )

    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'add')

    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'add')

    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'sub')

    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full((), other, dtype=self.dtype, device=self.device)
        return other._binary_op(self, 'sub')

    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'mul')

    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'mul')

    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'div')

    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full((), other, dtype=self.dtype, device=self.device)
        return other._binary_op(self, 'div')

    def __pow__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'pow')

    def __neg__(self) -> 'Tensor':
        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            lazy_op=UnaryOp('neg', self),
            requires_grad=self._requires_grad
        )

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication."""
        # Validate shapes
        if self.ndim < 1 or other.ndim < 1:
            raise ValueError("matmul requires at least 1D tensors")

        # Determine output shape
        if self.ndim == 1 and other.ndim == 1:
            out_shape = ()
        elif self.ndim == 1:
            out_shape = other.shape[:-2] + (other.shape[-1],)
        elif other.ndim == 1:
            out_shape = self.shape[:-1]
        else:
            # Batch matmul
            batch_shape = _broadcast_shapes(self.shape[:-2], other.shape[:-2])
            out_shape = batch_shape + (self.shape[-2], other.shape[-1])

        return Tensor(
            shape=out_shape,
            dtype=_promote_dtypes(self.dtype, other.dtype),
            device=self.device,
            lazy_op=BinaryOp('matmul', self, other),
            requires_grad=self._requires_grad or other._requires_grad
        )

    # ========================================================================
    # Comparison Operations
    # ========================================================================

    def __eq__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'eq')

    def __ne__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'ne')

    def __lt__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'lt')

    def __le__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'le')

    def __gt__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'gt')

    def __ge__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        return self._binary_op(other, 'ge')

    # ========================================================================
    # Reduction Operations
    # ========================================================================

    def _reduce(
        self,
        op: str,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> 'Tensor':
        """Create lazy reduction operation."""
        if isinstance(dim, int):
            dims = (dim,)
        else:
            dims = dim

        return Tensor(
            dtype=self.dtype,
            device=self.device,
            lazy_op=ReduceOp(op, self, dims, keepdim),
            requires_grad=self._requires_grad
        )

    def sum(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> 'Tensor':
        """Sum over dimensions."""
        return self._reduce('sum', dim, keepdim)

    def mean(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> 'Tensor':
        """Mean over dimensions."""
        return self._reduce('mean', dim, keepdim)

    def max(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False
    ) -> Union['Tensor', Tuple['Tensor', 'Tensor']]:
        """Max over dimensions."""
        return self._reduce('max', dim, keepdim)

    def min(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False
    ) -> Union['Tensor', Tuple['Tensor', 'Tensor']]:
        """Min over dimensions."""
        return self._reduce('min', dim, keepdim)

    def prod(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False
    ) -> 'Tensor':
        """Product over dimensions."""
        return self._reduce('prod', dim, keepdim)

    def std(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
        unbiased: bool = True
    ) -> 'Tensor':
        """Standard deviation."""
        return self._reduce('std', dim, keepdim)

    def var(
        self,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False,
        unbiased: bool = True
    ) -> 'Tensor':
        """Variance."""
        return self._reduce('var', dim, keepdim)

    def norm(
        self,
        p: float = 2,
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdim: bool = False
    ) -> 'Tensor':
        """Compute norm."""
        if p == 2:
            return (self ** 2).sum(dim, keepdim).sqrt()
        elif p == 1:
            return self.abs().sum(dim, keepdim)
        elif p == float('inf'):
            return self.abs().max(dim, keepdim)
        else:
            return (self.abs() ** p).sum(dim, keepdim) ** (1 / p)

    # ========================================================================
    # Unary Operations
    # ========================================================================

    def _unary(self, op: str) -> 'Tensor':
        """Create lazy unary operation."""
        return Tensor(
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
            lazy_op=UnaryOp(op, self),
            requires_grad=self._requires_grad
        )

    def abs(self) -> 'Tensor':
        return self._unary('abs')

    def sqrt(self) -> 'Tensor':
        return self._unary('sqrt')

    def rsqrt(self) -> 'Tensor':
        return self._unary('rsqrt')

    def exp(self) -> 'Tensor':
        return self._unary('exp')

    def log(self) -> 'Tensor':
        return self._unary('log')

    def log2(self) -> 'Tensor':
        return self._unary('log2')

    def log10(self) -> 'Tensor':
        return self._unary('log10')

    def sin(self) -> 'Tensor':
        return self._unary('sin')

    def cos(self) -> 'Tensor':
        return self._unary('cos')

    def tan(self) -> 'Tensor':
        return self._unary('tan')

    def tanh(self) -> 'Tensor':
        return self._unary('tanh')

    def sigmoid(self) -> 'Tensor':
        return self._unary('sigmoid')

    def relu(self) -> 'Tensor':
        return self._unary('relu')

    def gelu(self) -> 'Tensor':
        return self._unary('gelu')

    def softmax(self, dim: int = -1) -> 'Tensor':
        """Softmax along dimension."""
        exp_x = (self - self.max(dim, keepdim=True)).exp()
        return exp_x / exp_x.sum(dim, keepdim=True)

    def log_softmax(self, dim: int = -1) -> 'Tensor':
        """Log-softmax along dimension."""
        return self - self.exp().sum(dim, keepdim=True).log()

    # ========================================================================
    # String Representation
    # ========================================================================

    def __repr__(self) -> str:
        if self._lazy_op:
            return f"Tensor(shape={self.shape}, dtype={self.dtype}, lazy=True, device={self.device})"
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"

    def __str__(self) -> str:
        return self.__repr__()


# ============================================================================
# Helper Functions
# ============================================================================

def _broadcast_shapes(shape1: Shape, shape2: Shape) -> Shape:
    """Compute broadcast output shape."""
    result = []
    s1 = list(reversed(shape1))
    s2 = list(reversed(shape2))

    for i in range(max(len(s1), len(s2))):
        d1 = s1[i] if i < len(s1) else 1
        d2 = s2[i] if i < len(s2) else 1

        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")

    return tuple(reversed(result))


def _promote_dtypes(dtype1: DType, dtype2: DType) -> DType:
    """Determine output dtype for binary operations."""
    # Float > Int > Bool
    if dtype1.is_floating_point or dtype2.is_floating_point:
        # Use highest precision float
        if dtype1 == DType.float64 or dtype2 == DType.float64:
            return DType.float64
        return DType.float32

    if dtype1.is_integer or dtype2.is_integer:
        # Use 64-bit int
        return DType.int64

    return dtype1


class _nullcontext:
    """Null context manager for Python < 3.7."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        pass
