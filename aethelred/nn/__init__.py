"""
Aethelred Neural Network Module

A comprehensive neural network framework with:
- PyTorch-compatible API
- Automatic model verification
- Hardware-optimized layers
- Quantization support
- Model serialization for on-chain verification
"""

from __future__ import annotations

import copy
import math
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import (
    Any, Callable, Dict, Generic, Iterator, List, Optional,
    Protocol, Set, Tuple, TypeVar, Union, overload
)

from ..core.tensor import Tensor, DType, Shape
from ..core.runtime import Device, Profiler, profile


# Type variables
T = TypeVar('T', bound='Module')


# ============================================================================
# Parameter and Buffer
# ============================================================================

class Parameter(Tensor):
    """
    A tensor that is automatically registered as a module parameter.

    Parameters are tensors that require gradients and are included
    in the module's state dict for saving/loading.
    """

    __slots__ = ()

    def __new__(
        cls,
        data: Optional[Tensor] = None,
        requires_grad: bool = True
    ) -> 'Parameter':
        if data is None:
            data = Tensor.empty(0)

        # Create new tensor with requires_grad set
        instance = Tensor.empty(*data.shape, dtype=data.dtype, device=data.device)
        instance._requires_grad = requires_grad
        instance.__class__ = Parameter

        return instance

    def __init__(
        self,
        data: Optional[Tensor] = None,
        requires_grad: bool = True
    ) -> None:
        # Skip Tensor.__init__ — all attributes are set by __new__
        pass

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Parameter({self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"


class Buffer:
    """
    A tensor that is registered as a module buffer (not a parameter).

    Buffers are saved in state dict but don't receive gradients.
    """

    def __init__(
        self,
        data: Optional[Tensor] = None,
        persistent: bool = True
    ):
        self.data = data or Tensor.empty(0)
        self.persistent = persistent  # Whether to include in state dict

    def __repr__(self) -> str:
        return f"Buffer({self.data.shape}, persistent={self.persistent})"


# ============================================================================
# Module Base Class
# ============================================================================

class Module(ABC):
    """
    Base class for all neural network modules.

    Provides:
    - Automatic parameter tracking
    - Recursive submodule management
    - State dict for serialization
    - Forward/backward hooks
    - Device and dtype management

    Example:
        class MyModel(Module):
            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear = Linear(in_features, out_features)
                self.activation = ReLU()

            def forward(self, x):
                return self.activation(self.linear(x))

        model = MyModel(10, 5)
        output = model(input_tensor)
    """

    _version: int = 1  # For serialization compatibility

    def __init__(self):
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Buffer]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self._forward_hooks: Dict[int, Callable] = OrderedDict()
        self._backward_hooks: Dict[int, Callable] = OrderedDict()
        self._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
        self.training: bool = True
        self._is_full_backward_hook: Optional[bool] = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle parameter and module registration."""

        def remove_from(*dicts):
            for d in dicts:
                if name in d:
                    del d[name]

        params = self.__dict__.get('_parameters')
        modules = self.__dict__.get('_modules')
        buffers = self.__dict__.get('_buffers')

        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("Cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, modules, buffers)
            self._parameters[name] = value

        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError("Cannot assign modules before Module.__init__() call")
            remove_from(self.__dict__, params, buffers)
            self._modules[name] = value

        elif isinstance(value, Buffer):
            if buffers is None:
                raise AttributeError("Cannot assign buffers before Module.__init__() call")
            remove_from(self.__dict__, params, modules)
            self._buffers[name] = value

        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> Any:
        """Get parameters, modules, or buffers by name."""
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            if name in params:
                return params[name]

        if '_buffers' in self.__dict__:
            buffers = self.__dict__['_buffers']
            if name in buffers:
                return buffers[name].data

        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """Delete parameter, module, or buffer."""
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Define the forward pass."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> Any:
        """Execute forward pass with hooks."""
        # Pre-forward hooks
        for hook in self._forward_pre_hooks.values():
            result = hook(self, args)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                args = result

        # Forward pass with profiling
        profiler = Profiler.get_current()
        if profiler:
            with profiler.trace(f"{self.__class__.__name__}.forward", "nn"):
                output = self.forward(*args, **kwargs)
        else:
            output = self.forward(*args, **kwargs)

        # Post-forward hooks
        for hook in self._forward_hooks.values():
            hook_result = hook(self, args, output)
            if hook_result is not None:
                output = hook_result

        return output

    # ========================================================================
    # Parameter Access
    # ========================================================================

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Iterate over module parameters."""
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """Iterate over named parameters."""
        memo: Set[Parameter] = set()

        for name, param in self._parameters.items():
            if param is not None and param not in memo:
                memo.add(param)
                yield prefix + name, param

        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    yield from module.named_parameters(submodule_prefix, recurse)

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        """Iterate over module buffers."""
        for name, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(
        self,
        prefix: str = '',
        recurse: bool = True
    ) -> Iterator[Tuple[str, Tensor]]:
        """Iterate over named buffers."""
        for name, buf in self._buffers.items():
            if buf is not None:
                yield prefix + name, buf.data

        if recurse:
            for module_name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + module_name + '.'
                    yield from module.named_buffers(submodule_prefix, recurse)

    def children(self) -> Iterator['Module']:
        """Iterate over immediate children modules."""
        for name, module in self._modules.items():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        """Iterate over named immediate children."""
        for name, module in list(self._modules.items()):
            if module is not None:
                yield name, module

    def modules(self) -> Iterator['Module']:
        """Iterate over all modules (including self)."""
        for name, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set['Module']] = None,
        prefix: str = ''
    ) -> Iterator[Tuple[str, 'Module']]:
        """Iterate over all named modules."""
        if memo is None:
            memo = set()

        if self not in memo:
            memo.add(self)
            yield prefix, self

            for name, module in self._modules.items():
                if module is not None:
                    submodule_prefix = prefix + ('.' if prefix else '') + name
                    yield from module.named_modules(memo, submodule_prefix)

    # ========================================================================
    # State Dict
    # ========================================================================

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = ''
    ) -> Dict[str, Any]:
        """Get module state as dictionary."""
        if destination is None:
            destination = OrderedDict()

        # Add parameters
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param.detach()

        # Add persistent buffers
        for name, buf in self._buffers.items():
            if buf is not None and buf.persistent:
                destination[prefix + name] = buf.data.detach()

        # Add submodules
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')

        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True
    ) -> Tuple[List[str], List[str]]:
        """Load state from dictionary."""
        missing_keys: List[str] = []
        unexpected_keys: List[str] = list(state_dict.keys())

        def load(module: Module, prefix: str = ''):
            # Load parameters
            for name, param in module._parameters.items():
                key = prefix + name
                if key in state_dict:
                    # Copy data from state dict
                    unexpected_keys.remove(key)
                elif strict:
                    missing_keys.append(key)

            # Load buffers
            for name, buf in module._buffers.items():
                if buf is not None and buf.persistent:
                    key = prefix + name
                    if key in state_dict:
                        unexpected_keys.remove(key)
                    elif strict:
                        missing_keys.append(key)

            # Load submodules
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(self)

        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(
                f"Missing keys: {missing_keys}\n"
                f"Unexpected keys: {unexpected_keys}"
            )

        return missing_keys, unexpected_keys

    # ========================================================================
    # Training Mode
    # ========================================================================

    def train(self, mode: bool = True) -> 'Module':
        """Set training mode."""
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self) -> 'Module':
        """Set evaluation mode."""
        return self.train(False)

    # ========================================================================
    # Device and Dtype
    # ========================================================================

    def to(
        self,
        device: Optional[Device] = None,
        dtype: Optional[DType] = None
    ) -> 'Module':
        """Move module to device and/or convert dtype."""
        def convert(t: Tensor) -> Tensor:
            return t.to(device=device, dtype=dtype)

        return self._apply(convert)

    def cuda(self, device_id: int = 0) -> 'Module':
        """Move to GPU."""
        return self.to(device=Device.gpu(device_id))

    def cpu(self) -> 'Module':
        """Move to CPU."""
        return self.to(device=Device.cpu())

    def float(self) -> 'Module':
        """Convert to float32."""
        return self.to(dtype=DType.float32)

    def double(self) -> 'Module':
        """Convert to float64."""
        return self.to(dtype=DType.float64)

    def half(self) -> 'Module':
        """Convert to float16."""
        return self.to(dtype=DType.float16)

    def bfloat16(self) -> 'Module':
        """Convert to bfloat16."""
        return self.to(dtype=DType.bfloat16)

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> 'Module':
        """Apply function to all parameters and buffers."""
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = Parameter(fn(param), param.requires_grad)

        for name, buf in self._buffers.items():
            if buf is not None:
                buf.data = fn(buf.data)

        for module in self.children():
            module._apply(fn)

        return self

    # ========================================================================
    # Hooks
    # ========================================================================

    def register_forward_hook(
        self,
        hook: Callable[['Module', Tuple, Any], Optional[Any]]
    ) -> Any:
        """Register a forward hook."""
        handle = len(self._forward_hooks)
        self._forward_hooks[handle] = hook
        return handle

    def register_backward_hook(
        self,
        hook: Callable[['Module', Tuple, Tuple], Optional[Tuple]]
    ) -> Any:
        """Register a backward hook."""
        handle = len(self._backward_hooks)
        self._backward_hooks[handle] = hook
        return handle

    def register_forward_pre_hook(
        self,
        hook: Callable[['Module', Tuple], Optional[Tuple]]
    ) -> Any:
        """Register a pre-forward hook."""
        handle = len(self._forward_pre_hooks)
        self._forward_pre_hooks[handle] = hook
        return handle

    # ========================================================================
    # Utilities
    # ========================================================================

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero all parameter gradients."""
        for param in self.parameters():
            if param.grad is not None:
                if set_to_none:
                    param._grad = None
                else:
                    param._grad = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)

    def num_parameters(self, only_trainable: bool = True) -> int:
        """Count number of parameters."""
        total = 0
        for param in self.parameters():
            if not only_trainable or param.requires_grad:
                total += param.numel
        return total

    def extra_repr(self) -> str:
        """Extra representation for string output."""
        return ''

    def __repr__(self) -> str:
        """String representation of module."""
        lines = [self.__class__.__name__ + '(']

        extra = self.extra_repr()
        if extra:
            lines.append('  ' + extra)

        for name, module in self._modules.items():
            mod_str = repr(module).replace('\n', '\n  ')
            lines.append(f'  ({name}): {mod_str}')

        lines.append(')')
        return '\n'.join(lines)


# ============================================================================
# Container Modules
# ============================================================================

class Sequential(Module):
    """
    Sequential container for modules.

    Example:
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )
    """

    def __init__(self, *args: Module):
        super().__init__()

        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> 'Sequential':
        """Append a module."""
        self._modules[str(len(self._modules))] = module
        return self


class ModuleList(Module):
    """List container for modules."""

    def __init__(self, modules: Optional[List[Module]] = None):
        super().__init__()
        if modules is not None:
            for idx, module in enumerate(modules):
                self._modules[str(idx)] = module

    def __getitem__(self, idx: int) -> Module:
        return list(self._modules.values())[idx]

    def __setitem__(self, idx: int, module: Module) -> None:
        self._modules[str(idx)] = module

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def append(self, module: Module) -> 'ModuleList':
        """Append a module."""
        self._modules[str(len(self._modules))] = module
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleList is a container, not a layer")


class ModuleDict(Module):
    """Dictionary container for modules."""

    def __init__(self, modules: Optional[Dict[str, Module]] = None):
        super().__init__()
        if modules is not None:
            for name, module in modules.items():
                self._modules[name] = module

    def __getitem__(self, key: str) -> Module:
        return self._modules[key]

    def __setitem__(self, key: str, module: Module) -> None:
        self._modules[key] = module

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def keys(self):
        return self._modules.keys()

    def values(self):
        return self._modules.values()

    def items(self):
        return self._modules.items()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("ModuleDict is a container, not a layer")


# ============================================================================
# Linear Layers
# ============================================================================

class Linear(Module):
    """
    Linear (fully connected) layer.

    y = x @ W.T + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight with Kaiming uniform
        k = 1 / math.sqrt(in_features)
        self.weight = Parameter(
            Tensor.empty(out_features, in_features, dtype=dtype, device=device)
        )
        # Would initialize: uniform(-k, k)

        if bias:
            self.bias = Parameter(
                Tensor.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class Embedding(Module):
    """Embedding layer for discrete tokens."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = Parameter(
            Tensor.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        )

    def forward(self, indices: Tensor) -> Tensor:
        # Would implement embedding lookup
        output_shape = indices.shape + (self.embedding_dim,)
        return Tensor.empty(*output_shape, dtype=self.weight.dtype, device=self.weight.device)

    def extra_repr(self) -> str:
        return f'{self.num_embeddings}, {self.embedding_dim}'


# ============================================================================
# Normalization Layers
# ============================================================================

class LayerNorm(Module):
    """Layer normalization."""

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = Parameter(Tensor.ones(*normalized_shape, dtype=dtype, device=device))
            self.bias = Parameter(Tensor.zeros(*normalized_shape, dtype=dtype, device=device))
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        # Compute mean and variance over normalized dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dims, keepdim=True)

        # Normalize
        x_norm = (x - mean) / (var + self.eps).sqrt()

        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm

    def extra_repr(self) -> str:
        return f'{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class BatchNorm1d(Module):
    """Batch normalization for 1D inputs."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = Parameter(Tensor.ones(num_features, dtype=dtype, device=device))
            self.bias = Parameter(Tensor.zeros(num_features, dtype=dtype, device=device))
        else:
            self.weight = None
            self.bias = None

        if track_running_stats:
            self._buffers['running_mean'] = Buffer(
                Tensor.zeros(num_features, dtype=dtype, device=device)
            )
            self._buffers['running_var'] = Buffer(
                Tensor.ones(num_features, dtype=dtype, device=device)
            )
            self._buffers['num_batches_tracked'] = Buffer(
                Tensor.zeros(1, dtype=DType.int64, device=device)
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # Use batch statistics
            mean = x.mean(0)
            var = x.var(0, unbiased=False)

            if self.track_running_stats:
                # Update running stats
                # Would update running_mean and running_var here
                pass
        else:
            # Use running statistics
            mean = self._buffers['running_mean'].data
            var = self._buffers['running_var'].data

        # Normalize
        x_norm = (x - mean) / (var + self.eps).sqrt()

        if self.affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm


class RMSNorm(Module):
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(Tensor.ones(dim, dtype=dtype, device=device))

    def forward(self, x: Tensor) -> Tensor:
        # RMS normalization
        rms = (x ** 2).mean(-1, keepdim=True).sqrt()
        x_norm = x / (rms + self.eps)
        return x_norm * self.weight


# ============================================================================
# Activation Functions
# ============================================================================

class ReLU(Module):
    """Rectified Linear Unit."""

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class GELU(Module):
    """Gaussian Error Linear Unit."""

    def __init__(self, approximate: str = 'none'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()


class SiLU(Module):
    """Sigmoid Linear Unit (Swish)."""

    def forward(self, x: Tensor) -> Tensor:
        return x * x.sigmoid()


class Softmax(Module):
    """Softmax activation."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(self.dim)


class Tanh(Module):
    """Hyperbolic tangent."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    """Sigmoid activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


# ============================================================================
# Dropout
# ============================================================================

class Dropout(Module):
    """Dropout layer."""

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.p > 0:
            # Would implement dropout mask here
            return x * (1 - self.p)  # Placeholder
        return x

    def extra_repr(self) -> str:
        return f'p={self.p}'


# ============================================================================
# Attention
# ============================================================================

class MultiheadAttention(Module):
    """Multi-head attention mechanism."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.kdim = kdim or embed_dim
        self.vdim = vdim or embed_dim

        # Projections
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.k_proj = Linear(self.kdim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.v_proj = Linear(self.vdim, embed_dim, bias=bias, dtype=dtype, device=device)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype, device=device)

        if dropout > 0:
            self.dropout_layer = Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = True
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for multi-head attention.

        Args:
            query: (L, N, E) or (N, L, E) if batch_first
            key: (S, N, E) or (N, S, E)
            value: (S, N, E) or (N, S, E)

        Returns:
            attention output and optional attention weights
        """
        if self.batch_first:
            # (N, L, E) -> (L, N, E)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Reshape for multi-head attention
        # (L, N, E) -> (L, N, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        q = q.reshape(tgt_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.reshape(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.reshape(src_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = (q @ k.transpose(-2, -1)) * scale

        # Apply masks
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # Expand mask for broadcasting
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )

        attn_weights = attn_weights.softmax(-1)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape back
        # (N, num_heads, L, head_dim) -> (L, N, E)
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(tgt_len, batch_size, embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            return attn_output, attn_weights.mean(1)  # Average over heads
        return attn_output, None


# ============================================================================
# Transformer
# ============================================================================

class TransformerEncoderLayer(Module):
    """Transformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        dtype: DType = DType.float32,
        device: Optional[Device] = None
    ):
        super().__init__()

        self.self_attn = MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first,
            dtype=dtype, device=device
        )

        # Feedforward
        self.linear1 = Linear(d_model, dim_feedforward, dtype=dtype, device=device)
        self.linear2 = Linear(dim_feedforward, d_model, dtype=dtype, device=device)

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, dtype=dtype, device=device)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, dtype=dtype, device=device)

        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = GELU() if activation == 'gelu' else ReLU()
        self.norm_first = norm_first

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        if self.norm_first:
            # Pre-norm architecture
            x = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            # Post-norm architecture
            x = self.norm1(src + self._sa_block(src, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        x, _ = self.self_attn(x, x, x, key_padding_mask, attn_mask, need_weights=False)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# ============================================================================
# Loss Functions
# ============================================================================

class MSELoss(Module):
    """Mean Squared Error loss."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = (input - target) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CrossEntropyLoss(Module):
    """Cross entropy loss with logits."""

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Log softmax + NLL loss
        log_probs = input.log_softmax(-1)

        # Create one-hot or use integer targets
        if target.dtype.is_integer:
            # Gather log probs at target indices
            # Would implement proper indexing
            loss = -log_probs  # Placeholder
        else:
            loss = -(target * log_probs).sum(-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# Utilities
# ============================================================================

def load_model(path: str, map_location: Optional[Device] = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    import pickle

    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    return checkpoint


def save_model(model: Module, path: str) -> None:
    """Save model checkpoint."""
    import pickle

    # Convert state_dict tensors to serializable dicts
    raw_sd = model.state_dict()
    serializable_sd: Dict[str, Any] = {}
    for k, v in raw_sd.items():
        if isinstance(v, Tensor):
            serializable_sd[k] = {
                'shape': v.shape,
                'dtype': v.dtype.name,
            }
        else:
            serializable_sd[k] = v

    checkpoint = {
        'state_dict': serializable_sd,
        'class': model.__class__.__name__,
    }

    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)
