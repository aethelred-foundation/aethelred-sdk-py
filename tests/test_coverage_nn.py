"""Comprehensive tests for aethelred/nn/__init__.py to achieve 95%+ coverage.

Covers:
- Parameter, Buffer
- Module base class: __setattr__, __getattr__, __delattr__, __call__,
  parameters, named_parameters, buffers, named_buffers, children,
  named_children, modules, named_modules, state_dict, load_state_dict,
  train, eval, to, cuda, cpu, float, double, half, bfloat16, _apply,
  hooks, zero_grad, num_parameters, extra_repr, __repr__
- Sequential, ModuleList, ModuleDict
- Linear, Embedding
- LayerNorm, BatchNorm1d, RMSNorm
- ReLU, GELU, SiLU, Softmax, Tanh, Sigmoid
- Dropout
- MultiheadAttention
- TransformerEncoderLayer
- MSELoss, CrossEntropyLoss
- load_model, save_model
"""

from __future__ import annotations

import os
import pickle
import tempfile
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest

from aethelred.core.tensor import Tensor, DType
from aethelred.core.runtime import Device, DeviceType, Profiler
from aethelred.nn import (
    Parameter,
    Buffer,
    Module,
    Sequential,
    ModuleList,
    ModuleDict,
    Linear,
    Embedding,
    LayerNorm,
    BatchNorm1d,
    RMSNorm,
    ReLU,
    GELU,
    SiLU,
    Softmax,
    Tanh,
    Sigmoid,
    Dropout,
    MultiheadAttention,
    TransformerEncoderLayer,
    MSELoss,
    CrossEntropyLoss,
    load_model,
    save_model,
)


# ---------------------------------------------------------------------------
# Concrete Module for testing abstract Module
# ---------------------------------------------------------------------------

class SimpleModule(Module):
    def __init__(self, in_f=4, out_f=2):
        super().__init__()
        self.linear = Linear(in_f, out_f)

    def forward(self, x):
        return self.linear(x)


class BareModule(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# ============================================================================
# Parameter
# ============================================================================


class TestParameter:
    def test_creation_with_data(self):
        t = Tensor.empty(3, 4)
        p = Parameter(t)
        assert p.shape == (3, 4)
        assert p.requires_grad is True

    def test_creation_no_data(self):
        p = Parameter()
        assert p.shape == (0,)

    def test_no_grad(self):
        t = Tensor.empty(3)
        p = Parameter(t, requires_grad=False)
        assert p.requires_grad is False

    def test_repr(self):
        t = Tensor.empty(3, 4)
        p = Parameter(t)
        r = repr(p)
        assert "Parameter" in r
        assert "(3, 4)" in r


# ============================================================================
# Buffer
# ============================================================================


class TestBuffer:
    def test_creation(self):
        buf = Buffer()
        assert buf.data is not None
        assert buf.persistent is True

    def test_creation_with_data(self):
        t = Tensor.empty(5)
        buf = Buffer(t)
        assert buf.data is t

    def test_non_persistent(self):
        buf = Buffer(persistent=False)
        assert buf.persistent is False

    def test_repr(self):
        buf = Buffer()
        r = repr(buf)
        assert "Buffer" in r
        assert "persistent=True" in r


# ============================================================================
# Module
# ============================================================================


class TestModule:
    def test_init(self):
        m = BareModule()
        assert m.training is True

    def test_setattr_parameter(self):
        m = BareModule()
        p = Parameter(Tensor.empty(3))
        m.my_param = p
        assert "my_param" in m._parameters

    def test_setattr_module(self):
        m = BareModule()
        child = BareModule()
        m.child = child
        assert "child" in m._modules

    def test_setattr_buffer(self):
        m = BareModule()
        buf = Buffer(Tensor.empty(3))
        m.my_buf = buf
        assert "my_buf" in m._buffers

    def test_setattr_regular(self):
        m = BareModule()
        m.some_value = 42
        assert m.some_value == 42

    def test_getattr_parameter(self):
        m = BareModule()
        p = Parameter(Tensor.empty(3))
        m._parameters["weight"] = p
        assert m.weight is p

    def test_getattr_buffer(self):
        m = BareModule()
        buf = Buffer(Tensor.empty(3))
        m._buffers["running_mean"] = buf
        val = m.running_mean
        assert val is buf.data

    def test_getattr_module(self):
        m = BareModule()
        child = BareModule()
        m._modules["child"] = child
        assert m.child is child

    def test_getattr_missing(self):
        m = BareModule()
        with pytest.raises(AttributeError):
            _ = m.nonexistent

    def test_delattr_parameter(self):
        m = BareModule()
        m.my_param = Parameter(Tensor.empty(3))
        del m.my_param
        assert "my_param" not in m._parameters

    def test_delattr_buffer(self):
        m = BareModule()
        m.my_buf = Buffer(Tensor.empty(3))
        del m.my_buf
        assert "my_buf" not in m._buffers

    def test_delattr_module(self):
        m = BareModule()
        m.child = BareModule()
        del m.child
        assert "child" not in m._modules

    def test_delattr_regular(self):
        m = BareModule()
        m.val = 42
        del m.val
        with pytest.raises(AttributeError):
            _ = m.val

    def test_call(self):
        m = BareModule()
        t = Tensor.empty(3)
        result = m(t)
        assert result is t

    def test_call_with_profiler(self):
        profiler = Profiler(enabled=True)
        m = BareModule()
        t = Tensor.empty(3)
        with profiler:
            result = m(t)
        assert result is t

    def test_parameters(self):
        m = SimpleModule(4, 2)
        params = list(m.parameters())
        assert len(params) >= 1

    def test_parameters_no_recurse(self):
        m = SimpleModule(4, 2)
        params = list(m.parameters(recurse=False))
        assert len(params) == 0  # params are in child module

    def test_named_parameters(self):
        m = SimpleModule(4, 2)
        named = list(m.named_parameters())
        names = [n for n, p in named]
        assert any("weight" in n for n in names)

    def test_buffers(self):
        m = BareModule()
        m.my_buf = Buffer(Tensor.empty(3))
        bufs = list(m.buffers())
        assert len(bufs) == 1

    def test_named_buffers(self):
        m = BareModule()
        m.my_buf = Buffer(Tensor.empty(3))
        named = list(m.named_buffers())
        assert len(named) == 1
        assert named[0][0] == "my_buf"

    def test_children(self):
        m = SimpleModule(4, 2)
        children = list(m.children())
        assert len(children) == 1

    def test_named_children(self):
        m = SimpleModule(4, 2)
        named = list(m.named_children())
        assert len(named) == 1
        assert named[0][0] == "linear"

    def test_modules(self):
        m = SimpleModule(4, 2)
        mods = list(m.modules())
        assert len(mods) >= 2  # self + linear

    def test_named_modules(self):
        m = SimpleModule(4, 2)
        named = list(m.named_modules())
        names = [n for n, mod in named]
        assert "" in names  # self
        assert "linear" in names

    def test_state_dict(self):
        m = SimpleModule(4, 2)
        sd = m.state_dict()
        assert isinstance(sd, OrderedDict)
        assert any("weight" in k for k in sd.keys())

    def test_state_dict_with_buffer(self):
        m = BareModule()
        m.weight = Parameter(Tensor.empty(3))
        m.my_buf = Buffer(Tensor.empty(2))
        sd = m.state_dict()
        assert "weight" in sd
        assert "my_buf" in sd

    def test_state_dict_non_persistent_buffer(self):
        m = BareModule()
        m.my_buf = Buffer(Tensor.empty(2), persistent=False)
        sd = m.state_dict()
        assert "my_buf" not in sd

    def test_load_state_dict(self):
        m = SimpleModule(4, 2)
        sd = m.state_dict()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert len(missing) == 0
        assert len(unexpected) == 0

    def test_load_state_dict_strict_missing(self):
        m = SimpleModule(4, 2)
        with pytest.raises(RuntimeError, match="Missing keys"):
            m.load_state_dict({}, strict=True)

    def test_load_state_dict_strict_unexpected(self):
        m = BareModule()
        with pytest.raises(RuntimeError, match="Unexpected keys"):
            m.load_state_dict({"nonexistent": Tensor.empty(1)}, strict=True)

    def test_train(self):
        m = SimpleModule(4, 2)
        m.train(True)
        assert m.training is True
        for child in m.children():
            assert child.training is True

    def test_eval(self):
        m = SimpleModule(4, 2)
        m.eval()
        assert m.training is False

    def test_to_dtype(self):
        m = SimpleModule(4, 2)
        m.to(dtype=DType.float64)
        # should not raise

    def test_cuda(self):
        # No GPU, but should construct the Device call
        m = SimpleModule(4, 2)
        try:
            m.cuda(0)
        except RuntimeError:
            pass  # GPU not available

    def test_cpu(self):
        m = SimpleModule(4, 2)
        m.cpu()

    def test_float(self):
        m = SimpleModule(4, 2)
        m.float()

    def test_double(self):
        m = SimpleModule(4, 2)
        m.double()

    def test_half(self):
        m = SimpleModule(4, 2)
        m.half()

    def test_bfloat16(self):
        m = SimpleModule(4, 2)
        m.bfloat16()

    def test_register_forward_hook(self):
        m = BareModule()
        handle = m.register_forward_hook(lambda mod, inp, out: None)
        assert handle == 0
        assert len(m._forward_hooks) == 1

    def test_register_backward_hook(self):
        m = BareModule()
        handle = m.register_backward_hook(lambda mod, gin, gout: None)
        assert handle == 0
        assert len(m._backward_hooks) == 1

    def test_register_forward_pre_hook(self):
        m = BareModule()
        handle = m.register_forward_pre_hook(lambda mod, inp: None)
        assert handle == 0
        assert len(m._forward_pre_hooks) == 1

    def test_forward_hooks_called(self):
        m = BareModule()
        called = []
        m.register_forward_hook(lambda mod, inp, out: called.append(True))
        m(Tensor.empty(3))
        assert len(called) == 1

    def test_forward_hook_modifies_output(self):
        m = BareModule()
        replacement = Tensor.empty(5)
        m.register_forward_hook(lambda mod, inp, out: replacement)
        result = m(Tensor.empty(3))
        assert result is replacement

    def test_pre_forward_hook(self):
        m = BareModule()
        m.register_forward_pre_hook(lambda mod, inp: (Tensor.empty(5),))
        result = m(Tensor.empty(3))
        assert result.shape == (5,)

    def test_pre_forward_hook_non_tuple(self):
        m = BareModule()
        m.register_forward_pre_hook(lambda mod, inp: Tensor.empty(7))
        result = m(Tensor.empty(3))
        assert result.shape == (7,)

    def test_zero_grad(self):
        m = SimpleModule(4, 2)
        for p in m.parameters():
            p._grad = Tensor.zeros(*p.shape, dtype=p.dtype, device=p.device)
        m.zero_grad()
        for p in m.parameters():
            if p.grad is not None:
                assert p.grad.shape == p.shape

    def test_zero_grad_set_to_none(self):
        m = SimpleModule(4, 2)
        for p in m.parameters():
            p._grad = Tensor.zeros(*p.shape, dtype=p.dtype, device=p.device)
        m.zero_grad(set_to_none=True)
        for p in m.parameters():
            assert p.grad is None

    def test_num_parameters(self):
        m = SimpleModule(4, 2)
        n = m.num_parameters()
        assert n > 0

    def test_num_parameters_all(self):
        m = SimpleModule(4, 2)
        n = m.num_parameters(only_trainable=False)
        assert n > 0

    def test_repr(self):
        m = SimpleModule(4, 2)
        r = repr(m)
        assert "SimpleModule" in r
        assert "linear" in r


# ============================================================================
# Sequential
# ============================================================================


class TestSequential:
    def test_forward(self):
        seq = Sequential(BareModule(), BareModule())
        t = Tensor.empty(3)
        result = seq(t)
        assert result.shape == (3,)

    def test_getitem(self):
        m1 = BareModule()
        m2 = BareModule()
        seq = Sequential(m1, m2)
        assert seq[0] is m1
        assert seq[1] is m2

    def test_len(self):
        seq = Sequential(BareModule(), BareModule(), BareModule())
        assert len(seq) == 3

    def test_iter(self):
        modules = [BareModule(), BareModule()]
        seq = Sequential(*modules)
        for s, m in zip(seq, modules):
            assert s is m

    def test_append(self):
        seq = Sequential()
        m = BareModule()
        seq.append(m)
        assert len(seq) == 1


# ============================================================================
# ModuleList
# ============================================================================


class TestModuleList:
    def test_creation(self):
        ml = ModuleList([BareModule(), BareModule()])
        assert len(ml) == 2

    def test_creation_empty(self):
        ml = ModuleList()
        assert len(ml) == 0

    def test_getitem(self):
        m = BareModule()
        ml = ModuleList([m])
        assert ml[0] is m

    def test_setitem(self):
        ml = ModuleList([BareModule()])
        new_m = BareModule()
        ml[0] = new_m
        assert ml[0] is new_m

    def test_iter(self):
        modules = [BareModule(), BareModule()]
        ml = ModuleList(modules)
        for a, b in zip(ml, modules):
            assert a is b

    def test_append(self):
        ml = ModuleList()
        ml.append(BareModule())
        assert len(ml) == 1

    def test_forward_raises(self):
        ml = ModuleList()
        with pytest.raises(NotImplementedError):
            ml()


# ============================================================================
# ModuleDict
# ============================================================================


class TestModuleDict:
    def test_creation(self):
        md = ModuleDict({"a": BareModule(), "b": BareModule()})
        assert len(md) == 2

    def test_creation_empty(self):
        md = ModuleDict()
        assert len(md) == 0

    def test_getitem(self):
        m = BareModule()
        md = ModuleDict({"a": m})
        assert md["a"] is m

    def test_setitem(self):
        md = ModuleDict()
        m = BareModule()
        md["a"] = m
        assert md["a"] is m

    def test_contains(self):
        md = ModuleDict({"a": BareModule()})
        assert "a" in md
        assert "b" not in md

    def test_iter(self):
        md = ModuleDict({"a": BareModule(), "b": BareModule()})
        keys = list(md)
        assert "a" in keys
        assert "b" in keys

    def test_keys(self):
        md = ModuleDict({"a": BareModule()})
        assert "a" in md.keys()

    def test_values(self):
        m = BareModule()
        md = ModuleDict({"a": m})
        assert m in md.values()

    def test_items(self):
        m = BareModule()
        md = ModuleDict({"a": m})
        items = list(md.items())
        assert items[0] == ("a", m)

    def test_forward_raises(self):
        md = ModuleDict()
        with pytest.raises(NotImplementedError):
            md()


# ============================================================================
# Linear
# ============================================================================


class TestLinear:
    def test_creation(self):
        l = Linear(10, 5)
        assert l.in_features == 10
        assert l.out_features == 5
        assert l.weight is not None
        assert l.bias is not None

    def test_no_bias(self):
        l = Linear(10, 5, bias=False)
        assert l.bias is None

    def test_forward(self):
        l = Linear(4, 3)
        x = Tensor.empty(2, 4)
        out = l(x)
        assert out is not None

    def test_forward_no_bias(self):
        l = Linear(4, 3, bias=False)
        x = Tensor.empty(2, 4)
        out = l(x)
        assert out is not None

    def test_extra_repr(self):
        l = Linear(10, 5)
        r = l.extra_repr()
        assert "in_features=10" in r
        assert "out_features=5" in r
        assert "bias=True" in r


# ============================================================================
# Embedding
# ============================================================================


class TestEmbedding:
    def test_creation(self):
        e = Embedding(100, 32)
        assert e.num_embeddings == 100
        assert e.embedding_dim == 32

    def test_creation_with_padding(self):
        e = Embedding(100, 32, padding_idx=0)
        assert e.padding_idx == 0

    def test_forward(self):
        e = Embedding(100, 32)
        indices = Tensor.empty(5, dtype=DType.int64)
        out = e(indices)
        assert out.shape == (5, 32)

    def test_extra_repr(self):
        e = Embedding(100, 32)
        r = e.extra_repr()
        assert "100" in r
        assert "32" in r


# ============================================================================
# LayerNorm
# ============================================================================


class TestLayerNorm:
    def test_creation_int(self):
        ln = LayerNorm(10)
        assert ln.normalized_shape == (10,)

    def test_creation_tuple(self):
        ln = LayerNorm((3, 4))
        assert ln.normalized_shape == (3, 4)

    def test_affine(self):
        ln = LayerNorm(10)
        assert ln.weight is not None
        assert ln.bias is not None

    def test_no_affine(self):
        ln = LayerNorm(10, elementwise_affine=False)
        assert ln.weight is None
        assert ln.bias is None

    def test_forward(self):
        ln = LayerNorm(4)
        x = Tensor.empty(2, 4)
        out = ln(x)
        assert out is not None

    def test_forward_no_affine(self):
        ln = LayerNorm(4, elementwise_affine=False)
        x = Tensor.empty(2, 4)
        out = ln(x)
        assert out is not None

    def test_extra_repr(self):
        ln = LayerNorm(10, eps=1e-5)
        r = ln.extra_repr()
        assert "(10,)" in r


# ============================================================================
# BatchNorm1d
# ============================================================================


class TestBatchNorm1d:
    def test_creation(self):
        bn = BatchNorm1d(10)
        assert bn.num_features == 10
        assert bn.weight is not None
        assert bn.bias is not None

    def test_no_affine(self):
        bn = BatchNorm1d(10, affine=False)
        assert bn.weight is None
        assert bn.bias is None

    def test_no_tracking(self):
        bn = BatchNorm1d(10, track_running_stats=False)
        assert 'running_mean' not in bn._buffers

    def test_forward_training(self):
        bn = BatchNorm1d(4)
        bn.train()
        x = Tensor.empty(2, 4)
        out = bn(x)
        assert out is not None

    def test_forward_eval(self):
        bn = BatchNorm1d(4)
        bn.eval()
        x = Tensor.empty(2, 4)
        out = bn(x)
        assert out is not None


# ============================================================================
# RMSNorm
# ============================================================================


class TestRMSNorm:
    def test_creation(self):
        rn = RMSNorm(10)
        assert rn.dim == 10

    def test_forward(self):
        rn = RMSNorm(4)
        x = Tensor.empty(2, 4)
        out = rn(x)
        assert out is not None


# ============================================================================
# Activation Functions
# ============================================================================


class TestActivations:
    def test_relu(self):
        m = ReLU()
        x = Tensor.empty(3)
        out = m(x)
        assert out is not None

    def test_relu_inplace(self):
        m = ReLU(inplace=True)
        assert m.inplace is True

    def test_gelu(self):
        m = GELU()
        x = Tensor.empty(3)
        out = m(x)
        assert out is not None

    def test_gelu_approximate(self):
        m = GELU(approximate='tanh')
        assert m.approximate == 'tanh'

    def test_silu(self):
        m = SiLU()
        x = Tensor.empty(3)
        out = m(x)
        assert out is not None

    def test_softmax(self):
        m = Softmax(dim=-1)
        x = Tensor.empty(3, 4)
        out = m(x)
        assert out is not None

    def test_tanh(self):
        m = Tanh()
        x = Tensor.empty(3)
        out = m(x)
        assert out is not None

    def test_sigmoid(self):
        m = Sigmoid()
        x = Tensor.empty(3)
        out = m(x)
        assert out is not None


# ============================================================================
# Dropout
# ============================================================================


class TestDropout:
    def test_training(self):
        d = Dropout(p=0.5)
        d.train()
        x = Tensor.empty(10)
        out = d(x)
        assert out is not None

    def test_eval(self):
        d = Dropout(p=0.5)
        d.eval()
        x = Tensor.empty(10)
        out = d(x)
        assert out is x

    def test_zero_p(self):
        d = Dropout(p=0.0)
        d.train()
        x = Tensor.empty(10)
        out = d(x)
        assert out is x

    def test_extra_repr(self):
        d = Dropout(p=0.3)
        assert "p=0.3" in d.extra_repr()


# ============================================================================
# MultiheadAttention
# ============================================================================


class TestMultiheadAttention:
    def test_creation(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2)
        assert mha.embed_dim == 8
        assert mha.num_heads == 2
        assert mha.head_dim == 4

    def test_invalid_embed_dim(self):
        with pytest.raises(AssertionError):
            MultiheadAttention(embed_dim=7, num_heads=2)

    def test_with_dropout(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.1)
        assert mha.dropout_layer is not None

    def test_no_dropout(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0)
        assert mha.dropout_layer is None

    def test_custom_kv_dim(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2, kdim=16, vdim=16)
        assert mha.kdim == 16
        assert mha.vdim == 16

    def test_forward(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2)
        q = Tensor.empty(5, 2, 8)  # (L, N, E)
        k = Tensor.empty(5, 2, 8)
        v = Tensor.empty(5, 2, 8)
        out, weights = mha(q, k, v)
        assert out is not None
        assert weights is not None

    def test_forward_no_weights(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2)
        q = Tensor.empty(5, 2, 8)
        k = Tensor.empty(5, 2, 8)
        v = Tensor.empty(5, 2, 8)
        out, weights = mha(q, k, v, need_weights=False)
        assert out is not None
        assert weights is None

    def test_forward_batch_first(self):
        mha = MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        q = Tensor.empty(2, 5, 8)  # (N, L, E)
        k = Tensor.empty(2, 5, 8)
        v = Tensor.empty(2, 5, 8)
        out, weights = mha(q, k, v)
        assert out is not None


# ============================================================================
# TransformerEncoderLayer
# ============================================================================


class TestTransformerEncoderLayer:
    def test_creation_relu(self):
        layer = TransformerEncoderLayer(d_model=8, nhead=2, activation='relu')
        assert isinstance(layer.activation, ReLU)

    def test_creation_gelu(self):
        layer = TransformerEncoderLayer(d_model=8, nhead=2, activation='gelu')
        assert isinstance(layer.activation, GELU)

    def test_forward_post_norm(self):
        layer = TransformerEncoderLayer(d_model=8, nhead=2, norm_first=False)
        x = Tensor.empty(5, 2, 8)
        out = layer(x)
        assert out is not None

    def test_forward_pre_norm(self):
        layer = TransformerEncoderLayer(d_model=8, nhead=2, norm_first=True)
        x = Tensor.empty(5, 2, 8)
        out = layer(x)
        assert out is not None


# ============================================================================
# Loss Functions
# ============================================================================


class TestMSELoss:
    def test_mean(self):
        loss_fn = MSELoss(reduction='mean')
        x = Tensor.empty(3, 4)
        y = Tensor.empty(3, 4)
        loss = loss_fn(x, y)
        assert loss is not None

    def test_sum(self):
        loss_fn = MSELoss(reduction='sum')
        x = Tensor.empty(3, 4)
        y = Tensor.empty(3, 4)
        loss = loss_fn(x, y)
        assert loss is not None

    def test_none(self):
        loss_fn = MSELoss(reduction='none')
        x = Tensor.empty(3, 4)
        y = Tensor.empty(3, 4)
        loss = loss_fn(x, y)
        assert loss is not None


class TestCrossEntropyLoss:
    def test_creation(self):
        ce = CrossEntropyLoss()
        assert ce.reduction == 'mean'
        assert ce.label_smoothing == 0.0

    def test_forward_integer_target(self):
        ce = CrossEntropyLoss()
        logits = Tensor.empty(3, 10)
        targets = Tensor.empty(3, dtype=DType.int64)
        loss = ce(logits, targets)
        assert loss is not None

    def test_forward_float_target(self):
        ce = CrossEntropyLoss()
        logits = Tensor.empty(3, 10)
        targets = Tensor.empty(3, 10, dtype=DType.float32)
        loss = ce(logits, targets)
        assert loss is not None

    def test_sum_reduction(self):
        ce = CrossEntropyLoss(reduction='sum')
        logits = Tensor.empty(3, 10)
        targets = Tensor.empty(3, 10, dtype=DType.float32)
        loss = ce(logits, targets)
        assert loss is not None

    def test_none_reduction(self):
        ce = CrossEntropyLoss(reduction='none')
        logits = Tensor.empty(3, 10)
        targets = Tensor.empty(3, 10, dtype=DType.float32)
        loss = ce(logits, targets)
        assert loss is not None


# ============================================================================
# Utilities
# ============================================================================


class TestModelIO:
    def test_save_and_load(self):
        m = SimpleModule(4, 2)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            save_model(m, path)
            checkpoint = load_model(path)
            assert "state_dict" in checkpoint
            assert "class" in checkpoint
            assert checkpoint["class"] == "SimpleModule"
        finally:
            os.unlink(path)
