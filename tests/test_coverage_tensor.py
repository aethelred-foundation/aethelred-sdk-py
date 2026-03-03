"""Comprehensive tests for aethelred/core/tensor.py to achieve 95%+ coverage.

Covers:
- DType enum, properties, from_numpy
- TensorStorage dataclass
- LazyOp subclasses: BinaryOp, UnaryOp, ReduceOp
- Tensor class: factory methods, properties, shape ops, arithmetic, indexing
- _broadcast_shapes, _promote_dtypes helpers
- _nullcontext
"""

from __future__ import annotations

import math
from functools import reduce
from unittest.mock import patch, MagicMock

import pytest

from aethelred.core.tensor import (
    DType,
    TensorStorage,
    LazyOp,
    BinaryOp,
    UnaryOp,
    ReduceOp,
    Tensor,
    Shape,
    _broadcast_shapes,
    _promote_dtypes,
    _nullcontext,
)
from aethelred.core.runtime import Device, DeviceType, MemoryBlock, MemoryType


# ============================================================================
# DType
# ============================================================================


class TestDType:
    def test_itemsize_float(self):
        assert DType.float16.itemsize == 2
        assert DType.bfloat16.itemsize == 2
        assert DType.float32.itemsize == 4
        assert DType.float64.itemsize == 8

    def test_itemsize_int(self):
        assert DType.int8.itemsize == 1
        assert DType.uint8.itemsize == 1
        assert DType.int16.itemsize == 2
        assert DType.uint16.itemsize == 2
        assert DType.int32.itemsize == 4
        assert DType.uint32.itemsize == 4
        assert DType.int64.itemsize == 8
        assert DType.uint64.itemsize == 8

    def test_itemsize_bool(self):
        assert DType.bool_.itemsize == 1

    def test_itemsize_complex(self):
        assert DType.complex64.itemsize == 8
        assert DType.complex128.itemsize == 16

    def test_is_floating_point(self):
        assert DType.float32.is_floating_point is True
        assert DType.float16.is_floating_point is True
        assert DType.float64.is_floating_point is True
        assert DType.bfloat16.is_floating_point is True
        assert DType.int32.is_floating_point is False
        assert DType.bool_.is_floating_point is False

    def test_is_integer(self):
        assert DType.int8.is_integer is True
        assert DType.int16.is_integer is True
        assert DType.int32.is_integer is True
        assert DType.int64.is_integer is True
        assert DType.uint8.is_integer is True
        assert DType.uint16.is_integer is True
        assert DType.uint32.is_integer is True
        assert DType.uint64.is_integer is True
        assert DType.float32.is_integer is False

    def test_is_complex(self):
        assert DType.complex64.is_complex is True
        assert DType.complex128.is_complex is True
        assert DType.float32.is_complex is False

    def test_from_numpy(self):
        np = pytest.importorskip("numpy")
        assert DType.from_numpy(np.dtype(np.float32)) == DType.float32
        assert DType.from_numpy(np.dtype(np.float64)) == DType.float64
        assert DType.from_numpy(np.dtype(np.int32)) == DType.int32
        assert DType.from_numpy(np.dtype(np.int64)) == DType.int64
        assert DType.from_numpy(np.dtype(np.bool_)) == DType.bool_


# ============================================================================
# TensorStorage
# ============================================================================


class TestTensorStorage:
    def _make_storage(self):
        dev = Device(DeviceType.CPU, 0)
        blk = MemoryBlock(ptr=1000, size=256, memory_type=MemoryType.HOST, device=dev)
        return TensorStorage(data=blk, offset=0, device=dev, dtype=DType.float32)

    def test_add_ref(self):
        s = self._make_storage()
        assert s.ref_count == 1
        s.add_ref()
        assert s.ref_count == 2

    def test_release(self):
        s = self._make_storage()
        freed = s.release()
        assert freed is True

    def test_release_not_freed(self):
        s = self._make_storage()
        s.add_ref()
        freed = s.release()
        assert freed is False

    def test_data_ptr(self):
        s = self._make_storage()
        assert s.data_ptr == 1000

    def test_data_ptr_with_offset(self):
        s = self._make_storage()
        s.offset = 16
        assert s.data_ptr == 1016

    def test_nbytes(self):
        s = self._make_storage()
        assert s.nbytes == 256


# ============================================================================
# Tensor Factory Methods
# ============================================================================


class TestTensorFactoryMethods:
    def test_empty(self):
        t = Tensor.empty(3, 4, dtype=DType.float32)
        assert t.shape == (3, 4)
        assert t.dtype == DType.float32

    def test_empty_tuple_shape(self):
        t = Tensor.empty((3, 4))
        assert t.shape == (3, 4)

    def test_zeros(self):
        t = Tensor.zeros(2, 3)
        assert t.shape == (2, 3)

    def test_ones(self):
        t = Tensor.ones(5)
        assert t.shape == (5,)

    def test_full_float(self):
        t = Tensor.full((2, 3), 3.14)
        assert t.shape == (2, 3)
        assert t.dtype == DType.float32

    def test_full_int(self):
        t = Tensor.full((2,), 42)
        assert t.dtype == DType.int64

    def test_full_explicit_dtype(self):
        t = Tensor.full((2,), 42, dtype=DType.float32)
        assert t.dtype == DType.float32

    def test_randn(self):
        t = Tensor.randn(3, 4)
        assert t.shape == (3, 4)

    def test_rand(self):
        t = Tensor.rand(3, 4)
        assert t.shape == (3, 4)

    def test_arange_one_arg(self):
        t = Tensor.arange(10)
        assert t.shape == (10,)
        assert t.dtype == DType.int64

    def test_arange_two_arg(self):
        t = Tensor.arange(2, 10)
        assert t.shape == (8,)

    def test_arange_float(self):
        t = Tensor.arange(0.0, 1.0, 0.1)
        assert t.dtype == DType.float32

    def test_arange_negative(self):
        t = Tensor.arange(10, 0, -1)
        assert t.shape == (10,)

    def test_linspace(self):
        t = Tensor.linspace(0.0, 1.0, 10)
        assert t.shape == (10,)

    def test_eye(self):
        t = Tensor.eye(3)
        assert t.shape == (3, 3)

    def test_eye_rectangular(self):
        t = Tensor.eye(3, 4)
        assert t.shape == (3, 4)

    def test_from_numpy(self):
        np = pytest.importorskip("numpy")
        arr = np.zeros((2, 3), dtype=np.float32)
        t = Tensor.from_numpy(arr)
        assert t.shape == (2, 3)

    def test_from_numpy_list(self):
        np = pytest.importorskip("numpy")
        t = Tensor.from_numpy([1, 2, 3])
        assert t.shape == (3,)


# ============================================================================
# Tensor Properties
# ============================================================================


class TestTensorProperties:
    def test_shape(self):
        t = Tensor.empty(3, 4, 5)
        assert t.shape == (3, 4, 5)

    def test_strides(self):
        t = Tensor.empty(3, 4)
        strides = t.strides
        assert strides == (4, 1)

    def test_dtype_property(self):
        t = Tensor.empty(2, dtype=DType.float64)
        assert t.dtype == DType.float64

    def test_device_property(self):
        t = Tensor.empty(2)
        assert t.device is not None

    def test_ndim(self):
        t = Tensor.empty(2, 3, 4)
        assert t.ndim == 3

    def test_numel(self):
        t = Tensor.empty(2, 3, 4)
        assert t.numel == 24

    def test_nbytes(self):
        t = Tensor.empty(2, 3, dtype=DType.float32)
        assert t.nbytes == 24  # 6 * 4

    def test_requires_grad_default(self):
        t = Tensor.empty(2)
        assert t.requires_grad is False

    def test_requires_grad_setter(self):
        t = Tensor.empty(2)
        t.requires_grad = True
        assert t.requires_grad is True

    def test_grad_default(self):
        t = Tensor.empty(2)
        assert t.grad is None

    def test_is_contiguous(self):
        t = Tensor.empty(3, 4)
        assert t.is_contiguous is True

    def test_T(self):
        t = Tensor.empty(3, 4)
        t2 = t.T
        assert t2.shape == (4, 3)


# ============================================================================
# Tensor Shape Operations
# ============================================================================


class TestTensorShapeOps:
    def test_reshape(self):
        t = Tensor.empty(12)
        r = t.reshape(3, 4)
        assert r.shape == (3, 4)

    def test_reshape_tuple(self):
        t = Tensor.empty(12)
        r = t.reshape((3, 4))
        assert r.shape == (3, 4)

    def test_reshape_infer(self):
        t = Tensor.empty(12)
        r = t.reshape(3, -1)
        assert r.shape == (3, 4)

    def test_reshape_invalid(self):
        t = Tensor.empty(12)
        with pytest.raises(ValueError, match="Cannot reshape"):
            t.reshape(5, 5)

    def test_view(self):
        t = Tensor.empty(12)
        v = t.view(3, 4)
        assert v.shape == (3, 4)

    def test_flatten(self):
        t = Tensor.empty(2, 3, 4)
        f = t.flatten()
        assert f.shape == (24,)

    def test_flatten_partial(self):
        t = Tensor.empty(2, 3, 4)
        f = t.flatten(start_dim=1)
        assert f.shape == (2, 12)

    def test_squeeze(self):
        t = Tensor.empty(1, 3, 1, 4)
        s = t.squeeze()
        assert s.shape == (3, 4)

    def test_squeeze_dim(self):
        t = Tensor.empty(1, 3, 1, 4)
        s = t.squeeze(0)
        assert s.shape == (3, 1, 4)

    def test_squeeze_dim_not_1(self):
        t = Tensor.empty(3, 4)
        s = t.squeeze(0)
        assert s.shape == (3, 4)

    def test_unsqueeze(self):
        t = Tensor.empty(3, 4)
        u = t.unsqueeze(0)
        assert u.shape == (1, 3, 4)

    def test_unsqueeze_negative(self):
        t = Tensor.empty(3, 4)
        u = t.unsqueeze(-1)
        assert u.shape == (3, 4, 1)

    def test_expand(self):
        t = Tensor.empty(3, 1)
        e = t.expand(3, 4)
        # expand returns self as placeholder
        assert e is not None

    def test_transpose(self):
        t = Tensor.empty(3, 4)
        tr = t.transpose(0, 1)
        assert tr.shape == (4, 3)

    def test_transpose_negative(self):
        t = Tensor.empty(2, 3, 4)
        tr = t.transpose(-2, -1)
        assert tr.shape == (2, 4, 3)

    def test_permute(self):
        t = Tensor.empty(2, 3, 4)
        p = t.permute(2, 0, 1)
        assert p.shape == (4, 2, 3)


# ============================================================================
# Tensor Indexing
# ============================================================================


class TestTensorIndexing:
    def test_getitem_int(self):
        t = Tensor.empty(3, 4)
        s = t[0]
        assert s.shape == (4,)

    def test_getitem_negative_int(self):
        t = Tensor.empty(3, 4)
        s = t[-1]
        assert s.shape == (4,)

    def test_getitem_slice(self):
        t = Tensor.empty(10, 4)
        s = t[2:5]
        assert s.shape == (3, 4)

    def test_getitem_tuple_int(self):
        t = Tensor.empty(3, 4, 5)
        s = t[0]
        assert s.shape == (4, 5)
        s2 = s[1]
        assert s2.shape == (5,)

    def test_getitem_none(self):
        t = Tensor.empty(3, 4)
        s = t[(None,)]
        assert s.shape == (1, 3, 4)

    def test_getitem_invalid_type(self):
        t = Tensor.empty(3)
        with pytest.raises(TypeError, match="Unsupported index type"):
            t["invalid"]

    def test_setitem(self):
        t = Tensor.empty(3, 4)
        t[0] = 1.0  # Should not raise


# ============================================================================
# Tensor Arithmetic
# ============================================================================


class TestTensorArithmetic:
    def test_add(self):
        a = Tensor.empty(3, 4)
        b = Tensor.empty(3, 4)
        c = a + b
        assert c.shape == (3, 4)

    def test_add_scalar(self):
        a = Tensor.empty(3, 4)
        c = a + 1.0
        assert c.shape == (3, 4)

    def test_radd(self):
        a = Tensor.empty(3, 4)
        c = 1.0 + a
        assert c.shape == (3, 4)

    def test_sub(self):
        a = Tensor.empty(3, 4)
        b = Tensor.empty(3, 4)
        c = a - b
        assert c.shape == (3, 4)

    def test_rsub(self):
        a = Tensor.empty(3, 4)
        c = 1.0 - a
        assert c.shape == (3, 4)

    def test_mul(self):
        a = Tensor.empty(3, 4)
        c = a * 2.0
        assert c.shape == (3, 4)

    def test_rmul(self):
        a = Tensor.empty(3, 4)
        c = 2.0 * a
        assert c.shape == (3, 4)

    def test_truediv(self):
        a = Tensor.empty(3, 4)
        c = a / 2.0
        assert c.shape == (3, 4)

    def test_rtruediv(self):
        a = Tensor.empty(3, 4)
        c = 2.0 / a
        assert c.shape == (3, 4)

    def test_pow(self):
        a = Tensor.empty(3, 4)
        c = a ** 2
        assert c.shape == (3, 4)

    def test_neg(self):
        a = Tensor.empty(3, 4)
        c = -a
        assert c.shape == (3, 4)

    def test_matmul(self):
        a = Tensor.empty(3, 4)
        b = Tensor.empty(4, 5)
        c = a @ b
        assert c.shape == (3, 5)

    def test_matmul_1d(self):
        a = Tensor.empty(4)
        b = Tensor.empty(4)
        c = a @ b
        assert c.shape == ()

    def test_matmul_1d_2d(self):
        a = Tensor.empty(4)
        b = Tensor.empty(4, 5)
        c = a @ b
        assert c.shape == (5,)

    def test_matmul_2d_1d(self):
        a = Tensor.empty(3, 4)
        b = Tensor.empty(4)
        c = a @ b
        assert c.shape == (3,)

    def test_matmul_batch(self):
        a = Tensor.empty(2, 3, 4)
        b = Tensor.empty(2, 4, 5)
        c = a @ b
        assert c.shape == (2, 3, 5)

    def test_matmul_0d_raises(self):
        a = Tensor.full((), 1.0)
        b = Tensor.full((), 2.0)
        with pytest.raises(ValueError, match="matmul"):
            a @ b


# ============================================================================
# Tensor Comparison Operations
# ============================================================================


class TestTensorComparisons:
    def test_eq(self):
        a = Tensor.empty(3)
        c = a == 0
        assert c is not None

    def test_ne(self):
        a = Tensor.empty(3)
        c = a != 0
        assert c is not None

    def test_lt(self):
        a = Tensor.empty(3)
        c = a < 0
        assert c is not None

    def test_le(self):
        a = Tensor.empty(3)
        c = a <= 0
        assert c is not None

    def test_gt(self):
        a = Tensor.empty(3)
        c = a > 0
        assert c is not None

    def test_ge(self):
        a = Tensor.empty(3)
        c = a >= 0
        assert c is not None


# ============================================================================
# Tensor Reduction Operations
# ============================================================================


class TestTensorReductions:
    def test_sum(self):
        t = Tensor.empty(3, 4)
        s = t.sum()
        assert s.shape == ()

    def test_sum_dim(self):
        t = Tensor.empty(3, 4)
        s = t.sum(0)
        assert s.shape == (4,)

    def test_sum_keepdim(self):
        t = Tensor.empty(3, 4)
        s = t.sum(0, keepdim=True)
        assert s.shape == (1, 4)

    def test_mean(self):
        t = Tensor.empty(3, 4)
        m = t.mean()
        assert m.shape == ()

    def test_max(self):
        t = Tensor.empty(3, 4)
        mx = t.max()
        assert mx.shape == ()

    def test_min(self):
        t = Tensor.empty(3, 4)
        mn = t.min()
        assert mn.shape == ()

    def test_prod(self):
        t = Tensor.empty(3, 4)
        p = t.prod()
        assert p.shape == ()

    def test_std(self):
        t = Tensor.empty(3, 4)
        s = t.std()
        assert s is not None

    def test_var(self):
        t = Tensor.empty(3, 4)
        v = t.var()
        assert v is not None

    def test_norm_l2(self):
        t = Tensor.empty(3, 4)
        n = t.norm(p=2)
        assert n is not None

    def test_norm_l1(self):
        t = Tensor.empty(3, 4)
        n = t.norm(p=1)
        assert n is not None

    def test_norm_inf(self):
        t = Tensor.empty(3, 4)
        n = t.norm(p=float('inf'))
        assert n is not None

    def test_norm_general(self):
        t = Tensor.empty(3, 4)
        n = t.norm(p=3)
        assert n is not None


# ============================================================================
# Tensor Unary Operations
# ============================================================================


class TestTensorUnaryOps:
    def test_abs(self):
        t = Tensor.empty(3)
        assert t.abs().shape == (3,)

    def test_sqrt(self):
        t = Tensor.empty(3)
        assert t.sqrt().shape == (3,)

    def test_rsqrt(self):
        t = Tensor.empty(3)
        assert t.rsqrt().shape == (3,)

    def test_exp(self):
        t = Tensor.empty(3)
        assert t.exp().shape == (3,)

    def test_log(self):
        t = Tensor.empty(3)
        assert t.log().shape == (3,)

    def test_log2(self):
        t = Tensor.empty(3)
        assert t.log2().shape == (3,)

    def test_log10(self):
        t = Tensor.empty(3)
        assert t.log10().shape == (3,)

    def test_sin(self):
        t = Tensor.empty(3)
        assert t.sin().shape == (3,)

    def test_cos(self):
        t = Tensor.empty(3)
        assert t.cos().shape == (3,)

    def test_tan(self):
        t = Tensor.empty(3)
        assert t.tan().shape == (3,)

    def test_tanh(self):
        t = Tensor.empty(3)
        assert t.tanh().shape == (3,)

    def test_sigmoid(self):
        t = Tensor.empty(3)
        assert t.sigmoid().shape == (3,)

    def test_relu(self):
        t = Tensor.empty(3)
        assert t.relu().shape == (3,)

    def test_gelu(self):
        t = Tensor.empty(3)
        assert t.gelu().shape == (3,)


# ============================================================================
# Tensor Softmax Operations
# ============================================================================


class TestTensorSoftmax:
    def test_softmax(self):
        t = Tensor.empty(3, 4)
        s = t.softmax(-1)
        assert s is not None

    def test_log_softmax(self):
        t = Tensor.empty(3, 4)
        s = t.log_softmax(-1)
        assert s is not None


# ============================================================================
# Tensor Device Operations
# ============================================================================


class TestTensorDeviceOps:
    def test_to_no_change(self):
        t = Tensor.empty(3)
        t2 = t.to()
        assert t2 is t

    def test_to_same_device_same_dtype(self):
        t = Tensor.empty(3)
        t2 = t.to(device=t.device, dtype=t.dtype)
        assert t2 is t

    def test_to_different_dtype(self):
        t = Tensor.empty(3, dtype=DType.float32)
        t2 = t.to(dtype=DType.float64)
        assert t2.dtype == DType.float64

    def test_cpu(self):
        t = Tensor.empty(3)
        t2 = t.cpu()
        assert t2 is not None

    def test_contiguous(self):
        t = Tensor.empty(3, 4)
        c = t.contiguous()
        assert c is t  # already contiguous

    def test_clone(self):
        t = Tensor.empty(3, 4)
        c = t.clone()
        assert c.shape == t.shape

    def test_detach(self):
        t = Tensor.empty(3, requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False


# ============================================================================
# Tensor String Representation
# ============================================================================


class TestTensorRepr:
    def test_repr(self):
        t = Tensor.empty(3, 4)
        r = repr(t)
        assert "Tensor" in r
        assert "(3, 4)" in r

    def test_repr_lazy(self):
        a = Tensor.empty(3)
        b = a + 1
        r = repr(b)
        assert "lazy=True" in r

    def test_str(self):
        t = Tensor.empty(3)
        assert "Tensor" in str(t)


# ============================================================================
# LazyOp Subclasses
# ============================================================================


class TestBinaryOp:
    def test_realize(self):
        a = Tensor.empty(3, 4)
        b = Tensor.empty(3, 4)
        op = BinaryOp('add', a, b)
        result = op.realize()
        assert result.shape == (3, 4)

    def test_get_shape(self):
        a = Tensor.empty(3, 4)
        b = Tensor.empty(3, 4)
        op = BinaryOp('add', a, b)
        assert op.get_shape() == (3, 4)

    def test_get_dtype(self):
        a = Tensor.empty(3, dtype=DType.float32)
        b = Tensor.empty(3, dtype=DType.float64)
        op = BinaryOp('add', a, b)
        assert op.get_dtype() == DType.float64


class TestUnaryOp:
    def test_realize(self):
        a = Tensor.empty(3, 4)
        op = UnaryOp('neg', a)
        result = op.realize()
        assert result.shape == (3, 4)

    def test_get_shape(self):
        a = Tensor.empty(3, 4)
        op = UnaryOp('neg', a)
        assert op.get_shape() == (3, 4)

    def test_get_dtype(self):
        a = Tensor.empty(3, dtype=DType.float32)
        op = UnaryOp('neg', a)
        assert op.get_dtype() == DType.float32


class TestReduceOp:
    def test_realize(self):
        a = Tensor.empty(3, 4)
        op = ReduceOp('sum', a, None, False)
        result = op.realize()
        assert result.shape == ()

    def test_get_shape_no_dim(self):
        a = Tensor.empty(3, 4)
        op = ReduceOp('sum', a, None, False)
        assert op.get_shape() == ()

    def test_get_shape_no_dim_keepdim(self):
        a = Tensor.empty(3, 4)
        op = ReduceOp('sum', a, None, True)
        assert op.get_shape() == (1, 1)

    def test_get_shape_with_dim(self):
        a = Tensor.empty(3, 4)
        op = ReduceOp('sum', a, (0,), False)
        assert op.get_shape() == (4,)

    def test_get_shape_with_dim_keepdim(self):
        a = Tensor.empty(3, 4)
        op = ReduceOp('sum', a, (0,), True)
        assert op.get_shape() == (1, 4)

    def test_get_dtype_sum_int(self):
        a = Tensor.empty(3, dtype=DType.int32)
        op = ReduceOp('sum', a, None, False)
        assert op.get_dtype() == DType.int64

    def test_get_dtype_mean_int(self):
        a = Tensor.empty(3, dtype=DType.int32)
        op = ReduceOp('mean', a, None, False)
        assert op.get_dtype() == DType.int64

    def test_get_dtype_max_float(self):
        a = Tensor.empty(3, dtype=DType.float32)
        op = ReduceOp('max', a, None, False)
        assert op.get_dtype() == DType.float32


# ============================================================================
# Helper Functions
# ============================================================================


class TestBroadcastShapes:
    def test_same_shape(self):
        assert _broadcast_shapes((3, 4), (3, 4)) == (3, 4)

    def test_broadcast_1(self):
        assert _broadcast_shapes((3, 1), (1, 4)) == (3, 4)

    def test_different_ndim(self):
        assert _broadcast_shapes((4,), (3, 4)) == (3, 4)

    def test_incompatible(self):
        with pytest.raises(ValueError, match="Cannot broadcast"):
            _broadcast_shapes((3,), (4,))


class TestPromoteDtypes:
    def test_float_float(self):
        assert _promote_dtypes(DType.float32, DType.float32) == DType.float32

    def test_float_float64(self):
        assert _promote_dtypes(DType.float32, DType.float64) == DType.float64

    def test_int_int(self):
        assert _promote_dtypes(DType.int32, DType.int32) == DType.int64

    def test_float_int(self):
        assert _promote_dtypes(DType.float32, DType.int32) == DType.float32

    def test_bool_bool(self):
        assert _promote_dtypes(DType.bool_, DType.bool_) == DType.bool_


class TestNullcontext:
    def test_nullcontext(self):
        with _nullcontext() as v:
            assert v is None


# ============================================================================
# Tensor Lazy Evaluation
# ============================================================================


class TestTensorLazyEval:
    def test_realize(self):
        a = Tensor.empty(3, 4)
        b = a + 1
        # b is lazy
        assert b._lazy_op is not None
        b.realize()
        assert b._lazy_op is None

    def test_shape_from_lazy_op(self):
        a = Tensor.empty(3, 4)
        b = a + 1
        assert b.shape == (3, 4)  # from lazy op

    def test_dtype_from_lazy_op(self):
        a = Tensor.empty(3, dtype=DType.float32)
        b = a + 1
        assert b.dtype == DType.float32

    def test_is_contiguous_lazy(self):
        a = Tensor.empty(3)
        b = a + 1
        assert b.is_contiguous is True


# ============================================================================
# Tensor Data Access
# ============================================================================


class TestTensorDataAccess:
    def test_data_ptr(self):
        t = Tensor.empty(3)
        ptr = t.data_ptr
        assert isinstance(ptr, int)

    def test_data_ptr_no_storage(self):
        t = Tensor(storage=None, shape=(3,))
        with pytest.raises(RuntimeError, match="no storage"):
            t.data_ptr

    def test_numpy(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(3, 4, dtype=DType.float32)
        arr = t.numpy()
        assert arr.shape == (3, 4)

    def test_tolist(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(3)
        lst = t.tolist()
        assert isinstance(lst, list)

    def test_item_scalar(self):
        np = pytest.importorskip("numpy")
        t = Tensor.empty(1)
        # item() will work since Tensor.empty(1) has numel=1
        val = t.item()
        assert isinstance(val, (int, float))

    def test_item_non_scalar(self):
        t = Tensor.empty(3)
        with pytest.raises(ValueError, match="item()"):
            t.item()


# ============================================================================
# Tensor Compute Strides
# ============================================================================


class TestComputeStrides:
    def test_empty_shape(self):
        assert Tensor._compute_strides(()) == ()

    def test_1d(self):
        assert Tensor._compute_strides((5,)) == (1,)

    def test_2d(self):
        assert Tensor._compute_strides((3, 4)) == (4, 1)

    def test_3d(self):
        assert Tensor._compute_strides((2, 3, 4)) == (12, 4, 1)
