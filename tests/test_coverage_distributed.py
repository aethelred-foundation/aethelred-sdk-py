"""Comprehensive tests for aethelred/distributed/__init__.py to achieve 95%+ coverage.

Covers:
- Backend enum
- ProcessGroupInfo dataclass
- ProcessGroup: init, factory, P2P, collective ops, helper methods
- Request async handle
- DistributedDataParallel
- GradientBucket
- ZeROStage, ZeROOptimizer
- PipelineStage, PipelineParallel
- ColumnParallelLinear, RowParallelLinear
- GradientCompressor, TopKCompressor, PowerSGDCompressor
- ElasticTrainer
"""

from __future__ import annotations

import os
import pickle
import tempfile
import threading
from collections import defaultdict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aethelred.core.tensor import Tensor, DType
from aethelred.core.runtime import Device, DeviceType, Stream, Event
from aethelred.nn import Module, Parameter, Linear, Sequential
from aethelred.distributed import (
    Backend,
    ProcessGroupInfo,
    ProcessGroup,
    Request,
    DistributedDataParallel,
    GradientBucket,
    ZeROStage,
    ZeROOptimizer,
    PipelineStage,
    PipelineParallel,
    ColumnParallelLinear,
    RowParallelLinear,
    GradientCompressor,
    TopKCompressor,
    PowerSGDCompressor,
    ElasticTrainer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_pg():
    old = ProcessGroup._default_group
    yield
    ProcessGroup._default_group = old


class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# Backend
# ============================================================================


class TestBackend:
    def test_values(self):
        assert Backend.GLOO is not None
        assert Backend.NCCL is not None
        assert Backend.MPI is not None
        assert Backend.AETHELRED is not None


# ============================================================================
# ProcessGroupInfo
# ============================================================================


class TestProcessGroupInfo:
    def test_creation(self):
        info = ProcessGroupInfo(
            rank=0, world_size=4, local_rank=0,
            local_world_size=2, backend=Backend.GLOO
        )
        assert info.rank == 0
        assert info.world_size == 4
        assert info.is_initialized is False


# ============================================================================
# ProcessGroup
# ============================================================================


class TestProcessGroup:
    def test_creation(self):
        ProcessGroup._default_group = None
        pg = ProcessGroup(backend=Backend.GLOO, world_size=1, rank=0)
        assert pg.world_size == 1
        assert pg.rank == 0
        assert pg._initialized is False

    def test_creation_from_env(self):
        with patch.dict(os.environ, {
            'WORLD_SIZE': '2',
            'RANK': '1',
            'LOCAL_RANK': '0',
            'LOCAL_WORLD_SIZE': '2',
        }):
            pg = ProcessGroup()
            assert pg.world_size == 2
            assert pg.rank == 1

    def test_get_default(self):
        ProcessGroup._default_group = None
        pg = ProcessGroup.get_default()
        assert pg is not None
        pg2 = ProcessGroup.get_default()
        assert pg is pg2

    def test_initialize(self):
        pg = ProcessGroup(world_size=1, rank=0)
        pg.initialize()
        assert pg._initialized is True

    def test_initialize_idempotent(self):
        pg = ProcessGroup(world_size=1, rank=0)
        pg.initialize()
        pg.initialize()
        assert pg._initialized is True

    def test_destroy(self):
        pg = ProcessGroup(world_size=1, rank=0)
        pg.initialize()
        pg.destroy()
        assert pg._initialized is False

    def test_send(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        pg.send(t, dst=1)
        assert len(pg._send_buffers[1]) == 1

    def test_recv(self):
        pg = ProcessGroup(world_size=2, rank=1)
        t = Tensor.empty(4)
        pg.recv(t, src=0)  # Should not raise

    def test_isend(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        req = pg.isend(t, dst=1)
        assert isinstance(req, Request)

    def test_irecv(self):
        pg = ProcessGroup(world_size=2, rank=1)
        t = Tensor.empty(4)
        req = pg.irecv(t, src=0)
        assert isinstance(req, Request)

    def test_broadcast_single(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        result = pg.broadcast(t, src=0)
        assert result is None

    def test_broadcast_multi(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.broadcast(t, src=0)
        assert result is None

    def test_broadcast_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.broadcast(t, src=0, async_op=True)
        assert isinstance(result, Request)

    def test_reduce_single(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        result = pg.reduce(t, dst=0)
        assert result is None

    def test_reduce_multi(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.reduce(t, dst=0)
        assert result is None

    def test_reduce_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.reduce(t, dst=0, async_op=True)
        assert isinstance(result, Request)

    def test_all_reduce_single(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        result = pg.all_reduce(t)
        assert result is None

    def test_all_reduce_multi(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.all_reduce(t)
        assert result is None

    def test_all_reduce_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.all_reduce(t, async_op=True)
        assert isinstance(result, Request)

    def test_all_gather(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        out = [Tensor.empty(4)]
        result = pg.all_gather(out, t)
        assert result is None

    def test_all_gather_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        out = [Tensor.empty(4), Tensor.empty(4)]
        result = pg.all_gather(out, t, async_op=True)
        assert isinstance(result, Request)

    def test_gather(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        result = pg.gather(t)
        assert result is None

    def test_gather_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.gather(t, async_op=True)
        assert isinstance(result, Request)

    def test_scatter(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        result = pg.scatter(t)
        assert result is None

    def test_scatter_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.scatter(t, async_op=True)
        assert isinstance(result, Request)

    def test_reduce_scatter(self):
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        result = pg.reduce_scatter(t, [Tensor.empty(4)])
        assert result is None

    def test_reduce_scatter_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        t = Tensor.empty(4)
        result = pg.reduce_scatter(t, [Tensor.empty(4)], async_op=True)
        assert isinstance(result, Request)

    def test_barrier_single(self):
        pg = ProcessGroup(world_size=1, rank=0)
        result = pg.barrier()
        assert result is None

    def test_barrier_multi(self):
        pg = ProcessGroup(world_size=2, rank=0)
        result = pg.barrier()
        assert result is None

    def test_barrier_async(self):
        pg = ProcessGroup(world_size=2, rank=0)
        result = pg.barrier(async_op=True)
        assert isinstance(result, Request)

    def test_serialize_tensor(self):
        np = pytest.importorskip("numpy")
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        data = pg._serialize_tensor(t)
        assert isinstance(data, bytes)

    def test_deserialize_tensor(self):
        np = pytest.importorskip("numpy")
        pg = ProcessGroup(world_size=1, rank=0)
        t = Tensor.empty(4)
        data = pg._serialize_tensor(t)
        t2 = pg._deserialize_tensor(data)
        assert t2 is not None


# ============================================================================
# Request
# ============================================================================


class TestRequest:
    def test_creation(self):
        req = Request()
        assert req.is_completed() is False

    def test_wait_timeout(self):
        req = Request()
        result = req.wait(timeout=0.01)
        assert result is False

    def test_complete(self):
        req = Request()
        req._completed.set()
        assert req.is_completed() is True
        assert req.wait(timeout=1.0) is True


# ============================================================================
# DistributedDataParallel
# ============================================================================


class TestDistributedDataParallel:
    def test_creation(self):
        ProcessGroup._default_group = None
        model = SimpleNet()
        ddp = DistributedDataParallel(model)
        assert ddp.module is model

    def test_creation_with_device_ids(self):
        ProcessGroup._default_group = None
        model = SimpleNet()
        ddp = DistributedDataParallel(model, device_ids=[0])
        assert ddp.device_ids == [0]
        assert ddp.output_device == 0

    def test_forward(self):
        ProcessGroup._default_group = None
        model = SimpleNet()
        ddp = DistributedDataParallel(model)
        x = Tensor.empty(2, 4)
        out = ddp(x)
        assert out is not None

    def test_forward_no_broadcast_buffers(self):
        ProcessGroup._default_group = None
        model = SimpleNet()
        ddp = DistributedDataParallel(model, broadcast_buffers=False)
        x = Tensor.empty(2, 4)
        out = ddp(x)
        assert out is not None

    def test_reduce_gradients(self):
        ProcessGroup._default_group = None
        model = SimpleNet()
        ddp = DistributedDataParallel(model)
        ddp._reduce_gradients()  # Should not raise


# ============================================================================
# GradientBucket
# ============================================================================


class TestGradientBucket:
    def test_creation(self):
        gb = GradientBucket(max_size=1024)
        assert gb.max_size == 1024
        assert len(gb.params) == 0

    def test_add_param(self):
        gb = GradientBucket(max_size=10000)
        p = Parameter(Tensor.empty(4))
        result = gb.add_param(p)
        assert result is True
        assert len(gb.params) == 1

    def test_add_param_full(self):
        gb = GradientBucket(max_size=16)  # 16 bytes
        p1 = Parameter(Tensor.empty(4, dtype=DType.float32))  # 16 bytes
        gb.add_param(p1)
        p2 = Parameter(Tensor.empty(4, dtype=DType.float32))
        result = gb.add_param(p2)
        assert result is False

    def test_all_reduce(self):
        ProcessGroup._default_group = None
        pg = ProcessGroup(world_size=1, rank=0)
        gb = GradientBucket(max_size=10000)
        gb.all_reduce(pg)

    def test_mark_grad_ready(self):
        gb = GradientBucket(max_size=10000)
        p = Parameter(Tensor.empty(4))
        gb.add_param(p)
        done = gb.mark_grad_ready(p)
        assert done is True

    def test_mark_grad_ready_not_done(self):
        gb = GradientBucket(max_size=10000)
        p1 = Parameter(Tensor.empty(4))
        p2 = Parameter(Tensor.empty(4))
        gb.add_param(p1)
        gb.add_param(p2)
        done = gb.mark_grad_ready(p1)
        assert done is False


# ============================================================================
# ZeROStage & ZeROOptimizer
# ============================================================================


class TestZeROStage:
    def test_values(self):
        assert ZeROStage.DISABLED.value == 0
        assert ZeROStage.OPTIMIZER.value == 1
        assert ZeROStage.GRADIENTS.value == 2
        assert ZeROStage.PARAMETERS.value == 3


class TestZeROOptimizer:
    def _make_optimizer(self):
        model = SimpleNet()
        opt = MagicMock()
        opt.param_groups = [{'params': list(model.parameters()), 'lr': 0.01}]
        opt.step = MagicMock(return_value=None)
        opt.zero_grad = MagicMock()
        opt.state_dict = MagicMock(return_value={})
        opt.load_state_dict = MagicMock()
        return model, opt

    def test_creation_stage1(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.OPTIMIZER)
        assert zero.stage == ZeROStage.OPTIMIZER

    def test_creation_stage2(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.GRADIENTS)
        assert zero.stage == ZeROStage.GRADIENTS

    def test_creation_stage3(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.PARAMETERS)
        assert zero.stage == ZeROStage.PARAMETERS

    def test_step_stage1(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.OPTIMIZER)
        zero.step()
        opt.step.assert_called_once()

    def test_step_stage2(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.GRADIENTS)
        zero.step()

    def test_step_stage3(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.PARAMETERS)
        zero.step()

    def test_zero_grad(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.OPTIMIZER)
        zero.zero_grad()
        opt.zero_grad.assert_called_once()

    def test_zero_grad_set_to_none(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.OPTIMIZER)
        zero.zero_grad(set_to_none=True)

    def test_reduce_scatter_gradients(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.GRADIENTS)
        zero._reduce_scatter_gradients()

    def test_all_gather_parameters(self):
        ProcessGroup._default_group = None
        model, opt = self._make_optimizer()
        zero = ZeROOptimizer(opt, model, stage=ZeROStage.PARAMETERS)
        zero._all_gather_parameters()


# ============================================================================
# PipelineStage
# ============================================================================


class TestPipelineStage:
    def test_creation(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=0, num_stages=2)
        assert ps.stage_id == 0
        assert ps.num_stages == 2

    def test_forward(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=0, num_stages=2)
        x = Tensor.empty(2, 4)
        out = ps(x)
        assert out is not None

    def test_send_forward(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=0, num_stages=2)
        t = Tensor.empty(2, 4)
        ps.send_forward(t)  # Should send since stage_id < num_stages - 1

    def test_send_forward_last_stage(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=1, num_stages=2)
        t = Tensor.empty(2, 4)
        ps.send_forward(t)  # Should not send (last stage)

    def test_recv_forward_first_stage(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=0, num_stages=2)
        result = ps.recv_forward((2, 4), DType.float32)
        assert result is None  # First stage doesn't receive

    def test_recv_forward_middle_stage(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=1, num_stages=3)
        result = ps.recv_forward((2, 4), DType.float32)
        assert result is not None

    def test_send_backward(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=1, num_stages=2)
        t = Tensor.empty(2, 4)
        ps.send_backward(t)

    def test_send_backward_first_stage(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=0, num_stages=2)
        t = Tensor.empty(2, 4)
        ps.send_backward(t)  # Should not send

    def test_recv_backward(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=0, num_stages=2)
        result = ps.recv_backward((2, 4), DType.float32)
        assert result is not None

    def test_recv_backward_last_stage(self):
        ProcessGroup._default_group = None
        m = Linear(4, 4)
        ps = PipelineStage(m, stage_id=1, num_stages=2)
        result = ps.recv_backward((2, 4), DType.float32)
        assert result is None


# ============================================================================
# PipelineParallel
# ============================================================================


class TestPipelineParallel:
    def _make_modules(self):
        return [Linear(4, 4), Linear(4, 4)]

    def test_creation_gpipe(self):
        ProcessGroup._default_group = None
        mods = self._make_modules()
        pp = PipelineParallel(mods, num_microbatches=4, schedule='gpipe')
        assert pp.num_stages == 2
        assert pp.schedule == 'gpipe'

    def test_creation_1f1b(self):
        ProcessGroup._default_group = None
        mods = self._make_modules()
        pp = PipelineParallel(mods, num_microbatches=4, schedule='1f1b')
        assert pp.schedule == '1f1b'

    def test_forward_gpipe(self):
        ProcessGroup._default_group = None
        mods = self._make_modules()
        pp = PipelineParallel(mods, num_microbatches=2, schedule='gpipe')
        inputs = [Tensor.empty(2, 4) for _ in range(2)]
        outputs = pp.forward(inputs)
        assert isinstance(outputs, list)

    def test_forward_1f1b(self):
        ProcessGroup._default_group = None
        mods = self._make_modules()
        pp = PipelineParallel(mods, num_microbatches=4, schedule='1f1b')
        inputs = [Tensor.empty(2, 4) for _ in range(4)]
        outputs = pp.forward(inputs)
        assert isinstance(outputs, list)

    def test_forward_unknown_schedule(self):
        ProcessGroup._default_group = None
        mods = self._make_modules()
        pp = PipelineParallel(mods, num_microbatches=2, schedule='unknown')
        with pytest.raises(ValueError, match="Unknown schedule"):
            pp.forward([Tensor.empty(2, 4)])


# ============================================================================
# ColumnParallelLinear
# ============================================================================


class TestColumnParallelLinear:
    def test_creation(self):
        ProcessGroup._default_group = None
        cpl = ColumnParallelLinear(4, 4, bias=True)
        assert cpl.weight is not None

    def test_creation_no_bias(self):
        ProcessGroup._default_group = None
        cpl = ColumnParallelLinear(4, 4, bias=False)
        assert cpl.bias is None

    def test_forward_gather(self):
        ProcessGroup._default_group = None
        cpl = ColumnParallelLinear(4, 4, gather_output=True)
        x = Tensor.empty(2, 4)
        out = cpl(x)
        assert out is not None

    def test_forward_no_gather(self):
        ProcessGroup._default_group = None
        cpl = ColumnParallelLinear(4, 4, gather_output=False)
        x = Tensor.empty(2, 4)
        out = cpl(x)
        assert out is not None


# ============================================================================
# RowParallelLinear
# ============================================================================


class TestRowParallelLinear:
    def test_creation(self):
        ProcessGroup._default_group = None
        rpl = RowParallelLinear(4, 4, bias=True)
        assert rpl.weight is not None

    def test_creation_no_bias(self):
        ProcessGroup._default_group = None
        rpl = RowParallelLinear(4, 4, bias=False)
        assert rpl.bias is None

    def test_forward(self):
        ProcessGroup._default_group = None
        rpl = RowParallelLinear(4, 4)
        x = Tensor.empty(2, 4)
        out = rpl(x)
        assert out is not None

    def test_forward_input_parallel(self):
        ProcessGroup._default_group = None
        rpl = RowParallelLinear(4, 4, input_is_parallel=True)
        x = Tensor.empty(2, 4)
        out = rpl(x)
        assert out is not None


# ============================================================================
# Gradient Compressors
# ============================================================================


class TestTopKCompressor:
    def test_creation(self):
        c = TopKCompressor(ratio=0.01)
        assert c.ratio == 0.01

    def test_compress(self):
        c = TopKCompressor(ratio=0.5)
        t = Tensor.empty(100)
        data, meta = c.compress(t)
        assert isinstance(meta, dict)

    def test_decompress(self):
        c = TopKCompressor(ratio=0.5)
        result = c.decompress(None, {})
        assert result is not None


class TestPowerSGDCompressor:
    def test_creation(self):
        c = PowerSGDCompressor(rank=4, start_iter=10)
        assert c.rank == 4
        assert c.start_iter == 10

    def test_compress_early(self):
        c = PowerSGDCompressor(rank=4, start_iter=5)
        t = Tensor.empty(100)
        data, meta = c.compress(t)
        assert meta.get('compressed') is False
        assert data is t

    def test_compress_after_start(self):
        c = PowerSGDCompressor(rank=4, start_iter=0)
        t = Tensor.empty(100)
        data, meta = c.compress(t)
        assert meta.get('compressed') is True

    def test_decompress_not_compressed(self):
        c = PowerSGDCompressor()
        t = Tensor.empty(100)
        result = c.decompress(t, {'compressed': False})
        assert result is t

    def test_decompress_compressed(self):
        c = PowerSGDCompressor()
        result = c.decompress(None, {'compressed': True})
        assert result is not None


# ============================================================================
# ElasticTrainer
# ============================================================================


class TestElasticTrainer:
    def _make_trainer(self):
        ProcessGroup._default_group = None
        model = SimpleNet()
        opt = MagicMock()
        opt.param_groups = [{'params': list(model.parameters()), 'lr': 0.01}]
        opt.state_dict = MagicMock(return_value={
            'state': {},
            'param_groups': [{'lr': 0.01}]
        })
        opt.load_state_dict = MagicMock()
        return ElasticTrainer(model, opt, min_workers=1, max_workers=8)

    def test_creation(self):
        trainer = self._make_trainer()
        assert trainer.min_workers == 1
        assert trainer.max_workers == 8

    def test_save_checkpoint_rank0(self):
        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            trainer.checkpoint_dir = td
            path = trainer.save_checkpoint(step=10)
            assert path != ''
            assert os.path.exists(path)

    def test_save_checkpoint_not_rank0(self):
        trainer = self._make_trainer()
        trainer.process_group = ProcessGroup(world_size=2, rank=1)
        with tempfile.TemporaryDirectory() as td:
            trainer.checkpoint_dir = td
            path = trainer.save_checkpoint(step=10)
            assert path == ''

    def test_load_checkpoint(self):
        trainer = self._make_trainer()
        with tempfile.TemporaryDirectory() as td:
            trainer.checkpoint_dir = td
            path = trainer.save_checkpoint(step=10)
            step = trainer.load_checkpoint(path)
            assert step == 10

    def test_handle_worker_change_no_change(self):
        trainer = self._make_trainer()
        trainer.handle_worker_change()  # No change

    def test_handle_worker_change(self):
        trainer = self._make_trainer()
        with patch.dict(os.environ, {'WORLD_SIZE': '4'}):
            trainer.handle_worker_change()
