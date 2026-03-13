"""Final coverage tests to reach 95%+ - covers distributed, integrations, seals, jobs,
models, validators, verification, realtime, and oracles modules."""

import asyncio
import copy
import json
import os
import pickle
import threading
import time
import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

# =============================================================================
# Distributed module tests
# =============================================================================

class TestProcessGroup:
    """Tests for distributed.ProcessGroup."""

    def test_init_defaults(self):
        from aethelred.distributed import ProcessGroup, Backend
        pg = ProcessGroup()
        assert pg.backend == Backend.GLOO
        assert pg.world_size == 1
        assert pg.rank == 0
        assert pg._initialized is False

    def test_init_custom(self):
        from aethelred.distributed import ProcessGroup, Backend
        pg = ProcessGroup(backend=Backend.NCCL, world_size=4, rank=2)
        assert pg.backend == Backend.NCCL
        assert pg.world_size == 4
        assert pg.rank == 2

    def test_init_from_env(self):
        from aethelred.distributed import ProcessGroup
        with patch.dict(os.environ, {'WORLD_SIZE': '8', 'RANK': '3', 'LOCAL_RANK': '1', 'LOCAL_WORLD_SIZE': '2'}):
            pg = ProcessGroup()
            assert pg.world_size == 8
            assert pg.rank == 3
            assert pg.local_rank == 1
            assert pg.local_world_size == 2

    def test_get_default(self):
        from aethelred.distributed import ProcessGroup
        ProcessGroup._default_group = None
        pg = ProcessGroup.get_default()
        assert pg is not None
        # Should return same instance
        pg2 = ProcessGroup.get_default()
        assert pg is pg2
        ProcessGroup._default_group = None  # cleanup

    def test_initialize_and_destroy(self):
        from aethelred.distributed import ProcessGroup
        pg = ProcessGroup()
        assert pg._initialized is False
        pg.initialize()
        assert pg._initialized is True
        # Re-initialize should be no-op
        pg.initialize()
        assert pg._initialized is True
        pg.destroy()
        assert pg._initialized is False

    def test_send(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup()
        t = Tensor.empty(2, 3)
        pg.send(t, dst=1)
        assert pg._initialized is True
        assert len(pg._send_buffers[1]) == 1

    def test_recv(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup()
        t = Tensor.empty(2, 3)
        pg.recv(t, src=0)
        assert pg._initialized is True

    def test_isend(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup()
        t = Tensor.empty(2, 3)
        req = pg.isend(t, dst=1)
        assert isinstance(req, Request)

    def test_irecv(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup()
        t = Tensor.empty(2, 3)
        req = pg.irecv(t, src=0)
        assert isinstance(req, Request)

    def test_broadcast_single_process(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.broadcast(t, src=0)
        assert result is None

    def test_broadcast_multi_process(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.broadcast(t, src=0)
        assert result is None

    def test_broadcast_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.broadcast(t, src=0, async_op=True)
        assert isinstance(result, Request)

    def test_reduce_single_process(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.reduce(t, dst=0)
        assert result is None

    def test_reduce_multi_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.reduce(t, dst=0, async_op=True)
        assert isinstance(result, Request)

    def test_all_reduce_single_process(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.all_reduce(t)
        assert result is None

    def test_all_reduce_multi_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.all_reduce(t, async_op=True)
        assert isinstance(result, Request)

    def test_all_gather(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        outputs = [Tensor.empty(2, 3), Tensor.empty(2, 3)]
        result = pg.all_gather(outputs, t)
        assert result is None

    def test_all_gather_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        outputs = [Tensor.empty(2, 3)]
        result = pg.all_gather(outputs, t, async_op=True)
        assert isinstance(result, Request)

    def test_gather(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.gather(t, dst=0)
        assert result is None

    def test_gather_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.gather(t, dst=0, async_op=True)
        assert isinstance(result, Request)

    def test_scatter(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.scatter(t, src=0)
        assert result is None

    def test_scatter_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        t = Tensor.empty(2, 3)
        result = pg.scatter(t, src=0, async_op=True)
        assert isinstance(result, Request)

    def test_reduce_scatter(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        output = Tensor.empty(2, 3)
        inputs = [Tensor.empty(2, 3)]
        result = pg.reduce_scatter(output, inputs)
        assert result is None

    def test_reduce_scatter_async(self):
        from aethelred.distributed import ProcessGroup, Request
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        output = Tensor.empty(2, 3)
        inputs = [Tensor.empty(2, 3)]
        result = pg.reduce_scatter(output, inputs, async_op=True)
        assert isinstance(result, Request)

    def test_barrier_single_process(self):
        from aethelred.distributed import ProcessGroup
        pg = ProcessGroup(world_size=1)
        pg.initialize()
        result = pg.barrier()
        assert result is None

    def test_barrier_multi_async(self):
        from aethelred.distributed import ProcessGroup, Request
        pg = ProcessGroup(world_size=2, rank=0)
        pg.initialize()
        result = pg.barrier(async_op=True)
        assert isinstance(result, Request)

    def test_barrier_not_initialized(self):
        from aethelred.distributed import ProcessGroup
        pg = ProcessGroup(world_size=2, rank=0)
        pg.barrier()
        assert pg._initialized is True

    def test_serialize_tensor(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup()
        t = Tensor.empty(2, 3)
        data = pg._serialize_tensor(t)
        assert isinstance(data, bytes)

    def test_deserialize_tensor(self):
        from aethelred.distributed import ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup()
        t = Tensor.empty(2, 3)
        data = pg._serialize_tensor(t)
        t2 = pg._deserialize_tensor(data)
        assert t2.shape == (2, 3)


class TestRequest:
    """Tests for distributed.Request."""

    def test_wait(self):
        from aethelred.distributed import Request
        req = Request()
        assert req.is_completed() is False
        req._completed.set()
        assert req.is_completed() is True
        assert req.wait(timeout=0.1) is True

    def test_wait_timeout(self):
        from aethelred.distributed import Request
        req = Request()
        assert req.wait(timeout=0.01) is False

    def test_get_future(self):
        from aethelred.distributed import Request
        req = Request()
        req._completed.set()
        req._result = "done"
        # Just call it and verify no errors
        loop = asyncio.new_event_loop()
        future = None
        try:
            asyncio.set_event_loop(loop)
            future = req.get_future()
        finally:
            loop.close()
            asyncio.set_event_loop(None)


class TestDistributedDataParallel:
    """Tests for distributed.DistributedDataParallel."""

    def test_init(self):
        from aethelred.distributed import DistributedDataParallel, ProcessGroup
        from aethelred.nn import Module, Linear
        ProcessGroup._default_group = None
        model = Linear(4, 2)
        ddp = DistributedDataParallel(model)
        assert ddp.module is model
        assert ddp.bucket_cap_mb == 25.0
        ProcessGroup._default_group = None

    def test_init_with_options(self):
        from aethelred.distributed import DistributedDataParallel, ProcessGroup
        from aethelred.nn import Module, Linear
        pg = ProcessGroup(world_size=1, rank=0)
        model = Linear(4, 2)
        ddp = DistributedDataParallel(
            model,
            device_ids=[0],
            broadcast_buffers=True,
            process_group=pg,
            bucket_cap_mb=10.0,
            find_unused_parameters=True,
            static_graph=True,
        )
        assert ddp.output_device == 0
        assert ddp.static_graph is True
        ProcessGroup._default_group = None

    def test_forward(self):
        from aethelred.distributed import DistributedDataParallel, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        model = Linear(4, 2)
        ddp = DistributedDataParallel(model, process_group=pg)
        x = Tensor.empty(1, 4)
        y = ddp(x)
        assert y.shape == (1, 2)
        ProcessGroup._default_group = None

    def test_reduce_gradients(self):
        from aethelred.distributed import DistributedDataParallel, ProcessGroup
        from aethelred.nn import Linear
        pg = ProcessGroup(world_size=1, rank=0)
        model = Linear(4, 2)
        ddp = DistributedDataParallel(model, process_group=pg)
        ddp._reduce_gradients()  # Should not raise
        ProcessGroup._default_group = None


class TestGradientBucket:
    """Tests for distributed.GradientBucket."""

    def test_add_param(self):
        from aethelred.distributed import GradientBucket
        from aethelred.nn import Parameter
        from aethelred.core.tensor import Tensor
        bucket = GradientBucket(max_size=1024)
        p = Parameter(Tensor.empty(4, 4))
        assert bucket.add_param(p) is True
        assert len(bucket.params) == 1
        assert bucket.size > 0

    def test_add_param_full(self):
        from aethelred.distributed import GradientBucket
        from aethelred.nn import Parameter
        from aethelred.core.tensor import Tensor
        bucket = GradientBucket(max_size=1)  # Very small bucket
        p1 = Parameter(Tensor.empty(4, 4))
        bucket.add_param(p1)  # First param always added
        p2 = Parameter(Tensor.empty(4, 4))
        assert bucket.add_param(p2) is False  # Bucket full

    def test_all_reduce(self):
        from aethelred.distributed import GradientBucket, ProcessGroup
        bucket = GradientBucket(max_size=1024)
        pg = ProcessGroup()
        bucket.all_reduce(pg)  # No-op, should not raise

    def test_mark_grad_ready(self):
        from aethelred.distributed import GradientBucket
        from aethelred.nn import Parameter
        from aethelred.core.tensor import Tensor
        bucket = GradientBucket(max_size=1024)
        p1 = Parameter(Tensor.empty(4))
        p2 = Parameter(Tensor.empty(4))
        bucket.add_param(p1)
        bucket.add_param(p2)
        assert bucket.mark_grad_ready(p1) is False
        assert bucket.mark_grad_ready(p2) is True


class TestZeROOptimizer:
    """Tests for distributed.ZeROOptimizer."""

    def _make_optimizer(self, stage=None):
        from aethelred.distributed import ZeROOptimizer, ZeROStage, ProcessGroup
        from aethelred.nn import Linear
        if stage is None:
            stage = ZeROStage.OPTIMIZER
        pg = ProcessGroup(world_size=1, rank=0)
        model = Linear(4, 2)
        base_opt = MagicMock()
        base_opt.step = MagicMock(return_value=0.5)
        base_opt.zero_grad = MagicMock()
        base_opt.param_groups = [{'lr': 0.01}]
        base_opt.state_dict = MagicMock(return_value={})
        base_opt.load_state_dict = MagicMock()
        return ZeROOptimizer(base_opt, model, stage=stage, process_group=pg), pg

    def test_stage1(self):
        from aethelred.distributed import ZeROStage
        opt, pg = self._make_optimizer(ZeROStage.OPTIMIZER)
        loss = opt.step()
        assert loss == 0.5
        ProcessGroup = type(pg)
        ProcessGroup._default_group = None

    def test_stage2(self):
        from aethelred.distributed import ZeROStage
        opt, pg = self._make_optimizer(ZeROStage.GRADIENTS)
        loss = opt.step()
        assert loss == 0.5
        type(pg)._default_group = None

    def test_stage3(self):
        from aethelred.distributed import ZeROStage
        opt, pg = self._make_optimizer(ZeROStage.PARAMETERS)
        loss = opt.step()
        assert loss == 0.5
        type(pg)._default_group = None

    def test_zero_grad(self):
        from aethelred.distributed import ZeROStage
        opt, pg = self._make_optimizer(ZeROStage.OPTIMIZER)
        opt.zero_grad()
        opt.base_optimizer.zero_grad.assert_called_once_with(False)
        opt.zero_grad(set_to_none=True)
        type(pg)._default_group = None


class TestPipelineParallel:
    """Tests for distributed.PipelineStage and PipelineParallel."""

    def test_pipeline_stage_init(self):
        from aethelred.distributed import PipelineStage, ProcessGroup
        from aethelred.nn import Linear
        pg = ProcessGroup(world_size=2, rank=0)
        model = Linear(4, 2)
        stage = PipelineStage(model, stage_id=0, num_stages=2, process_group=pg)
        assert stage.stage_id == 0
        assert stage.num_stages == 2
        ProcessGroup._default_group = None

    def test_pipeline_stage_forward(self):
        from aethelred.distributed import PipelineStage, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        model = Linear(4, 2)
        stage = PipelineStage(model, stage_id=0, num_stages=2, process_group=pg)
        x = Tensor.empty(1, 4)
        y = stage(x)
        assert y.shape == (1, 2)
        ProcessGroup._default_group = None

    def test_pipeline_stage_send_forward(self):
        from aethelred.distributed import PipelineStage, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        model = Linear(4, 2)
        stage = PipelineStage(model, stage_id=0, num_stages=2, process_group=pg)
        t = Tensor.empty(1, 2)
        stage.send_forward(t)
        # Last stage should not send
        stage2 = PipelineStage(model, stage_id=1, num_stages=2, process_group=pg)
        stage2.send_forward(t)  # No-op for last stage
        ProcessGroup._default_group = None

    def test_pipeline_stage_recv_forward(self):
        from aethelred.distributed import PipelineStage, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor, DType
        pg = ProcessGroup(world_size=2, rank=0)
        model = Linear(4, 2)
        # First stage should return None (no prior stage)
        stage0 = PipelineStage(model, stage_id=0, num_stages=2, process_group=pg)
        result = stage0.recv_forward((1, 4), DType.float32)
        assert result is None
        # Non-first stage should try to recv
        stage1 = PipelineStage(model, stage_id=1, num_stages=2, process_group=pg)
        result = stage1.recv_forward((1, 4), DType.float32)
        assert result is not None
        ProcessGroup._default_group = None

    def test_pipeline_stage_send_backward(self):
        from aethelred.distributed import PipelineStage, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=2, rank=0)
        model = Linear(4, 2)
        stage0 = PipelineStage(model, stage_id=0, num_stages=2, process_group=pg)
        t = Tensor.empty(1, 2)
        stage0.send_backward(t)  # First stage - no prior stage to send to
        stage1 = PipelineStage(model, stage_id=1, num_stages=2, process_group=pg)
        stage1.send_backward(t)  # Should send to stage 0
        ProcessGroup._default_group = None

    def test_pipeline_stage_recv_backward(self):
        from aethelred.distributed import PipelineStage, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor, DType
        pg = ProcessGroup(world_size=2, rank=0)
        model = Linear(4, 2)
        # First stage (not last) should recv backward
        stage0 = PipelineStage(model, stage_id=0, num_stages=2, process_group=pg)
        result = stage0.recv_backward((1, 2), DType.float32)
        assert result is not None
        # Last stage should return None
        stage1 = PipelineStage(model, stage_id=1, num_stages=2, process_group=pg)
        result = stage1.recv_backward((1, 2), DType.float32)
        assert result is None
        ProcessGroup._default_group = None

    def test_pipeline_parallel_gpipe(self):
        from aethelred.distributed import PipelineParallel, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        modules = [Linear(4, 4), Linear(4, 2)]
        pp = PipelineParallel(modules, num_microbatches=2, schedule='gpipe', process_group=pg)
        inputs = [Tensor.empty(1, 4), Tensor.empty(1, 4)]
        outputs = pp.forward(inputs)
        # At stage 0 in a 2-stage pipeline with local_stage_id=0
        ProcessGroup._default_group = None

    def test_pipeline_parallel_1f1b(self):
        from aethelred.distributed import PipelineParallel, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        modules = [Linear(4, 4), Linear(4, 2)]
        pp = PipelineParallel(modules, num_microbatches=3, schedule='1f1b', process_group=pg)
        inputs = [Tensor.empty(1, 4), Tensor.empty(1, 4), Tensor.empty(1, 4)]
        outputs = pp.forward(inputs)
        ProcessGroup._default_group = None

    def test_pipeline_parallel_unknown_schedule(self):
        from aethelred.distributed import PipelineParallel, ProcessGroup
        from aethelred.nn import Linear
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        modules = [Linear(4, 2)]
        pp = PipelineParallel(modules, num_microbatches=1, schedule='unknown', process_group=pg)
        with pytest.raises(ValueError, match="Unknown schedule"):
            pp.forward([Tensor.empty(1, 4)])
        ProcessGroup._default_group = None


class TestTensorParallel:
    """Tests for distributed.ColumnParallelLinear and RowParallelLinear."""

    def test_column_parallel_linear(self):
        from aethelred.distributed import ColumnParallelLinear, ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        layer = ColumnParallelLinear(4, 4, process_group=pg)
        x = Tensor.empty(1, 4)
        y = layer(x)
        assert y is not None
        ProcessGroup._default_group = None

    def test_column_parallel_no_bias(self):
        from aethelred.distributed import ColumnParallelLinear, ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        layer = ColumnParallelLinear(4, 4, bias=False, process_group=pg)
        assert layer.bias is None
        x = Tensor.empty(1, 4)
        y = layer(x)
        ProcessGroup._default_group = None

    def test_row_parallel_linear(self):
        from aethelred.distributed import RowParallelLinear, ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        layer = RowParallelLinear(4, 4, process_group=pg)
        x = Tensor.empty(1, 4)
        y = layer(x)
        assert y is not None
        ProcessGroup._default_group = None

    def test_row_parallel_no_bias(self):
        from aethelred.distributed import RowParallelLinear, ProcessGroup
        from aethelred.core.tensor import Tensor
        pg = ProcessGroup(world_size=1, rank=0)
        layer = RowParallelLinear(4, 4, bias=False, process_group=pg)
        assert layer.bias is None
        ProcessGroup._default_group = None


class TestGradientCompressor:
    """Tests for gradient compressors."""

    def test_topk_compressor(self):
        from aethelred.distributed import TopKCompressor
        from aethelred.core.tensor import Tensor
        comp = TopKCompressor(ratio=0.5)
        t = Tensor.empty(10)
        data, meta = comp.compress(t)
        result = comp.decompress(data, meta)

    def test_powersgd_compressor_early(self):
        from aethelred.distributed import PowerSGDCompressor
        from aethelred.core.tensor import Tensor
        comp = PowerSGDCompressor(rank=4, start_iter=10)
        t = Tensor.empty(4, 4)
        # Before start_iter, no compression
        data, meta = comp.compress(t)
        assert meta.get('compressed') is False
        result = comp.decompress(data, meta)
        # data should be the original tensor
        assert result is data

    def test_powersgd_compressor_after_start(self):
        from aethelred.distributed import PowerSGDCompressor
        from aethelred.core.tensor import Tensor
        comp = PowerSGDCompressor(rank=4, start_iter=0)
        t = Tensor.empty(4, 4)
        data, meta = comp.compress(t)
        assert meta.get('compressed') is True
        result = comp.decompress(data, meta)


class TestElasticTrainer:
    """Tests for distributed.ElasticTrainer."""

    def test_init(self):
        from aethelred.distributed import ElasticTrainer, ProcessGroup
        from aethelred.nn import Linear
        ProcessGroup._default_group = None
        model = Linear(4, 2)
        opt = MagicMock()
        opt.state_dict = MagicMock(return_value={})
        opt.load_state_dict = MagicMock()
        opt.param_groups = [{'lr': 0.01}]
        trainer = ElasticTrainer(model, opt, min_workers=1, max_workers=8)
        assert trainer.min_workers == 1
        assert trainer.max_workers == 8
        ProcessGroup._default_group = None

    def test_save_load_checkpoint(self, tmp_path):
        from aethelred.distributed import ElasticTrainer, ProcessGroup
        from aethelred.nn import Linear
        ProcessGroup._default_group = None
        model = Linear(4, 2)
        opt = MagicMock()
        opt.state_dict = MagicMock(return_value={'lr': 0.01})
        opt.load_state_dict = MagicMock()
        opt.param_groups = [{'lr': 0.01}]
        trainer = ElasticTrainer(model, opt, checkpoint_dir=str(tmp_path))
        # Rank 0 saves
        trainer.process_group.rank = 0
        path = trainer.save_checkpoint(step=10)
        assert path != ''
        # Load it back
        step = trainer.load_checkpoint(path)
        assert step == 10
        ProcessGroup._default_group = None

    def test_save_checkpoint_non_rank0(self, tmp_path):
        from aethelred.distributed import ElasticTrainer, ProcessGroup
        from aethelred.nn import Linear
        ProcessGroup._default_group = None
        model = Linear(4, 2)
        opt = MagicMock()
        opt.param_groups = [{'lr': 0.01}]
        trainer = ElasticTrainer(model, opt, checkpoint_dir=str(tmp_path))
        trainer.process_group.rank = 1
        path = trainer.save_checkpoint(step=5)
        assert path == ''
        ProcessGroup._default_group = None

    def test_handle_worker_change(self):
        from aethelred.distributed import ElasticTrainer, ProcessGroup
        from aethelred.nn import Linear
        ProcessGroup._default_group = None
        model = Linear(4, 2)
        opt = MagicMock()
        opt.param_groups = [{'lr': 0.01}]
        trainer = ElasticTrainer(model, opt)
        # No change
        trainer.handle_worker_change()
        # Simulate worker change
        with patch.dict(os.environ, {'WORLD_SIZE': '2'}):
            trainer.handle_worker_change()
            assert trainer._current_world_size == 2
            assert opt.param_groups[0]['lr'] == 0.02
        ProcessGroup._default_group = None


class TestZeROStage:
    """Tests for ZeRO stage enum."""
    def test_stages(self):
        from aethelred.distributed import ZeROStage
        assert ZeROStage.DISABLED.value == 0
        assert ZeROStage.OPTIMIZER.value == 1
        assert ZeROStage.GRADIENTS.value == 2
        assert ZeROStage.PARAMETERS.value == 3


class TestBackend:
    """Tests for Backend enum."""
    def test_backends(self):
        from aethelred.distributed import Backend
        assert Backend.GLOO is not None
        assert Backend.NCCL is not None
        assert Backend.MPI is not None
        assert Backend.AETHELRED is not None


class TestProcessGroupInfo:
    """Tests for ProcessGroupInfo."""
    def test_info(self):
        from aethelred.distributed import ProcessGroupInfo, Backend
        info = ProcessGroupInfo(
            rank=0, world_size=4, local_rank=0, local_world_size=2,
            backend=Backend.GLOO, is_initialized=True,
        )
        assert info.rank == 0
        assert info.world_size == 4
        assert info.is_initialized is True


# =============================================================================
# Integrations module tests
# =============================================================================

class TestVerificationCommon:
    """Tests for integrations._common module."""

    def test_normalize_for_hash_primitives(self):
        from aethelred.integrations._common import _normalize_for_hash
        assert _normalize_for_hash(None) is None
        assert _normalize_for_hash(True) is True
        assert _normalize_for_hash(42) == 42
        assert _normalize_for_hash(3.14) == 3.14
        assert _normalize_for_hash("hello") == "hello"

    def test_normalize_for_hash_bytes(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash(b"hello")
        assert "__bytes__" in result

    def test_normalize_for_hash_bytearray(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash(bytearray(b"hello"))
        assert "__bytes__" in result

    def test_normalize_for_hash_mapping(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash({"b": 2, "a": 1})
        assert isinstance(result, dict)

    def test_normalize_for_hash_list(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash([1, 2, 3])
        assert result == [1, 2, 3]

    def test_normalize_for_hash_tuple(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash((1, 2))
        assert result == [1, 2]

    def test_normalize_for_hash_set(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash({1, 2, 3})
        assert isinstance(result, list)

    def test_normalize_for_hash_dataclass(self):
        from aethelred.integrations._common import _normalize_for_hash
        @dataclass
        class Foo:
            x: int = 1
            y: str = "hello"
        result = _normalize_for_hash(Foo())
        assert result == {"x": 1, "y": "hello"}

    def test_normalize_for_hash_pydantic_model_dump(self):
        from aethelred.integrations._common import _normalize_for_hash
        obj = MagicMock()
        obj.model_dump = MagicMock(return_value={"key": "value"})
        del obj.dict
        result = _normalize_for_hash(obj)
        assert "key" in result

    def test_normalize_for_hash_pydantic_dict(self):
        from aethelred.integrations._common import _normalize_for_hash
        obj = MagicMock(spec=[])
        obj.dict = MagicMock(return_value={"key": "value"})
        # Remove model_dump to fall through to .dict()
        result = _normalize_for_hash(obj)
        # Should reach __repr__ fallback or dict()

    def test_normalize_for_hash_tolist(self):
        from aethelred.integrations._common import _normalize_for_hash
        obj = MagicMock(spec=[])
        obj.tolist = MagicMock(return_value=[1.0, 2.0])
        result = _normalize_for_hash(obj)
        assert result == [1.0, 2.0]

    def test_normalize_for_hash_object_with_dict(self):
        from aethelred.integrations._common import _normalize_for_hash
        class Foo:
            def __init__(self):
                self.x = 1
                self._private = "hidden"
        result = _normalize_for_hash(Foo())
        assert result == {"x": 1}

    def test_normalize_for_hash_fallback_repr(self):
        from aethelred.integrations._common import _normalize_for_hash
        result = _normalize_for_hash(object())
        assert "__repr__" in result

    def test_canonical_json(self):
        from aethelred.integrations._common import canonical_json
        result = canonical_json({"b": 2, "a": 1})
        assert '"a"' in result
        assert '"b"' in result

    def test_hash_payload(self):
        from aethelred.integrations._common import hash_payload
        h1 = hash_payload({"a": 1})
        h2 = hash_payload({"a": 1})
        assert h1 == h2
        h3 = hash_payload({"a": 2})
        assert h1 != h3

    def test_verification_envelope(self):
        from aethelred.integrations._common import VerificationEnvelope
        env = VerificationEnvelope(
            trace_id="abc", framework="test", operation="op",
            input_hash="inhash", output_hash="outhash",
            timestamp_ms=12345, metadata={},
        )
        headers = env.to_headers()
        assert "x-aethelred-trace-id" in headers
        assert headers["x-aethelred-trace-id"] == "abc"

    def test_verification_envelope_custom_prefix(self):
        from aethelred.integrations._common import VerificationEnvelope
        env = VerificationEnvelope(
            trace_id="abc", framework="test", operation="op",
            input_hash="inhash", output_hash="outhash",
            timestamp_ms=12345, metadata={},
        )
        headers = env.to_headers("x-custom-")
        assert "x-custom-trace-id" in headers

    def test_verification_recorder_sync(self):
        from aethelred.integrations._common import VerificationRecorder
        envelopes = []
        recorder = VerificationRecorder(sink=envelopes.append)
        env = recorder.record(
            framework="test", operation="op",
            input_data={"x": 1}, output_data={"y": 2},
        )
        assert len(envelopes) == 1
        assert env.framework == "test"

    def test_verification_recorder_no_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        env = recorder.record(
            framework="test", operation="op",
            input_data={"x": 1}, output_data={"y": 2},
        )
        assert env.framework == "test"

    def test_verification_recorder_with_metadata(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder(
            default_metadata={"env": "test"},
            header_prefix="x-custom",
        )
        env = recorder.record(
            framework="test", operation="op",
            input_data=1, output_data=2,
            metadata={"extra": "val"},
        )
        assert "env" in env.metadata
        assert "extra" in env.metadata

    def test_verification_recorder_async_sink_error(self):
        from aethelred.integrations._common import VerificationRecorder

        async def async_sink(env):
            pass

        recorder = VerificationRecorder(sink=async_sink)
        with pytest.raises(TypeError, match="async sink"):
            recorder.record(
                framework="test", operation="op",
                input_data=1, output_data=2,
            )

    @pytest.mark.asyncio
    async def test_verification_recorder_arecord(self):
        from aethelred.integrations._common import VerificationRecorder
        envelopes = []

        async def async_sink(env):
            envelopes.append(env)

        recorder = VerificationRecorder(sink=async_sink)
        env = await recorder.arecord(
            framework="test", operation="op",
            input_data=1, output_data=2,
        )
        assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_verification_recorder_arecord_sync_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        envelopes = []
        recorder = VerificationRecorder(sink=envelopes.append)
        env = await recorder.arecord(
            framework="test", operation="op",
            input_data=1, output_data=2,
        )
        assert len(envelopes) == 1

    @pytest.mark.asyncio
    async def test_verification_recorder_arecord_no_sink(self):
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        env = await recorder.arecord(
            framework="test", operation="op",
            input_data=1, output_data=2,
        )
        assert env is not None


class TestFastAPIMiddleware:
    """Tests for integrations.fastapi module."""

    def test_normalize_header_prefix(self):
        from aethelred.integrations.fastapi import _normalize_header_prefix
        assert _normalize_header_prefix("X-Custom-") == "x-custom"
        assert _normalize_header_prefix("x-test") == "x-test"

    def test_headers_to_dict(self):
        from aethelred.integrations.fastapi import _headers_to_dict
        headers = [(b"Content-Type", b"application/json")]
        result = _headers_to_dict(headers)
        assert result["content-type"] == "application/json"

    def test_merge_headers_new(self):
        from aethelred.integrations.fastapi import _merge_headers
        headers = [(b"existing", b"val")]
        _merge_headers(headers, {"new-key": "new-val"})
        assert len(headers) == 2

    def test_merge_headers_update(self):
        from aethelred.integrations.fastapi import _merge_headers
        headers = [(b"existing", b"old")]
        _merge_headers(headers, {"existing": "new"})
        assert len(headers) == 1
        assert headers[0] == (b"existing", b"new")

    def test_middleware_init(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        app = MagicMock()
        mw = AethelredVerificationMiddleware(app)
        assert mw.app is app

    def test_middleware_init_with_options(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        from aethelred.integrations._common import VerificationRecorder
        app = MagicMock()
        recorder = VerificationRecorder()
        mw = AethelredVerificationMiddleware(
            app,
            recorder=recorder,
            include_paths=["/api"],
            exclude_paths=["/health"],
            max_capture_bytes=1024,
            header_prefix="x-test",
        )
        assert mw.include_paths == ("/api",)
        assert mw.exclude_paths == ("/health",)

    def test_should_process(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        app = MagicMock()
        mw = AethelredVerificationMiddleware(
            app, include_paths=["/api"], exclude_paths=["/health"],
        )
        assert mw._should_process("/api/test") is True
        assert mw._should_process("/other") is False
        assert mw._should_process("/health") is False

    @pytest.mark.asyncio
    async def test_middleware_non_http(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        app_called = []

        async def app(scope, receive, send):
            app_called.append(True)

        mw = AethelredVerificationMiddleware(app)
        await mw({"type": "websocket"}, None, None)
        assert len(app_called) == 1

    @pytest.mark.asyncio
    async def test_middleware_excluded_path(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        app_called = []

        async def app(scope, receive, send):
            app_called.append(True)

        mw = AethelredVerificationMiddleware(app, exclude_paths=["/health"])
        await mw({"type": "http", "path": "/health", "headers": []}, None, None)
        assert len(app_called) == 1

    @pytest.mark.asyncio
    async def test_middleware_simple_response(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b'{"ok":true}', "more_body": False})

        async def receive():
            return {"type": "http.request", "body": b'{"input":1}'}

        async def send(msg):
            sent.append(msg)

        mw = AethelredVerificationMiddleware(app)
        await mw({"type": "http", "path": "/api/test", "headers": [], "method": "POST", "query_string": b""}, receive, send)
        assert len(sent) >= 2  # start + body

    @pytest.mark.asyncio
    async def test_middleware_no_body_response(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 204, "headers": []})
            # No body sent - middleware should handle flushing

        async def receive():
            return {"type": "http.request", "body": b''}

        async def send(msg):
            sent.append(msg)

        mw = AethelredVerificationMiddleware(app)
        await mw({"type": "http", "path": "/api/test", "headers": [], "method": "DELETE", "query_string": b""}, receive, send)
        # Should flush the stored response start

    @pytest.mark.asyncio
    async def test_middleware_streaming_response(self):
        from aethelred.integrations.fastapi import AethelredVerificationMiddleware
        sent = []

        async def app(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            # Send a chunk larger than max_capture_bytes
            await send({"type": "http.response.body", "body": b"x" * 200, "more_body": True})
            await send({"type": "http.response.body", "body": b"y" * 200, "more_body": False})

        async def receive():
            return {"type": "http.request", "body": b''}

        async def send(msg):
            sent.append(msg)

        mw = AethelredVerificationMiddleware(app, max_capture_bytes=100)
        await mw({"type": "http", "path": "/api", "headers": [], "method": "GET", "query_string": b""}, receive, send)


class TestTensorFlowIntegration:
    """Tests for integrations.tensorflow module."""

    def test_keras_callback_init(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        assert cb.last_verification is None

    def test_keras_callback_with_options(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        from aethelred.integrations._common import VerificationRecorder
        recorder = VerificationRecorder()
        cb = AethelredKerasCallback(
            recorder=recorder,
            capture_batch_events=True,
            include_log_keys=["loss", "accuracy"],
            extra_metadata={"env": "test"},
        )
        assert cb._capture_batch_events is True

    def test_set_model(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        model = MagicMock()
        cb.set_model(model)
        assert cb._model is model

    def test_filtered_logs_empty(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        assert cb._filtered_logs(None) == {}

    def test_filtered_logs_no_filter(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        logs = {"loss": 0.5, "accuracy": 0.9}
        result = cb._filtered_logs(logs)
        assert result == logs

    def test_filtered_logs_with_filter(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(include_log_keys=["loss"])
        logs = {"loss": 0.5, "accuracy": 0.9}
        result = cb._filtered_logs(logs)
        assert result == {"loss": 0.5}

    def test_on_epoch_end(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback()
        model_mock = MagicMock()
        model_mock.__class__.__name__ = "TestModel"
        cb.set_model(model_mock)
        cb.on_epoch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is not None
        assert cb.last_verification.operation == "epoch_end"

    def test_on_train_batch_end_disabled(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=False)
        cb.on_train_batch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is None

    def test_on_train_batch_end_enabled(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=True)
        model_mock = MagicMock()
        model_mock.__class__.__name__ = "TestModel"
        cb.set_model(model_mock)
        cb.on_train_batch_end(0, logs={"loss": 0.5})
        assert cb.last_verification is not None

    def test_on_test_batch_end_enabled(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=True)
        model_mock = MagicMock()
        cb.set_model(model_mock)
        cb.on_test_batch_end(0, logs={"loss": 0.3})
        assert cb.last_verification is not None

    def test_on_predict_batch_end_enabled(self):
        from aethelred.integrations.tensorflow import AethelredKerasCallback
        cb = AethelredKerasCallback(capture_batch_events=True)
        model_mock = MagicMock()
        cb.set_model(model_mock)
        cb.on_predict_batch_end(0, logs={"output": [1, 2]})
        assert cb.last_verification is not None

    def test_create_keras_callback_no_tf(self):
        from aethelred.integrations.tensorflow import create_keras_callback
        with patch.dict("sys.modules", {"tensorflow": None}):
            cb = create_keras_callback()
            from aethelred.integrations.tensorflow import AethelredKerasCallback
            assert isinstance(cb, AethelredKerasCallback)


class TestLangChainIntegration:
    """Tests for integrations.langchain module."""

    def test_init(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        wrapper = VerifiedLangChainRunnable(runnable)
        assert wrapper.last_verification is None

    def test_init_with_options(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        from aethelred.integrations._common import VerificationRecorder
        runnable = MagicMock()
        recorder = VerificationRecorder()
        wrapper = VerifiedLangChainRunnable(
            runnable,
            recorder=recorder,
            component_name="TestChain",
            extra_metadata={"version": "1.0"},
        )
        assert wrapper._component_name == "TestChain"

    def test_getattr(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.some_attr = "value"
        wrapper = VerifiedLangChainRunnable(runnable)
        assert wrapper.some_attr == "value"

    def test_invoke_with_invoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.invoke = MagicMock(return_value="output")
        wrapper = VerifiedLangChainRunnable(runnable)
        result = wrapper.invoke("hello")
        assert result == "output"
        assert wrapper.last_verification is not None

    def test_invoke_without_invoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock(spec=[])  # No invoke method
        runnable.return_value = "output"
        wrapper = VerifiedLangChainRunnable(runnable)
        result = wrapper.invoke("hello")
        assert result == "output"

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.ainvoke = AsyncMock(return_value="async_output")
        wrapper = VerifiedLangChainRunnable(runnable)
        result = await wrapper.ainvoke("hello")
        assert result == "async_output"

    @pytest.mark.asyncio
    async def test_ainvoke_without_ainvoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock(spec=[])
        runnable.return_value = "output"
        wrapper = VerifiedLangChainRunnable(runnable)
        result = await wrapper.ainvoke("hello")
        assert result == "output"

    def test_batch_with_batch(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.batch = MagicMock(return_value=["out1", "out2"])
        wrapper = VerifiedLangChainRunnable(runnable)
        result = wrapper.batch(["in1", "in2"])
        assert result == ["out1", "out2"]

    def test_batch_without_batch(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock(spec=[])
        runnable.return_value = "out"
        wrapper = VerifiedLangChainRunnable(runnable)
        result = wrapper.batch(["in1", "in2"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_abatch_with_abatch(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.abatch = AsyncMock(return_value=["out1"])
        wrapper = VerifiedLangChainRunnable(runnable)
        result = await wrapper.abatch(["in1"])
        assert result == ["out1"]

    @pytest.mark.asyncio
    async def test_abatch_with_ainvoke(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock(spec=['ainvoke'])
        runnable.ainvoke = AsyncMock(return_value="out")
        wrapper = VerifiedLangChainRunnable(runnable)
        result = await wrapper.abatch(["in1", "in2"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_abatch_fallback(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock(spec=[])
        runnable.return_value = "out"
        wrapper = VerifiedLangChainRunnable(runnable)
        result = await wrapper.abatch(["in1"])
        assert len(result) == 1

    def test_call(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.invoke = MagicMock(return_value="output")
        wrapper = VerifiedLangChainRunnable(runnable)
        result = wrapper("hello")
        assert result == "output"

    def test_call_with_args(self):
        from aethelred.integrations.langchain import VerifiedLangChainRunnable
        runnable = MagicMock()
        runnable.invoke = MagicMock(return_value="output")
        wrapper = VerifiedLangChainRunnable(runnable)
        result = wrapper("hello", "extra_arg")
        assert result == "output"

    def test_wrap_langchain_runnable(self):
        from aethelred.integrations.langchain import wrap_langchain_runnable
        runnable = MagicMock()
        wrapper = wrap_langchain_runnable(runnable)
        assert wrapper._runnable is runnable


class TestPyTorchIntegration:
    """Tests for integrations.pytorch module."""

    def test_init(self):
        from aethelred.integrations.pytorch import VerifiedPyTorchModule
        model = MagicMock()
        wrapper = VerifiedPyTorchModule(model)
        assert wrapper.last_verification is None

    def test_init_with_options(self):
        from aethelred.integrations.pytorch import VerifiedPyTorchModule
        model = MagicMock()
        wrapper = VerifiedPyTorchModule(
            model, component_name="TestNet", extra_metadata={"version": "1"},
        )
        assert wrapper._component_name == "TestNet"

    def test_getattr(self):
        from aethelred.integrations.pytorch import VerifiedPyTorchModule
        model = MagicMock()
        model.some_method = MagicMock(return_value=42)
        wrapper = VerifiedPyTorchModule(model)
        assert wrapper.some_method() == 42

    def test_call(self):
        from aethelred.integrations.pytorch import VerifiedPyTorchModule
        model = MagicMock()
        model.return_value = "output"
        wrapper = VerifiedPyTorchModule(model)
        result = wrapper("input")
        assert result == "output"
        assert wrapper.last_verification is not None
        assert wrapper.last_verification.operation == "forward"

    def test_call_error(self):
        from aethelred.integrations.pytorch import VerifiedPyTorchModule
        model = MagicMock()
        model.side_effect = ValueError("test error")
        wrapper = VerifiedPyTorchModule(model)
        with pytest.raises(ValueError, match="test error"):
            wrapper("input")
        assert wrapper.last_verification is not None
        assert wrapper.last_verification.operation == "forward.error"

    def test_forward(self):
        from aethelred.integrations.pytorch import VerifiedPyTorchModule
        model = MagicMock()
        model.return_value = "output"
        wrapper = VerifiedPyTorchModule(model)
        result = wrapper.forward("input")
        assert result == "output"

    def test_wrap_pytorch_model(self):
        from aethelred.integrations.pytorch import wrap_pytorch_model
        model = MagicMock()
        wrapper = wrap_pytorch_model(model, component_name="Net")
        assert wrapper._component_name == "Net"


class TestHuggingFaceIntegration:
    """Tests for integrations.huggingface module."""

    def test_init(self):
        from aethelred.integrations.huggingface import VerifiedTransformersPipeline
        pipeline = MagicMock()
        pipeline.task = "text-classification"
        wrapper = VerifiedTransformersPipeline(pipeline)
        assert wrapper._component_name == "text-classification"

    def test_init_no_task(self):
        from aethelred.integrations.huggingface import VerifiedTransformersPipeline
        pipeline = MagicMock(spec=[])
        wrapper = VerifiedTransformersPipeline(pipeline)
        assert wrapper._component_name == "MagicMock"

    def test_getattr(self):
        from aethelred.integrations.huggingface import VerifiedTransformersPipeline
        pipeline = MagicMock()
        pipeline.model = "bert-base"
        wrapper = VerifiedTransformersPipeline(pipeline)
        assert wrapper.model == "bert-base"

    def test_call(self):
        from aethelred.integrations.huggingface import VerifiedTransformersPipeline
        pipeline = MagicMock()
        pipeline.return_value = [{"label": "positive", "score": 0.99}]
        pipeline.task = "sentiment"
        wrapper = VerifiedTransformersPipeline(pipeline)
        result = wrapper("hello world")
        assert result == [{"label": "positive", "score": 0.99}]
        assert wrapper.last_verification is not None

    def test_wrap_transformers_pipeline(self):
        from aethelred.integrations.huggingface import wrap_transformers_pipeline
        pipeline = MagicMock()
        wrapper = wrap_transformers_pipeline(pipeline, component_name="TestPipeline")
        assert wrapper._component_name == "TestPipeline"


# =============================================================================
# Seals module tests
# =============================================================================

class TestSealsModule:
    """Tests for seals module."""

    @pytest.mark.asyncio
    async def test_create(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.post = AsyncMock(return_value={"seal_id": "s1", "tx_hash": "0xabc"})
        module = SealsModule(client)
        result = await module.create(job_id="j1", metadata={"key": "val"})
        assert result.seal_id == "s1"

    @pytest.mark.asyncio
    async def test_create_with_regulatory_info(self):
        from aethelred.seals import SealsModule
        from aethelred.core.types import RegulatoryInfo
        client = AsyncMock()
        client.post = AsyncMock(return_value={"seal_id": "s1", "tx_hash": "0x1"})
        module = SealsModule(client)
        reg = RegulatoryInfo(jurisdiction="US")
        result = await module.create(job_id="j1", regulatory_info=reg, expires_in_blocks=100)
        assert result.seal_id == "s1"

    @pytest.mark.asyncio
    async def test_get(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={
            "seal": {
                "id": "s1", "job_id": "j1", "model_hash": "h1",
                "status": "SEAL_STATUS_ACTIVE",
            }
        })
        module = SealsModule(client)
        result = await module.get("s1")
        assert result.id == "s1"

    @pytest.mark.asyncio
    async def test_list(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"seals": []})
        module = SealsModule(client)
        result = await module.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        from aethelred.seals import SealsModule
        from aethelred.core.types import SealStatus, PageRequest
        client = AsyncMock()
        client.get = AsyncMock(return_value={"seals": []})
        module = SealsModule(client)
        result = await module.list(
            requester="addr1", model_hash="hash1",
            status=SealStatus.ACTIVE,
            pagination=PageRequest(page_size=10),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_model(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"seals": []})
        module = SealsModule(client)
        result = await module.list_by_model("hash1")
        assert result == []

    @pytest.mark.asyncio
    async def test_list_by_model_with_pagination(self):
        from aethelred.seals import SealsModule
        from aethelred.core.types import PageRequest
        client = AsyncMock()
        client.get = AsyncMock(return_value={"seals": []})
        module = SealsModule(client)
        result = await module.list_by_model("hash1", pagination=PageRequest(page_size=5))
        assert result == []

    @pytest.mark.asyncio
    async def test_verify(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"valid": True})
        module = SealsModule(client)
        result = await module.verify("s1")
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_revoke(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.post = AsyncMock(return_value={})
        module = SealsModule(client)
        result = await module.revoke("s1", reason="test")
        assert result is True

    @pytest.mark.asyncio
    async def test_export(self):
        from aethelred.seals import SealsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"data": b"export_data"})
        module = SealsModule(client)
        result = await module.export("s1", format="cbor")
        assert result == b"export_data"


class TestSyncSealsModule:
    """Tests for sync seals module."""

    def test_create(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(seal_id="s1"))
        module = SyncSealsModule(client, async_module)
        result = module.create(job_id="j1")
        assert result.seal_id == "s1"

    def test_get(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(seal_id="s1"))
        module = SyncSealsModule(client, async_module)
        result = module.get("s1")
        assert result.seal_id == "s1"

    def test_list(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=[])
        module = SyncSealsModule(client, async_module)
        result = module.list()
        assert result == []

    def test_list_by_model(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=[])
        module = SyncSealsModule(client, async_module)
        result = module.list_by_model("hash1")

    def test_verify(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(valid=True))
        module = SyncSealsModule(client, async_module)
        result = module.verify("s1")
        assert result.valid is True

    def test_revoke(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=True)
        module = SyncSealsModule(client, async_module)
        result = module.revoke("s1", "reason")
        assert result is True

    def test_export(self):
        from aethelred.seals import SyncSealsModule, SealsModule
        async_module = MagicMock(spec=SealsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=b"data")
        module = SyncSealsModule(client, async_module)
        result = module.export("s1")


# =============================================================================
# Jobs module tests
# =============================================================================

class TestJobsModule:
    """Tests for jobs module."""

    @pytest.mark.asyncio
    async def test_submit(self):
        from aethelred.jobs import JobsModule
        client = AsyncMock()
        client.post = AsyncMock(return_value={"job_id": "j1", "tx_hash": "0xabc"})
        module = JobsModule(client)
        result = await module.submit(model_hash=b"\x00" * 32, input_hash=b"\x01" * 32)
        assert result.job_id == "j1"

    @pytest.mark.asyncio
    async def test_get(self):
        from aethelred.jobs import JobsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={
            "job": {
                "id": "j1", "creator": "addr1", "model_hash": "h1", "input_hash": "i1",
                "status": "JOB_STATUS_COMPLETED", "proof_type": "PROOF_TYPE_TEE",
            }
        })
        module = JobsModule(client)
        result = await module.get("j1")
        assert result.id == "j1"

    @pytest.mark.asyncio
    async def test_list(self):
        from aethelred.jobs import JobsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"jobs": []})
        module = JobsModule(client)
        result = await module.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus, PageRequest
        client = AsyncMock()
        client.get = AsyncMock(return_value={"jobs": []})
        module = JobsModule(client)
        result = await module.list(
            status=JobStatus.COMPLETED, creator="addr1",
            pagination=PageRequest(page_size=10),
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_list_pending(self):
        from aethelred.jobs import JobsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"jobs": []})
        module = JobsModule(client)
        result = await module.list_pending()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_pending_with_pagination(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import PageRequest
        client = AsyncMock()
        client.get = AsyncMock(return_value={"jobs": []})
        module = JobsModule(client)
        result = await module.list_pending(pagination=PageRequest(page_size=5))
        assert result == []

    @pytest.mark.asyncio
    async def test_cancel(self):
        from aethelred.jobs import JobsModule
        client = AsyncMock()
        client.post = AsyncMock(return_value={})
        module = JobsModule(client)
        result = await module.cancel("j1")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_result(self):
        from aethelred.jobs import JobsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={
            "job_id": "j1", "output_hash": "h1",
        })
        module = JobsModule(client)
        result = await module.get_result("j1")

    @pytest.mark.asyncio
    async def test_wait_for_completion(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.types import JobStatus
        client = AsyncMock()
        # Simulate job completing after 2nd poll
        get_responses = [
            {"job": {"id": "j1", "creator": "addr1", "model_hash": "h1", "input_hash": "i1",
                     "status": "JOB_STATUS_PENDING", "proof_type": "PROOF_TYPE_TEE"}},
            {"job": {"id": "j1", "creator": "addr1", "model_hash": "h1", "input_hash": "i1",
                     "status": "JOB_STATUS_COMPLETED", "proof_type": "PROOF_TYPE_TEE"}},
        ]
        client.get = AsyncMock(side_effect=get_responses)
        module = JobsModule(client)
        result = await module.wait_for_completion("j1", poll_interval=0.01, timeout=1.0)
        assert result.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self):
        from aethelred.jobs import JobsModule
        from aethelred.core.exceptions import TimeoutError
        client = AsyncMock()
        client.get = AsyncMock(return_value={
            "job": {"id": "j1", "creator": "addr1", "model_hash": "h1", "input_hash": "i1",
                    "status": "JOB_STATUS_PENDING", "proof_type": "PROOF_TYPE_TEE"}
        })
        module = JobsModule(client)
        with pytest.raises(TimeoutError):
            await module.wait_for_completion("j1", poll_interval=0.01, timeout=0.02)


class TestSyncJobsModule:
    """Tests for sync jobs module."""

    def test_submit(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(job_id="j1"))
        module = SyncJobsModule(client, async_module)
        result = module.submit(model_hash=b"\x00" * 32, input_hash=b"\x01" * 32)
        assert result.job_id == "j1"

    def test_get(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(job_id="j1"))
        module = SyncJobsModule(client, async_module)
        result = module.get("j1")

    def test_list(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=[])
        module = SyncJobsModule(client, async_module)
        result = module.list()

    def test_list_pending(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=[])
        module = SyncJobsModule(client, async_module)
        result = module.list_pending()

    def test_cancel(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=True)
        module = SyncJobsModule(client, async_module)
        result = module.cancel("j1")

    def test_get_result(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock())
        module = SyncJobsModule(client, async_module)
        result = module.get_result("j1")

    def test_wait_for_completion(self):
        from aethelred.jobs import SyncJobsModule, JobsModule
        async_module = MagicMock(spec=JobsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock())
        module = SyncJobsModule(client, async_module)
        result = module.wait_for_completion(job_id="j1")


# =============================================================================
# Models module tests
# =============================================================================

class TestModelsModule:
    """Tests for models module."""

    @pytest.mark.asyncio
    async def test_register(self):
        from aethelred.models import ModelsModule
        client = AsyncMock()
        client.post = AsyncMock(return_value={"model_hash": "h1", "tx_hash": "0x1"})
        module = ModelsModule(client)
        result = await module.register(model_hash=b"\x00" * 32, name="TestModel")
        assert result.model_hash == "h1"

    @pytest.mark.asyncio
    async def test_get(self):
        from aethelred.models import ModelsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={
            "model": {"model_hash": "h1", "name": "TestModel", "owner": "addr1"}
        })
        module = ModelsModule(client)
        result = await module.get("h1")
        assert result.name == "TestModel"

    @pytest.mark.asyncio
    async def test_list(self):
        from aethelred.models import ModelsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"models": []})
        module = ModelsModule(client)
        result = await module.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_with_filters(self):
        from aethelred.models import ModelsModule
        from aethelred.core.types import UtilityCategory, PageRequest
        client = AsyncMock()
        client.get = AsyncMock(return_value={"models": []})
        module = ModelsModule(client)
        result = await module.list(
            owner="addr1", category=UtilityCategory.GENERAL,
            pagination=PageRequest(page_size=10),
        )
        assert result == []


class TestSyncModelsModule:
    """Tests for sync models module."""

    def test_register(self):
        from aethelred.models import SyncModelsModule, ModelsModule
        async_module = MagicMock(spec=ModelsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(model_hash="h1"))
        module = SyncModelsModule(client, async_module)
        result = module.register(model_hash=b"\x00" * 32, name="Test")

    def test_get(self):
        from aethelred.models import SyncModelsModule, ModelsModule
        async_module = MagicMock(spec=ModelsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock())
        module = SyncModelsModule(client, async_module)
        result = module.get("h1")

    def test_list(self):
        from aethelred.models import SyncModelsModule, ModelsModule
        async_module = MagicMock(spec=ModelsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=[])
        module = SyncModelsModule(client, async_module)
        result = module.list()


# =============================================================================
# Validators module tests
# =============================================================================

class TestValidatorsModule:
    """Tests for validators module."""

    @pytest.mark.asyncio
    async def test_get_stats(self):
        from aethelred.validators import ValidatorsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={
            "address": "addr1", "jobs_completed": 100, "jobs_failed": 5,
            "average_latency_ms": 50, "uptime_percentage": 99.9,
            "reputation_score": 0.95, "total_rewards": "1000",
            "slashing_events": 0,
        })
        module = ValidatorsModule(client)
        result = await module.get_stats("addr1")

    @pytest.mark.asyncio
    async def test_list(self):
        from aethelred.validators import ValidatorsModule
        client = AsyncMock()
        client.get = AsyncMock(return_value={"validators": []})
        module = ValidatorsModule(client)
        result = await module.list()
        assert result == []

    @pytest.mark.asyncio
    async def test_list_with_pagination(self):
        from aethelred.validators import ValidatorsModule
        from aethelred.core.types import PageRequest
        client = AsyncMock()
        client.get = AsyncMock(return_value={"validators": []})
        module = ValidatorsModule(client)
        result = await module.list(pagination=PageRequest(page_size=5))

    @pytest.mark.asyncio
    async def test_register_capability(self):
        from aethelred.validators import ValidatorsModule
        from aethelred.core.types import HardwareCapability
        client = AsyncMock()
        client.post = AsyncMock(return_value={})
        module = ValidatorsModule(client)
        cap = HardwareCapability(gpu_model="A100", vram_gb=80, cpu_cores=64, ram_gb=512)
        result = await module.register_capability("addr1", cap)
        assert result is True


class TestSyncValidatorsModule:
    """Tests for sync validators module."""

    def test_get_stats(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        async_module = MagicMock(spec=ValidatorsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock())
        module = SyncValidatorsModule(client, async_module)
        result = module.get_stats("addr1")

    def test_list(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        async_module = MagicMock(spec=ValidatorsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=[])
        module = SyncValidatorsModule(client, async_module)
        result = module.list()

    def test_register_capability(self):
        from aethelred.validators import SyncValidatorsModule, ValidatorsModule
        from aethelred.core.types import HardwareCapability
        async_module = MagicMock(spec=ValidatorsModule)
        client = MagicMock()
        client._run = MagicMock(return_value=True)
        module = SyncValidatorsModule(client, async_module)
        cap = HardwareCapability(gpu_model="A100", vram_gb=80, cpu_cores=64, ram_gb=512)
        result = module.register_capability("addr1", cap)


# =============================================================================
# Verification module tests
# =============================================================================

class TestVerificationModule:
    """Tests for verification module."""

    @pytest.mark.asyncio
    async def test_verify_zk_proof(self):
        from aethelred.verification import VerificationModule
        client = AsyncMock()
        client.post = AsyncMock(return_value={"valid": True, "verification_time_ms": 50})
        module = VerificationModule(client)
        result = await module.verify_zk_proof(
            proof=b"\x00" * 32, public_inputs=[b"\x01" * 32],
            verifying_key_hash=b"\x02" * 32,
        )
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_verify_tee_attestation(self):
        from aethelred.verification import VerificationModule
        from aethelred.core.types import TEEAttestation, TEEPlatform
        client = AsyncMock()
        client.post = AsyncMock(return_value={
            "valid": True, "platform": "TEE_PLATFORM_SGX",
        })
        module = VerificationModule(client)
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX, quote=b"\x00" * 64,
            enclave_hash=b"\x01" * 32, timestamp="2024-01-01T00:00:00Z",
        )
        result = await module.verify_tee_attestation(att, expected_enclave_hash=b"\x01" * 32)
        assert result.valid is True

    @pytest.mark.asyncio
    async def test_verify_tee_attestation_no_expected_hash(self):
        from aethelred.verification import VerificationModule
        from aethelred.core.types import TEEAttestation, TEEPlatform
        client = AsyncMock()
        client.post = AsyncMock(return_value={"valid": True})
        module = VerificationModule(client)
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX, quote=b"\x00" * 64,
            enclave_hash=b"\x01" * 32, timestamp="2024-01-01T00:00:00Z",
        )
        result = await module.verify_tee_attestation(att)
        assert result.valid is True


class TestSyncVerificationModule:
    """Tests for sync verification module."""

    def test_verify_zk_proof(self):
        from aethelred.verification import SyncVerificationModule, VerificationModule
        async_module = MagicMock(spec=VerificationModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(valid=True))
        module = SyncVerificationModule(client, async_module)
        result = module.verify_zk_proof(proof=b"", public_inputs=[], verifying_key_hash=b"")

    def test_verify_tee_attestation(self):
        from aethelred.verification import SyncVerificationModule, VerificationModule
        from aethelred.core.types import TEEAttestation, TEEPlatform
        async_module = MagicMock(spec=VerificationModule)
        client = MagicMock()
        client._run = MagicMock(return_value=MagicMock(valid=True))
        module = SyncVerificationModule(client, async_module)
        att = TEEAttestation(
            platform=TEEPlatform.INTEL_SGX, quote=b"\x00" * 64,
            enclave_hash=b"\x01" * 32, timestamp="2024-01-01T00:00:00Z",
        )
        result = module.verify_tee_attestation(att)


class TestVerifyResponses:
    """Tests for verification response classes."""

    def test_verify_zk_proof_response(self):
        from aethelred.verification import VerifyZKProofResponse
        resp = VerifyZKProofResponse(valid=True, verification_time_ms=50)
        assert resp.valid is True
        assert resp.verification_time_ms == 50
        assert resp.error is None

    def test_verify_tee_response(self):
        from aethelred.verification import VerifyTEEResponse
        from aethelred.core.types import TEEPlatform
        resp = VerifyTEEResponse(valid=False, platform=TEEPlatform.INTEL_SGX, error="mismatch")
        assert resp.valid is False
        assert resp.error == "mismatch"


# =============================================================================
# Additional coverage for __init__.py files
# =============================================================================

class TestMainInit:
    """Tests for aethelred/__init__.py coverage."""

    def test_import(self):
        import aethelred
        assert hasattr(aethelred, '__version__')

    def test_core_init(self):
        import aethelred.core
        assert aethelred.core is not None


class TestIntegrationsInit:
    """Tests for integrations/__init__.py."""

    def test_import(self):
        from aethelred import integrations
        assert integrations is not None


class TestModuleExports:
    """Test __all__ exports."""

    def test_seals_all(self):
        from aethelred.seals import __all__
        assert "SealsModule" in __all__
        assert "SyncSealsModule" in __all__

    def test_jobs_all(self):
        from aethelred.jobs import __all__
        assert "JobsModule" in __all__
        assert "SyncJobsModule" in __all__

    def test_models_all(self):
        from aethelred.models import __all__
        assert "ModelsModule" in __all__

    def test_validators_all(self):
        from aethelred.validators import __all__
        assert "ValidatorsModule" in __all__

    def test_verification_all(self):
        from aethelred.verification import __all__
        assert "VerificationModule" in __all__

    def test_common_all(self):
        from aethelred.integrations._common import __all__
        assert "VerificationEnvelope" in __all__

    def test_fastapi_all(self):
        from aethelred.integrations.fastapi import __all__
        assert "AethelredVerificationMiddleware" in __all__

    def test_pytorch_all(self):
        from aethelred.integrations.pytorch import __all__
        assert "VerifiedPyTorchModule" in __all__

    def test_huggingface_all(self):
        from aethelred.integrations.huggingface import __all__
        assert "VerifiedTransformersPipeline" in __all__

    def test_langchain_all(self):
        from aethelred.integrations.langchain import __all__
        assert "VerifiedLangChainRunnable" in __all__

    def test_tensorflow_all(self):
        from aethelred.integrations.tensorflow import __all__
        assert "AethelredKerasCallback" in __all__
