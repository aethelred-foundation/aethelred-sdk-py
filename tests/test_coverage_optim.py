"""Comprehensive tests for aethelred/optim/__init__.py to achieve 95%+ coverage.

Covers:
- Optimizer base class
- SGD, Adam, AdamW, Lion, LAMB
- LRScheduler, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
- OneCycleLR, WarmupLR
- clip_grad_norm_, clip_grad_value_
"""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from aethelred.core.tensor import Tensor, DType
from aethelred.nn import Parameter, Linear
from aethelred.optim import (
    Optimizer,
    SGD,
    Adam,
    AdamW,
    Lion,
    LAMB,
    LRScheduler,
    StepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    WarmupLR,
    clip_grad_norm_,
    clip_grad_value_,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(n=3, shape=(4,)):
    """Create a list of Parameters for testing."""
    params = []
    for _ in range(n):
        t = Tensor.empty(*shape, dtype=DType.float32)
        p = Parameter(t, requires_grad=True)
        p._grad = Tensor.ones(*shape, dtype=DType.float32)
        params.append(p)
    return params


def _make_params_no_grad(n=2, shape=(4,)):
    """Create params without gradients."""
    params = []
    for _ in range(n):
        t = Tensor.empty(*shape, dtype=DType.float32)
        p = Parameter(t, requires_grad=True)
        params.append(p)
    return params


# ============================================================================
# Optimizer Base
# ============================================================================


class TestOptimizer:
    def test_empty_params_raises(self):
        with pytest.raises(ValueError, match="empty parameter list"):
            SGD([], lr=0.01)

    def test_add_param_group(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        new_params = _make_params(1)
        opt.add_param_group({'params': new_params, 'lr': 0.001})
        assert len(opt.param_groups) == 2

    def test_add_param_group_single(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        p = _make_params(1)[0]
        opt.add_param_group({'params': p})
        assert len(opt.param_groups) == 2
        assert isinstance(opt.param_groups[-1]['params'], list)

    def test_zero_grad(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        opt.zero_grad()
        for g in opt.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    assert p.grad.shape == p.shape

    def test_zero_grad_set_to_none(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        opt.zero_grad(set_to_none=True)
        for g in opt.param_groups:
            for p in g['params']:
                assert p.grad is None

    def test_state_dict(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        sd = opt.state_dict()
        assert 'state' in sd
        assert 'param_groups' in sd

    def test_load_state_dict(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        sd = opt.state_dict()
        opt.load_state_dict(sd)

    def test_dict_param_groups(self):
        params1 = _make_params(2)
        params2 = _make_params(2)
        groups = [
            {'params': params1, 'lr': 0.01},
            {'params': params2, 'lr': 0.001}
        ]
        opt = SGD(groups, lr=0.01)
        assert len(opt.param_groups) == 2


# ============================================================================
# SGD
# ============================================================================


class TestSGD:
    def test_basic_step(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        loss = opt.step()
        assert loss is None

    def test_with_closure(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        loss = opt.step(closure=lambda: 1.0)
        assert loss == 1.0

    def test_with_momentum(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01, momentum=0.9)
        opt.step()
        opt.step()  # Second step uses momentum buffer

    def test_with_weight_decay(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01, weight_decay=0.01)
        opt.step()

    def test_nesterov(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01, momentum=0.9, nesterov=True)
        opt.step()
        opt.step()

    def test_nesterov_invalid(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="Nesterov"):
            SGD(params, lr=0.01, nesterov=True, momentum=0.0)

    def test_nesterov_with_dampening(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="Nesterov"):
            SGD(params, lr=0.01, nesterov=True, momentum=0.9, dampening=0.1)

    def test_no_grad_skip(self):
        params = _make_params_no_grad(2)
        opt = SGD(params, lr=0.01)
        opt.step()  # Should skip params without grad


# ============================================================================
# Adam
# ============================================================================


class TestAdam:
    def test_basic_step(self):
        params = _make_params(2)
        opt = Adam(params, lr=0.001)
        opt.step()

    def test_with_closure(self):
        params = _make_params(2)
        opt = Adam(params, lr=0.001)
        loss = opt.step(closure=lambda: 2.0)
        assert loss == 2.0

    def test_amsgrad(self):
        params = _make_params(2)
        opt = Adam(params, lr=0.001, amsgrad=True)
        opt.step()
        opt.step()

    def test_weight_decay(self):
        params = _make_params(2)
        opt = Adam(params, lr=0.001, weight_decay=0.01)
        opt.step()

    def test_invalid_lr(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="learning rate"):
            Adam(params, lr=-1.0)

    def test_invalid_beta1(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="beta1"):
            Adam(params, lr=0.001, betas=(1.5, 0.999))

    def test_invalid_beta2(self):
        params = _make_params(2)
        with pytest.raises(ValueError, match="beta2"):
            Adam(params, lr=0.001, betas=(0.9, 1.5))

    def test_no_grad_skip(self):
        params = _make_params_no_grad(2)
        opt = Adam(params, lr=0.001)
        opt.step()

    def test_multiple_steps(self):
        params = _make_params(2)
        opt = Adam(params, lr=0.001)
        for _ in range(5):
            opt.step()


# ============================================================================
# AdamW
# ============================================================================


class TestAdamW:
    def test_creation(self):
        params = _make_params(2)
        opt = AdamW(params, lr=0.001, weight_decay=0.01)
        assert opt.defaults['weight_decay'] == 0.01

    def test_step(self):
        params = _make_params(2)
        opt = AdamW(params, lr=0.001)
        opt.step()


# ============================================================================
# Lion
# ============================================================================


class TestLion:
    def test_basic_step(self):
        params = _make_params(2)
        opt = Lion(params, lr=1e-4)
        opt.step()

    def test_with_closure(self):
        params = _make_params(2)
        opt = Lion(params, lr=1e-4)
        loss = opt.step(closure=lambda: 3.0)
        assert loss == 3.0

    def test_weight_decay(self):
        params = _make_params(2)
        opt = Lion(params, lr=1e-4, weight_decay=0.01)
        opt.step()

    def test_multiple_steps(self):
        params = _make_params(2)
        opt = Lion(params, lr=1e-4)
        opt.step()
        opt.step()

    def test_no_grad_skip(self):
        params = _make_params_no_grad(2)
        opt = Lion(params, lr=1e-4)
        opt.step()


# ============================================================================
# LAMB
# ============================================================================


class TestLAMB:
    def test_basic_step(self):
        params = _make_params(2)
        opt = LAMB(params, lr=0.001)
        opt.step()

    def test_with_closure(self):
        params = _make_params(2)
        opt = LAMB(params, lr=0.001)
        loss = opt.step(closure=lambda: 4.0)
        assert loss == 4.0

    def test_multiple_steps(self):
        params = _make_params(2)
        opt = LAMB(params, lr=0.001)
        for _ in range(3):
            opt.step()

    def test_no_grad_skip(self):
        params = _make_params_no_grad(2)
        opt = LAMB(params, lr=0.001)
        opt.step()


# ============================================================================
# LRScheduler
# ============================================================================


class TestStepLR:
    def test_creation(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = StepLR(opt, step_size=10, gamma=0.1)
        assert sched.step_size == 10
        assert sched.gamma == 0.1

    def test_step(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        initial_lr = opt.param_groups[0]['lr']
        sched.step()  # epoch 0
        sched.step()  # epoch 1
        sched.step()  # epoch 2 -> decay
        new_lr = opt.param_groups[0]['lr']
        assert new_lr < initial_lr

    def test_state_dict(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = StepLR(opt, step_size=10)
        sd = sched.state_dict()
        assert 'last_epoch' in sd

    def test_load_state_dict(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = StepLR(opt, step_size=10)
        sched.step()
        sd = sched.state_dict()
        sched2 = StepLR(opt, step_size=10)
        sched2.load_state_dict(sd)
        assert sched2.last_epoch == sd['last_epoch']

    def test_step_with_epoch(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = StepLR(opt, step_size=10)
        sched.step(epoch=5)
        assert sched.last_epoch == 5


class TestCosineAnnealingLR:
    def test_creation(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=100)
        assert sched.T_max == 100

    def test_get_lr_epoch_0(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=100)
        lrs = sched.get_lr()
        assert abs(lrs[0] - 0.1) < 1e-3

    def test_step(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingLR(opt, T_max=10)
        for _ in range(5):
            sched.step()
        lr = opt.param_groups[0]['lr']
        assert lr < 0.1


class TestCosineAnnealingWarmRestarts:
    def test_creation(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingWarmRestarts(opt, T_0=10)
        assert sched.T_0 == 10

    def test_step(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2)
        for _ in range(15):
            sched.step()

    def test_step_with_epoch(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingWarmRestarts(opt, T_0=10)
        sched.step(epoch=3)
        assert sched.T_cur == 3

    def test_restart(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = CosineAnnealingWarmRestarts(opt, T_0=3, T_mult=1)
        for _ in range(10):
            sched.step()


class TestOneCycleLR:
    def test_creation_scalar_max_lr(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=100)
        assert sched.max_lrs == [0.1]

    def test_creation_list_max_lr(self):
        params1 = _make_params(2)
        params2 = _make_params(2)
        opt = SGD([{'params': params1, 'lr': 0.01}, {'params': params2, 'lr': 0.01}], lr=0.01)
        sched = OneCycleLR(opt, max_lr=[0.1, 0.2], total_steps=100)
        assert sched.max_lrs == [0.1, 0.2]

    def test_warmup_phase(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=100, pct_start=0.3)
        for _ in range(10):
            sched.step()
        lr = opt.param_groups[0]['lr']
        assert lr > 0

    def test_annealing_phase_cos(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=100, pct_start=0.3, anneal_strategy='cos')
        for _ in range(50):
            sched.step()

    def test_annealing_phase_linear(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.01)
        sched = OneCycleLR(opt, max_lr=0.1, total_steps=100, pct_start=0.3, anneal_strategy='linear')
        for _ in range(50):
            sched.step()


class TestWarmupLR:
    def test_warmup_phase(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = WarmupLR(opt, warmup_steps=10)
        sched.step()
        lr = opt.param_groups[0]['lr']
        assert lr < 0.1

    def test_post_warmup_no_total(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = WarmupLR(opt, warmup_steps=2)
        for _ in range(5):
            sched.step()
        lr = opt.param_groups[0]['lr']
        assert lr == 0.1

    def test_post_warmup_with_decay(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = WarmupLR(opt, warmup_steps=5, total_steps=20)
        for _ in range(15):
            sched.step()
        lr = opt.param_groups[0]['lr']
        assert lr < 0.1

    def test_last_epoch_nondefault(self):
        params = _make_params(2)
        opt = SGD(params, lr=0.1)
        sched = WarmupLR(opt, warmup_steps=10, last_epoch=5)
        assert sched.last_epoch >= 5


# ============================================================================
# Gradient Clipping
# ============================================================================


class TestClipGradNorm:
    def test_clip(self):
        params = _make_params(2)
        total_norm = clip_grad_norm_(params, max_norm=1.0)
        assert isinstance(total_norm, float)

    def test_empty_params(self):
        total_norm = clip_grad_norm_([], max_norm=1.0)
        assert total_norm == 0.0

    def test_no_grad_params(self):
        params = _make_params_no_grad(2)
        total_norm = clip_grad_norm_(params, max_norm=1.0)
        assert total_norm == 0.0

    def test_large_norm_clips(self):
        params = _make_params(2)
        total_norm = clip_grad_norm_(params, max_norm=0.001)
        assert total_norm >= 0


class TestClipGradValue:
    def test_clip(self):
        params = _make_params(2)
        clip_grad_value_(params, clip_value=1.0)

    def test_no_grad_params(self):
        params = _make_params_no_grad(2)
        clip_grad_value_(params, clip_value=1.0)
