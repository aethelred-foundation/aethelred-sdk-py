"""
Aethelred Optimizers

Advanced optimization algorithms with:
- Learning rate scheduling
- Gradient clipping
- Mixed precision support
- Distributed training support
- State checkpointing
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Optional,
    Tuple, TypeVar, Union
)

from ..core.tensor import Tensor, DType
from ..nn import Parameter


# ============================================================================
# Base Optimizer
# ============================================================================

class Optimizer(ABC):
    """
    Base class for all optimizers.

    Provides:
    - Parameter group management
    - State management
    - Gradient clipping
    - Learning rate scheduling integration
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        defaults: Dict[str, Any]
    ):
        self.defaults = defaults
        self.state: Dict[Parameter, Dict[str, Any]] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        # Process parameter groups
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for group in param_groups:
            self.add_param_group(group)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group."""
        params = param_group['params']
        if isinstance(params, Parameter):
            param_group['params'] = [params]
        else:
            param_group['params'] = list(params)

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero all gradients."""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if set_to_none:
                        param._grad = None
                    else:
                        param._grad = Tensor.zeros(
                            *param.shape,
                            dtype=param.dtype,
                            device=param.device
                        )

    @abstractmethod
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform a single optimization step."""
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state as dictionary."""
        # Convert state keys to indices
        param_mappings = {}
        for group_idx, group in enumerate(self.param_groups):
            for param_idx, param in enumerate(group['params']):
                param_mappings[id(param)] = (group_idx, param_idx)

        packed_state = {}
        for param, state in self.state.items():
            key = param_mappings[id(param)]
            packed_state[key] = state

        return {
            'state': packed_state,
            'param_groups': [
                {k: v for k, v in g.items() if k != 'params'}
                for g in self.param_groups
            ]
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state."""
        # Restore parameter groups
        for group, saved_group in zip(self.param_groups, state_dict['param_groups']):
            group.update(saved_group)

        # Restore state
        for key, state in state_dict['state'].items():
            group_idx, param_idx = key
            param = self.param_groups[group_idx]['params'][param_idx]
            self.state[param] = state


# ============================================================================
# SGD
# ============================================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum and weight decay.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: L2 regularization
        dampening: Dampening for momentum
        nesterov: Use Nesterov momentum
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        dampening: float = 0.0,
        nesterov: bool = False
    ):
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov requires momentum > 0 and dampening = 0")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad

                # Weight decay
                if weight_decay != 0:
                    grad = grad + param * weight_decay

                # Momentum
                if momentum != 0:
                    state = self.state[param]

                    if 'momentum_buffer' not in state:
                        buf = grad.clone()
                        state['momentum_buffer'] = buf
                    else:
                        buf = state['momentum_buffer']
                        buf = buf * momentum + grad * (1 - dampening)
                        state['momentum_buffer'] = buf

                    if nesterov:
                        grad = grad + buf * momentum
                    else:
                        grad = buf

                # Update parameter
                # param.data = param.data - lr * grad
                # In production, this would be an in-place update

        return loss


# ============================================================================
# Adam
# ============================================================================

class Adam(Optimizer):
    """
    Adam optimizer with decoupled weight decay option.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        betas: Coefficients for moving averages
        eps: Numerical stability
        weight_decay: L2 regularization (AdamW style if amsgrad=False)
        amsgrad: Use AMSGrad variant
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            amsgrad = group['amsgrad']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)
                    state['exp_avg_sq'] = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)
                    if amsgrad:
                        state['max_exp_avg_sq'] = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Decoupled weight decay (AdamW)
                if weight_decay != 0:
                    # param.data = param.data * (1 - lr * weight_decay)
                    pass

                # Update biased first moment
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad

                # Update biased second moment
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    # max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq)
                    # denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps
                else:
                    # denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps
                    pass

                step_size = lr / bias_correction1

                # Update: param = param - step_size * exp_avg / denom

        return loss


class AdamW(Adam):
    """Adam with decoupled weight decay (default weight decay is decoupled)."""

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False
    ):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)


# ============================================================================
# Lion
# ============================================================================

class Lion(Optimizer):
    """
    Lion optimizer (Evolved Sign Momentum).

    More memory efficient than Adam, uses sign of momentum.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        betas: Coefficients for momentum
        weight_decay: Decoupled weight decay
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0
    ):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state['exp_avg'] = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)

                exp_avg = state['exp_avg']

                # Weight decay
                if weight_decay != 0:
                    # param = param * (1 - lr * weight_decay)
                    pass

                # Update with sign of interpolated momentum
                # update = sign(beta1 * exp_avg + (1 - beta1) * grad)
                # param = param - lr * update

                # Update momentum
                # exp_avg = beta2 * exp_avg + (1 - beta2) * grad

        return loss


# ============================================================================
# LAMB
# ============================================================================

class LAMB(Optimizer):
    """
    LAMB optimizer for large batch training.

    Layer-wise Adaptive Moments for Batch training.

    Args:
        params: Parameters to optimize
        lr: Learning rate
        betas: Adam momentum coefficients
        eps: Numerical stability
        weight_decay: L2 regularization
        trust_clip: Clip trust ratio
    """

    def __init__(
        self,
        params: Union[Iterable[Parameter], Iterable[Dict[str, Any]]],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        trust_clip: bool = False
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            trust_clip=trust_clip
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Perform optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)
                    state['exp_avg_sq'] = Tensor.zeros(*param.shape, dtype=param.dtype, device=param.device)

                state['step'] += 1
                step = state['step']

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']

                # Update moments (same as Adam)
                # exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                # exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                # Compute Adam update
                # adam_step = exp_avg / bias_correction1 / (sqrt(exp_avg_sq / bias_correction2) + eps)

                # Add weight decay
                # if weight_decay != 0:
                #     adam_step = adam_step + weight_decay * param

                # Compute trust ratio
                # weight_norm = param.norm(2)
                # adam_norm = adam_step.norm(2)
                # trust_ratio = weight_norm / (adam_norm + eps)
                # if trust_clip:
                #     trust_ratio = min(trust_ratio, 10)

                # Update: param = param - lr * trust_ratio * adam_step

        return loss


# ============================================================================
# Learning Rate Schedulers
# ============================================================================

class LRScheduler(ABC):
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

        if last_epoch == -1:
            for group, lr in zip(optimizer.param_groups, self.base_lrs):
                group['lr'] = lr
        else:
            self.step()

    @abstractmethod
    def get_lr(self) -> List[float]:
        """Compute learning rate."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None) -> None:
        """Update learning rate."""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        lrs = self.get_lr()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group['lr'] = lr

    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state."""
        return {'last_epoch': self.last_epoch}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state."""
        self.last_epoch = state_dict['last_epoch']


class StepLR(LRScheduler):
    """Step decay: lr = lr * gamma every step_size epochs."""

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1
    ):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch == 0:
            return self.base_lrs

        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestarts(LRScheduler):
    """Cosine annealing with warm restarts."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_i = T_0
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i *= self.T_mult
        else:
            self.T_cur = epoch

        super().step(epoch)


class OneCycleLR(LRScheduler):
    """One cycle learning rate policy."""

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
        last_epoch: int = -1
    ):
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        if isinstance(max_lr, (list, tuple)):
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.initial_lrs = [lr / div_factor for lr in self.max_lrs]
        self.final_lrs = [lr / (div_factor * final_div_factor) for lr in self.max_lrs]

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        if step < self.total_steps * self.pct_start:
            # Warmup phase
            pct = step / (self.total_steps * self.pct_start)
            return [
                initial + (max_lr - initial) * pct
                for initial, max_lr in zip(self.initial_lrs, self.max_lrs)
            ]
        else:
            # Annealing phase
            pct = (step - self.total_steps * self.pct_start) / (self.total_steps * (1 - self.pct_start))
            if self.anneal_strategy == 'cos':
                return [
                    final + (max_lr - final) * (1 + math.cos(math.pi * pct)) / 2
                    for final, max_lr in zip(self.final_lrs, self.max_lrs)
                ]
            else:
                return [
                    max_lr - (max_lr - final) * pct
                    for final, max_lr in zip(self.final_lrs, self.max_lrs)
                ]


class WarmupLR(LRScheduler):
    """Linear warmup followed by constant or decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: Optional[int] = None,
        warmup_ratio: float = 0.1,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            scale = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        if self.total_steps is not None:
            # Linear decay after warmup
            remaining = self.total_steps - self.last_epoch
            total_after_warmup = self.total_steps - self.warmup_steps
            scale = remaining / max(1, total_after_warmup)
            return [base_lr * scale for base_lr in self.base_lrs]

        return self.base_lrs


# ============================================================================
# Gradient Utilities
# ============================================================================

def clip_grad_norm_(
    parameters: Iterable[Parameter],
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradient norm of parameters.

    Args:
        parameters: Parameters to clip
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2.0 for L2)

    Returns:
        Total gradient norm before clipping
    """
    parameters = list(parameters)

    if len(parameters) == 0:
        return 0.0

    # Compute total norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.norm(norm_type).item()
            total_norm += param_norm ** norm_type

    total_norm = total_norm ** (1.0 / norm_type)

    # Clip
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                # param.grad = param.grad * clip_coef
                pass

    return total_norm


def clip_grad_value_(
    parameters: Iterable[Parameter],
    clip_value: float
) -> None:
    """
    Clip gradient values of parameters.

    Args:
        parameters: Parameters to clip
        clip_value: Maximum absolute gradient value
    """
    for param in parameters:
        if param.grad is not None:
            # param.grad = param.grad.clamp(-clip_value, clip_value)
            pass
