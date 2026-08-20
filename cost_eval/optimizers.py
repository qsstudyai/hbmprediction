"""Versioned optimizer state and step-workspace contracts."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_OPTIMIZERS = frozenset({"adamw", "muon", "muonadam", "muon_adam"})


@dataclass(frozen=True)
class OptimizerStateBytes:
    parameter: int
    gradient: int
    master_weight: int
    optimizer_state: int


def state_bytes(weight, optimizer) -> OptimizerStateBytes:
    numel = weight.local_numel
    parameter = numel * weight.dtype_bytes
    if not weight.trainable:
        return OptimizerStateBytes(parameter, 0, 0, 0)
    gradient = numel * max(weight.dtype_bytes, optimizer.gradient_bytes)
    if not optimizer.is_muon:
        master = 0 if weight.dtype_bytes == 4 else numel * optimizer.master_weight_bytes
        return OptimizerStateBytes(
            parameter, gradient, master,
            numel * optimizer.optimizer_state_bytes,
        )

    fraction = optimizer.muon_parameter_fraction
    muon_numel = round(numel * fraction)
    adam_numel = numel - muon_numel
    muon_master = muon_numel * optimizer.muon_main_parameter_bytes
    muon_state = muon_numel * optimizer.muon_momentum_bytes
    adam_master = 0 if weight.dtype_bytes == 4 else adam_numel * 4
    adam_state = adam_numel * 8
    return OptimizerStateBytes(
        parameter, gradient,
        muon_master + adam_master,
        muon_state + adam_state,
    )


def optimizer_step_workspace_bytes(
    optimizer, largest_gradient_bytes: int, optimizer_offload: bool = False
) -> int:
    if largest_gradient_bytes < 0:
        raise ValueError("largest_gradient_bytes 不能为负数")
    if optimizer.name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"unsupported optimizer contract: {optimizer.type}")
    if optimizer.is_muon:
        # Newton-Schulz keeps input/output plus one matrix temporary live;
        # iteration count changes time, not simultaneous matrix count.
        workspace = 3 * largest_gradient_bytes
    else:
        workspace = largest_gradient_bytes
    if optimizer_offload:
        # One prefetched state chunk accompanies the gradient chunk.
        workspace += largest_gradient_bytes
    return workspace
