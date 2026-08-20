"""M5：每个 stage 的参数、梯度与优化器持久态。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StaticBreakdown:
    parameter: int
    gradient: int
    master_weight: int
    optimizer_state: int

    @property
    def total(self):
        return sum(self.__dict__.values())


@dataclass(frozen=True)
class StageStaticMemory:
    stage: int
    persistent_bytes: int
    breakdown: StaticBreakdown
    edge_parameter_bytes: int = 0
    decoder_persistent_bytes: int = 0
    host_pinned_bytes: int = 0
    host_breakdown: StaticBreakdown | None = None

    @property
    def total(self): return self.persistent_bytes
    def __int__(self): return self.persistent_bytes
    def __eq__(self, other):
        if isinstance(other, int):
            return self.persistent_bytes == other or self.decoder_persistent_bytes == other
        return super().__eq__(other)
    def __floordiv__(self, other):
        return self.decoder_persistent_bytes // other


def _unique_layer_params(layer):
    seen = set()
    for op in layer.ops:
        for weight in op.params:
            identity = weight.storage_key
            if identity not in seen:
                seen.add(identity)
                yield weight


def _state_bytes(weight, optimizer) -> StaticBreakdown:
    """Persistent bytes for one local FSDP shard, honoring parameter dtype."""
    from .optimizers import state_bytes
    state = state_bytes(weight, optimizer)
    return StaticBreakdown(
        state.parameter, state.gradient,
        state.master_weight, state.optimizer_state,
    )


def _add_breakdown(left, right):
    return StaticBreakdown(*(
        getattr(left, field) + getattr(right, field)
        for field in ("parameter", "gradient", "master_weight", "optimizer_state")
    ))


def _sharded_group_state(weights, degree, alignment_bytes, optimizer):
    """Shard a flattened dtype/trainability group with explicit padding."""
    weights = tuple(weights)
    if not weights:
        return StaticBreakdown(0, 0, 0, 0)
    representative = weights[0]
    dtype_bytes = representative.dtype_bytes
    alignment_numel = max(1, (alignment_bytes + dtype_bytes - 1) // dtype_bytes)
    total_numel = sum(weight.local_numel for weight in weights)
    shard_unit = degree * alignment_numel
    padded_numel = ((total_numel + shard_unit - 1) // shard_unit) * shard_unit
    shard = representative.__class__(
        **{**representative.__dict__, "local_numel": padded_numel // degree}
    )
    return _state_bytes(shard, optimizer)


class StaticMem:
    def compute(self, graph, optimizer, pm, cpu_offload: bool = None):
        if not hasattr(pm, "pc"):
            from .parallel_model import ParallelModel
            pc = pm
            layer_count = sum(len(v) for v in graph.stages.values())
            pm = ParallelModel(pc, layer_count)
        from .offload_model import OffloadPolicy
        policy = OffloadPolicy.from_parallel(pm.pc)
        if cpu_offload is True:
            policy = OffloadPolicy(True, True, True)
        result = {}
        for stage, layers in graph.stages.items():
            decoder_breakdown = StaticBreakdown(0, 0, 0, 0)
            for layer in layers:
                groups = {}
                for weight in _unique_layer_params(layer):
                    degree = 1 if weight.fsdp_replicated else (
                        pm.efsdp_degree() if weight.is_expert else pm.fsdp_degree()
                    )
                    key = (
                        degree, weight.dtype_bytes, weight.trainable,
                        weight.is_expert, weight.fsdp_replicated,
                    )
                    groups.setdefault(key, []).append(weight)
                for key, weights in groups.items():
                    decoder_breakdown = _add_breakdown(
                        decoder_breakdown,
                        _sharded_group_state(
                            weights, key[0],
                            pm.pc.fsdp_flatten_alignment_bytes,
                            optimizer,
                        ),
                    )
            edge_breakdown = StaticBreakdown(0, 0, 0, 0)
            edge_global_bytes = 0
            seen_edge_storage = set()
            edge_weights = (
                tensor
                for op in (graph.edge_ops or {}).get(stage, ())
                for tensor in op.params
            ) if getattr(graph, "edge_ops", None) is not None else iter(
                (graph.stage_params or {}).get(stage, ())
            )
            for weight in edge_weights:
                identity = weight.storage_key
                if identity in seen_edge_storage:
                    continue
                seen_edge_storage.add(identity)
                degree = 1 if weight.fsdp_replicated else pm.fsdp_degree()
                edge_breakdown = _add_breakdown(
                    edge_breakdown,
                    _sharded_group_state(
                        (weight,), degree,
                        pm.pc.fsdp_flatten_alignment_bytes,
                        optimizer,
                    ),
                )
                from math import prod
                edge_global_bytes += prod(weight.global_shape) * weight.dtype_bytes
            breakdown = _add_breakdown(decoder_breakdown, edge_breakdown)
            breakdown, host_breakdown = policy.split(breakdown)
            decoder_resident, _ = policy.split(decoder_breakdown)
            total = breakdown.total
            result[stage] = StageStaticMemory(
                stage, total, breakdown, edge_global_bytes,
                decoder_resident.total,
                host_breakdown.total,
                host_breakdown,
            )
        return result
