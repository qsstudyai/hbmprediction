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
            identity = weight.storage_id or weight.name
            if identity not in seen:
                seen.add(identity)
                yield weight


def _state_bytes(weight, optimizer) -> StaticBreakdown:
    """Persistent bytes for one local FSDP shard, honoring parameter dtype."""
    if not weight.trainable:
        return StaticBreakdown(weight.local_bytes, 0, 0, 0)
    numel = weight.local_numel
    parameter = numel * weight.dtype_bytes
    gradient = numel * weight.dtype_bytes
    master = 0 if weight.dtype_bytes == 4 else numel * optimizer.master_weight_bytes
    state = numel * optimizer.optimizer_state_bytes
    return StaticBreakdown(parameter, gradient, master, state)


def _add_breakdown(left, right):
    return StaticBreakdown(*(
        getattr(left, field) + getattr(right, field)
        for field in ("parameter", "gradient", "master_weight", "optimizer_state")
    ))


class StaticMem:
    def compute(self, graph, optimizer, pm, cpu_offload: bool = None):
        if not hasattr(pm, "pc"):
            from .parallel_model import ParallelModel
            pc = pm
            layer_count = sum(len(v) for v in graph.stages.values())
            pm = ParallelModel(pc, layer_count)
        if cpu_offload is None:
            cpu_offload = pm.pc.cpu_offload
        result = {}
        for stage, layers in graph.stages.items():
            decoder_breakdown = StaticBreakdown(0, 0, 0, 0)
            for layer in layers:
                for weight in _unique_layer_params(layer):
                    degree = (
                        pm.efsdp_degree()
                        if weight.is_expert
                        else pm.fsdp_degree()
                    )
                    if weight.local_numel % degree:
                        raise ValueError(
                            f"{weight.name} local_numel={weight.local_numel} "
                            f"不被 FSDP degree={degree} 整除"
                        )
                    shard = weight.__class__(
                        **{**weight.__dict__, "local_numel": weight.local_numel // degree}
                    )
                    decoder_breakdown = _add_breakdown(
                        decoder_breakdown, _state_bytes(shard, optimizer)
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
                identity = weight.storage_id or weight.name
                if identity in seen_edge_storage:
                    continue
                seen_edge_storage.add(identity)
                degree = pm.fsdp_degree()
                if weight.local_numel % degree:
                    raise ValueError(f"{weight.name} 不被 dense FSDP degree={degree} 整除")
                shard = weight.__class__(
                    **{**weight.__dict__, "local_numel": weight.local_numel // degree}
                )
                edge_breakdown = _add_breakdown(
                    edge_breakdown, _state_bytes(shard, optimizer)
                )
                from math import prod
                edge_global_bytes += prod(weight.global_shape) * weight.dtype_bytes
            breakdown = _add_breakdown(decoder_breakdown, edge_breakdown)
            total = 0 if cpu_offload else breakdown.total
            if cpu_offload:
                breakdown = StaticBreakdown(0, 0, 0, 0)
            result[stage] = StageStaticMemory(
                stage, total, breakdown, edge_global_bytes,
                0 if cpu_offload else decoder_breakdown.total,
            )
        return result
