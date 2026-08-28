"""M6：沿 1F1B 时间线追踪张量存活并计算峰值。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace

from .event_schedule import Event, build_1f1b, build_interleaved_1f1b
from .memory_actions import forward_layer_actions
from .memory_ledger import MemoryLedger
from .trace import TraceActivation, TraceTensor


@dataclass
class Buckets:
    persistent: int = 0
    act_live: int = 0
    gather_buf: int = 0
    grad_buf: int = 0
    recomp_scratch: int = 0
    swap_buf: int = 0
    workspace: int = 0

    def total(self) -> int:
        return sum(self.__dict__.values())


@dataclass(frozen=True)
class MemBreakdown:
    persistent: int
    act_live: int
    gather_buf: int
    grad_buf: int
    recomp_scratch: int
    swap_buf: int
    workspace: int
    framework: int

    @property
    def total(self) -> int:
        return sum(self.__dict__.values())


@dataclass(frozen=True)
class StagePeak:
    stage: int
    peak_bytes: int
    breakdown: MemBreakdown
    peak_event: str
    oom: bool
    bucket_peaks: "BucketPeaks" = None
    physical_dynamic_peak_bytes: int = 0
    allocator_pool_peak_bytes: int = 0
    device_baseline_bytes: int = 0
    untracked_runtime_point_bytes: int = 0
    fragmentation_point_bytes: int = 0
    total_point_bytes: int = 0
    safe_upper_bytes: int = 0
    oom_status: str = "risky"


@dataclass(frozen=True)
class BucketPeaks:
    persistent: int = 0
    act_live: int = 0
    gather_buf: int = 0
    grad_buf: int = 0
    recomp_scratch: int = 0
    swap_buf: int = 0
    workspace: int = 0


@dataclass(frozen=True)
class LayerMemoryPlan:
    resident_bytes: int
    offloaded_bytes: int
    recompute_scratch_bytes: int
    recomputed_ops: frozenset[str]
    recompute_comm_bytes: int = 0
    recomputed_comm_ops: frozenset[str] = frozenset()
    resident_keys: frozenset[str] = frozenset()
    offloaded_keys: frozenset[str] = frozenset()
    recompute_keys: frozenset[str] = frozenset()

    @property
    def resident(self): return self.resident_bytes
    @property
    def offloaded(self): return self.offloaded_bytes
    @property
    def recomputed(self): return self.recompute_scratch_bytes


def _unique_tensors(tensors):
    unique = {}
    for tensor in tensors:
        unique.setdefault(tensor.storage_key, tensor)
    return unique.values()


def _layer_memory_plan(layer, recompute, swap) -> LayerMemoryPlan:
    op_names = [op.name for op in layer.ops]
    recomputed = recompute.recomputed_ops(layer.layer_id, op_names)
    swapped = swap.swapped_ops(layer.layer_id, op_names) - recomputed
    comm_recomputed = recompute.recomputed_comm_ops(layer.layer_id, layer.ops)

    tensors = {}
    saves_by_op = {}
    for op in layer.ops:
        saves_by_op[op.name] = {tensor.storage_key for tensor in op.saves}
        for tensor in (*op.inputs, op.output, *op.saves):
            if not tensor.is_weight:
                tensors.setdefault(tensor.storage_key, tensor)

    resident_names = set().union(
        *(
            saves_by_op[op.name]
            for op in layer.ops
            if op.name not in recomputed
        ),
        set(),
    )
    scratch_names = set().union(
        *(saves_by_op[name] for name in recomputed), set()
    )

    # 每段连续重计算区域只保留其首个非权重输入作为 checkpoint。
    checkpoints = set()
    previous_selected = False
    for op in layer.ops:
        selected = op.name in recomputed
        if selected and not previous_selected:
            checkpoint = next(
                (tensor for tensor in op.inputs if not tensor.is_weight), None
            )
            if checkpoint is not None:
                tensors.setdefault(checkpoint.storage_key, checkpoint)
                checkpoints.add(checkpoint.storage_key)
        previous_selected = selected
    resident_names |= checkpoints
    scratch_names -= resident_names

    # Communication recompute discards the collective input/output activation
    # after forward and materializes it again immediately before backward.
    comm_tensor_names = set()
    comm_bytes = 0
    for op in layer.ops:
        if op.name in comm_recomputed:
            comm_tensor_names.update(
                item.output_value_id or item.output_tensor
                for item in op.collectives
                if item.output_value_id is not None or item.output_tensor is not None
            )
            comm_bytes += sum(item.volume_bytes for item in op.collectives)
    resident_names -= comm_tensor_names

    swap_candidates = set().union(
        *(saves_by_op[name] for name in swapped), set()
    )
    unswapped = set().union(
        *(
            saves_by_op[op.name]
            for op in layer.ops
            if op.name not in recomputed and op.name not in swapped
        ),
        set(),
    )
    offloaded_names = swap_candidates - unswapped - checkpoints
    resident_names -= offloaded_names

    def byte_sum(names) -> int:
        return sum(tensors[name].local_bytes for name in names)

    return LayerMemoryPlan(
        byte_sum(resident_names),
        byte_sum(offloaded_names),
        byte_sum(scratch_names),
        recomputed,
        comm_bytes,
        comm_recomputed,
        frozenset(resident_names),
        frozenset(offloaded_names),
        frozenset(scratch_names),
    )


def _save_plan(layer, recompute, swap):
    """Public save/recompute/swap partition helper."""
    return _layer_memory_plan(layer, recompute, swap)


def _layer_fsdp_buffer_bytes(layer, pm) -> int:
    total = 0
    for tensor in _unique_tensors(
        tensor for op in layer.ops for tensor in op.params
    ):
        degree = 1 if tensor.fsdp_replicated else (
            pm.efsdp_degree() if tensor.is_expert else pm.fsdp_degree()
        )
        if degree > 1 or pm.pc.parameter_offload:
            total += tensor.local_bytes
    return total


def _layer_grad_buffer_bytes(layer, pm, gradient_bytes: int | None = None) -> int:
    """Full gradients materialized before FSDP/eFSDP reduce-scatter."""
    total = 0
    for tensor in _unique_tensors(
        tensor for op in layer.ops for tensor in op.params
    ):
        if not tensor.trainable:
            continue
        degree = 1 if tensor.fsdp_replicated else (
            pm.efsdp_degree() if tensor.is_expert else pm.fsdp_degree()
        )
        if degree > 1 or pm.pc.gradient_offload:
            dtype_bytes = max(
                tensor.dtype_bytes, gradient_bytes or tensor.dtype_bytes
            )
            total += tensor.local_numel * dtype_bytes
    return total


def _layer_workspace(layer) -> int:
    return max((
        op.workspace_bytes + max(
            (comm.volume_bytes for comm in op.collectives), default=0
        )
        for op in layer.ops
    ), default=0)


def _op_fsdp_buffer_bytes(op, pm) -> int:
    if pm.fsdp_degree() == 1 and not pm.pc.parameter_offload:
        return 0
    return sum(
        tensor.local_bytes for tensor in _unique_tensors(op.params)
        if not tensor.fsdp_replicated or pm.pc.parameter_offload
    )


def _op_grad_buffer_bytes(op, pm, gradient_bytes: int | None = None) -> int:
    if pm.fsdp_degree() == 1 and not pm.pc.gradient_offload:
        return 0
    return sum(
        tensor.local_numel
        * max(tensor.dtype_bytes, gradient_bytes or tensor.dtype_bytes)
        for tensor in _unique_tensors(op.params)
        if tensor.trainable and (not tensor.fsdp_replicated or pm.pc.gradient_offload)
    )


def _edge_saved_bytes(ops) -> int:
    return sum(
        tensor.local_bytes
        for tensor in _unique_tensors(
            tensor for op in ops for tensor in op.saves if not tensor.is_weight
        )
    )


class MemTimeline:
    def simulate(
        self,
        graph,
        recompute,
        swap,
        pm,
        static_persistent,
        framework_reserve: int,
        max_device_memory: int,
        gradient_bytes: int | None = None,
        optimizer=None,
        trace=None,
    ) -> dict[int, StagePeak]:
        result = {}
        for stage, layers in graph.stages.items():
            static_value = static_persistent.get(stage, 0)
            bucket = Buckets(persistent=int(static_value) if not isinstance(static_value, int) else static_value)
            high = Buckets(persistent=bucket.persistent)
            layer_ids = [layer.layer_id for layer in layers]
            by_id = {layer.layer_id: layer for layer in layers}
            layer_fsdp_buffer = {
                layer.layer_id: _layer_fsdp_buffer_bytes(layer, pm)
                for layer in layers
            }
            layer_grad_buffer = {
                layer.layer_id: _layer_grad_buffer_bytes(
                    layer, pm, gradient_bytes
                )
                for layer in layers
            }
            edge_ops = tuple((graph.edge_ops or {}).get(stage, ()))
            prefix_ops = tuple(op for op in edge_ops if op.name == "embedding")
            suffix_ops = tuple(op for op in edge_ops if op.name != "embedding")
            edge_resident = _edge_saved_bytes(edge_ops)
            suffix_resident = _edge_saved_bytes(suffix_ops)
            memory_plans = {
                layer.layer_id: _layer_memory_plan(
                    layer, recompute, swap
                )
                for layer in layers
            }
            pp_boundary_bytes = 0
            if pm.degree("pp") > 1:
                pp_boundary_bytes = max(
                    (layer.checkpoint_bytes for layer in layers), default=0
                )
            peak = -1
            peak_event = "initial"
            peak_breakdown = None
            schedule_event = None
            pinned: dict[tuple[int, int], tuple[int, int]] = {}

            def record(
                tag: str,
                *,
                layer_id: int | None = None,
                op_name: str = "",
                action=None,
                live_allocations=None,
            ) -> None:
                nonlocal peak, peak_event, peak_breakdown
                current = bucket.total() + framework_reserve
                for name, value in bucket.__dict__.items():
                    setattr(high, name, max(getattr(high, name), value))
                new_peak = current > peak
                if new_peak:
                    peak = current
                    peak_event = tag
                    peak_breakdown = MemBreakdown(
                        bucket.persistent,
                        bucket.act_live,
                        bucket.gather_buf,
                        bucket.grad_buf,
                        bucket.recomp_scratch,
                        bucket.swap_buf,
                        bucket.workspace,
                        framework_reserve,
                    )
                if trace is not None:
                    live = tuple(
                        TraceTensor(key, size, allocation_bucket)
                        for key, (size, allocation_bucket) in sorted(
                            (live_allocations or {}).items()
                        )
                    )
                    activation_state = tuple(
                        TraceActivation(mb, lid, resident, offloaded)
                        for (mb, lid), (resident, offloaded) in sorted(
                            pinned.items()
                        )
                    )
                    phase = (
                        schedule_event.kind.lower()
                        if schedule_event is not None
                        else "optimizer" if tag.startswith("optimizer")
                        else "initial"
                    )
                    trace.record(
                        stage=stage,
                        schedule_kind=(
                            schedule_event.kind
                            if schedule_event is not None else ""
                        ),
                        microbatch=(
                            schedule_event.mb
                            if schedule_event is not None else None
                        ),
                        chunk=(
                            schedule_event.chunk
                            if schedule_event is not None else None
                        ),
                        phase=phase,
                        layer_id=layer_id,
                        op_name=op_name or (
                            "" if action is None else action.op_name
                        ),
                        tag=tag,
                        action=action,
                        buckets={
                            **bucket.__dict__, "framework": framework_reserve
                        },
                        total_active_bytes=current,
                        new_peak=new_peak,
                        live_tensors=live,
                        pinned_activations=activation_state,
                    )

            def gather_window(index: int, backward: bool = False, ids=None) -> int:
                ids = layer_ids if ids is None else ids
                depth = pm.pc.prefetch_depth
                if backward:
                    lo, hi = max(0, index - depth), index + 1
                else:
                    lo, hi = index, min(len(ids), index + depth + 1)
                return sum(
                    layer_fsdp_buffer[ids[i]] for i in range(lo, hi)
                )

            keep_gathered = not pm.pc.reshard_after_forward
            gathered_ids: set[int] = set()
            record("initial")

            for event in build_interleaved_1f1b(
                stage,
                pm.degree("pp"),
                pm.pc.num_microbatches,
                pm.pc.interleave,
            ):
                schedule_event = event
                active_layer_ids = [
                    layer_id for index, layer_id in enumerate(layer_ids)
                    if index % pm.pc.interleave == event.chunk
                ]
                if not active_layer_ids:
                    raise ValueError(
                        f"stage {stage} 没有足够层数支持 interleave={pm.pc.interleave}"
                    )
                first_chunk = event.chunk == 0
                last_chunk = event.chunk == pm.pc.interleave - 1
                if event.kind == "FWD":
                    for op in prefix_ops if first_chunk else ():
                        bucket.gather_buf = _op_fsdp_buffer_bytes(op, pm)
                        record(f"fwd_edge_gather@{op.name}")
                        bucket.workspace = op.workspace_bytes + max(
                            (comm.volume_bytes for comm in op.collectives), default=0
                        )
                        record(f"fwd_edge_op@{op.name}")
                        bucket.workspace = 0
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    for index, layer_id in enumerate(active_layer_ids):
                        layer = by_id[layer_id]
                        if keep_gathered:
                            upper = min(
                                len(active_layer_ids),
                                index + pm.pc.prefetch_depth + 1,
                            )
                            gathered_ids.update(active_layer_ids[index:upper])
                            bucket.gather_buf = sum(
                                layer_fsdp_buffer[item]
                                for item in gathered_ids
                            )
                        else:
                            bucket.gather_buf = gather_window(index, ids=active_layer_ids)
                        record(f"fwd_gather@{layer_id}", layer_id=layer_id)

                        plan = memory_plans[layer_id]
                        resident = plan.resident_bytes
                        offloaded = plan.offloaded_bytes
                        # Replay explicit value/workspace/communication
                        # allocations so saved tensors from early operators
                        # overlap later operator workspaces at the real peak.
                        base_act_live = bucket.act_live
                        ledger = MemoryLedger()
                        for action in forward_layer_actions(
                            layer, plan.resident_keys
                        ):
                            trace_action = action
                            if (
                                action.kind in {"FREE", "MOVE_OUT"}
                                and action.key in ledger.allocations
                            ):
                                size, allocation_bucket = ledger.allocations[
                                    action.key
                                ]
                                trace_action = replace(
                                    action,
                                    size_bytes=size,
                                    bucket=allocation_bucket,
                                )
                            ledger.apply(action)
                            local = ledger.buckets
                            bucket.act_live = base_act_live + local.get("act_live", 0)
                            bucket.workspace = (
                                local.get("workspace", 0)
                                + local.get("communication", 0)
                            )
                            record(
                                action.owner or f"fwd_action@{layer_id}",
                                layer_id=layer_id,
                                op_name=action.op_name,
                                action=trace_action,
                                live_allocations=ledger.allocations,
                            )
                        if ledger.live_keys != plan.resident_keys:
                            raise ValueError(
                                f"layer {layer_id} forward ledger resident 不守恒: "
                                f"{sorted(ledger.live_keys)} != {sorted(plan.resident_keys)}"
                            )
                        bucket.workspace = 0
                        bucket.act_live = base_act_live + resident
                        pinned[(event.mb, layer_id)] = (resident, offloaded)
                        record(f"fwd_end@{layer_id}", layer_id=layer_id)
                    if not keep_gathered:
                        bucket.gather_buf = 0
                    if last_chunk and suffix_ops:
                        suffix_retained = frozenset(
                            tensor.storage_key
                            for op in suffix_ops for tensor in op.saves
                            if not tensor.is_weight
                        )
                        suffix_ledger = MemoryLedger()
                        base_act_live = bucket.act_live
                        by_op_name = {op.name: op for op in suffix_ops}
                        for action in forward_layer_actions(
                            SimpleNamespace(
                                layer_id=-(stage + 1), ops=suffix_ops
                            ),
                            suffix_retained,
                        ):
                            trace_action = action
                            if (
                                action.kind in {"FREE", "MOVE_OUT"}
                                and action.key in suffix_ledger.allocations
                            ):
                                size, allocation_bucket = (
                                    suffix_ledger.allocations[action.key]
                                )
                                trace_action = replace(
                                    action,
                                    size_bytes=size,
                                    bucket=allocation_bucket,
                                )
                            suffix_ledger.apply(action)
                            local = suffix_ledger.buckets
                            bucket.act_live = (
                                base_act_live + local.get("act_live", 0)
                            )
                            bucket.workspace = (
                                local.get("workspace", 0)
                                + local.get("communication", 0)
                            )
                            op = by_op_name.get(action.op_name)
                            bucket.gather_buf = (
                                _op_fsdp_buffer_bytes(op, pm) if op else 0
                            )
                            record(
                                "loss_logits"
                                if action.op_name == "lm_head"
                                else action.owner,
                                layer_id=-(stage + 1),
                                op_name=action.op_name,
                                action=trace_action,
                                live_allocations=suffix_ledger.allocations,
                            )
                        if suffix_ledger.live_keys != suffix_retained:
                            raise ValueError("edge forward ledger resident 不守恒")
                        bucket.workspace = 0
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    if last_chunk:
                        pinned[(event.mb, -1)] = (edge_resident, 0)
                        # The suffix ledger already left its saved tensors
                        # resident; add only edge values outside that ledger
                        # (currently embedding/token inputs).
                        bucket.act_live += edge_resident - suffix_resident
                        record("fwd_end")
                        if pp_boundary_bytes:
                            bucket.workspace = pp_boundary_bytes
                            record(f"pp_send@stage{stage}")
                            bucket.workspace = 0
                else:
                    if last_chunk and pp_boundary_bytes:
                        bucket.workspace = pp_boundary_bytes
                        record(f"pp_recv_grad@stage{stage}")
                        bucket.workspace = 0
                    for op in reversed(suffix_ops if last_chunk else ()):
                        bucket.gather_buf = _op_fsdp_buffer_bytes(op, pm)
                        record(f"bwd_edge_gather@{op.name}")
                        bucket.grad_buf = _op_grad_buffer_bytes(
                            op, pm, gradient_bytes
                        )
                        bucket.workspace = op.backward_workspace_bytes + max(
                            (comm.volume_bytes for comm in op.collectives),
                            default=0,
                        )
                        if bucket.workspace:
                            record(
                                "loss_bwd_dlogits"
                                if op.name.startswith("loss_")
                                else f"bwd_edge_op@{op.name}",
                                layer_id=-(stage + 1),
                                op_name=op.name,
                            )
                        bucket.workspace = 0
                        record(
                            f"bwd_edge_grad@{op.name}",
                            layer_id=-(stage + 1),
                            op_name=op.name,
                        )
                        bucket.grad_buf = 0
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    for reverse_index, layer_id in enumerate(reversed(active_layer_ids)):
                        layer = by_id[layer_id]
                        index = len(active_layer_ids) - 1 - reverse_index
                        if keep_gathered:
                            bucket.gather_buf = sum(
                                layer_fsdp_buffer[item]
                                for item in gathered_ids
                            )
                        else:
                            bucket.gather_buf = gather_window(index, backward=True, ids=active_layer_ids)
                        record(f"bwd_gather@{layer_id}", layer_id=layer_id)

                        resident, _ = pinned[(event.mb, layer_id)]
                        plan = memory_plans[layer_id]
                        reverse_ids = list(reversed(active_layer_ids))
                        window = reverse_ids[
                            reverse_index : reverse_index
                            + swap.default_prefetch
                        ]
                        bucket.swap_buf = sum(
                            pinned[(event.mb, item)][1] for item in window
                        )
                        if bucket.swap_buf:
                            record(
                                f"bwd_swap_prefetch@{layer_id}",
                                layer_id=layer_id,
                            )
                        bucket.swap_buf = 0
                        if plan.recomputed_ops:
                            bucket.recomp_scratch = (
                                plan.recompute_scratch_bytes
                            )
                            record(
                                f"bwd_recompute@{layer_id}",
                                layer_id=layer_id,
                            )
                            bucket.recomp_scratch = 0
                        if plan.recomputed_comm_ops:
                            bucket.recomp_scratch = plan.recompute_comm_bytes
                            record(
                                f"bwd_recompute_comm@{layer_id}",
                                layer_id=layer_id,
                            )
                            bucket.recomp_scratch = 0

                        # Backward kernels and their communication staging run
                        # while the layer's saved activations are still live.
                        for op in reversed(layer.ops):
                            bucket.workspace = op.backward_workspace_bytes + max(
                                (comm.volume_bytes for comm in op.collectives),
                                default=0,
                            )
                            if bucket.workspace or trace is not None:
                                record(
                                    f"bwd_op@{layer_id}:{op.name}",
                                    layer_id=layer_id,
                                    op_name=op.name,
                                )
                        bucket.workspace = 0

                        bucket.grad_buf = layer_grad_buffer[layer_id]
                        record(f"bwd_grad@{layer_id}", layer_id=layer_id)
                        bucket.grad_buf = 0
                        bucket.act_live -= resident
                        pinned.pop((event.mb, layer_id))
                        record(
                            f"bwd_activation_release@{layer_id}",
                            layer_id=layer_id,
                        )
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    if last_chunk:
                        resident, _ = pinned.pop((event.mb, -1), (0, 0))
                        bucket.act_live -= resident
                        record("bwd_edge_activation_release", layer_id=-1)
                    for op in reversed(prefix_ops if first_chunk else ()):
                        bucket.gather_buf = _op_fsdp_buffer_bytes(op, pm)
                        record(f"bwd_edge_gather@{op.name}")
                        bucket.grad_buf = _op_grad_buffer_bytes(
                            op, pm, gradient_bytes
                        )
                        record(f"bwd_edge_grad@{op.name}")
                        bucket.grad_buf = 0
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    record("bwd_end")

            if optimizer is not None:
                from .optimizers import optimizer_step_workspace_bytes
                largest_gradient = max(
                    (*layer_grad_buffer.values(), *(
                        _op_grad_buffer_bytes(op, pm, gradient_bytes)
                        for op in edge_ops
                    )),
                    default=0,
                )
                bucket.workspace = optimizer_step_workspace_bytes(
                    optimizer, largest_gradient,
                    pm.pc.optimizer_offload,
                )
                if bucket.workspace:
                    record(f"optimizer_step@stage{stage}")
                bucket.workspace = 0

            assert peak_breakdown is not None
            result[stage] = StagePeak(
                stage,
                peak,
                peak_breakdown,
                peak_event,
                peak > max_device_memory,
                BucketPeaks(**high.__dict__),
            )
        return result
