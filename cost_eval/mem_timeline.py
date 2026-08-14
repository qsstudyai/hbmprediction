"""M6：沿 1F1B 时间线追踪张量存活并计算峰值。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    kind: str
    mb: int
    layer: int = -1
    chunk: int = 0


def build_1f1b(stage: int, pp: int, m: int = None, microbatches: int = None) -> list[Event]:
    if m is None:
        m = microbatches
    if not (0 <= stage < pp) or m <= 0:
        raise ValueError("stage/pp/m 参数非法")
    warmup = min(pp - 1 - stage, m)
    events = [Event("FWD", mb) for mb in range(warmup)]
    fwd_mb, bwd_mb = warmup, 0
    while bwd_mb < m:
        if fwd_mb < m:
            events.append(Event("FWD", fwd_mb))
            fwd_mb += 1
        events.append(Event("BWD", bwd_mb))
        bwd_mb += 1
    return events


def build_interleaved_1f1b(
    stage: int,
    pp: int,
    m: int,
    interleave: int,
) -> list[Event]:
    """Build a deterministic virtual-stage interleaved 1F1B schedule.

    Each physical stage is split into ``interleave`` virtual chunks.  Every
    chunk gets the normal 1F1B schedule for the physical pipeline and the
    chunk schedules are merged round-robin.  The result keeps the real
    microbatch dependencies explicit while allowing the timed builder and the
    memory ledger to account for chunk-local activation lifetimes.
    """

    if interleave <= 0:
        raise ValueError("interleave 必须为正整数")
    if interleave == 1:
        return build_1f1b(stage, pp, m)
    # Expand each physical 1F1B macro in virtual-chunk order.  Forward work
    # enters chunk 0 -> chunk 1 -> ..., while backward work drains in the
    # reverse order.  Merging independent per-chunk schedules (the old
    # implementation) could issue ``FWD c0, BWD c0, FWD c1, BWD c1`` and
    # artificially serialize an interleaved pipeline.
    result: list[Event] = []
    for event in build_1f1b(stage, pp, m):
        chunks = range(interleave) if event.kind == "FWD" else range(interleave - 1, -1, -1)
        result.extend(Event(event.kind, event.mb, event.layer, chunk) for chunk in chunks)
    return result


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

    @property
    def resident(self): return self.resident_bytes
    @property
    def offloaded(self): return self.offloaded_bytes
    @property
    def recomputed(self): return self.recompute_scratch_bytes


def _unique_tensors(tensors):
    unique = {}
    for tensor in tensors:
        unique.setdefault(tensor.storage_id or tensor.name, tensor)
    return unique.values()


def _layer_memory_plan(layer, recompute, swap) -> LayerMemoryPlan:
    op_names = [op.name for op in layer.ops]
    recomputed = recompute.recomputed_ops(layer.layer_id, op_names)
    swapped = swap.swapped_ops(layer.layer_id, op_names) - recomputed
    comm_recomputed = recompute.recomputed_comm_ops(layer.layer_id, layer.ops)

    tensors = {}
    saves_by_op = {}
    for op in layer.ops:
        saves_by_op[op.name] = {tensor.name for tensor in op.saves}
        for tensor in (*op.inputs, op.output, *op.saves):
            if not tensor.is_weight:
                tensors.setdefault(tensor.name, tensor)

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
                tensors.setdefault(checkpoint.name, checkpoint)
                checkpoints.add(checkpoint.name)
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
                item.output_tensor for item in op.collectives
                if item.output_tensor is not None
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
    )


def _save_plan(layer, recompute, swap):
    """Public save/recompute/swap partition helper."""
    return _layer_memory_plan(layer, recompute, swap)


def _layer_fsdp_buffer_bytes(layer, pm) -> int:
    total = 0
    for tensor in _unique_tensors(
        tensor for op in layer.ops for tensor in op.params
    ):
        degree = (
            pm.efsdp_degree() if tensor.is_expert else pm.fsdp_degree()
        )
        if degree > 1 or pm.pc.cpu_offload:
            total += tensor.local_bytes
    return total


def _layer_workspace(layer) -> int:
    return max((
        op.workspace_bytes + max(
            (comm.volume_bytes for comm in op.collectives), default=0
        )
        for op in layer.ops
    ), default=0)


def _op_fsdp_buffer_bytes(op, pm) -> int:
    if pm.fsdp_degree() == 1 and not pm.pc.cpu_offload:
        return 0
    return sum(tensor.local_bytes for tensor in _unique_tensors(op.params))


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
            edge_ops = tuple((graph.edge_ops or {}).get(stage, ()))
            prefix_ops = tuple(op for op in edge_ops if op.name == "embedding")
            suffix_ops = tuple(op for op in edge_ops if op.name != "embedding")
            edge_resident = _edge_saved_bytes(edge_ops)
            memory_plans = {
                layer.layer_id: _layer_memory_plan(
                    layer, recompute, swap
                )
                for layer in layers
            }
            peak = -1
            peak_event = "initial"
            peak_breakdown = None

            def record(tag: str) -> None:
                nonlocal peak, peak_event, peak_breakdown
                current = bucket.total() + framework_reserve
                for name, value in bucket.__dict__.items():
                    setattr(high, name, max(getattr(high, name), value))
                if current > peak:
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
            pinned: dict[tuple[int, int], tuple[int, int]] = {}
            record("initial")

            for event in build_interleaved_1f1b(
                stage,
                pm.degree("pp"),
                pm.pc.num_microbatches,
                pm.pc.interleave,
            ):
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
                            gathered_ids.update(layer_ids[index:upper])
                            bucket.gather_buf = sum(
                                layer_fsdp_buffer[item]
                                for item in gathered_ids
                            )
                        else:
                            bucket.gather_buf = gather_window(index, ids=active_layer_ids)
                        record(f"fwd_gather@{layer_id}")

                        bucket.workspace = _layer_workspace(layer)
                        record(f"fwd_op@{layer_id}")
                        bucket.workspace = 0

                        plan = memory_plans[layer_id]
                        resident = plan.resident_bytes
                        offloaded = plan.offloaded_bytes
                        pinned[(event.mb, layer_id)] = (resident, offloaded)
                        bucket.act_live += resident
                        record(f"fwd_end@{layer_id}")
                    if not keep_gathered:
                        bucket.gather_buf = 0
                    for op in suffix_ops if last_chunk else ():
                        bucket.gather_buf = _op_fsdp_buffer_bytes(op, pm)
                        record(f"fwd_edge_gather@{op.name}")
                        bucket.workspace = op.workspace_bytes + max(
                            (comm.volume_bytes for comm in op.collectives), default=0
                        )
                        record("loss_logits" if op.name == "lm_head" else f"fwd_edge_op@{op.name}")
                        bucket.workspace = 0
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    if last_chunk:
                        pinned[(event.mb, -1)] = (edge_resident, 0)
                        bucket.act_live += edge_resident
                        record("fwd_end")
                else:
                    for op in reversed(suffix_ops if last_chunk else ()):
                        bucket.gather_buf = _op_fsdp_buffer_bytes(op, pm)
                        record(f"bwd_edge_gather@{op.name}")
                        bucket.grad_buf = bucket.gather_buf
                        record(f"bwd_edge_grad@{op.name}")
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
                        record(f"bwd_gather@{layer_id}")

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
                            record(f"bwd_swap_prefetch@{layer_id}")
                        bucket.swap_buf = 0
                        if plan.recomputed_ops:
                            bucket.recomp_scratch = (
                                plan.recompute_scratch_bytes
                            )
                            record(f"bwd_recompute@{layer_id}")
                            bucket.recomp_scratch = 0
                        if plan.recomputed_comm_ops:
                            bucket.recomp_scratch = plan.recompute_comm_bytes
                            record(f"bwd_recompute_comm@{layer_id}")
                            bucket.recomp_scratch = 0

                        bucket.grad_buf = layer_fsdp_buffer[layer_id]
                        record(f"bwd_grad@{layer_id}")
                        bucket.grad_buf = 0
                        bucket.act_live -= resident
                        pinned.pop((event.mb, layer_id))
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    if last_chunk:
                        resident, _ = pinned.pop((event.mb, -1), (0, 0))
                        bucket.act_live -= resident
                    for op in reversed(prefix_ops if first_chunk else ()):
                        bucket.gather_buf = _op_fsdp_buffer_bytes(op, pm)
                        record(f"bwd_edge_gather@{op.name}")
                        bucket.grad_buf = bucket.gather_buf
                        record(f"bwd_edge_grad@{op.name}")
                        bucket.grad_buf = 0
                        if not keep_gathered:
                            bucket.gather_buf = 0
                    record("bwd_end")

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
