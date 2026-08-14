from cost_eval.layers import build_dense_decoder, build_moe_decoder
from cost_eval.mem_timeline import (
    MemTimeline,
    _layer_memory_plan,
    build_1f1b,
)
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.parallel_model import ParallelModel
from cost_eval.report import Evaluator
from cost_eval.shape_eval import ShapeEval
from cost_eval.specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)
from cost_eval.static_mem import StaticMem


def dense_setup(parallel=None):
    dims = DimTable(
        H=64,
        F=128,
        n_heads=4,
        n_kv=4,
        head_dim=16,
        S=128,
        B=1,
        vocab=100,
        n_layers=4,
    )
    spec = ModelSpec(
        "dense",
        dims,
        ("dense",) * 4,
        {"dense": build_dense_decoder(dims)},
    )
    pc = parallel or ParallelConfig(num_microbatches=2)
    pm = ParallelModel(pc, 4)
    graph = ShapeEval().resolve(spec, pm)
    persistent = StaticMem().compute(
        graph, OptimizerSpec.adamw(), pm, pc.cpu_offload
    )
    return dims, spec, pc, pm, graph, persistent


def simulate(pc, pm, graph, persistent, recompute=None, swap=None):
    return MemTimeline().simulate(
        graph,
        recompute or RecomputeSpec(),
        swap or SwapSpec(),
        pm,
        persistent,
        framework_reserve=4096,
        max_device_memory=10**12,
    )


def test_1f1b_schedule_has_warmup_and_balanced_counts():
    events = build_1f1b(stage=0, pp=4, m=8)
    prefix = []
    for event in events:
        if event.kind != "FWD":
            break
        prefix.append(event)
    assert len(prefix) == 4
    assert sum(event.kind == "FWD" for event in events) == 8
    assert sum(event.kind == "BWD" for event in events) == 8


def test_recompute_and_swap_reduce_peak_and_breakdown_reconciles():
    _, _, pc, pm, graph, persistent = dense_setup()
    baseline = simulate(pc, pm, graph, persistent)[0]
    recomputed = simulate(
        pc, pm, graph, persistent, RecomputeSpec("full")
    )[0]
    swapped = simulate(
        pc, pm, graph, persistent, swap=SwapSpec(True)
    )[0]
    assert recomputed.peak_bytes < baseline.peak_bytes
    assert swapped.peak_bytes < baseline.peak_bytes
    assert recomputed.breakdown.total == recomputed.peak_bytes
    assert recomputed.peak_event.startswith(("bwd_", "fwd_"))


def test_single_device_has_no_fsdp_transient_buffers():
    _, _, pc, pm, graph, persistent = dense_setup(
        ParallelConfig(num_microbatches=1)
    )
    peak = simulate(pc, pm, graph, persistent)[0]
    assert peak.breakdown.gather_buf == 0
    assert peak.breakdown.grad_buf == 0


def test_op_level_recompute_and_swap_partition_saved_tensors():
    _, _, _, _, graph, _ = dense_setup()
    layer = graph.stages[0][0]
    baseline = _layer_memory_plan(
        layer, RecomputeSpec(), SwapSpec()
    )
    selective = _layer_memory_plan(
        layer,
        RecomputeSpec(
            "select", select_ops={0: {"fc1", "swiglu"}}
        ),
        SwapSpec(),
    )
    full = _layer_memory_plan(
        layer, RecomputeSpec("full"), SwapSpec()
    )
    op_swap = _layer_memory_plan(
        layer,
        RecomputeSpec(),
        SwapSpec(True, swap_ops={0: {"flash"}}),
    )
    assert full.resident_bytes < selective.resident_bytes
    assert selective.resident_bytes < baseline.resident_bytes
    assert selective.recompute_scratch_bytes > 0
    assert 0 < op_swap.offloaded_bytes < baseline.resident_bytes


def test_fsdp_never_reshard_retains_more_gather_memory():
    always_pc = ParallelConfig(
        dp_shard=2,
        reshard_after_forward="always",
        prefetch_depth=0,
        num_microbatches=1,
    )
    _, _, _, always_pm, always_graph, always_persistent = dense_setup(always_pc)
    always = simulate(
        always_pc, always_pm, always_graph, always_persistent
    )[0]

    never_pc = ParallelConfig(
        dp_shard=2,
        reshard_after_forward="never",
        prefetch_depth=0,
        num_microbatches=1,
    )
    _, _, _, never_pm, never_graph, never_persistent = dense_setup(never_pc)
    never = simulate(never_pc, never_pm, never_graph, never_persistent)[0]
    assert never.peak_bytes > always.peak_bytes
    assert never.breakdown.gather_buf > 0


def test_oom_and_dense_evaluator_end_to_end():
    _, spec, _, _, _, _ = dense_setup()
    report = Evaluator(
        spec,
        ParallelConfig(tp=2, dp_shard=2, num_microbatches=2),
        OptimizerSpec.adamw(),
        HardwareSpec(max_device_memory=1),
        RecomputeSpec(),
        SwapSpec(),
    ).evaluate()
    assert report.oom
    assert report.tightest_stage == 0
    assert report.per_stage[0].breakdown.persistent > 0


def test_moe_evaluator_end_to_end():
    dims = DimTable(
        H=64,
        F=128,
        n_heads=4,
        n_kv=4,
        head_dim=16,
        S=64,
        B=1,
        vocab=100,
        n_layers=2,
        n_experts=8,
        topk=2,
        moe_F=128,
    )
    spec = ModelSpec(
        "moe",
        dims,
        ("moe", "moe"),
        {"moe": build_moe_decoder(dims)},
    )
    report = Evaluator(
        spec,
        ParallelConfig(tp=2, dp_shard=2, ep=4),
        OptimizerSpec.adamw(),
        HardwareSpec(max_device_memory=80 * 2**30),
        RecomputeSpec(),
        SwapSpec(),
    ).evaluate()
    assert not report.oom
    assert report.per_stage[0].peak_bytes > 0
