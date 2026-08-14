"""Run end-to-end functional checks with human-readable progress output."""

from __future__ import annotations

from dataclasses import asdict

from cost_eval import (
    DimTable,
    Evaluator,
    HardwareSpec,
    ModelSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)
from cost_eval.layers import build_dense_decoder, build_moe_decoder


GIB = 2**30
MIB = 2**20


def gib(value: int) -> str:
    return f"{value / GIB:.4f} GiB"


def build_dense_model(n_layers: int = 8) -> ModelSpec:
    dims = DimTable(
        H=1024,
        F=4096,
        n_heads=16,
        n_kv=16,
        head_dim=64,
        S=1024,
        B=1,
        vocab=32000,
        n_layers=n_layers,
    )
    return ModelSpec(
        "functional-dense",
        dims,
        ("dense",) * n_layers,
        {"dense": build_dense_decoder(dims)},
    )


def build_moe_model() -> ModelSpec:
    dims = DimTable(
        H=512,
        F=1024,
        n_heads=8,
        n_kv=8,
        head_dim=64,
        S=512,
        B=1,
        vocab=32000,
        n_layers=2,
        n_experts=8,
        topk=2,
        moe_F=1024,
    )
    return ModelSpec(
        "functional-moe",
        dims,
        ("moe",) * dims.n_layers,
        {"moe": build_moe_decoder(dims)},
    )


def evaluate(
    model: ModelSpec,
    parallel: ParallelConfig,
    recompute: RecomputeSpec | None = None,
    swap: SwapSpec | None = None,
    max_memory: int = 8 * GIB,
):
    return Evaluator(
        model,
        parallel,
        OptimizerSpec.adamw(),
        HardwareSpec(max_memory, framework_reserve=256 * MIB),
        recompute,
        swap,
    ).evaluate()


def print_report(name: str, parallel: ParallelConfig, report) -> None:
    print(f"\n[RUN] {name}")
    print(
        "  Config: "
        f"DP={parallel.dp_replicate}, FSDP={parallel.dp_shard}, "
        f"TP={parallel.tp}, PP={parallel.pp}, CP={parallel.cp}, "
        f"EP={parallel.ep}, microbatches={parallel.num_microbatches}"
    )
    for stage in report.per_stage:
        print(
            f"  Stage {stage.stage}: peak={gib(stage.peak_bytes)}, "
            f"event={stage.peak_event}, OOM={stage.oom}"
        )
        for bucket, value in asdict(stage.breakdown).items():
            print(f"    {bucket:16s} {gib(value)}")
    print(f"  Tightest stage: {report.tightest_stage}")
    print(f"  Final OOM: {report.oom}")


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        raise AssertionError(message)
    print(f"  [PASS] {message}")


def main() -> None:
    print("=" * 72)
    print("HBM Prediction functional test")
    print("=" * 72)
    passed = 0

    dense = build_dense_model()

    tp1_config = ParallelConfig(tp=1, dp_shard=2, num_microbatches=4)
    tp1_report = evaluate(dense, tp1_config)
    print_report("Dense baseline", tp1_config, tp1_report)
    check(tp1_report.per_stage[0].peak_bytes > 0, "Dense evaluation returns a positive peak")
    passed += 1

    tp4_config = ParallelConfig(tp=4, dp_shard=2, num_microbatches=4)
    tp4_report = evaluate(dense, tp4_config)
    print_report("Tensor parallel scaling", tp4_config, tp4_report)
    check(
        tp4_report.per_stage[0].peak_bytes < tp1_report.per_stage[0].peak_bytes,
        "TP=4 uses less peak memory than TP=1",
    )
    passed += 1

    pipeline_config = ParallelConfig(tp=4, pp=2, dp_shard=2, num_microbatches=6)
    normal_report = evaluate(dense, pipeline_config)
    recompute_report = evaluate(dense, pipeline_config, RecomputeSpec("full"))
    print_report("Pipeline without recompute", pipeline_config, normal_report)
    print_report("Pipeline with full recompute", pipeline_config, recompute_report)
    check(
        recompute_report.per_stage[0].peak_bytes < normal_report.per_stage[0].peak_bytes,
        "Full recompute reduces the stage-0 activation peak",
    )
    passed += 1

    oom_config = ParallelConfig(tp=1)
    oom_report = evaluate(dense, oom_config, max_memory=1)
    print_report("Forced OOM", oom_config, oom_report)
    check(oom_report.oom, "A one-byte memory limit is reported as OOM")
    passed += 1

    moe = build_moe_model()
    moe_config = ParallelConfig(tp=2, ep=4, dp_shard=2, num_microbatches=2)
    moe_report = evaluate(moe, moe_config, swap=SwapSpec(enabled=True))
    print_report("MoE with activation swap", moe_config, moe_report)
    check(
        moe_report.per_stage[0].bucket_peaks.swap_buf > 0,
        "MoE swap contributes a visible prefetch buffer",
    )
    passed += 1

    print("\n" + "=" * 72)
    print(f"Functional test result: PASS ({passed}/5 checks passed)")
    print("=" * 72)


if __name__ == "__main__":
    main()
