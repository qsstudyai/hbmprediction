"""运行两个离线内存评估样例：Dense 与 MoE。"""

from dataclasses import asdict
from json import dumps

from cost_eval.layers import build_dense_decoder, build_moe_decoder
from cost_eval.model_spec import DimTable, ModelSpec
from cost_eval.report import Evaluator
from cost_eval.specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)

GIB = 2**30


def evaluate_dense():
    dims = DimTable(
        H=4096,
        F=11008,
        n_heads=32,
        n_kv=32,
        head_dim=128,
        S=4096,
        B=1,
        vocab=32000,
        n_layers=4,
    )
    spec = ModelSpec(
        "llama-like",
        dims,
        ("dense",) * dims.n_layers,
        {"dense": build_dense_decoder(dims)},
    )
    return Evaluator(
        spec,
        ParallelConfig(
            tp=8,
            dp_shard=8,
            sequence_parallel=True,
            num_microbatches=4,
        ),
        OptimizerSpec.adamw(),
        HardwareSpec(max_device_memory=60 * GIB, framework_reserve=1 * GIB),
        RecomputeSpec("full"),
        SwapSpec(),
    ).evaluate()


def evaluate_moe():
    dims = DimTable(
        H=1024,
        F=4096,
        n_heads=16,
        n_kv=4,
        head_dim=64,
        S=2048,
        B=1,
        vocab=32000,
        n_layers=4,
        n_experts=16,
        topk=2,
        moe_F=4096,
    )
    spec = ModelSpec(
        "moe-like",
        dims,
        ("moe",) * dims.n_layers,
        {"moe": build_moe_decoder(dims)},
    )
    return Evaluator(
        spec,
        ParallelConfig(
            tp=4,
            ep=8,
            dp_shard=4,
            sequence_parallel=True,
            num_microbatches=2,
        ),
        OptimizerSpec.adamw(),
        HardwareSpec(max_device_memory=80 * GIB, framework_reserve=1 * GIB),
        RecomputeSpec(),
        SwapSpec(enable=True, swap_layers={0, 1}),
    ).evaluate()


if __name__ == "__main__":
    reports = {
        "dense": asdict(evaluate_dense()),
        "moe": asdict(evaluate_moe()),
    }
    print(dumps(reports, ensure_ascii=False, indent=2))
