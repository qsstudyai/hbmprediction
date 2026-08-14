"""P0 评估器门面和可解释峰值显存报告。"""

from __future__ import annotations

from dataclasses import dataclass

from .mem_timeline import MemTimeline, StagePeak
from .parallel_model import ParallelModel
from .shape_eval import ShapeEval
from .static_mem import StaticMem


@dataclass(frozen=True)
class PeakMemoryReport:
    per_stage: tuple[StagePeak, ...]
    tightest_stage: int
    oom: bool
    static_per_stage: dict | None = None
    summary: "ModelSummary | None" = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelSummary:
    name: str
    num_layers: int
    world_size: int


class Evaluator:
    """离线计算一个并行训练配置的逐 stage 峰值显存。"""

    def __init__(
        self,
        model_spec,
        parallel_config,
        optimizer,
        hardware,
        recompute=None,
        swap=None,
        warnings=(),
    ):
        self.spec = model_spec
        self.pc = parallel_config
        self.optimizer = optimizer
        self.hardware = hardware
        from .specs import RecomputeSpec, SwapSpec

        self.recompute = recompute or RecomputeSpec()
        self.swap = swap or SwapSpec()
        self.warnings = tuple(warnings)

    def evaluate(self) -> PeakMemoryReport:
        pm = ParallelModel(self.pc, self.spec.dims.n_layers)
        graph = ShapeEval().resolve(self.spec, pm)
        persistent = StaticMem().compute(
            graph, self.optimizer, pm, self.pc.cpu_offload
        )
        peaks = MemTimeline().simulate(
            graph,
            self.recompute,
            self.swap,
            pm,
            persistent,
            self.hardware.framework_reserve,
            self.hardware.max_device_memory,
        )
        per_stage = tuple(peaks[stage] for stage in sorted(peaks))
        tightest = max(per_stage, key=lambda item: item.peak_bytes).stage
        return PeakMemoryReport(
            per_stage=per_stage,
            tightest_stage=tightest,
            oom=any(item.oom for item in per_stage),
            static_per_stage=persistent,
            summary=ModelSummary(
                self.spec.name,
                self.spec.dims.total_layers,
                self.pc.world_size,
            ),
            warnings=self.warnings,
        )
