"""P0 评估器门面和可解释峰值显存报告。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .allocator_model import AllocatorModel
from .mem_timeline import MemTimeline, StagePeak
from .parallel_model import ParallelModel
from .shape_eval import ShapeEval
from .static_mem import StaticMem
from .trace import MemoryTrace


@dataclass(frozen=True)
class PeakMemoryReport:
    per_stage: tuple[StagePeak, ...]
    tightest_stage: int
    oom: bool
    static_per_stage: dict | None = None
    summary: "ModelSummary | None" = None
    warnings: tuple[str, ...] = ()
    schema_version: str = "hbmprediction.report.v2"
    tightest_rank: int = 0
    tightest_event: str = ""
    oom_status: str = "risky"
    capability_status: str = "experimental"
    fingerprints: dict | None = None
    per_rank: tuple["RankPeak", ...] = ()


@dataclass(frozen=True)
class ModelSummary:
    name: str
    num_layers: int
    world_size: int


@dataclass(frozen=True)
class RankPeak:
    rank: int
    stage: int
    peak_bytes: int
    safe_upper_bytes: int
    peak_event: str
    oom_status: str


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
        legacy_warning = (
            (
                "hardware.framework_reserve is deprecated; it is mapped to "
                "untracked_runtime_point_bytes"
            ),
        ) if hardware.framework_reserve else ()
        self.warnings = tuple(warnings) + legacy_warning

    def evaluate(self, trace: MemoryTrace | None = None) -> PeakMemoryReport:
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
            0,
            self.hardware.max_device_memory,
            self.optimizer.gradient_bytes,
            self.optimizer,
            trace,
        )
        allocator = AllocatorModel()
        capability_status = self._capability_status()
        composed = {}
        for stage, physical in peaks.items():
            estimate = allocator.estimate(
                physical.peak_bytes, physical.breakdown, self.hardware
            )
            if trace is not None:
                trace.add_stage_composition(
                    stage,
                    peak_event=physical.peak_event,
                    physical_active_peak_bytes=(
                        estimate.physical_active_peak_bytes
                    ),
                    allocator_pool_peak_bytes=(
                        estimate.allocator_pool_peak_bytes
                    ),
                    device_baseline_bytes=estimate.device_baseline_bytes,
                    untracked_runtime_point_bytes=(
                        estimate.untracked_runtime_point_bytes
                    ),
                    fragmentation_point_bytes=(
                        estimate.fragmentation_point_bytes
                    ),
                    total_point_bytes=estimate.total_point_bytes,
                    safe_upper_bytes=estimate.safe_upper_bytes,
                )
            usable = int(self.hardware.usable_device_memory)
            if estimate.total_point_bytes > usable:
                status = "predicted_oom"
            elif capability_status == "unsupported":
                status = "unsupported"
            elif estimate.safe_upper_bytes > usable:
                status = "risky"
            elif capability_status != "validated":
                status = "risky"
            else:
                status = "definitely_safe"
            overhead = estimate.total_point_bytes - estimate.physical_active_peak_bytes
            breakdown = replace(physical.breakdown, framework=overhead)
            composed[stage] = replace(
                physical,
                peak_bytes=estimate.total_point_bytes,
                breakdown=breakdown,
                oom=status == "predicted_oom",
                physical_dynamic_peak_bytes=estimate.physical_active_peak_bytes,
                allocator_pool_peak_bytes=estimate.allocator_pool_peak_bytes,
                device_baseline_bytes=estimate.device_baseline_bytes,
                untracked_runtime_point_bytes=estimate.untracked_runtime_point_bytes,
                fragmentation_point_bytes=estimate.fragmentation_point_bytes,
                total_point_bytes=estimate.total_point_bytes,
                safe_upper_bytes=estimate.safe_upper_bytes,
                oom_status=status,
            )
        peaks = composed
        per_stage = tuple(peaks[stage] for stage in sorted(peaks))
        tightest = max(per_stage, key=lambda item: item.peak_bytes).stage
        tightest_peak = peaks[tightest]
        per_rank = tuple(
            RankPeak(
                rank, pm.stage_of_rank(rank),
                peaks[pm.stage_of_rank(rank)].peak_bytes,
                peaks[pm.stage_of_rank(rank)].safe_upper_bytes,
                peaks[pm.stage_of_rank(rank)].peak_event,
                peaks[pm.stage_of_rank(rank)].oom_status,
            )
            for rank in range(pm.world_size)
        )
        tightest_rank = max(per_rank, key=lambda item: item.peak_bytes).rank
        statuses = {item.oom_status for item in per_stage}
        overall_status = (
            "predicted_oom" if "predicted_oom" in statuses
            else "unsupported" if "unsupported" in statuses
            else "risky" if "risky" in statuses
            else "definitely_safe"
        )
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
            tightest_rank=tightest_rank,
            tightest_event=tightest_peak.peak_event,
            oom_status=overall_status,
            capability_status=capability_status,
            fingerprints={
                "hardware": self.hardware.hardware_profile,
                "runtime": self.hardware.runtime_profile,
                "source": self.hardware.source_profile,
                "model_source_commit": self.spec.capabilities.get("source_commit"),
            },
            per_rank=per_rank,
        )

    def evaluate_with_trace(
        self, capture_live_tensors: bool = True
    ) -> tuple[PeakMemoryReport, MemoryTrace]:
        """Evaluate and retain every modeled memory transition."""

        trace = MemoryTrace(capture_live_tensors=capture_live_tensors)
        return self.evaluate(trace=trace), trace

    def _capability_status(self) -> str:
        caps = self.spec.capabilities
        if caps.get("contract_status") == "unsupported":
            return "unsupported"
        if (
            caps.get("model_family") == "deepseek_v4"
            and (
                caps.get("probe_status") != "passed"
                or caps.get("hbm_validation_status") != "validated"
            )
        ):
            return "unsupported"
        profiles_known = all(
            value != "unknown" for value in (
                self.hardware.hardware_profile,
                self.hardware.runtime_profile,
                self.hardware.source_profile,
            )
        )
        if profiles_known and caps.get("hbm_validation_status") == "validated":
            return "validated"
        return "experimental"
