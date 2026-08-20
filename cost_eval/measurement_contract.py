"""互斥、可审计的真实 NPU HBM 测量合同。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


MEASUREMENT_SCHEMA = "hbmprediction.measurement-contract.v1"


def _int_mapping(value: Mapping[Any, Any], field: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} 必须是映射")
    result: dict[int, int] = {}
    for key, item in value.items():
        try:
            converted_key = int(key)
            converted_value = int(item)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} 包含非法值: {key!r}={item!r}") from exc
        if converted_key < 0 or converted_value < 0:
            raise ValueError(f"{field} 不能包含负数")
        result[converted_key] = converted_value
    return result


@dataclass(frozen=True)
class HBMComponents:
    """同一采集窗口内互斥的设备显存分量。"""

    device_baseline_bytes: int
    model_active_peak_bytes: int
    allocator_pool_peak_bytes: int
    untracked_runtime_bytes: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in self.__dict__.values()):
            raise ValueError("HBM 分量不能为负数")

    @property
    def dynamic_total_bytes(self) -> int:
        return max(
            self.model_active_peak_bytes, self.allocator_pool_peak_bytes
        ) + self.untracked_runtime_bytes

    @property
    def nominal_total_bytes(self) -> int:
        return self.device_baseline_bytes + self.dynamic_total_bytes

    def validate_dynamic_total(
        self, measured: int, tolerance_bytes: int = 2 * 2**20
    ) -> None:
        if measured < 0:
            raise ValueError("measured dynamic HBM 不能为负数")
        delta = abs(self.dynamic_total_bytes - measured)
        if delta > tolerance_bytes:
            raise ValueError(
                "HBM 分量不守恒: "
                f"max(active,pool)+runtime={self.dynamic_total_bytes}, "
                f"measured={measured}, delta={delta}"
            )


@dataclass(frozen=True)
class RankHBMMeasurement:
    rank: int
    stage: int
    baseline_bytes: int
    peak_used_bytes: int
    peak_delta_bytes: int
    sample_count: int

    def __post_init__(self) -> None:
        if self.rank < 0 or self.stage < 0 or self.sample_count < 0:
            raise ValueError("rank/stage/sample_count 不能为负数")
        if min(
            self.baseline_bytes, self.peak_used_bytes, self.peak_delta_bytes
        ) < 0:
            raise ValueError("per-rank HBM 字节数不能为负数")
        expected = self.peak_used_bytes - self.baseline_bytes
        # npu-smi samples are MiB-quantized and the saved baseline can use a
        # neighbouring idle sample.  Allow one quantization unit.
        if abs(expected - self.peak_delta_bytes) > 2**20:
            raise ValueError(
                f"rank {self.rank} peak_used-baseline 与 peak_delta 不守恒"
            )


@dataclass(frozen=True)
class HBMMeasurement:
    name: str
    per_stage_peak_bytes: Mapping[int, int]
    per_rank: tuple[RankHBMMeasurement, ...]
    components: HBMComponents
    oom: bool
    quality_status: str
    quality_issues: tuple[str, ...]
    collection_attempt_id: Optional[str] = None

    @classmethod
    def from_case_result(cls, root: Mapping[str, Any]) -> "HBMMeasurement":
        if not isinstance(root, Mapping):
            raise ValueError("case_result 根节点必须是映射")
        measured = root.get("measured")
        if not isinstance(measured, Mapping):
            raise ValueError("case_result 缺少 measured 映射")
        stage_peaks = _int_mapping(
            measured.get("per_stage_peak_bytes", {}),
            "measured.per_stage_peak_bytes",
        )
        if not stage_peaks and not bool(measured.get("oom")):
            raise ValueError("非 OOM case 必须包含 per_stage_peak_bytes")

        raw_ranks = measured.get("per_device", {})
        if not isinstance(raw_ranks, Mapping):
            raise ValueError("measured.per_device 必须是映射")
        ranks = []
        for raw_rank, value in raw_ranks.items():
            if not isinstance(value, Mapping):
                raise ValueError(f"per_device[{raw_rank!r}] 必须是映射")
            ranks.append(RankHBMMeasurement(
                rank=int(raw_rank),
                stage=int(value.get("stage", 0)),
                baseline_bytes=int(value.get("baseline_bytes", 0)),
                peak_used_bytes=int(value.get("peak_used_bytes", 0)),
                peak_delta_bytes=int(value.get("peak_delta_bytes", 0)),
                sample_count=int(value.get("sample_count", 0)),
            ))
        ranks.sort(key=lambda item: item.rank)

        raw_components = measured.get("hbm_components")
        if not isinstance(raw_components, Mapping):
            raise ValueError("measured.hbm_components 必须是映射")
        components = HBMComponents(
            device_baseline_bytes=int(
                raw_components.get(
                    "device_baseline_bytes",
                    measured.get("device_baseline_bytes", 0),
                )
            ),
            model_active_peak_bytes=int(
                raw_components.get("model_dynamic_peak_bytes", 0)
            ),
            allocator_pool_peak_bytes=int(
                raw_components.get("allocator_pool_peak_bytes", 0)
            ),
            untracked_runtime_bytes=int(
                raw_components.get("untracked_runtime_bytes", 0)
            ),
        )
        if stage_peaks:
            components.validate_dynamic_total(max(stage_peaks.values()))

        quality = root.get("measurement_quality") or measured.get(
            "measurement_quality", {}
        )
        if isinstance(quality, Mapping) and "hbm" in quality:
            quality = quality.get("hbm", {})
        if not isinstance(quality, Mapping):
            quality = {}
        issues = quality.get("issues", ())
        if isinstance(issues, str):
            issues = (issues,)
        return cls(
            name=str(root.get("name", "case")),
            per_stage_peak_bytes=stage_peaks,
            per_rank=tuple(ranks),
            components=components,
            oom=bool(measured.get("oom", False)),
            quality_status=str(quality.get("status", "unknown")),
            quality_issues=tuple(str(item) for item in issues),
            collection_attempt_id=(
                str(root["collection_attempt_id"])
                if root.get("collection_attempt_id") is not None
                else None
            ),
        )

    @property
    def worst_dynamic_bytes(self) -> int:
        if self.per_stage_peak_bytes:
            return max(self.per_stage_peak_bytes.values())
        return 0

    @property
    def worst_total_bytes(self) -> int:
        if self.per_rank:
            return max(item.peak_used_bytes for item in self.per_rank)
        return self.components.nominal_total_bytes

    def eligibility_issues(
        self, expected_rank_count: Optional[int] = None, min_samples: int = 20
    ) -> tuple[str, ...]:
        issues = list(self.quality_issues)
        if self.quality_status != "valid":
            issues.append(f"quality_status={self.quality_status}")
        if expected_rank_count is not None and len(self.per_rank) != expected_rank_count:
            issues.append(
                f"rank_count={len(self.per_rank)} expected={expected_rank_count}"
            )
        for item in self.per_rank:
            if item.sample_count < min_samples:
                issues.append(
                    f"rank={item.rank} sample_count={item.sample_count} < {min_samples}"
                )
        if not self.per_rank and not self.oom:
            issues.append("missing per-rank HBM measurements")
        return tuple(dict.fromkeys(issues))

    def validate_stage_aggregation(self, tolerance_bytes: int = 2**20) -> None:
        by_stage: dict[int, int] = {}
        for item in self.per_rank:
            by_stage[item.stage] = max(
                by_stage.get(item.stage, 0), item.peak_delta_bytes
            )
        if set(by_stage) != set(self.per_stage_peak_bytes):
            raise ValueError(
                "per-rank stage 与 per_stage_peak_bytes 不一致: "
                f"{sorted(by_stage)} != {sorted(self.per_stage_peak_bytes)}"
            )
        for stage, value in by_stage.items():
            measured = self.per_stage_peak_bytes[stage]
            if abs(value - measured) > tolerance_bytes:
                raise ValueError(
                    f"stage {stage} max-rank peak 与 stage peak 不一致"
                )
