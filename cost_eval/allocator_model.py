"""Auditable allocator/runtime composition for total device HBM."""

from __future__ import annotations

from dataclasses import dataclass


def _round_up(value: int, granularity: int) -> int:
    return 0 if value == 0 else ((value + granularity - 1) // granularity) * granularity


@dataclass(frozen=True)
class AllocatorEstimate:
    physical_active_peak_bytes: int
    allocator_pool_peak_bytes: int
    device_baseline_bytes: int
    untracked_runtime_point_bytes: int
    fragmentation_point_bytes: int
    total_point_bytes: int
    safe_upper_bytes: int


class AllocatorModel:
    """First-version bucket model; no fitted global multiplier is used."""

    def estimate(self, active_bytes: int, breakdown, hardware) -> AllocatorEstimate:
        if active_bytes < 0:
            raise ValueError("physical active 不能为负数")
        granularity = hardware.allocator_granularity_bytes
        components = (
            breakdown.persistent,
            breakdown.act_live,
            breakdown.gather_buf,
            breakdown.grad_buf,
            breakdown.recomp_scratch,
            breakdown.swap_buf,
            breakdown.workspace,
        )
        fragmentation = sum(
            _round_up(value, granularity) - value for value in components
        )
        pool = max(
            active_bytes,
            hardware.allocator_pool_point_bytes,
            active_bytes + hardware.allocator_pool_slack_point_bytes,
        )
        runtime = (
            hardware.untracked_runtime_point_bytes
            if hardware.untracked_runtime_point_bytes
            else hardware.framework_reserve
        )
        total = (
            hardware.device_baseline_bytes
            + max(active_bytes, pool)
            + runtime
            + fragmentation
        )
        safe = total + hardware.calibrated_upper_margin_bytes + hardware.ood_margin_bytes
        return AllocatorEstimate(
            active_bytes,
            pool,
            hardware.device_baseline_bytes,
            runtime,
            fragmentation,
            total,
            safe,
        )
