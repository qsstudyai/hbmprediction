"""Explicit calibration-only allocator/runtime profile selection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path


def layout_key(pc) -> str:
    return (
        f"tp{pc.tp}-cp{pc.cp}-pp{pc.pp}-ep{pc.ep}-"
        f"dp{pc.dp_shard}x{pc.dp_replicate}-i{pc.interleave}"
    )


@dataclass(frozen=True)
class RuntimeProfile:
    family: str
    layout: str
    sample_count: int
    runtime_point: int
    runtime_p90: int
    pool_slack_point: int
    pool_slack_p90: int
    baseline_point: int
    baseline_p90: int
    physical_shortfall_p90: int
    label_sha256: tuple[str, ...]


class RuntimeProfileRegistry:
    def __init__(self, profiles, profile_id: str):
        self.profiles = {(item.family, item.layout): item for item in profiles}
        self.profile_id = profile_id

    @classmethod
    def load(cls, path) -> "RuntimeProfileRegistry":
        path = Path(path)
        root = json.loads(path.read_text())
        if root.get("schema") != "hbmprediction.runtime-profiles.v1":
            raise ValueError("unsupported runtime profile schema")
        profiles = []
        for item in root.get("profiles", ()):
            profiles.append(RuntimeProfile(
                str(item["family"]), str(item["layout"]),
                int(item["sample_count"]),
                int(item["untracked_runtime_point_bytes"]),
                int(item["untracked_runtime_p90_bytes"]),
                int(item["allocator_pool_slack_point_bytes"]),
                int(item["allocator_pool_slack_p90_bytes"]),
                int(item.get("device_baseline_point_bytes", 0)),
                int(item.get("device_baseline_p90_bytes", 0)),
                int(item.get("physical_active_shortfall_p90_bytes", 0)),
                tuple(item["label_sha256"]),
            ))
        profile_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(profiles, f"sha256:{profile_sha256}")

    def apply(self, hardware, family: str, pc):
        profile = self.profiles.get((family, layout_key(pc)))
        if profile is None:
            # OOD stays explicit and widens only the interval.
            return replace(
                hardware,
                runtime_profile=f"{self.profile_id}:missing:{family}:{layout_key(pc)}",
                ood_margin_bytes=max(hardware.ood_margin_bytes, 2 * 2**30),
            )
        if profile.sample_count < 2:
            # A singleton may inform the point, but receives a broad interval.
            singleton_margin = 2 * 2**30
        else:
            singleton_margin = 0
        upper = (
            profile.runtime_p90 - profile.runtime_point
            + profile.pool_slack_p90 - profile.pool_slack_point
            + profile.baseline_p90 - profile.baseline_point
            + profile.physical_shortfall_p90
            + singleton_margin
        )
        return replace(
            hardware,
            untracked_runtime_point_bytes=profile.runtime_point,
            allocator_pool_slack_point_bytes=profile.pool_slack_point,
            device_baseline_bytes=profile.baseline_point,
            calibrated_upper_margin_bytes=max(
                hardware.calibrated_upper_margin_bytes, upper
            ),
            runtime_profile=f"{self.profile_id}:{family}:{profile.layout}",
        )
