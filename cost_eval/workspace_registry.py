"""Version-aware kernel workspace profiles with formula fallback metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkspaceKey:
    family: str
    op_type: str
    fusion_variant: str
    dtype_bytes: int
    local_shape_bucket: tuple[int, ...]
    tp: int
    cp: int
    ep: int
    mindspore_version: str
    cann_version: str
    hardware: str


@dataclass(frozen=True)
class WorkspaceProfile:
    key: WorkspaceKey
    point_bytes: int
    upper_bytes: int
    source: str
    sample_count: int
    content_sha256: str

    def __post_init__(self) -> None:
        if self.point_bytes < 0 or self.upper_bytes < self.point_bytes:
            raise ValueError("workspace profile 区间非法")
        if self.sample_count <= 0 or len(self.content_sha256) != 64:
            raise ValueError("workspace profile 缺少可审计样本/hash")


class WorkspaceRegistry:
    def __init__(self, profiles=()):
        self._profiles = {profile.key: profile for profile in profiles}

    def lookup(self, key: WorkspaceKey) -> WorkspaceProfile | None:
        return self._profiles.get(key)
