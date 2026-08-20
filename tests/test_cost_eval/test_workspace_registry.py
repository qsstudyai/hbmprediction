import pytest

from cost_eval.workspace_registry import (
    WorkspaceKey, WorkspaceProfile, WorkspaceRegistry,
)


def _key(ms="2.7", cann="9.0"):
    return WorkspaceKey(
        "qwen3", "flash_attn", "fused", 2, (1024, 1, 8, 128),
        2, 1, 1, ms, cann, "910B2",
    )


def test_registry_requires_exact_versioned_key():
    profile = WorkspaceProfile(
        _key(), 100, 120, "profiler", 20, "a" * 64
    )
    registry = WorkspaceRegistry((profile,))
    assert registry.lookup(_key()) == profile
    assert registry.lookup(_key(ms="2.8")) is None


def test_profile_rejects_unbounded_or_unhashed_values():
    with pytest.raises(ValueError):
        WorkspaceProfile(_key(), 120, 100, "profiler", 20, "a" * 64)
    with pytest.raises(ValueError):
        WorkspaceProfile(_key(), 100, 120, "profiler", 0, "bad")
