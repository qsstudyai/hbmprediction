"""Checked allocation/free ledger with simultaneous-peak snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from .memory_actions import MemoryAction


@dataclass(frozen=True)
class LedgerPeak:
    bytes: int
    owner: str
    buckets: dict[str, int]


class MemoryLedger:
    def __init__(self) -> None:
        self._live: dict[str, tuple[int, str]] = {}
        self._aliases: dict[str, str] = {}
        self._buckets: dict[str, int] = {}
        self.peak = LedgerPeak(0, "initial", {})

    @property
    def live_bytes(self) -> int:
        return sum(size for size, _ in self._live.values())

    @property
    def buckets(self) -> dict[str, int]:
        return dict(self._buckets)

    @property
    def live_keys(self) -> frozenset[str]:
        return frozenset(self._live)

    def _record(self, owner: str) -> None:
        current = self.live_bytes
        if current > self.peak.bytes:
            self.peak = LedgerPeak(current, owner, self.buckets)

    def apply(self, action: MemoryAction) -> None:
        if action.kind in {"ALLOC", "MOVE_IN"}:
            if action.key in self._live or action.key in self._aliases:
                raise ValueError(f"重复分配: {action.key}")
            self._live[action.key] = (action.size_bytes, action.bucket)
            self._buckets[action.bucket] = self._buckets.get(action.bucket, 0) + action.size_bytes
        elif action.kind == "ALIAS":
            target = action.alias_of
            if not target or target not in self._live:
                raise ValueError(f"alias target 不存在: {target}")
            if action.key in self._live or action.key in self._aliases:
                raise ValueError(f"重复 alias: {action.key}")
            self._aliases[action.key] = target
        elif action.kind in {"FREE", "MOVE_OUT"}:
            if action.key in self._aliases:
                self._aliases.pop(action.key)
            elif action.key in self._live:
                if action.key in self._aliases.values():
                    raise ValueError(f"仍有 alias 引用，不能释放: {action.key}")
                size, bucket = self._live.pop(action.key)
                self._buckets[bucket] -= size
                if self._buckets[bucket] < 0:
                    raise ValueError(f"bucket 出现负数: {bucket}")
            else:
                raise ValueError(f"释放未分配值: {action.key}")
        self._record(action.owner or f"{action.kind}@{action.key}")

    def replay(self, actions, allowed_live=frozenset()) -> LedgerPeak:
        for action in actions:
            self.apply(action)
        leaked = self.live_keys - frozenset(allowed_live)
        if leaked:
            raise ValueError("transient ledger leak: " + ", ".join(sorted(leaked)))
        return self.peak
