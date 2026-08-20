"""Rank-local pipeline event schedules used by the HBM ledger."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    kind: str
    mb: int
    layer: int = -1
    chunk: int = 0


def build_1f1b(
    stage: int, pp: int, m: int | None = None, microbatches: int | None = None
) -> list[Event]:
    if m is None:
        m = microbatches
    if not (0 <= stage < pp) or m is None or m <= 0:
        raise ValueError("stage/pp/m 参数非法")
    warmup = min(pp - 1 - stage, m)
    events = [Event("FWD", mb) for mb in range(warmup)]
    fwd_mb, bwd_mb = warmup, 0
    while bwd_mb < m:
        if fwd_mb < m:
            events.append(Event("FWD", fwd_mb))
            fwd_mb += 1
        events.append(Event("BWD", bwd_mb))
        bwd_mb += 1
    return events


def build_interleaved_1f1b(
    stage: int, pp: int, m: int, interleave: int
) -> list[Event]:
    if interleave <= 0:
        raise ValueError("interleave 必须为正整数")
    if interleave == 1:
        return build_1f1b(stage, pp, m)
    result: list[Event] = []
    for event in build_1f1b(stage, pp, m):
        chunks = (
            range(interleave)
            if event.kind == "FWD"
            else range(interleave - 1, -1, -1)
        )
        result.extend(
            Event(event.kind, event.mb, event.layer, chunk) for chunk in chunks
        )
    return result
