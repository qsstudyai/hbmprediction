"""Auditable, optional per-event memory traces for HBM predictions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class TraceTensor:
    """One allocation visible in the layer-local value ledger."""

    key: str
    size_bytes: int
    bucket: str


@dataclass(frozen=True)
class TraceActivation:
    """Saved activation state pinned until a matching backward event."""

    microbatch: int
    layer_id: int
    resident_bytes: int
    offloaded_bytes: int


@dataclass(frozen=True)
class MemoryTraceEvent:
    index: int
    stage_step: int
    stage: int
    schedule_kind: str
    microbatch: int | None
    chunk: int | None
    phase: str
    layer_id: int | None
    op_name: str
    tag: str
    action_kind: str
    tensor_key: str
    tensor_bytes: int
    action_bucket: str
    buckets: Mapping[str, int]
    bucket_deltas: Mapping[str, int]
    total_active_bytes: int
    delta_total_bytes: int
    new_peak: bool
    live_tensors: tuple[TraceTensor, ...] = ()
    pinned_activations: tuple[TraceActivation, ...] = ()


class MemoryTrace:
    """Collect stage-local timeline snapshots and export JSON or CSV.

    ``capture_live_tensors`` records the complete layer-local allocation set
    after every action.  It is intentionally optional because a full-model
    trace can be very large.  Pinned activation summaries are always retained.
    """

    schema_version = "hbmprediction.memory-trace.v1"

    def __init__(self, capture_live_tensors: bool = False) -> None:
        self.capture_live_tensors = bool(capture_live_tensors)
        self.events: list[MemoryTraceEvent] = []
        self.stage_compositions: dict[int, dict[str, int | str]] = {}
        self._stage_steps: dict[int, int] = {}
        self._previous_buckets: dict[int, dict[str, int]] = {}
        self._previous_totals: dict[int, int] = {}

    def record(
        self,
        *,
        stage: int,
        schedule_kind: str,
        microbatch: int | None,
        chunk: int | None,
        phase: str,
        layer_id: int | None,
        op_name: str,
        tag: str,
        action=None,
        buckets: Mapping[str, int],
        total_active_bytes: int,
        new_peak: bool,
        live_tensors: Iterable[TraceTensor] = (),
        pinned_activations: Iterable[TraceActivation] = (),
    ) -> None:
        current = {str(name): int(value) for name, value in buckets.items()}
        previous = self._previous_buckets.get(
            stage, {name: 0 for name in current}
        )
        deltas = {
            name: current.get(name, 0) - previous.get(name, 0)
            for name in current
        }
        previous_total = self._previous_totals.get(stage, 0)
        stage_step = self._stage_steps.get(stage, 0)
        event = MemoryTraceEvent(
            index=len(self.events),
            stage_step=stage_step,
            stage=stage,
            schedule_kind=schedule_kind,
            microbatch=microbatch,
            chunk=chunk,
            phase=phase,
            layer_id=layer_id,
            op_name=op_name,
            tag=tag,
            action_kind="" if action is None else str(action.kind),
            tensor_key="" if action is None else str(action.key),
            tensor_bytes=0 if action is None else int(action.size_bytes),
            action_bucket="" if action is None else str(action.bucket),
            buckets=current,
            bucket_deltas=deltas,
            total_active_bytes=int(total_active_bytes),
            delta_total_bytes=int(total_active_bytes) - previous_total,
            new_peak=bool(new_peak),
            live_tensors=(
                tuple(live_tensors) if self.capture_live_tensors else ()
            ),
            pinned_activations=tuple(pinned_activations),
        )
        self.events.append(event)
        self._stage_steps[stage] = stage_step + 1
        self._previous_buckets[stage] = current
        self._previous_totals[stage] = int(total_active_bytes)

    def add_stage_composition(self, stage: int, **components) -> None:
        self.stage_compositions[int(stage)] = {
            str(name): value for name, value in components.items()
        }

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "capture_live_tensors": self.capture_live_tensors,
            "events": [asdict(event) for event in self.events],
            "stage_compositions": {
                str(stage): values
                for stage, values in sorted(self.stage_compositions.items())
            },
        }

    def write_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def write_csv(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        bucket_names = (
            "persistent", "act_live", "gather_buf", "grad_buf",
            "recomp_scratch", "swap_buf", "workspace", "framework",
        )
        fields = [
            "index", "stage_step", "stage", "schedule_kind",
            "microbatch", "chunk", "phase", "layer_id", "op_name", "tag",
            "action_kind", "tensor_key", "tensor_bytes", "action_bucket",
            *bucket_names,
            *(f"delta_{name}" for name in bucket_names),
            "total_active_bytes", "delta_total_bytes", "new_peak",
            "live_tensors", "pinned_activations",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for event in self.events:
                row = {
                    "index": event.index,
                    "stage_step": event.stage_step,
                    "stage": event.stage,
                    "schedule_kind": event.schedule_kind,
                    "microbatch": event.microbatch,
                    "chunk": event.chunk,
                    "phase": event.phase,
                    "layer_id": event.layer_id,
                    "op_name": event.op_name,
                    "tag": event.tag,
                    "action_kind": event.action_kind,
                    "tensor_key": event.tensor_key,
                    "tensor_bytes": event.tensor_bytes,
                    "action_bucket": event.action_bucket,
                    "total_active_bytes": event.total_active_bytes,
                    "delta_total_bytes": event.delta_total_bytes,
                    "new_peak": event.new_peak,
                    "live_tensors": json.dumps(
                        [asdict(item) for item in event.live_tensors],
                        ensure_ascii=False,
                    ),
                    "pinned_activations": json.dumps(
                        [asdict(item) for item in event.pinned_activations],
                        ensure_ascii=False,
                    ),
                }
                for name in bucket_names:
                    row[name] = event.buckets.get(name, 0)
                    row[f"delta_{name}"] = event.bucket_deltas.get(name, 0)
                writer.writerow(row)
        return path

    def write(self, path) -> Path:
        path = Path(path)
        if path.suffix.lower() == ".json":
            return self.write_json(path)
        if path.suffix.lower() == ".csv":
            return self.write_csv(path)
        raise ValueError("memory trace 输出扩展名必须是 .json 或 .csv")
