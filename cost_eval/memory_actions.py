"""Translate resolved operators into explicit transient-memory actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MemoryAction:
    kind: str
    key: str
    size_bytes: int = 0
    bucket: str = "workspace"
    owner: str = ""
    alias_of: str | None = None
    op_name: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"ALLOC", "FREE", "ALIAS", "MOVE_IN", "MOVE_OUT"}:
            raise ValueError(f"未知 memory action: {self.kind}")
        if self.size_bytes < 0:
            raise ValueError("memory action size 不能为负数")


def forward_layer_actions(layer, retained_keys: Iterable[str]) -> tuple[MemoryAction, ...]:
    """Create a value-liveness ledger for one layer forward pass.

    Outputs live through their final consumer. Saved values selected by the
    recompute/swap policy remain resident at layer exit. Kernel workspace and
    collective staging are explicit, separately owned allocations.
    """

    retained = set(retained_keys)
    last_use: dict[str, int] = {}
    for index, op in enumerate(layer.ops):
        for tensor in (*op.inputs, *op.saves):
            if not tensor.is_weight:
                last_use[tensor.storage_key] = index

    live: set[str] = set()
    sizes: dict[str, int] = {}
    actions: list[MemoryAction] = []

    def ensure(tensor, index: int, role: str) -> None:
        if tensor.is_weight:
            return
        key = tensor.storage_key
        sizes.setdefault(key, tensor.local_bytes)
        if key not in live:
            bucket = "act_live" if key in retained else "workspace"
            actions.append(MemoryAction(
                "ALLOC", key, tensor.local_bytes, bucket,
                f"fwd_{role}@{layer.layer_id}:{index}",
                op_name=op.name,
            ))
            live.add(key)

    for index, op in enumerate(layer.ops):
        for tensor in op.inputs:
            ensure(tensor, index, "input")
        ensure(op.output, index, "output")
        for tensor in op.saves:
            ensure(tensor, index, "save")

        if op.workspace_bytes:
            key = f"workspace:{layer.layer_id}:{index}:{op.name}"
            actions.append(MemoryAction(
                "ALLOC", key, op.workspace_bytes, "workspace",
                f"fwd_op@{layer.layer_id}:{op.name}",
                op_name=op.name,
            ))
        comm_bytes = max(
            (comm.volume_bytes for comm in op.collectives), default=0
        )
        if comm_bytes:
            key = f"comm:{layer.layer_id}:{index}:{op.name}"
            actions.append(MemoryAction(
                "ALLOC", key, comm_bytes, "communication",
                f"fwd_comm@{layer.layer_id}:{op.name}",
                op_name=op.name,
            ))
            actions.append(MemoryAction(
                "FREE", key, owner=f"fwd_comm_end@{layer.layer_id}:{op.name}",
                op_name=op.name,
            ))
        if op.workspace_bytes:
            key = f"workspace:{layer.layer_id}:{index}:{op.name}"
            actions.append(MemoryAction(
                "FREE", key, owner=f"fwd_op_end@{layer.layer_id}:{op.name}",
                op_name=op.name,
            ))

        for key in tuple(live):
            if key not in retained and last_use.get(key, index) <= index:
                actions.append(MemoryAction(
                    "FREE", key, owner=f"fwd_value_end@{layer.layer_id}:{op.name}",
                    op_name=op.name,
                ))
                live.remove(key)

    for key in tuple(live):
        if key not in retained:
            actions.append(MemoryAction("FREE", key, owner=f"fwd_layer_end@{layer.layer_id}"))
            live.remove(key)
    return tuple(actions)
