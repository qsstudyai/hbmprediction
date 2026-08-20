"""Saved-tensor contract shared by MoE dispatcher variants."""

from __future__ import annotations

from dataclasses import dataclass

from ..model_spec import TensorRef


@dataclass(frozen=True)
class MoEDispatchTensors:
    topk_scores: TensorRef
    topk_indices: TensorRef
    token_counts: TensorRef
    expert_offsets: TensorRef
    permute_map: TensorRef
    inverse_map: TensorRef


def build_dispatch_tensors(prefix: str = "router") -> MoEDispatchTensors:
    return MoEDispatchTensors(
        TensorRef(f"{prefix}_topk_scores", ("S", "B", "topk"), {0: "sp"}, dtype_bytes=4),
        TensorRef(f"{prefix}_topk_indices", ("S", "B", "topk"), {0: "sp"}, dtype_bytes=4),
        TensorRef(f"{prefix}_token_counts", ("n_experts",), {0: "ep"}, dtype_bytes=4),
        TensorRef(f"{prefix}_expert_offsets", ("n_experts+1",), dtype_bytes=4),
        TensorRef(f"{prefix}_permute_map", ("T_routed",), {0: "ep"}, dtype_bytes=4),
        TensorRef(f"{prefix}_inverse_map", ("T_routed",), {0: "ep"}, dtype_bytes=4),
    )
