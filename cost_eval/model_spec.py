"""M1：声明式算子图及其内存契约。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Mapping, Optional, Sequence, Union

DimExpr = Union[int, str]


class OpType(Enum):
    MATMUL = "matmul"
    FLASH_ATTN = "flash_attn"
    ELEMENTWISE = "elementwise"
    NORM = "norm"
    ROPE = "rope"
    MOE_ROUTER = "moe_router"
    MOE_GEMM = "moe_gemm"
    DISPATCH = "dispatch"
    COMBINE = "combine"


@dataclass(frozen=True)
class DimTable:
    H: int
    F: int
    n_heads: int
    n_kv: int
    head_dim: int
    S: int
    B: int
    vocab: int
    n_layers: int
    n_experts: int = 0
    topk: int = 0
    n_shared: int = 0
    moe_F: int = 0
    capacity_factor: float = 1.0
    dtype_bytes: int = 2
    param_dtype_bytes: int = 2
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    shared_F: int = 0
    n_mtp_layers: int = 0
    use_shared_expert_gating: bool = False
    tie_word_embeddings: bool = False
    hc_mult: int = 0
    enable_hyper_connections: bool = False

    def __post_init__(self) -> None:
        required = {
            "H": self.H,
            "F": self.F,
            "n_heads": self.n_heads,
            "n_kv": self.n_kv,
            "head_dim": self.head_dim,
            "S": self.S,
            "B": self.B,
            "vocab": self.vocab,
            "n_layers": self.n_layers,
            "dtype_bytes": self.dtype_bytes,
            "param_dtype_bytes": self.param_dtype_bytes,
        }
        invalid = [name for name, value in required.items() if value <= 0]
        if invalid:
            raise ValueError(f"维度必须为正数: {', '.join(invalid)}")
        if self.capacity_factor <= 0:
            raise ValueError("capacity_factor 必须为正数")
        if self.hc_mult < 0:
            raise ValueError("hc_mult 不能为负数")

    def as_dict(self) -> dict[str, Union[int, float]]:
        values = dict(self.__dict__)
        values["T_routed"] = ceil(
            self.S * self.B * self.topk * self.capacity_factor
        )
        return values

    @property
    def total_layers(self) -> int:
        return self.n_layers + self.n_mtp_layers


@dataclass(frozen=True)
class TensorRef:
    """符号 shape 与 placement；shard 为 ``维度索引 -> mesh 轴``。"""

    name: str
    shape: tuple[DimExpr, ...]
    shard: Mapping[int, str] = field(default_factory=dict)
    is_weight: bool = False
    partial: Optional[str] = None
    dtype_bytes: Optional[int] = None
    storage_id: Optional[str] = None
    trainable: bool = True
    swappable: bool = True
    recomputable: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "shard", dict(self.shard))
        for dim_index in self.shard:
            if dim_index < 0 or dim_index >= len(self.shape):
                raise ValueError(f"{self.name} 的 shard 维度越界: {dim_index}")

    def has_ep(self) -> bool:
        return "ep" in self.shard.values()

    @property
    def is_expert(self) -> bool:
        return self.has_ep()


@dataclass(frozen=True)
class OpSpec:
    name: str
    type: OpType
    inputs: Sequence[TensorRef]
    output: TensorRef
    params: Sequence[TensorRef] = field(default_factory=tuple)
    saves: Sequence[TensorRef] = field(default_factory=tuple)
    workspace: Optional[DimExpr] = None
    attrs: Mapping[str, object] = field(default_factory=dict)
    module_paths: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class LayerSpec:
    ops: Sequence[OpSpec]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dims: DimTable
    layer_pattern: Sequence[str]
    layer_specs: Mapping[str, LayerSpec]
    capabilities: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.layer_pattern) != self.dims.total_layers:
            raise ValueError(
                "layer_pattern 长度必须等于 decoder+MTP 层数: "
                f"{len(self.layer_pattern)} != {self.dims.total_layers}"
            )
        missing = sorted(set(self.layer_pattern) - set(self.layer_specs))
        if missing:
            raise ValueError(f"缺少 layer spec: {', '.join(missing)}")
        object.__setattr__(self, "capabilities", dict(self.capabilities or {}))

    def get_layer(self, layer_type: str) -> LayerSpec:
        return self.layer_specs[layer_type]
