"""MindFormers PyNative-aligned evaluator configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


def _normalize_op_map(value) -> dict[int, frozenset[str]]:
    result = {}
    for layer, names in dict(value).items():
        if isinstance(names, str):
            names = (names,)
        result[int(layer)] = frozenset(str(name) for name in names)
    return result


@dataclass(frozen=True)
class ParallelConfig:
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1
    pp: int = 1
    ep: int = 1
    sequence_parallel: bool = False
    # MindFormers source field. ``reshard_after_forward`` below is the resolved
    # runtime boolean (default=True without PP, False with PP).
    reshard_after_forward_policy: str = "default"
    reshard_after_forward: object = None
    cpu_offload: bool = False
    microbatch: int = 1
    interleave: int = 1
    pipeline_schedule: str = "1f1b"
    pipeline_interleave: Optional[int] = None
    layers_per_stage: Optional[Sequence[int]] = None
    prefetch_depth: int = 1
    num_microbatches: int = 1
    dense_fsdp_shard_size: Optional[int] = None
    context_parallel_method: str = "colossal"
    ulysses_degree_in_cp: Optional[int] = None
    moe_token_dispatcher: str = "alltoall"
    enable_loss_parallel: bool = False
    parameter_offload: bool = False
    gradient_offload: bool = False
    optimizer_offload: bool = False
    fsdp_flatten_alignment_bytes: int = 1

    def __post_init__(self) -> None:
        policy = self.reshard_after_forward_policy
        if isinstance(self.reshard_after_forward, str):
            policy = self.reshard_after_forward
        if policy not in {"always", "never", "default"}:
            raise ValueError("reshard_after_forward_policy 必须是 always、never 或 default")
        object.__setattr__(self, "reshard_after_forward_policy", policy)
        object.__setattr__(
            self,
            "reshard_after_forward",
            policy == "always" or (policy == "default" and self.pp == 1),
        )
        if self.pipeline_interleave is not None:
            object.__setattr__(self, "interleave", self.pipeline_interleave)
        object.__setattr__(self, "pipeline_interleave", self.interleave)
        if self.cpu_offload:
            object.__setattr__(self, "parameter_offload", True)
            object.__setattr__(self, "gradient_offload", True)
            object.__setattr__(self, "optimizer_offload", True)
        degrees = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.microbatch,
            self.interleave,
            self.num_microbatches,
        )
        if any(value <= 0 for value in degrees):
            raise ValueError("所有并行度和 batch 数必须为正整数")
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth 不能为负数")
        if self.fsdp_flatten_alignment_bytes <= 0:
            raise ValueError("fsdp_flatten_alignment_bytes 必须为正数")
        if self.pipeline_schedule != "1f1b":
            raise ValueError("当前显存时间线仅支持 MindFormers 1f1b schedule")
        if self.context_parallel_method not in {"colossal", "ulysses", "hybrid"}:
            raise ValueError("未知 context_parallel_method")
        if self.moe_token_dispatcher not in {
            "alltoall", "alltoall_deredundency", "alltoall_zero_redundancy"
        }:
            raise ValueError(
                f"unsupported MoE dispatcher contract: {self.moe_token_dispatcher}"
            )
        ulysses = self.ulysses_degree_in_cp
        if self.context_parallel_method == "ulysses":
            ulysses = self.cp if ulysses is None else ulysses
            if ulysses != self.cp:
                raise ValueError("Ulysses CP requires ulysses_degree_in_cp == cp")
        elif self.context_parallel_method == "hybrid":
            if ulysses is None or not (1 < ulysses < self.cp) or self.cp % ulysses:
                raise ValueError(
                    "Hybrid CP requires 1 < ulysses_degree_in_cp < cp and divisibility"
                )
        else:
            ulysses = 1
        object.__setattr__(self, "ulysses_degree_in_cp", ulysses)
        full_fsdp = self.dp_shard * self.cp
        if self.dense_fsdp_shard_size is not None and (
            self.dense_fsdp_shard_size < 1
            or full_fsdp % self.dense_fsdp_shard_size
        ):
            raise ValueError("dense_fsdp_shard_size 必须整除 dp_shard*cp")
        if self.layers_per_stage is not None:
            object.__setattr__(
                self, "layers_per_stage", tuple(self.layers_per_stage)
            )

    @property
    def world_size(self) -> int:
        return self.dp_replicate * self.dp_shard * self.cp * self.tp * self.pp

    @property
    def fsdp_degree(self) -> int:
        return self.dense_fsdp_shard_size or self.dp_shard * self.cp

    @property
    def expert_fsdp_degree(self) -> int:
        # MindFormers only activates the sparse eFSDP mesh for EP > 1.  With
        # ep=1 experts remain inside the ordinary FSDP-wrapped decoder layer.
        if self.ep == 1:
            return self.fsdp_degree
        region = self.dp_shard * self.cp * self.tp
        if region % self.ep:
            raise ValueError(f"ep({self.ep}) 必须整除 dp_shard*cp*tp={region}")
        return region // self.ep

    @property
    def virtual_pipeline_size(self) -> int:
        """Number of virtual pipeline chunks on each physical stage."""

        return self.interleave


@dataclass(frozen=True)
class OptimizerSpec:
    type: str = "AdamW"
    state_bytes_per_param: int = 16
    parameter_bytes: int = 2
    gradient_bytes: int = 2
    master_weight_bytes: int = 4
    optimizer_state_bytes: int = 8
    muon_ns_steps: int = 0
    muon_parameter_fraction: float = 0.0
    muon_momentum_bytes: int = 0
    muon_main_parameter_bytes: int = 0

    def __post_init__(self) -> None:
        if self.state_bytes_per_param <= 0:
            raise ValueError("state_bytes_per_param 必须为正数")
        state = self.state_bytes_per_param - (
            self.parameter_bytes + self.gradient_bytes + self.master_weight_bytes
        )
        if state < 0:
            raise ValueError("optimizer byte breakdown exceeds state_bytes_per_param")
        if self.muon_ns_steps < 0 or self.muon_momentum_bytes < 0 or self.muon_main_parameter_bytes < 0:
            raise ValueError("Muon 参数不能为负数")
        if not 0 <= self.muon_parameter_fraction <= 1:
            raise ValueError("muon_parameter_fraction 必须位于 [0, 1]")
        object.__setattr__(self, "optimizer_state_bytes", state)

    @property
    def name(self) -> str:
        return self.type.lower()

    @property
    def bytes_per_parameter(self) -> int:
        return self.state_bytes_per_param

    @property
    def is_muon(self) -> bool:
        return self.name in {"muon", "muonadam", "muon_adam"} or self.muon_ns_steps > 0

    @classmethod
    def adamw(cls, fp32_grad: bool = False) -> "OptimizerSpec":
        return cls("AdamW", 18 if fp32_grad else 16, 2, 4 if fp32_grad else 2, 4, 8)


@dataclass(frozen=True)
class HardwareSpec:
    max_device_memory: int
    framework_reserve: int = 0
    usable_device_memory: Optional[int] = None
    device_baseline_bytes: int = 0
    allocator_pool_point_bytes: int = 0
    allocator_pool_slack_point_bytes: int = 0
    untracked_runtime_point_bytes: int = 0
    allocator_granularity_bytes: int = 512
    calibrated_upper_margin_bytes: int = 0
    ood_margin_bytes: int = 0
    hardware_profile: str = "unknown"
    runtime_profile: str = "unknown"
    source_profile: str = "unknown"

    def __post_init__(self) -> None:
        numeric = (
            self.framework_reserve,
            self.device_baseline_bytes,
            self.allocator_pool_point_bytes,
            self.allocator_pool_slack_point_bytes,
            self.untracked_runtime_point_bytes,
            self.calibrated_upper_margin_bytes,
            self.ood_margin_bytes,
        )
        if self.max_device_memory <= 0 or any(value < 0 for value in numeric):
            raise ValueError("设备内存必须为正数，框架预留不能为负数")
        if self.allocator_granularity_bytes <= 0:
            raise ValueError("allocator_granularity_bytes 必须为正数")
        usable = self.usable_device_memory or self.max_device_memory
        if usable <= 0 or usable > self.max_device_memory:
            raise ValueError("usable_device_memory 必须位于 (0, max_device_memory]")
        object.__setattr__(self, "usable_device_memory", usable)


@dataclass(frozen=True)
class RecomputeSpec:
    mode: str = "none"
    full_layers: frozenset[int] = field(default_factory=frozenset)
    select_ops: Mapping[int, frozenset[str]] = field(default_factory=dict)
    exclude_ops: object = field(default_factory=frozenset)
    exclude_ops_by_layer: Mapping[int, frozenset[str]] = field(default_factory=dict, init=False)
    select_modules: Mapping[str, frozenset[int]] = field(default_factory=dict)
    recompute_comm: bool = False
    comm_select_modules: Mapping[str, frozenset[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = self.mode.lower()
        if mode not in {"none", "full", "select"}:
            raise ValueError("recompute mode 仅支持 none、full 或 select")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "full_layers", frozenset(self.full_layers))
        object.__setattr__(
            self, "select_ops", _normalize_op_map(self.select_ops)
        )
        if isinstance(self.exclude_ops, Mapping):
            by_layer = _normalize_op_map(self.exclude_ops)
            global_exclude = by_layer.get(-1, frozenset())
        else:
            by_layer = {}
            global_exclude = frozenset(str(x) for x in self.exclude_ops)
        object.__setattr__(self, "exclude_ops", global_exclude)
        object.__setattr__(self, "exclude_ops_by_layer", by_layer)
        object.__setattr__(self, "select_modules", {
            str(name): frozenset(layers)
            for name, layers in dict(self.select_modules).items()
        })
        object.__setattr__(self, "comm_select_modules", {
            str(name): frozenset(layers)
            for name, layers in dict(self.comm_select_modules).items()
        })

    @property
    def layers(self) -> frozenset[int]:
        return self.full_layers

    def is_full(self, layer_id: int) -> bool:
        return self.mode == "full" and (
            not self.full_layers or layer_id in self.full_layers
        )

    def recomputed_ops(
        self, layer_id: int, available_ops: Sequence[str]
    ) -> frozenset[str]:
        available = frozenset(available_ops)
        if self.is_full(layer_id):
            selected = available
        elif self.mode == "select":
            selected = set(
                self.select_ops.get(-1, frozenset())
                | self.select_ops.get(layer_id, frozenset())
            )
            unknown = selected - set(available)
            if unknown:
                raise ValueError(f"layer {layer_id} 的 select_ops 包含未知算子: {', '.join(sorted(unknown))}")
            groups = {
                "attention": {"ln1", "qkv", "rope", "flash", "o_proj", "add1"},
                "self_attention": {"ln1", "qkv", "rope", "flash", "o_proj", "add1"},
                "mlp": {"ln2", "fc1", "swiglu", "fc2", "add2", "router", "dispatch", "expert_fc1", "expert_swiglu", "expert_fc2", "combine"},
                "transformer_layer": set(available),
            }
            for module, layers in self.select_modules.items():
                if not layers or layer_id in layers:
                    selected |= groups.get(module.split(".")[-1], set())
            selected = frozenset(selected & set(available))
        else:
            selected = frozenset()
        excluded = frozenset(
            self.exclude_ops
            | self.exclude_ops_by_layer.get(-1, frozenset())
            | self.exclude_ops_by_layer.get(layer_id, frozenset())
        )
        aliases = {"flash_attn": "flash", "matmul": None}
        normalized_excluded = set()
        for name in excluded:
            alias = aliases.get(name, name)
            if alias is None:
                normalized_excluded |= {op for op in available if op in {"qkv", "o_proj", "fc1", "fc2", "expert_fc1", "expert_fc2"}}
            elif alias in available:
                normalized_excluded.add(alias)
        excluded = frozenset(normalized_excluded)
        return frozenset(selected - excluded)

    def recomputed_comm_ops(self, layer_id: int, ops) -> frozenset[str]:
        """Resolve MindFormers recompute_comm module selectors to graph ops."""
        if not self.recompute_comm:
            return frozenset()
        selected = set()
        for op in ops:
            if not op.collectives:
                continue
            paths = set(op.module_paths) | {op.name}
            for module, layers in self.comm_select_modules.items():
                suffix = module.split(".")[-1]
                if (not layers or layer_id in layers) and any(
                    path == module
                    or path.endswith(module)
                    or path.split(".")[-1] == suffix
                    for path in paths
                ):
                    selected.add(op.name)
        return frozenset(selected)


@dataclass(frozen=True)
class SwapSpec:
    enable: bool = False
    default_prefetch: int = 1
    swap_layers: frozenset[int] = field(default_factory=frozenset)
    swap_ops: Mapping[int, frozenset[str]] = field(default_factory=dict)
    enabled: Optional[bool] = None
    layers: Optional[frozenset[int]] = None
    op_layers: Mapping[str, frozenset[int]] = field(default_factory=dict)
    prefetch_depth: Optional[int] = None
    op_names: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.enabled is not None:
            object.__setattr__(self, "enable", self.enabled)
        object.__setattr__(self, "enabled", self.enable)
        if self.layers is not None:
            object.__setattr__(self, "swap_layers", frozenset(self.layers))
        object.__setattr__(self, "layers", frozenset(self.swap_layers))
        if self.prefetch_depth is not None:
            object.__setattr__(self, "default_prefetch", self.prefetch_depth)
        object.__setattr__(self, "prefetch_depth", self.default_prefetch)
        if self.default_prefetch <= 0:
            raise ValueError("default_prefetch 必须为正数")
        object.__setattr__(self, "swap_layers", frozenset(self.swap_layers))
        object.__setattr__(self, "swap_ops", _normalize_op_map(self.swap_ops))
        object.__setattr__(self, "op_layers", {
            str(name): frozenset(layers)
            for name, layers in dict(self.op_layers).items()
        })
        if self.op_names:
            current = dict(self.swap_ops)
            current[-1] = frozenset(current.get(-1, frozenset()) | self.op_names)
            object.__setattr__(self, "swap_ops", current)

    def swaps(self, layer_id: int) -> bool:
        return self.enable and (
            (not self.swap_layers and not self.swap_ops)
            or layer_id in self.swap_layers
        )

    def swapped_ops(
        self, layer_id: int, available_ops: Sequence[str]
    ) -> frozenset[str]:
        if not self.enable:
            return frozenset()
        available = frozenset(available_ops)
        if self.swaps(layer_id):
            selected = available
        else:
            selected = frozenset(
                self.swap_ops.get(-1, frozenset())
                | self.swap_ops.get(layer_id, frozenset())
            )
            groups = {
                "attention": {"qkv", "rope", "flash", "o_proj"},
                "self_attention": {"qkv", "rope", "flash", "o_proj"},
                "mlp": {"fc1", "swiglu", "fc2", "router", "dispatch", "expert_fc1", "expert_swiglu", "expert_fc2", "combine"},
                "transformer_layer": set(available),
            }
            selected = set(selected)
            if "flash_attn" in selected:
                selected.remove("flash_attn")
                selected.add("flash")
            for module, layers in self.op_layers.items():
                if layer_id in layers:
                    selected |= groups.get(module.split(".")[-1], set())
            selected = frozenset(selected & set(available))
        unknown = selected - available
        if unknown:
            raise ValueError(
                f"layer {layer_id} 的 swap_ops 包含未知算子: "
                f"{', '.join(sorted(unknown))}"
            )
        return frozenset(selected)
