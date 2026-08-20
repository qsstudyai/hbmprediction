"""M3：mesh 度数关系与流水线层分配。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from .specs import ParallelConfig


class ParallelModel:
    class _StageLayers(Mapping):
        def __init__(self, owner):
            self.owner = owner
        def __getitem__(self, stage):
            return tuple(self.owner._stage_layers(stage))
        def __iter__(self):
            return iter(range(self.owner.pc.pp))
        def __len__(self):
            return self.owner.pc.pp
        def __call__(self, stage):
            return list(self.owner._stage_layers(stage))

    @classmethod
    def build(cls, pc, dims, world_size=None):
        return cls(pc, dims.n_layers, world_size)

    def __init__(
        self,
        pc: ParallelConfig,
        n_layers: int,
        world_size: Optional[int] = None,
    ):
        if n_layers <= 0:
            raise ValueError("n_layers 必须为正数")
        if pc.pp > n_layers:
            raise ValueError("当前评估器要求 pp 不大于 n_layers")
        self.pc = pc
        self.n_layers = n_layers
        self.expected_world_size = (
            pc.world_size
        )
        self.world_size = world_size or self.expected_world_size
        if self.world_size != self.expected_world_size:
            raise ValueError(
                f"world_size={self.world_size} 与并行配置要求的 "
                f"{self.expected_world_size} 不一致"
            )
        expert_region = pc.dp_shard * pc.cp * pc.tp
        if expert_region % pc.ep:
            raise ValueError(
                f"ep({pc.ep}) 必须整除 dp_shard*cp*tp={expert_region}"
            )
        # MindFormers only creates a separate eFSDP wrapper when EP > 1.
        # With ep=1, routed experts stay inside the ordinary FSDP-wrapped
        # decoder layer and therefore shard over the dense fsdp mesh
        # (dp_shard * cp), not over the hypothetical sparse mesh that would
        # additionally include TP.  Treating ep=1 as expert_region here
        # over-shards experts whenever TP > 1 and underestimates persistent
        # parameter/optimizer memory.
        self._efsdp = (
            pc.fsdp_degree if pc.ep == 1 else expert_region // pc.ep
        )
        self._mapping = self._build_layer_to_stage()
        if pc.interleave > 1:
            stage_counts = [self._mapping.count(stage) for stage in range(pc.pp)]
            if any(count < pc.interleave for count in stage_counts):
                raise ValueError(
                    "pipeline interleave 要求每个 physical stage 至少有 "
                    f"{pc.interleave} 层，当前为 {stage_counts}"
                )
        self.stage_layers = self._StageLayers(self)

    def degree(self, axis: str) -> int:
        if axis == "sp":
            return self.pc.tp if self.pc.sequence_parallel else 1
        values = {
            "tp": self.pc.tp,
            "cp": self.pc.cp,
            "ep": self.pc.ep,
            "dp_shard": self.pc.dp_shard,
            "dp_replicate": self.pc.dp_replicate,
            "pp": self.pc.pp,
        }
        try:
            return values[axis]
        except KeyError as exc:
            raise ValueError(f"未知 mesh 轴: {axis}") from exc

    def fsdp_degree(self) -> int:
        return self.pc.fsdp_degree

    def efsdp_degree(self) -> int:
        return self._efsdp

    def stage_of(self, layer_id: int) -> int:
        return self._mapping[layer_id]

    def stage_of_rank(self, rank: int) -> int:
        if rank < 0 or rank >= self.world_size:
            raise ValueError(f"无效 rank: {rank}")
        # MindFormers lays the pipeline axis out as the innermost rank axis in
        # the evaluator-facing world mesh.
        return rank % self.pc.pp

    def _stage_layers(self, stage: int) -> list[int]:
        if stage < 0 or stage >= self.pc.pp:
            raise ValueError(f"无效 stage: {stage}")
        return [
            layer
            for layer, assigned in enumerate(self._mapping)
            if assigned == stage
        ]

    def _build_layer_to_stage(self) -> list[int]:
        if self.pc.layers_per_stage is not None:
            raw = list(self.pc.layers_per_stage)
            if len(raw) != self.pc.pp:
                raise ValueError("layers_per_stage 的长度必须等于 pp")
            if raw and all(isinstance(item, int) for item in raw):
                counts = raw
                if sum(counts) != self.n_layers:
                    raise ValueError("layers_per_stage 总和必须等于 n_layers")
                return [stage for stage, count in enumerate(counts) for _ in range(count)]
            mapping = [-1] * self.n_layers
            for stage, layer_ids in enumerate(raw):
                for layer_id in layer_ids:
                    if not 0 <= int(layer_id) < self.n_layers or mapping[int(layer_id)] != -1:
                        raise ValueError("pipeline layer mapping 包含越界或重复层")
                    mapping[int(layer_id)] = stage
            if -1 in mapping:
                raise ValueError("pipeline layer mapping 未覆盖全部层")
            return mapping
        else:
            # MTP and other auxiliary layers can make total_layers not
            # divisible by PP.  MindFormers' auto policy assigns the
            # remainder to the earliest stages; retaining that policy keeps
            # the schedule legal without inventing a fake dropped layer.
            base, remainder = divmod(self.n_layers, self.pc.pp)
            counts = [base + int(stage < remainder) for stage in range(self.pc.pp)]
        return [
            stage
            for stage, count in enumerate(counts)
            for _ in range(count)
        ]
