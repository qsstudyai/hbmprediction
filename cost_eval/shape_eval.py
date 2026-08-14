"""M4：安全符号求值、切分代入、placement 变换与 resolved graph。"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from math import prod
from typing import Mapping, Optional, Union

from .model_spec import DimExpr, DimTable, ModelSpec, OpType, TensorRef

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}
_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def eval_expr(expr: DimExpr, dims: DimTable) -> int:
    """仅允许整数、已知维度符号以及 ``+ - * //``。"""

    if isinstance(expr, int) and not isinstance(expr, bool):
        return expr
    if not isinstance(expr, str):
        raise ValueError(f"维度表达式必须是整数或字符串: {expr!r}")
    env = dims.as_dict() if hasattr(dims, "as_dict") else dict(dims)

    def evaluate(node: ast.AST) -> Union[int, float]:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"未知符号: {node.id}")
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
            return _UNARYOPS[type(node.op)](evaluate(node.operand))
        raise ValueError(f"非法表达式节点: {ast.dump(node)}")

    try:
        value = evaluate(ast.parse(expr, mode="eval"))
    except (SyntaxError, ZeroDivisionError) as exc:
        raise ValueError(f"非法维度表达式: {expr!r}") from exc
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"维度表达式必须得到非负整数: {expr!r} -> {value!r}")
    return value


@dataclass(frozen=True)
class Placement:
    shard: tuple[tuple[int, str], ...] = ()
    partial: Optional[str] = None

    def __init__(self, shard=(), partial=None):
        items = tuple(sorted(shard.items())) if isinstance(shard, Mapping) else tuple(shard)
        object.__setattr__(self, "shard", items)
        object.__setattr__(self, "partial", partial)

    @staticmethod
    def of(tensor: TensorRef) -> "Placement":
        return Placement(tensor.shard, tensor.partial)

    @property
    def shards(self):
        return self.shard


@dataclass(frozen=True)
class CommSpec:
    ctype: str
    volume_bytes: int
    group_axis: str
    phase: str = "fwd"
    output_tensor: Optional[str] = None
    rank_group: tuple[int, ...] = ()
    topology_class: Optional[str] = None
    algorithm: Optional[str] = None

    @property
    def kind(self):
        return self.ctype


def detect_reshard(
    src: Optional[Placement],
    dst: Placement,
    numel: Optional[int] = None,
    dtype_bytes: Optional[int] = None,
) -> Optional[CommSpec]:
    if isinstance(src, ResolvedTensor):
        source_tensor, dest_tensor = src, dst
        src, dst = source_tensor.placement, dest_tensor.placement
        numel, dtype_bytes = dest_tensor.local_numel, dest_tensor.dtype_bytes
    if src is None or src == dst:
        return None
    src_axes = {axis for _, axis in src.shard}
    dst_axes = {axis for _, axis in dst.shard}
    axis = src.partial or next(iter(src_axes or dst_axes), None)
    if axis is None:
        return None
    if src.partial and not dst.partial:
        ctype = "reduce_scatter" if dst.shard else "all_reduce"
    elif src.shard and not dst.shard and not dst.partial:
        ctype = "all_gather"
    elif src.shard and dst.shard:
        ctype = "all_to_all"
    elif not src.shard and dst.shard:
        return None
    else:
        return None
    return CommSpec(ctype, int(numel or 0) * int(dtype_bytes or 0), axis)


@dataclass(frozen=True)
class ResolvedTensor:
    name: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    local_numel: int
    dtype_bytes: int
    is_weight: bool
    is_expert: bool = False
    placement: Placement = Placement()
    storage_id: Optional[str] = None
    trainable: bool = True
    swappable: bool = True
    recomputable: bool = True

    @property
    def local_bytes(self) -> int:
        return self.local_numel * self.dtype_bytes

    @property
    def tid(self):
        return self.name


def resolve_tensor(tensor: TensorRef, dims: DimTable, pm) -> ResolvedTensor:
    global_sizes = tuple(eval_expr(dim, dims) for dim in tensor.shape)
    sizes = list(global_sizes)
    effective_shard = []
    # MindFormers CP always partitions sequence activations; sequence parallel
    # additionally partitions the same dimension over TP.
    if tensor.shard.get(0) == "sp" and pm.degree("cp") > 1:
        if sizes[0] % pm.degree("cp"):
            raise ValueError(f"{tensor.name} dim0 不被 cp 整除")
        sizes[0] //= pm.degree("cp")
        effective_shard.append((0, "cp"))
    for dim_index, axis in tensor.shard.items():
        degree = pm.degree(axis)
        if sizes[dim_index] % degree:
            raise ValueError(
                f"{tensor.name} dim{dim_index}={sizes[dim_index]} "
                f"不被 {axis}={degree} 整除"
            )
        sizes[dim_index] //= degree
        if degree > 1:
            effective_shard.append((dim_index, "tp" if axis == "sp" else axis))
    if (
        "deredund" in pm.pc.moe_token_dispatcher
        or "zero_redundancy" in pm.pc.moe_token_dispatcher
    ) and tensor.name in {
        "dispatched", "expert_gate", "expert_act", "expert_out"
    } and pm.degree("tp") > 1:
        degree = pm.degree("tp")
        if sizes[0] % degree:
            raise ValueError(f"{tensor.name} routed-token dim is not divisible by tp={degree}")
        sizes[0] //= degree
        effective_shard.append((0, "tp"))
    partial = tensor.partial
    if partial is not None and pm.degree(partial) == 1:
        partial = None
    dtype_bytes = tensor.dtype_bytes or (
        getattr(dims, "param_dtype_bytes", 2) if tensor.is_weight
        else getattr(dims, "dtype_bytes", dict(dims).get("dtype_bytes", 2) if isinstance(dims, Mapping) else 2)
    )
    return ResolvedTensor(
        tensor.name,
        global_sizes,
        tuple(sizes),
        prod(sizes) if sizes else 1,
        dtype_bytes,
        tensor.is_weight,
        tensor.has_ep(),
        Placement(effective_shard, partial),
        tensor.storage_id or tensor.name,
        tensor.trainable,
        tensor.swappable,
        tensor.recomputable,
    )


@dataclass(frozen=True)
class ResolvedOp:
    name: str
    type: OpType
    inputs: tuple[ResolvedTensor, ...]
    output: ResolvedTensor
    params: tuple[ResolvedTensor, ...]
    saves: tuple[ResolvedTensor, ...]
    workspace_bytes: int
    collectives: tuple[CommSpec, ...]
    module_paths: tuple[str, ...] = ()
    attrs: Mapping[str, object] = None


@dataclass(frozen=True)
class ResolvedLayer:
    layer_id: int
    layer_type: str
    ops: tuple[ResolvedOp, ...]

    @property
    def activation_bytes(self):
        unique = {t.name: t for op in self.ops for t in op.saves}
        return sum(t.local_bytes for t in unique.values())

    @property
    def checkpoint_bytes(self):
        first = next((t for op in self.ops for t in op.inputs if not t.is_weight), None)
        return first.local_bytes if first else 0


@dataclass(frozen=True)
class ResolvedGraph:
    stages: Mapping[int, tuple[ResolvedLayer, ...]]
    stage_params: Mapping[int, tuple[ResolvedTensor, ...]] = None
    stage_output_bytes: Mapping[int, int] = None
    edge_ops: Mapping[int, tuple[ResolvedOp, ...]] = None


class ShapeEval:
    def resolve(self, spec: ModelSpec, pm) -> ResolvedGraph:
        stages: dict[int, list[ResolvedLayer]] = {}
        for layer_id, layer_type in enumerate(spec.layer_pattern):
            layer_spec = spec.get_layer(layer_type)
            produced: dict[str, Placement] = {}
            resolved_ops: list[ResolvedOp] = []
            for op in layer_spec.ops:
                inputs = tuple(
                    resolve_tensor(tensor, spec.dims, pm)
                    for tensor in op.inputs
                )
                output = resolve_tensor(op.output, spec.dims, pm)
                params = tuple(
                    resolve_tensor(tensor, spec.dims, pm)
                    for tensor in op.params
                )

                # 一张张量被多个 backward 节点引用时只保存一份。
                unique_saves: dict[str, ResolvedTensor] = {}
                for tensor in op.saves:
                    unique_saves.setdefault(
                        tensor.name, resolve_tensor(tensor, spec.dims, pm)
                    )

                collectives = []
                for tensor, resolved in zip(op.inputs, inputs):
                    collective = detect_reshard(
                        produced.get(tensor.name),
                        resolved.placement,
                        resolved.local_numel,
                        resolved.dtype_bytes,
                    )
                    if collective is not None:
                        collectives.append(CommSpec(
                            collective.ctype,
                            collective.volume_bytes,
                            collective.group_axis,
                            collective.phase,
                            resolved.name,
                        ))
                if op.type == OpType.FLASH_ATTN and pm.degree("cp") > 1:
                    if pm.pc.context_parallel_method == "colossal":
                        collectives.append(CommSpec("ring_p2p", output.local_bytes, "cp", output_tensor=output.name))
                    elif pm.pc.context_parallel_method == "ulysses":
                        collectives.append(CommSpec("all_to_all", output.local_bytes, "cp", output_tensor=output.name))
                    else:
                        # Hybrid CP composes an inner Ulysses all-to-all with
                        # an outer ring over cp / ulysses_degree ranks.
                        collectives.extend((
                            CommSpec("all_to_all", output.local_bytes, "ulysses_cp", output_tensor=output.name),
                            CommSpec("ring_p2p", output.local_bytes, "ring_cp", output_tensor=output.name),
                        ))
                if op.type in {OpType.DISPATCH, OpType.COMBINE} and pm.degree("ep") > 1:
                    collectives.append(CommSpec("all_to_all", output.local_bytes, "ep", output_tensor=output.name))
                workspace = (
                    eval_expr(op.workspace, spec.dims)
                    if op.workspace is not None
                    else 0
                )
                if op.type in {OpType.DISPATCH, OpType.COMBINE}:
                    # Local permute/unpermute storage coexists with the
                    # all-to-all communication buffer at dispatcher peaks.
                    workspace += output.local_bytes
                resolved_ops.append(
                    ResolvedOp(
                        op.name,
                        op.type,
                        inputs,
                        output,
                        params,
                        tuple(unique_saves.values()),
                        workspace,
                        tuple(collectives),
                        tuple(op.module_paths),
                        dict(op.attrs),
                    )
                )
                produced[op.output.name] = output.placement

            stage = pm.pc.pp - 1 if layer_type == "mtp" else pm.stage_of(layer_id)
            stages.setdefault(stage, []).append(
                ResolvedLayer(layer_id, layer_type, tuple(resolved_ops))
            )
        resolved_stages = {stage: tuple(layers) for stage, layers in stages.items()}
        # Edge modules use the same ResolvedOp contract as decoder operations.
        # This keeps their parameters, FSDP gathers and temporary outputs on the
        # same execution timeline instead of a static-only side channel.
        embedding_weight = resolve_tensor(TensorRef(
            "embedding_weight", ("vocab", "H"), {0: "tp"}, True,
            storage_id="token_embedding"
        ), spec.dims, pm)
        token_ids = resolve_tensor(TensorRef(
            "token_ids", ("S", "B"), {0: "sp"}, dtype_bytes=4,
            trainable=False, swappable=False, recomputable=False,
        ), spec.dims, pm)
        embedded = resolve_tensor(TensorRef(
            "embedded", ("S", "B", "H"), {0: "sp"}
        ), spec.dims, pm)
        embedding_op = ResolvedOp(
            "embedding", OpType.ELEMENTWISE, (token_ids,), embedded,
            (embedding_weight,), (token_ids,), 0, (), ("embedding",),
        )

        final_norm_weight = resolve_tensor(TensorRef(
            "final_norm_weight", ("H",), is_weight=True, dtype_bytes=4,
            storage_id="final_norm",
        ), spec.dims, pm)
        final_input = resolve_tensor(TensorRef(
            "final_hidden", ("S", "B", "H"), {0: "sp"}
        ), spec.dims, pm)
        normalized = resolve_tensor(TensorRef(
            "normalized_hidden", ("S", "B", "H"), {0: "sp"}
        ), spec.dims, pm)
        final_norm_op = ResolvedOp(
            "final_norm", OpType.NORM, (final_input,), normalized,
            (final_norm_weight,), (final_input,), 0, (), ("decoder.final_layernorm",),
        )
        lm_head_weight = resolve_tensor(TensorRef(
            "lm_head_weight", ("vocab", "H"), {0: "tp"}, True,
            storage_id="token_embedding" if spec.dims.tie_word_embeddings else "lm_head",
        ), spec.dims, pm)
        logits_shard = {0: "sp"}
        if pm.pc.enable_loss_parallel:
            logits_shard[2] = "tp"
        logits = resolve_tensor(TensorRef(
            "logits", ("S", "B", "vocab"), logits_shard,
        ), spec.dims, pm)
        lm_collectives = ()
        if pm.degree("tp") > 1 and not pm.pc.enable_loss_parallel:
            lm_collectives = (CommSpec(
                "all_gather", logits.local_bytes, "tp", output_tensor=logits.name
            ),)
        lm_head_op = ResolvedOp(
            "lm_head", OpType.MATMUL, (normalized,), logits,
            (lm_head_weight,), (normalized,), logits.local_bytes,
            lm_collectives, ("output_layer",),
        )
        edge_ops = {stage: tuple() for stage in resolved_stages}
        edge_ops[0] = (embedding_op,)
        edge_ops[pm.pc.pp - 1] = edge_ops.get(pm.pc.pp - 1, ()) + (
            final_norm_op, lm_head_op,
        )
        stage_params = {
            stage: tuple({
                tensor.storage_id or tensor.name: tensor
                for op in ops for tensor in op.params
            }.values())
            for stage, ops in edge_ops.items()
        }
        return ResolvedGraph(
            resolved_stages,
            stage_params,
            {pm.pc.pp - 1: logits.local_bytes},
            edge_ops,
        )
