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
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.FloorDiv) and left % right:
                raise ValueError(
                    f"维度表达式要求整除，实际为 {left}//{right}"
                )
            return _BINOPS[type(node.op)](left, right)
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
    # Machine identity of the collective result.  ``output_tensor`` remains a
    # human-readable compatibility field used by existing reports.
    output_value_id: Optional[str] = None

    @property
    def kind(self):
        return self.ctype


def detect_reshards(
    src: Optional[Placement],
    dst: Placement,
    numel: Optional[int] = None,
    dtype_bytes: Optional[int] = None,
) -> tuple[CommSpec, ...]:
    """Return the deterministic 0..N collective plan for a placement change.

    Placement is compared per mesh axis.  Adding a shard is local; removing a
    shard needs an all-gather; moving the same mesh axis to another tensor
    dimension needs an all-to-all.  A partial value is materialized first.
    """

    if isinstance(dst, ResolvedTensor):
        dest_tensor = dst
        dst = dest_tensor.placement
        numel, dtype_bytes = dest_tensor.local_numel, dest_tensor.dtype_bytes
    if isinstance(src, ResolvedTensor):
        src = src.placement
    if src is None or src == dst:
        return ()
    volume = int(numel or 0) * int(dtype_bytes or 0)
    src_by_axis = {axis: dim for dim, axis in src.shard}
    dst_by_axis = {axis: dim for dim, axis in dst.shard}
    result: list[CommSpec] = []

    if src.partial and src.partial != dst.partial:
        partial_axis = src.partial
        ctype = "reduce_scatter" if partial_axis in dst_by_axis else "all_reduce"
        result.append(CommSpec(ctype, volume, partial_axis))

    # Sorting is required: communication plans must not depend on hash/set
    # iteration order.
    for axis in sorted(set(src_by_axis) | set(dst_by_axis)):
        src_dim = src_by_axis.get(axis)
        dst_dim = dst_by_axis.get(axis)
        if src_dim is None or src_dim == dst_dim:
            continue
        if dst_dim is None:
            result.append(CommSpec("all_gather", volume, axis))
        else:
            result.append(CommSpec("all_to_all", volume, axis))
    return tuple(result)


def detect_reshard(
    src: Optional[Placement],
    dst: Placement,
    numel: Optional[int] = None,
    dtype_bytes: Optional[int] = None,
) -> Optional[CommSpec]:
    """Compatibility wrapper for callers that expect at most one action."""

    actions = detect_reshards(src, dst, numel, dtype_bytes)
    return actions[0] if actions else None


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
    fsdp_replicated: bool = False
    logical_name: Optional[str] = None
    value_id: Optional[str] = None

    @property
    def local_bytes(self) -> int:
        return self.local_numel * self.dtype_bytes

    @property
    def tid(self):
        # Compatibility/display identifier.  Accounting uses ``storage_key``.
        return self.name

    @property
    def storage_key(self) -> str:
        """Identity used for byte de-duplication.

        Only an explicit storage id aliases otherwise distinct values.
        """

        return self.storage_id or self.value_id or self.name


def _placement_value_id(tensor: TensorRef, placement: Placement) -> str:
    if tensor.value_id:
        return tensor.value_id
    shards = ",".join(f"{dim}:{axis}" for dim, axis in placement.shard) or "replicated"
    partial = placement.partial or "complete"
    dtype = tensor.dtype_bytes if tensor.dtype_bytes is not None else "default"
    return f"{tensor.name}|shard={shards}|partial={partial}|dtype={dtype}"


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
    placement = Placement(effective_shard, partial)
    return ResolvedTensor(
        tensor.name,
        global_sizes,
        tuple(sizes),
        prod(sizes) if sizes else 1,
        dtype_bytes,
        tensor.is_weight,
        tensor.has_ep(),
        placement,
        tensor.storage_id,
        tensor.trainable,
        tensor.swappable,
        tensor.recomputable,
        tensor.fsdp_replicated,
        tensor.name,
        _placement_value_id(tensor, placement),
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
    backward_workspace_bytes: int = 0
    workspace_source: str = "none"
    workspace_upper_bytes: int = 0


@dataclass(frozen=True)
class ResolvedLayer:
    layer_id: int
    layer_type: str
    ops: tuple[ResolvedOp, ...]

    @property
    def activation_bytes(self):
        unique = {t.storage_key: t for op in self.ops for t in op.saves}
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
    def __init__(self, workspace_registry=None):
        self.workspace_registry = workspace_registry

    def resolve(self, spec: ModelSpec, pm) -> ResolvedGraph:
        stages: dict[int, list[ResolvedLayer]] = {}
        for layer_id, layer_type in enumerate(spec.layer_pattern):
            layer_spec = spec.get_layer(layer_type)
            produced: dict[str, ResolvedTensor] = {}
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
                    saved = resolve_tensor(tensor, spec.dims, pm)
                    unique_saves.setdefault(
                        saved.storage_key, saved
                    )

                collectives = []
                for tensor, resolved in zip(op.inputs, inputs):
                    actions = detect_reshards(
                        produced.get(tensor.name),
                        resolved,
                        resolved.local_numel,
                        resolved.dtype_bytes,
                    )
                    for collective in actions:
                        collectives.append(CommSpec(
                            collective.ctype,
                            collective.volume_bytes,
                            collective.group_axis,
                            collective.phase,
                            resolved.name,
                            output_value_id=resolved.value_id,
                        ))
                if op.type in {OpType.FLASH_ATTN, OpType.SPARSE_ATTN} and pm.degree("cp") > 1:
                    if pm.pc.context_parallel_method == "colossal":
                        collectives.append(CommSpec("ring_p2p", output.local_bytes, "cp", output_tensor=output.name, output_value_id=output.value_id))
                    elif pm.pc.context_parallel_method == "ulysses":
                        collectives.append(CommSpec("all_to_all", output.local_bytes, "cp", output_tensor=output.name, output_value_id=output.value_id))
                    else:
                        # Hybrid CP composes an inner Ulysses all-to-all with
                        # an outer ring over cp / ulysses_degree ranks.
                        collectives.extend((
                            CommSpec("all_to_all", output.local_bytes, "ulysses_cp", output_tensor=output.name, output_value_id=output.value_id),
                            CommSpec("ring_p2p", output.local_bytes, "ring_cp", output_tensor=output.name, output_value_id=output.value_id),
                        ))
                if op.type in {OpType.DISPATCH, OpType.COMBINE} and pm.degree("ep") > 1:
                    collectives.append(CommSpec("all_to_all", output.local_bytes, "ep", output_tensor=output.name, output_value_id=output.value_id))
                workspace = (
                    eval_expr(op.workspace, spec.dims)
                    if op.workspace is not None
                    else 0
                )
                if op.attrs.get("workspace_like_output"):
                    workspace += output.local_bytes
                if op.type in {OpType.DISPATCH, OpType.COMBINE}:
                    # Local permute/unpermute storage coexists with the
                    # all-to-all communication buffer at dispatcher peaks.
                    workspace += output.local_bytes
                workspace_source = "formula" if workspace else "none"
                workspace_upper = workspace
                if self.workspace_registry is not None:
                    from .workspace_registry import WorkspaceKey
                    key = WorkspaceKey(
                        str(spec.capabilities.get("model_family", spec.name)),
                        op.type.value,
                        str(op.attrs.get("fusion_variant", "default")),
                        output.dtype_bytes,
                        output.local_shape,
                        pm.degree("tp"), pm.degree("cp"), pm.degree("ep"),
                        str(spec.capabilities.get("mindspore_version", "unknown")),
                        str(spec.capabilities.get("cann_version", "unknown")),
                        str(spec.capabilities.get("hardware", "unknown")),
                    )
                    profile = self.workspace_registry.lookup(key)
                    if profile is not None:
                        workspace = profile.point_bytes
                        workspace_upper = profile.upper_bytes
                        workspace_source = profile.source
                backward_workspace = (
                    eval_expr(op.backward_workspace, spec.dims)
                    if op.backward_workspace is not None
                    else 0
                )
                multiplier = int(
                    op.attrs.get(
                        "backward_workspace_largest_input_multiplier",
                        int(bool(op.attrs.get("backward_workspace_like_largest_input"))),
                    )
                )
                if multiplier:
                    backward_workspace += multiplier * max(
                        (tensor.local_bytes for tensor in inputs), default=0
                    )
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
                        backward_workspace,
                        workspace_source,
                        workspace_upper,
                    )
                )
                produced[op.output.name] = output

            is_mtp = layer_type == "mtp" or layer_type.startswith("deepseek_v4_mtp")
            stage = pm.pc.pp - 1 if is_mtp else pm.stage_of(layer_id)
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
                "all_gather", logits.local_bytes, "tp", output_tensor=logits.name,
                output_value_id=logits.value_id,
            ),)
        lm_head_op = ResolvedOp(
            "lm_head", OpType.MATMUL, (normalized,), logits,
            (lm_head_weight,), (normalized,), 0,
            lm_collectives, ("output_layer",),
        )
        from .layers.loss import build_loss_ops

        loss_specs = build_loss_ops(
            spec.dims,
            str(spec.capabilities.get("loss_variant", "fallback_cross_entropy")),
            pm.pc.enable_loss_parallel,
        )
        loss_ops = []
        edge_produced = {"logits": logits}
        for loss_spec in loss_specs:
            loss_inputs = tuple(
                resolve_tensor(tensor, spec.dims, pm)
                for tensor in loss_spec.inputs
            )
            loss_output = resolve_tensor(loss_spec.output, spec.dims, pm)
            loss_saves = {}
            for tensor in loss_spec.saves:
                saved = resolve_tensor(tensor, spec.dims, pm)
                loss_saves.setdefault(saved.storage_key, saved)
            loss_collectives = []
            for tensor, resolved in zip(loss_spec.inputs, loss_inputs):
                for collective in detect_reshards(
                    edge_produced.get(tensor.name), resolved
                ):
                    loss_collectives.append(CommSpec(
                        collective.ctype, collective.volume_bytes,
                        collective.group_axis, collective.phase,
                        resolved.name, output_value_id=resolved.value_id,
                    ))
            loss_workspace = (
                eval_expr(loss_spec.workspace, spec.dims)
                if loss_spec.workspace is not None else 0
            )
            loss_backward_workspace = (
                eval_expr(loss_spec.backward_workspace, spec.dims)
                if loss_spec.backward_workspace is not None else 0
            )
            multiplier = int(
                loss_spec.attrs.get(
                    "backward_workspace_largest_input_multiplier",
                    int(bool(loss_spec.attrs.get("backward_workspace_like_largest_input"))),
                )
            )
            if multiplier:
                loss_backward_workspace += multiplier * max(
                    (tensor.local_bytes for tensor in loss_inputs), default=0
                )
            loss_ops.append(ResolvedOp(
                loss_spec.name, loss_spec.type, loss_inputs, loss_output, (),
                tuple(loss_saves.values()), loss_workspace,
                tuple(loss_collectives), tuple(loss_spec.module_paths),
                dict(loss_spec.attrs), loss_backward_workspace,
            ))
            edge_produced[loss_spec.output.name] = loss_output
        edge_ops = {stage: tuple() for stage in resolved_stages}
        edge_ops[0] = (embedding_op,)
        edge_ops[pm.pc.pp - 1] = edge_ops.get(pm.pc.pp - 1, ()) + (
            final_norm_op, lm_head_op, *loss_ops,
        )
        stage_params = {
            stage: tuple({
                tensor.storage_key: tensor
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
