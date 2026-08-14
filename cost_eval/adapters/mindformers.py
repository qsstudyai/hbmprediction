"""Offline adapter for MindFormers PyNative TrainConfig-compatible YAML."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..layers import (
    build_dense_decoder, build_moe_decoder,
    build_mla_dense_decoder, build_mla_moe_decoder, build_mtp_layer,
)
from ..model_spec import DimTable, ModelSpec
from ..report import Evaluator
from ..specs import HardwareSpec, OptimizerSpec, ParallelConfig, RecomputeSpec, SwapSpec


@dataclass(frozen=True)
class MindFormersInputs:
    model_spec: ModelSpec
    parallel_config: ParallelConfig
    optimizer: OptimizerSpec
    hardware: HardwareSpec
    recompute: RecomputeSpec
    swap: SwapSpec
    warnings: tuple[str, ...] = ()
    source: str = ""

    def evaluator(self) -> Evaluator:
        return Evaluator(
            self.model_spec,
            self.parallel_config,
            self.optimizer,
            self.hardware,
            self.recompute,
            self.swap,
            warnings=self.warnings,
        )


class MindFormersAdapter:
    """Convert MindFormers PyNative configuration data without importing MindSpore."""

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        world_size: int,
        framework_reserve: int = 0,
    ) -> MindFormersInputs:
        source = Path(path)
        data = load_mindformers_yaml(source)
        return cls.from_mapping(data, world_size, framework_reserve, str(source.resolve()))

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        world_size: int,
        framework_reserve: int = 0,
        source: str = "<mapping>",
    ) -> MindFormersInputs:
        if world_size < 1:
            raise ValueError("world_size must be positive")
        parallelism = _section(data, "parallelism")
        training = _section(data, "training")
        model = _section(data, "model")
        context = _section(data, "context")
        optimizer_data = _section(data, "optimizer")
        recompute_data = _section(data, "recompute")
        recompute_comm_data = _section(data, "recompute_comm")
        swap_data = _section(data, "swap")
        warnings: list[str] = []
        shard_strategy = str(parallelism.get("data_parallel_shard_strategy", "optim_grads_params"))
        if shard_strategy != "optim_grads_params":
            raise ValueError("P0 currently supports MindFormers data_parallel_shard_strategy='optim_grads_params' only")

        tp = _positive_int(
            parallelism.get("tensor_parallel", parallelism.get("model_parallel", 1)),
            "tensor_parallel",
        )
        cp = _positive_int(parallelism.get("context_parallel", 1), "context_parallel")
        pp = _positive_int(
            parallelism.get("pipeline_parallel", parallelism.get("pipeline_stage", 1)),
            "pipeline_parallel",
        )
        ep = _positive_int(parallelism.get("expert_parallel", 1), "expert_parallel")
        base = tp * cp * pp
        if world_size % base:
            raise ValueError("world_size must be divisible by tp * cp * pp")
        data_parallel = world_size // base
        requested_shard = int(parallelism.get("data_parallel_shard", -1))
        if requested_shard < 0:
            dp_shard, dp_replicate = data_parallel, 1
        else:
            if requested_shard < 1 or data_parallel % requested_shard:
                raise ValueError("data_parallel_shard must evenly divide the data-parallel domain")
            dp_shard = requested_shard
            dp_replicate = data_parallel // requested_shard
        if tp > 1 and not bool(parallelism.get("sequence_parallel", False)):
            raise ValueError("MindFormers PyNative requires sequence_parallel=True when TP > 1")
        local_batch = _positive_int(training.get("local_batch_size", 1), "local_batch_size")
        global_batch = _positive_int(training.get("global_batch_size", local_batch), "global_batch_size")
        batch_unit = data_parallel * local_batch
        if global_batch < batch_unit or global_batch % batch_unit:
            raise ValueError(
                "training.global_batch_size must be divisible by "
                "data_parallel * training.local_batch_size"
            )
        # The legacy static-graph LLaMA runner defines ``micro_batch_num``
        # after accounting for its interleave factor.  Native PyNative
        # configs do not expose this training-level field and keep the usual
        # global-batch/data-parallel definition.  Preserve both contracts so
        # the offline DAG has the same microbatch count as the real runner.
        training_interleave = _positive_int(
            training.get("micro_batch_interleave_num", 1),
            "micro_batch_interleave_num",
        )
        effective_batch_unit = batch_unit * training_interleave
        if global_batch < effective_batch_unit or global_batch % effective_batch_unit:
            raise ValueError(
                "training.global_batch_size must be divisible by "
                "data_parallel * training.local_batch_size * "
                "training.micro_batch_interleave_num"
            )
        num_microbatches = global_batch // effective_batch_unit

        dims, pattern, model_warnings = _build_model_shape(model, training)
        warnings.extend(model_warnings)
        mla = bool(model.get("multi_latent_attention", False))
        if mla:
            pattern = tuple(
                "mla_moe" if item == "moe" else "mla_dense"
                for item in pattern
            )
            layer_specs = {"mla_dense": build_mla_dense_decoder(dims)}
            if "mla_moe" in pattern:
                layer_specs["mla_moe"] = build_mla_moe_decoder(dims)
        else:
            layer_specs = {"dense": build_dense_decoder(dims)}
            if "moe" in pattern:
                layer_specs["moe"] = build_moe_decoder(dims)
        if dims.n_mtp_layers:
            pattern = pattern + ("mtp",) * dims.n_mtp_layers
            layer_specs["mtp"] = build_mtp_layer(dims)
        runtime_mode = str(
            data.get("runtime_mode")
            or data.get("run_mode")
            or data.get("graph_mode")
            or ("pynative" if data.get("use_pynative", True) else "static_graph")
        ).lower()
        capabilities = {
            "runtime_mode": runtime_mode,
            "mla": mla,
            "mtp_layers": dims.n_mtp_layers,
            "hyper_connections": dims.enable_hyper_connections,
            "hc_mult": dims.hc_mult,
            "grouped_gemm": bool(model.get("moe_grouped_gemm", model.get("grouped_gemm", False))),
            "token_dispatcher": str(parallelism.get("moe_token_dispatcher_type", "alltoall")),
            "optimizer_type": str(optimizer_data.get("type", "AdamW")).lower(),
        }
        model_spec = ModelSpec(
            str(model.get("model_type") or model.get("architectures") or "mindformers-model"),
            dims,
            pattern,
            layer_specs,
            capabilities,
        )
        cp_method = str(parallelism.get("context_parallel_method", "colossal"))
        dispatcher = str(parallelism.get("moe_token_dispatcher_type", "alltoall"))
        supported_dispatchers = {"alltoall", "alltoall_deredundency", "alltoall_zero_redundancy"}
        if dispatcher not in supported_dispatchers:
            warnings.append(f"moe_token_dispatcher_type={dispatcher!r} uses baseline all-to-all memory volume in P0")

        layers_per_stage = _parse_stage_layers(
            parallelism.get("pipeline_parallel_layers_per_stage"), dims.n_layers
        )
        # PpLayerSetting receives decoder-layer count only.  DeepSeek models
        # append MTP auxiliary layers after that mapping, on the final
        # physical stage; mirror that runtime placement so the offline
        # ParallelModel still covers ``dims.total_layers``.
        if layers_per_stage is not None and dims.n_mtp_layers:
            layers_per_stage = tuple(
                (*stage_layers, *range(dims.n_layers, dims.total_layers))
                if stage == pp - 1 else stage_layers
                for stage, stage_layers in enumerate(layers_per_stage)
            )
        parallel = ParallelConfig(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            cp=cp,
            tp=tp,
            pp=pp,
            ep=ep,
            sequence_parallel=bool(parallelism.get("sequence_parallel", False)),
            cpu_offload=bool(parallelism.get("cpu_offload", False)),
            num_microbatches=num_microbatches,
            reshard_after_forward_policy=str(
                parallelism.get("reshard_after_forward_policy", "default")
            ).lower(),
            dense_fsdp_shard_size=_optional_int(parallelism.get("dense_fsdp_shard_size")),
            context_parallel_method=cp_method,
            ulysses_degree_in_cp=_optional_int(parallelism.get("ulysses_degree_in_cp")),
            pipeline_schedule=str(parallelism.get("pipeline_parallel_schedule", "1f1b")),
            pipeline_interleave=_positive_int(
                parallelism.get("pipeline_parallel_interleave_num", 1),
                "pipeline_parallel_interleave_num",
            ),
            layers_per_stage=layers_per_stage,
            moe_token_dispatcher=dispatcher,
            enable_loss_parallel=bool(parallelism.get("enable_loss_parallel", False)),
        )

        optimizer = _build_optimizer(optimizer_data, model, warnings)
        hardware = HardwareSpec(
            parse_memory_bytes(context.get("max_device_memory", "59GB")),
            framework_reserve,
        )
        recompute = _build_recompute(
            recompute_data,
            recompute_comm_data,
            dims.total_layers,
            warnings,
        )
        swap = _build_swap(swap_data, dims.total_layers, warnings)
        return MindFormersInputs(
            model_spec, parallel, optimizer, hardware, recompute, swap,
            tuple(warnings), source,
        )


def load_mindformers_yaml(path: Path) -> dict[str, Any]:
    """Read the top-level TrainConfig sections used by the offline evaluator.

    PyYAML is used when available. The fallback parser supports the scalar and
    inline-list subset used by MindFormers' evaluator-facing fields.
    """
    if not path.exists():
        raise FileNotFoundError(f"MindFormers config not found: {path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _load_yaml_subset(path.read_text(encoding="utf-8"))
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("MindFormers YAML root must be a mapping")
    return loaded


def parse_memory_bytes(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGT]?B)\s*", str(value), re.I)
    if not match:
        raise ValueError(f"invalid max_device_memory: {value!r}")
    units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}
    return int(float(match.group(1)) * units[match.group(2).upper()])


def _build_model_shape(
    model: Mapping[str, Any], training: Mapping[str, Any]
) -> tuple[DimTable, tuple[str, ...], list[str]]:
    warnings: list[str] = []
    hidden = _required_alias(model, "hidden_size")
    heads = _required_alias(model, "num_attention_heads")
    layers = _required_alias(model, "num_layers", "num_hidden_layers")
    # DeepSeek-V4's PyNative config names the routed expert width
    # ``moe_intermediate_size`` and does not repeat it as a generic FFN
    # field.  It is the only FFN width available for the MoE layer shape, so
    # use it as the base width when the conventional aliases are absent.
    ffn = _required_alias(
        model, "ffn_hidden_size", "intermediate_size", "moe_intermediate_size"
    )
    seq = _required_alias(model, "seq_length", "max_sequence_length")
    vocab = _required_alias(model, "vocab_size", "padded_vocab_size")
    kv_heads = int(model.get("num_query_groups") or model.get("num_key_value_heads") or heads)
    head_dim = int(model.get("kv_channels") or hidden // heads)
    experts = int(model.get("num_moe_experts") or model.get("n_routed_experts") or 0)
    topk = int(model.get("moe_router_topk") or model.get("num_experts_per_tok") or 1)
    moe_ffn = int(model.get("moe_ffn_hidden_size") or model.get("moe_intermediate_size") or ffn)
    batch = int(training.get("local_batch_size", 1))
    hc_enabled = bool(
        model.get("enable_hyper_connections", model.get("hyper_connections", False))
        or model.get("enable_hc_head", False)
    )
    hc_mult = int(
        model.get("hc_mult")
        or model.get("hyper_connections_multiplier")
        or (2 if hc_enabled else 0)
    )
    dims = DimTable(
        H=hidden,
        F=ffn,
        n_heads=heads,
        n_kv=kv_heads,
        head_dim=head_dim,
        S=seq,
        B=batch,
        vocab=vocab,
        n_layers=layers,
        n_experts=experts,
        topk=topk,
        moe_F=moe_ffn if experts else 0,
        n_shared=int(model.get("num_shared_experts") or model.get("n_shared_experts") or 0),
        shared_F=int(model.get("moe_shared_expert_intermediate_size") or 0),
        dtype_bytes=_dtype_bytes(model.get("compute_dtype", "bfloat16")),
        param_dtype_bytes=_dtype_bytes(model.get("params_dtype", "bfloat16")),
        q_lora_rank=int(model.get("q_lora_rank") or 0),
        kv_lora_rank=int(model.get("kv_lora_rank") or 0),
        qk_nope_head_dim=int(model.get("qk_nope_head_dim") or 0),
        qk_rope_head_dim=int(model.get("qk_rope_head_dim") or 0),
        v_head_dim=int(model.get("v_head_dim") or head_dim),
        n_mtp_layers=int(model.get("mtp_num_layers") or model.get("num_nextn_predict_layers") or 0),
        use_shared_expert_gating=bool(model.get("use_shared_expert_gating", False)),
        tie_word_embeddings=bool(model.get("tie_word_embeddings", False)),
        hc_mult=hc_mult,
        enable_hyper_connections=hc_enabled,
    )
    pattern = _moe_pattern(model, layers, experts)
    if dims.n_shared and not dims.shared_F:
        warnings.append("shared experts configured without moe_shared_expert_intermediate_size; shared branch omitted")
    return dims, pattern, warnings


def _moe_pattern(model: Mapping[str, Any], layers: int, experts: int) -> tuple[str, ...]:
    if not experts:
        return ("dense",) * layers
    first_dense = int(model.get("first_k_dense_replace") or 0)
    if first_dense:
        return ("dense",) * first_dense + ("moe",) * (layers - first_dense)
    frequency = model.get("moe_layer_freq", 1)
    if isinstance(frequency, list):
        if len(frequency) != layers:
            raise ValueError("moe_layer_freq list length must equal number of layers")
        return tuple("moe" if int(item) else "dense" for item in frequency)
    frequency = _positive_int(frequency, "moe_layer_freq")
    return tuple("moe" if index % frequency == 0 else "dense" for index in range(layers))


def _build_optimizer(
    config: Mapping[str, Any], model: Mapping[str, Any], warnings: list[str]
) -> OptimizerSpec:
    name = str(config.get("type", "AdamW"))
    parameter_bytes = _dtype_bytes(model.get("params_dtype", "bfloat16"))
    master_bytes = 0 if parameter_bytes == 4 else 4
    if name.lower() in {"muon", "muonadam", "muon_adam"}:
        ns_steps = int(
            config.get("muon_ns_steps", config.get("newton_schulz_steps", config.get("ns_steps", 5)))
        )
        fraction = float(config.get("muon_parameter_fraction", config.get("muon_fraction", 1.0)))
        momentum_bytes = int(config.get("muon_momentum_bytes", config.get("momentum_bytes", 4)))
        main_bytes = int(config.get("muon_main_parameter_bytes", config.get("main_parameter_bytes", 4)))
        total = parameter_bytes + parameter_bytes + main_bytes + momentum_bytes
        return OptimizerSpec(
            name,
            total,
            parameter_bytes=parameter_bytes,
            gradient_bytes=parameter_bytes,
            master_weight_bytes=main_bytes,
            optimizer_state_bytes=momentum_bytes,
            muon_ns_steps=ns_steps,
            muon_parameter_fraction=fraction,
            muon_momentum_bytes=momentum_bytes,
            muon_main_parameter_bytes=main_bytes,
        )
    if name.lower() != "adamw":
        warnings.append(f"optimizer {name!r} uses the AdamW byte model in P0")
    total = parameter_bytes + parameter_bytes + master_bytes + 8
    return OptimizerSpec(
        name, total,
        parameter_bytes=parameter_bytes,
        gradient_bytes=parameter_bytes,
        master_weight_bytes=master_bytes,
        optimizer_state_bytes=8,
    )


def _build_recompute(
    config: Mapping[str, Any],
    recompute_comm: Mapping[str, Any],
    n_layers: int,
    warnings: list[str],
) -> RecomputeSpec:
    mode = str(config.get("mode", "None")).lower()
    raw_excluded = config.get("exclude_op") or []
    if isinstance(raw_excluded, str):
        raw_excluded = [raw_excluded]
    excluded = frozenset(str(item) for item in raw_excluded)
    comm_enabled = bool(recompute_comm.get("enable", False))
    if comm_enabled and not recompute_comm.get("select_module"):
        raise ValueError("MindFormers recompute_comm.enable=True requires select_module")
    comm_modules = _parse_module_selection(
        recompute_comm.get("select_module"), n_layers
    ) if comm_enabled else {}
    if mode == "select":
        modules = _parse_module_selection(config.get("select_module"), n_layers)
        if not modules:
            raise ValueError("MindFormers select recompute requires select_module")
        return RecomputeSpec(
            "select", select_modules=modules, exclude_ops=excluded,
            recompute_comm=comm_enabled, comm_select_modules=comm_modules,
        )
    if mode == "full":
        selected = _parse_layer_selection(config.get("full_recompute_layer"), n_layers)
        if not selected:
            raise ValueError("MindFormers full recompute requires full_recompute_layer")
        return RecomputeSpec(
            "full", selected, exclude_ops=excluded,
            recompute_comm=comm_enabled, comm_select_modules=comm_modules,
        )
    return RecomputeSpec(
        "none", exclude_ops=excluded, recompute_comm=comm_enabled,
        comm_select_modules=comm_modules,
    )


def _build_swap(
    config: Mapping[str, Any], n_layers: int, warnings: list[str]
) -> SwapSpec:
    if not bool(config.get("enable", False)):
        return SwapSpec()
    if config.get("op_swap"):
        op_layers = _parse_module_selection(config.get("op_swap"), n_layers)
    else:
        op_layers = {}
    selected = _parse_layer_selection(config.get("layer_swap"), n_layers)
    prefetch = _positive_int(config.get("default_prefetch", 1), "default_prefetch")
    targeted = set(selected).union(*(set(v) for v in op_layers.values()), set())
    if prefetch >= n_layers or any(layer + prefetch >= n_layers for layer in targeted):
        raise ValueError("MindFormers swap prefetch target exceeds the model layer range")
    known = {"attention", "self_attention", "mlp", "transformer_layer"}
    for module in op_layers:
        if module.split(".")[-1] not in known:
            warnings.append(f"swap module pattern {module!r} has no exact P0 op-group mapping")
    return SwapSpec(
        enabled=True,
        layers=selected,
        prefetch_depth=prefetch,
        op_layers=op_layers,
    )


def _parse_stage_layers(value: Any, n_layers: int) -> tuple[tuple[int, ...], ...] | None:
    if value is None or str(value).lower() == "auto":
        return None
    if not isinstance(value, list):
        raise ValueError("pipeline_parallel_layers_per_stage must be auto or a list")
    stages = tuple(tuple(sorted(_parse_layer_selection(stage, n_layers))) for stage in value)
    return stages


def _parse_layer_selection(value: Any, n_layers: int) -> frozenset[int]:
    if value is None:
        return frozenset()
    entries = value if isinstance(value, (list, tuple, set)) else [value]
    selected: set[int] = set()
    for entry in entries:
        if isinstance(entry, int):
            selected.add(entry)
            continue
        if isinstance(entry, Mapping):
            nested = entry.get("layers") or entry.get("layer") or entry.get("layer_id")
            selected.update(_parse_layer_selection(nested, n_layers))
            continue
        text = str(entry).strip()
        # MindFormers represents interleaved explicit PP placement as one
        # comma-separated string per physical stage, e.g. ``"0-4,9-12"``.
        # Parse each virtual-chunk range independently rather than treating
        # the entire string as an unknown scalar.
        if "," in text:
            for part in text.split(","):
                if part.strip():
                    selected.update(_parse_layer_selection(part.strip(), n_layers))
            continue
        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
        if match:
            start, end = map(int, match.groups())
            selected.update(range(start, end + 1))
        elif text.isdigit():
            selected.add(int(text))
    if any(layer < 0 or layer >= n_layers for layer in selected):
        raise ValueError("layer selection contains an out-of-range layer")
    return frozenset(selected)


def _parse_module_selection(value: Any, n_layers: int) -> dict[str, frozenset[int]]:
    if value is None:
        return {}
    result: dict[str, frozenset[int]] = {}
    entries = value if isinstance(value, list) else [value]
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        if "module" in entry or "op" in entry or "op_name" in entry or "name" in entry:
            name = str(
                entry.get("module") or entry.get("op") or entry.get("op_name") or entry.get("name")
            )
            layers = entry.get("layers") or entry.get("layer") or entry.get("layer_id")
            result[name] = _parse_layer_selection(layers, n_layers)
        else:
            for name, layers in entry.items():
                result[str(name)] = _parse_layer_selection(layers, n_layers)
    return result


def _load_yaml_subset(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if indent == 0 and content.endswith(":"):
            section = content[:-1].strip()
            current = {}
            result[section] = current
        elif indent > 0 and current is not None and not content.startswith("-") and ":" in content:
            key, raw_value = content.split(":", 1)
            if raw_value.strip():
                current[key.strip()] = _parse_scalar(raw_value.strip())
    return result


def _strip_comment(line: str) -> str:
    quote: str | None = None
    for index, char in enumerate(line):
        if char in {'"', "'"}:
            quote = None if quote == char else char if quote is None else quote
        elif char == "#" and quote is None:
            return line[:index]
    return line


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"\'')


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"MindFormers section {name!r} must be a mapping")
    return value


def _required_alias(data: Mapping[str, Any], *names: str) -> int:
    for name in names:
        if data.get(name) is not None:
            return int(data[name])
    raise ValueError(f"MindFormers model config requires one of: {', '.join(names)}")


def _positive_int(value: Any, name: str) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be >= 1")
    return converted


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _dtype_bytes(value: Any) -> int:
    name = str(value).lower().replace("mindspore.", "")
    sizes = {"float16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2, "float32": 4, "fp32": 4}
    if name not in sizes:
        raise ValueError(f"unsupported parameter dtype: {value!r}")
    return sizes[name]
