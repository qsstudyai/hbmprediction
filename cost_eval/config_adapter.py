"""把专家配置字典、JSON 或常用 YAML 子集转换为评估器输入。"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .layers import build_dense_decoder, build_moe_decoder
from .model_spec import DimTable, ModelSpec
from .report import Evaluator
from .specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)

_PARALLEL_ALIASES = {
    "tensor_parallel": "tp",
    "pipeline_parallel": "pp",
    "expert_parallel": "ep",
    "context_parallel": "cp",
    "data_parallel_shard": "dp_shard",
    "data_parallel_replicate": "dp_replicate",
}
_BYTE_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 2**10,
    "mib": 2**20,
    "gib": 2**30,
    "tib": 2**40,
}


def parse_bytes(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)\s*",
            value,
            flags=re.IGNORECASE,
        )
        if match:
            return int(float(match.group(1)) * _BYTE_UNITS[match.group(2).lower()])
    raise ValueError(f"无法解析内存字节数: {value!r}")


def _parse_yaml_scalar(text: str) -> Any:
    text = text.strip()
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if not text:
        return {}
    if text[0:1] in {"'", '"'}:
        return ast.literal_eval(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return (
            []
            if not inner
            else [_parse_yaml_scalar(item) for item in inner.split(",")]
        )
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result = {}
        if inner:
            for item in inner.split(","):
                key, value = item.split(":", 1)
                result[str(_parse_yaml_scalar(key))] = _parse_yaml_scalar(value)
        return result
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _simple_yaml_load(text: str) -> Any:
    """解析本项目配置所需的 YAML 子集：嵌套映射、标量与标量列表。"""

    lines = []
    for line_number, raw in enumerate(text.splitlines(), 1):
        content = raw.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        if "\t" in content[:indent]:
            raise ValueError(f"YAML 第 {line_number} 行不能使用 tab 缩进")
        lines.append((indent, content.strip(), line_number))
    if not lines:
        return {}

    def parse_block(index: int, indent: int):
        is_list = lines[index][1].startswith("- ")
        container = [] if is_list else {}
        while index < len(lines):
            current_indent, content, line_number = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"YAML 第 {line_number} 行缩进不合法")
            if is_list:
                if not content.startswith("- "):
                    break
                item = content[2:].strip()
                if not item:
                    if index + 1 >= len(lines):
                        container.append(None)
                        index += 1
                    else:
                        child_indent = lines[index + 1][0]
                        child, index = parse_block(index + 1, child_indent)
                        container.append(child)
                else:
                    container.append(_parse_yaml_scalar(item))
                    index += 1
            else:
                if content.startswith("- ") or ":" not in content:
                    raise ValueError(f"YAML 第 {line_number} 行应为 key: value")
                key, raw_value = content.split(":", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                if raw_value:
                    container[key] = _parse_yaml_scalar(raw_value)
                    index += 1
                elif index + 1 < len(lines) and lines[index + 1][0] > indent:
                    child_indent = lines[index + 1][0]
                    child, index = parse_block(index + 1, child_indent)
                    container[key] = child
                else:
                    container[key] = {}
                    index += 1
        return container, index

    result, final_index = parse_block(0, lines[0][0])
    if final_index != len(lines):
        raise ValueError(f"YAML 第 {lines[final_index][2]} 行无法解析")
    return result


def _section(config: Mapping[str, Any], *names: str) -> dict:
    for name in names:
        if name in config:
            value = config[name]
            if not isinstance(value, Mapping):
                raise ValueError(f"{name} 必须是映射")
            return dict(value)
    return {}


@dataclass(frozen=True)
class EvaluationInputs:
    model_spec: ModelSpec
    parallel: ParallelConfig
    optimizer: OptimizerSpec
    hardware: HardwareSpec
    recompute: RecomputeSpec
    swap: SwapSpec

    def evaluator(self) -> Evaluator:
        return Evaluator(
            self.model_spec,
            self.parallel,
            self.optimizer,
            self.hardware,
            self.recompute,
            self.swap,
        )


class ConfigAdapter:
    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> EvaluationInputs:
        model = _section(config, "model", "model_spec")
        model_type = str(model.pop("type", "dense")).lower()
        model_name = str(model.pop("name", config.get("name", model_type)))
        pattern = model.pop(
            "layer_pattern", config.get("layer_pattern")
        )
        dim_names = {item.name for item in fields(DimTable)}
        unknown_model = set(model) - dim_names
        if unknown_model:
            raise ValueError(
                f"未知模型配置项: {', '.join(sorted(unknown_model))}"
            )
        try:
            dims = DimTable(**model)
        except TypeError as exc:
            raise ValueError(f"模型维度配置不完整: {exc}") from exc

        if pattern is None:
            pattern = [model_type] * dims.n_layers
        if isinstance(pattern, str):
            raise ValueError("layer_pattern 必须是层类型列表，不能是字符串")
        pattern = tuple(str(item).lower() for item in pattern)
        builders = {
            "dense": build_dense_decoder,
            "moe": build_moe_decoder,
        }
        unknown_types = set(pattern) - set(builders)
        if unknown_types:
            raise ValueError(
                f"未知层类型: {', '.join(sorted(unknown_types))}"
            )
        layer_specs = {
            layer_type: builders[layer_type](dims)
            for layer_type in set(pattern)
        }
        model_spec = ModelSpec(
            model_name,
            dims,
            pattern,
            layer_specs,
            capabilities={
                "runtime_mode": str(config.get("runtime_mode", "pynative")).lower(),
                "mla": bool(getattr(dims, "q_lora_rank", 0) or getattr(dims, "kv_lora_rank", 0)),
                "hyper_connections": bool(getattr(dims, "enable_hyper_connections", False)),
                "hc_mult": int(getattr(dims, "hc_mult", 0)),
            },
        )

        parallel_values = _section(
            config, "parallel", "parallelism", "parallel_config"
        )
        parallel_values = {
            _PARALLEL_ALIASES.get(key, key): value
            for key, value in parallel_values.items()
        }
        allowed_parallel = {item.name for item in fields(ParallelConfig)}
        unknown_parallel = set(parallel_values) - allowed_parallel
        if unknown_parallel:
            raise ValueError(
                f"未知并行配置项: {', '.join(sorted(unknown_parallel))}"
            )
        parallel = ParallelConfig(**parallel_values)

        optimizer_values = _section(config, "optimizer")
        optimizer_type = str(optimizer_values.pop("type", "AdamW"))
        fp32_grad = bool(optimizer_values.pop("fp32_grad", False))
        if optimizer_type.lower() in {"muon", "muonadam", "muon_adam"}:
            parameter_bytes = int(optimizer_values.pop("parameter_bytes", 2))
            gradient_bytes = int(optimizer_values.pop("gradient_bytes", parameter_bytes))
            main_bytes = int(optimizer_values.pop("muon_main_parameter_bytes", 4))
            momentum_bytes = int(optimizer_values.pop("muon_momentum_bytes", 4))
            ns_value = optimizer_values.pop("muon_ns_steps", None)
            if ns_value is None:
                ns_value = optimizer_values.pop("newton_schulz_steps", 5)
            ns_steps = int(ns_value)
            fraction = float(optimizer_values.pop("muon_parameter_fraction", 1.0))
            optimizer = OptimizerSpec(
                optimizer_type,
                parameter_bytes + gradient_bytes + main_bytes + momentum_bytes,
                parameter_bytes=parameter_bytes,
                gradient_bytes=gradient_bytes,
                master_weight_bytes=main_bytes,
                optimizer_state_bytes=momentum_bytes,
                muon_ns_steps=ns_steps,
                muon_parameter_fraction=fraction,
                muon_momentum_bytes=momentum_bytes,
                muon_main_parameter_bytes=main_bytes,
            )
        elif "state_bytes_per_param" in optimizer_values:
            optimizer = OptimizerSpec(
                optimizer_type,
                int(optimizer_values.pop("state_bytes_per_param")),
            )
        elif optimizer_type.lower() == "adamw":
            optimizer = OptimizerSpec.adamw(fp32_grad)
        else:
            raise ValueError(
                "非 AdamW 优化器必须提供 state_bytes_per_param"
            )
        if optimizer_values:
            raise ValueError(
                f"未知优化器配置项: {', '.join(sorted(optimizer_values))}"
            )

        hardware_values = _section(config, "hardware", "context")
        if "max_device_memory" not in hardware_values:
            raise ValueError("hardware.max_device_memory 是必填项")
        hardware = HardwareSpec(
            max_device_memory=parse_bytes(hardware_values.pop("max_device_memory")),
            framework_reserve=parse_bytes(hardware_values.pop("framework_reserve", 0)),
            usable_device_memory=(
                parse_bytes(hardware_values.pop("usable_device_memory"))
                if "usable_device_memory" in hardware_values else None
            ),
            device_baseline_bytes=parse_bytes(hardware_values.pop("device_baseline", 0)),
            allocator_pool_point_bytes=parse_bytes(hardware_values.pop("allocator_pool_point", 0)),
            allocator_pool_slack_point_bytes=parse_bytes(hardware_values.pop("allocator_pool_slack_point", 0)),
            untracked_runtime_point_bytes=parse_bytes(hardware_values.pop("untracked_runtime_point", 0)),
            allocator_granularity_bytes=parse_bytes(hardware_values.pop("allocator_granularity", 512)),
            calibrated_upper_margin_bytes=parse_bytes(hardware_values.pop("calibrated_upper_margin", 0)),
            ood_margin_bytes=parse_bytes(hardware_values.pop("ood_margin", 0)),
            hardware_profile=str(hardware_values.pop("hardware_profile", "unknown")),
            runtime_profile=str(hardware_values.pop("runtime_profile", "unknown")),
            source_profile=str(hardware_values.pop("source_profile", "unknown")),
        )
        if hardware_values:
            raise ValueError(
                f"未知硬件配置项: {', '.join(sorted(hardware_values))}"
            )

        recompute = RecomputeSpec(**_section(config, "recompute"))
        swap = SwapSpec(**_section(config, "swap"))
        return EvaluationInputs(
            model_spec,
            parallel,
            optimizer,
            hardware,
            recompute,
            swap,
        )

    @classmethod
    def load(cls, path) -> EvaluationInputs:
        config = load_data_file(path)
        if not isinstance(config, Mapping):
            raise ValueError("配置根节点必须是映射")
        return cls.from_dict(config)


def load_data_file(path) -> Any:
    """读取 JSON 或本项目支持的 YAML 子集。"""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _simple_yaml_load(text)
    raise ValueError("配置文件扩展名必须是 .json、.yaml 或 .yml")
