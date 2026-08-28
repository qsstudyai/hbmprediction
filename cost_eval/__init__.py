"""离线并行训练峰值显存评估器。"""

from .config_adapter import ConfigAdapter, EvaluationInputs
from .model_spec import DimTable, LayerSpec, ModelSpec, OpSpec, OpType, TensorRef
from .report import Evaluator, ModelSummary, PeakMemoryReport, RankPeak
from .specs import (
    HardwareSpec,
    OptimizerSpec,
    ParallelConfig,
    RecomputeSpec,
    SwapSpec,
)
from .trace import MemoryTrace, MemoryTraceEvent, TraceActivation, TraceTensor

__all__ = [
    "ConfigAdapter",
    "DimTable",
    "EvaluationInputs",
    "Evaluator",
    "HardwareSpec",
    "LayerSpec",
    "ModelSpec",
    "ModelSummary",
    "MemoryTrace",
    "MemoryTraceEvent",
    "OpSpec",
    "OpType",
    "OptimizerSpec",
    "ParallelConfig",
    "PeakMemoryReport",
    "RankPeak",
    "RecomputeSpec",
    "SwapSpec",
    "TensorRef",
    "TraceActivation",
    "TraceTensor",
]
