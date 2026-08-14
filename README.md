# HBM Prediction

面向 MindFormers PyNative 并行训练的离线峰值显存预测器。

给定模型结构、并行策略、优化器、重计算、Swap 和设备显存限制，预测器输出：

- 每个 Pipeline stage 的峰值显存及发生事件；
- 参数、梯度、优化器状态、激活、workspace、FSDP gather 和 Swap buffer 拆解；
- 最紧张的 stage 和 OOM 判断；
- 配置适配过程发现的未精确建模项。

## 支持范围

- Dense、MoE、DeepSeek MLA/shared expert/MTP；
- DP/HSDP、TP、PP、CP、EP 和 sequence parallel；
- MindFormers 1F1B 与 interleave stage placement；
- FSDP reshard/prefetch、CPU offload；
- full/select/exclude recompute、通信重计算；
- layer/op 级 activation swap；
- JSON/YAML 通用配置和 MindFormers PyNative YAML 适配。

## 安装和运行

```bash
python -m pip install -e '.[test]'
python -m cost_eval examples/configs/dense.yaml
```

CLI 输出 JSON，其中 `per_stage[].peak_bytes` 是各 stage 峰值，`oom` 是最终
OOM 判断。也可以直接使用 Python API：

```python
from cost_eval import ConfigAdapter

inputs = ConfigAdapter.load("examples/configs/dense.yaml")
report = inputs.evaluator().evaluate()
print(report.per_stage[0].peak_bytes, report.oom)
```

MindFormers 配置使用独立适配器：

```python
from cost_eval.adapters import MindFormersAdapter

inputs = MindFormersAdapter.from_yaml("pynative_model.yaml", world_size=8)
report = inputs.evaluator().evaluate()
```

## 验证

```bash
python -m pytest
python run_p0_tests.py
python run_functional_tests.py
python examples/memory_report.py
```

`calibration.py` 可用一组实测峰值显存检查预测误差：

```bash
python -m cost_eval.calibration profile.json -o calibration-report.json
```

## 代码结构

- `cost_eval/model_spec.py`：声明式算子图与 tensor 内存契约；
- `cost_eval/layers/`：Dense、MoE、DeepSeek 图构建；
- `cost_eval/shape_eval.py`：并行切分后的本地 shape 和通信生命周期；
- `cost_eval/static_mem.py`：参数、梯度和优化器静态显存；
- `cost_eval/mem_timeline.py`：前反向事件驱动峰值仿真；
- `cost_eval/report.py`：统一评估入口与结果；
- `cost_eval/config_adapter.py`：通用配置适配；
- `cost_eval/adapters/mindformers.py`：MindFormers PyNative 适配；
- `cost_eval/calibration.py`：P0 显存误差校准报告。

原始设计说明保留在 `specs/`，P0 实施计划保留在 `plans/`。其中总体架构文档
可能提到后续性能建模；这些后续模块不属于本项目实现范围。

面向 DeepSeek V3/V4、Qwen3 PyNative 动态图、真实 NPU 泛化验证以及
DP/HSDP、重计算和 CPU offload 的生产化方案见
[`specs/2026-08-14-hbm-prediction-production-design.md`](specs/2026-08-14-hbm-prediction-production-design.md)。
