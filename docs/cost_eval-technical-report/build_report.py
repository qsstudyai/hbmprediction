#!/usr/bin/env python3
"""Build the canonical portable artifact for the cost_eval technical report.

The narrative is intentionally hand-authored from the implementation.  The
module and API inventories are extracted from the current Python source tree so
the report cannot silently omit a newly added class or function.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parents[1]
PACKAGE = ROOT / "cost_eval"

CATEGORIES = {
    "包入口": "入口与公开 API",
    "MindFormers 适配": "MindFormers 配置适配",
    "核心数据模型与配置": "配置与核心数据模型",
    "模型与算子图": "模型与算子图",
    "形状、静态态与时间线": "形状、静态显存与时间线",
    "报告、测量与校准": "报告、测量与校准",
    "运行时源码合同": "运行时源码合同",
    "运行时校准与溯源": "运行时校准与溯源",
}

MODULES = [
    ("cost_eval/__init__.py", "包入口", "公开 API 重导出"),
    ("cost_eval/__main__.py", "包入口", "命令行评估入口"),
    ("cost_eval/adapters/__init__.py", "MindFormers 适配", "适配器公开 API"),
    ("cost_eval/adapters/mindformers.py", "MindFormers 适配", "把 MindFormers TrainConfig 转为模型图、并行、优化器、硬件、重计算和 swap 合同"),
    ("cost_eval/allocator_model.py", "形状、静态态与时间线", "把物理存活峰值与 allocator、设备基线、运行时及安全裕量组合成总 HBM"),
    ("cost_eval/calibration.py", "报告、测量与校准", "预测与真实 profiling 的逐 stage、逐桶、OOM 和峰值事件对账"),
    ("cost_eval/config_adapter.py", "核心数据模型与配置", "通用 JSON/YAML 配置适配"),
    ("cost_eval/event_schedule.py", "形状、静态态与时间线", "生成 1F1B 和交错流水事件"),
    ("cost_eval/layers/__init__.py", "模型与算子图", "内置层图公开 API"),
    ("cost_eval/layers/deepseek.py", "模型与算子图", "DeepSeek-V3 MLA、MoE、共享专家和 MTP 图"),
    ("cost_eval/layers/deepseek_v4.py", "模型与算子图", "DeepSeek-V4 legacy MLA、hybrid CSA、mHC、hash router 和 MTP 图"),
    ("cost_eval/layers/dense.py", "模型与算子图", "Dense GQA、FlashAttention、SwiGLU 与通用 Hyper-Connections 图"),
    ("cost_eval/layers/loss.py", "模型与算子图", "融合与回退交叉熵的 logits、保存张量和反向工作区"),
    ("cost_eval/layers/moe.py", "模型与算子图", "通用 attention + pure-EP routed experts 图"),
    ("cost_eval/layers/moe_dispatch.py", "模型与算子图", "MoE 路由、permute 和 inverse-map 辅助张量"),
    ("cost_eval/layers/qwen3.py", "模型与算子图", "Qwen3 GQA、Q/K RMSNorm、RoPE、FlashAttention 和 SwiGLU 图"),
    ("cost_eval/measurement_contract.py", "报告、测量与校准", "真实 NPU HBM 的互斥分量、逐 rank 样本和质量合同"),
    ("cost_eval/mem_timeline.py", "形状、静态态与时间线", "沿逐 stage 1F1B 事件追踪七个物理显存桶并捕获同时峰值"),
    ("cost_eval/memory_actions.py", "形状、静态态与时间线", "把单层前向解析为显式 ALLOC/FREE/通信工作区动作"),
    ("cost_eval/memory_ledger.py", "形状、静态态与时间线", "执行分配/释放账本，检查别名、泄漏和桶守恒"),
    ("cost_eval/model_spec.py", "核心数据模型与配置", "声明符号维度、张量 placement、算子和模型层型"),
    ("cost_eval/offload_model.py", "形状、静态态与时间线", "把参数、梯度、master weight 和优化器状态拆到 HBM 或 host pinned"),
    ("cost_eval/optimizers.py", "形状、静态态与时间线", "AdamW/Muon 持久状态字节和 optimizer step 临时工作区"),
    ("cost_eval/parallel_model.py", "核心数据模型与配置", "校验并行 mesh，计算 FSDP/eFSDP 度数和 PP 层分配"),
    ("cost_eval/report.py", "报告、测量与校准", "统一评估门面、OOM 状态、逐 stage/逐 rank 输出"),
    ("cost_eval/runtime_profiles.py", "运行时校准与溯源", "按模型家族和并行布局加载 allocator/runtime 校准档案"),
    ("cost_eval/shape_eval.py", "形状、静态态与时间线", "求值符号 shape、代入切分、检测通信并解析成每卡字节图"),
    ("cost_eval/source_contracts/__init__.py", "运行时源码合同", "源码合同公开 API"),
    ("cost_eval/source_contracts/deepseek_v3.py", "运行时源码合同", "DeepSeek-V3 运行时来源、参数公式与并行依据"),
    ("cost_eval/source_contracts/deepseek_v4.py", "运行时源码合同", "DeepSeek-V4 固定 commit/hash、变体公式和覆盖校验"),
    ("cost_eval/source_contracts/qwen3.py", "运行时源码合同", "Qwen3 固定 commit、数据类型和 saved-tensor 合同"),
    ("cost_eval/source_fingerprint.py", "运行时校准与溯源", "生成稳定的 Python 源码树 SHA-256"),
    ("cost_eval/specs.py", "核心数据模型与配置", "并行、优化器、硬件、重计算和 swap 配置合同"),
    ("cost_eval/static_mem.py", "形状、静态态与时间线", "计算每个 PP stage 的参数、梯度、master weight 和优化器持久态"),
    ("cost_eval/validation_gate.py", "报告、测量与校准", "P0/P1 数值门、覆盖率门和生产证据缺口"),
    ("cost_eval/workspace_registry.py", "运行时校准与溯源", "按算子、形状、版本和硬件精确匹配 kernel workspace 档案"),
]

CLASS_DESCRIPTIONS = {
    "MindFormersInputs": "不可变输入包，保存 MindFormers 适配后的模型图、并行、优化器、硬件、重计算、swap、告警和来源。",
    "MindFormersAdapter": "无须导入 MindSpore 的适配门面，把 TrainConfig YAML/字典转换为可执行的离线评估输入。",
    "AllocatorEstimate": "总 HBM 组合结果，分别保留物理存活峰值、allocator 池、基线、运行时、碎片、点估计和安全上界。",
    "AllocatorModel": "allocator/runtime 组合器；使用可审计的分桶公式，不使用全局拟合倍率。",
    "MetricComparison": "一对预测值和实测值的有符号误差、绝对误差与相对误差。",
    "StageCalibration": "一个 stage 的峰值、事件、分桶误差对账结果。",
    "CaseCalibration": "一个 profiling case 的逐 stage 对账、OOM 混淆结果和元数据。",
    "CalibrationSummary": "多个 case 的均值/P50/P90/最大误差、事件命中率和 OOM 混淆矩阵汇总。",
    "CalibrationReport": "完整校准报告，由逐 case 明细和总体摘要组成。",
    "CalibrationRunner": "执行预测与实测对账、聚合误差统计并读取批量 profiling 配置。",
    "EvaluationInputs": "通用配置适配后的六类评估输入，并能直接构造 Evaluator。",
    "ConfigAdapter": "通用 JSON/YAML 配置适配器，适合项目自有而非 MindFormers 原生格式。",
    "Event": "rank-local 流水事件；记录前向/反向、microbatch、层和 virtual chunk。",
    "MoEDispatchTensors": "MoE dispatcher 必须保存的 top-k、计数、偏移、置换和逆置换张量集合。",
    "HBMComponents": "同一采集窗口内互斥的设备基线、模型 active、allocator pool 和未追踪运行时分量。",
    "RankHBMMeasurement": "单个 rank 的 stage、空闲基线、峰值、增量和采样数量。",
    "HBMMeasurement": "一个真实 NPU case 的逐 stage/逐 rank HBM、组件、OOM 与测量质量。",
    "Buckets": "时间线当前时刻的七个物理显存桶，可原地增减。",
    "MemBreakdown": "峰值时刻的只读显存快照；七个物理桶外加 framework/运行时综合开销。",
    "StagePeak": "一个 stage 的峰值字节、峰值事件、分桶、点估计、安全上界和 OOM 状态。",
    "BucketPeaks": "各物理桶在整个时间线上的独立最大值；不能相加当成同时峰值。",
    "LayerMemoryPlan": "一层保存张量在 resident、swap、recompute scratch 和通信重算之间的互斥划分。",
    "MemTimeline": "核心逐事件仿真器，沿每个 stage 的交错 1F1B 时间线寻找同时存活峰值。",
    "MemoryAction": "一条带所有者、字节、桶和别名信息的分配/释放/迁移动作。",
    "LedgerPeak": "MemoryLedger 回放期间观察到的局部最高存活字节及分桶快照。",
    "MemoryLedger": "严格的存活值账本，拒绝重复分配、非法释放、悬空别名和未授权泄漏。",
    "OpType": "评估器识别的算子类别枚举，用于通信、workspace 和模型能力分支。",
    "DimTable": "模型全局维度表，并派生 routed-token 容量与 decoder+MTP 总层数。",
    "TensorRef": "符号张量合同：shape、placement、dtype、权重/训练/换出/重算及显式存储别名。",
    "OpSpec": "声明式算子合同：输入、输出、参数、反向保存张量、workspace、模块路径和属性。",
    "LayerSpec": "一个层型的有序 OpSpec 序列。",
    "ModelSpec": "模型名称、维度、逐层类型模式、层型图和能力/来源元数据。",
    "OffloadPolicy": "参数、梯度和优化器三类状态的独立 CPU offload 开关。",
    "OptimizerStateBytes": "一个本地权重 shard 的参数、梯度、master weight 和优化器状态字节。",
    "ParallelModel": "并行 mesh 的已解析视图，提供 FSDP/eFSDP 度数、层到 stage、rank 到 stage 映射。",
    "ParallelModel._StageLayers": "把 stage 层列表同时暴露为 Mapping 和可调用对象的兼容包装器。",
    "PeakMemoryReport": "评估器最终报告：逐 stage/rank 峰值、最紧位置、OOM、能力状态、告警和指纹。",
    "ModelSummary": "模型名、总层数和 world size 的简要信息。",
    "RankPeak": "单 rank 继承其物理 stage 的峰值、上界、事件和 OOM 状态。",
    "Evaluator": "顶层评估门面，串联并行解析、形状解析、静态态、时间线和 allocator 组合。",
    "RuntimeProfile": "一个模型家族+并行布局的实测 runtime/pool/baseline 分位数档案。",
    "RuntimeProfileRegistry": "加载并选择 runtime profile；缺档时显式标 OOD 并扩大上界。",
    "Placement": "张量在各 mesh 轴上的分片位置以及 partial 状态。",
    "CommSpec": "一次 collective 的类型、字节量、通信轴、阶段和输出值身份。",
    "ResolvedTensor": "已代入真实整数和并行切分后的本地张量，可直接给出每卡字节。",
    "ResolvedOp": "已解析算子，包含本地输入/输出/参数/saves、collective 和前后向 workspace。",
    "ResolvedLayer": "带全局 layer_id 的已解析算子序列，并能汇总保存激活和 checkpoint。",
    "ResolvedGraph": "按 PP stage 组织的 decoder 层、edge 模块、参数和输出字节图。",
    "ShapeEval": "把声明式 ModelSpec 解析成每卡 ResolvedGraph。",
    "ParallelConfig": "并行与训练调度合同，校验 DP/CP/TP/PP/EP、SP、offload、prefetch 和 interleave。",
    "OptimizerSpec": "优化器持久字节合同，支持 AdamW 与 Muon/混合参数分区。",
    "HardwareSpec": "设备容量、可用容量、allocator、runtime、校准裕量和来源指纹。",
    "RecomputeSpec": "full/select 激活重计算及 communication recompute 的选择与排除规则。",
    "SwapSpec": "按层/算子选择保存激活换出及反向预取深度。",
    "StaticBreakdown": "持久态四桶：参数、梯度、master weight、优化器状态。",
    "StageStaticMemory": "单 stage 的 HBM 持久态、edge 参数、decoder 持久态和 host pinned 拆分。",
    "StaticMem": "按 stage、权重组和 FSDP 对齐规则计算持久态。",
    "GateResult": "一个验收门的名称、状态、观测值、阈值和证据范围。",
    "WorkspaceKey": "kernel workspace 的精确匹配键：家族、算子、融合、shape、并行、版本和硬件。",
    "WorkspaceProfile": "workspace 点估计/上界及样本数、来源和内容 hash。",
    "WorkspaceRegistry": "WorkspaceKey 到 WorkspaceProfile 的精确查找表。",
}

FUNCTION_DESCRIPTIONS = {
    "main": "解析命令行参数，运行相应评估或校准，并输出 JSON；阈值失败时返回非零退出码。",
    "MindFormersInputs.evaluator": "把已适配的全部输入和告警组装成 Evaluator。",
    "MindFormersAdapter.from_yaml": "读取 MindFormers YAML，并以文件绝对路径作为来源交给 from_mapping。",
    "MindFormersAdapter.from_mapping": "完成 world-size/批量/并行校验，识别模型家族，构图并生成所有评估合同。",
    "load_mindformers_yaml": "优先用 PyYAML 安全加载；没有依赖时退回到受限顶层 YAML 解析器。",
    "parse_memory_bytes": "把整数或 KB/MB/GB/TB 字符串按二进制单位换算为字节。",
    "_build_model_shape": "从 MindFormers model/training 字段解析 DimTable、dense/MoE 层模式和缺失能力告警。",
    "_is_deepseek_v4": "根据 model_type/architectures 名称识别 DeepSeek-V4。",
    "_is_qwen3": "根据 model_type/architectures 名称识别 Qwen3。",
    "_int_alias_default": "依次读取多个整数别名，均缺失时返回默认值。",
    "_moe_pattern": "按专家数、前置 dense 层、频率或逐层列表生成 dense/MoE 层模式。",
    "_build_optimizer": "按 MindFormers FP32 梯度累积合同构造 AdamW 或 Muon 的逐参数字节配置。",
    "_build_recompute": "把 full/select、排除算子和 communication recompute 模块选择转成 RecomputeSpec。",
    "_build_swap": "把 layer_swap/op_swap 和 prefetch 配置转成 SwapSpec，并拒绝越界预取。",
    "_parse_stage_layers": "解析显式 PP stage 层范围；auto 返回 None。",
    "_parse_layer_selection": "把整数、列表、映射、区间和逗号区间统一成合法 layer-id 集合。",
    "_parse_module_selection": "把模块名与层选择配置统一成 module -> layer-id 集合。",
    "_load_yaml_subset": "无 PyYAML 时只读取评估器需要的顶层 section 和标量字段。",
    "_strip_comment": "删除引号外的 YAML # 注释。",
    "_parse_scalar": "把受限 YAML 标量解析为 null、布尔、字面量、数值、列表或字符串。",
    "_section": "读取并校验一个配置 section 必须是映射。",
    "_required_alias": "从候选字段中返回第一个存在的必填整数，否则报错。",
    "_positive_int": "转为整数并要求至少为 1。",
    "_optional_int": "把可空值转为 int 或 None。",
    "_dtype_bytes": "把 fp16/bf16/fp32 名称映射为 2/4 字节。",
    "_round_up": "把非零字节向上对齐到 allocator 粒度。",
    "AllocatorModel.estimate": "逐桶计算对齐碎片，再组合 pool、设备基线、运行时、校准和 OOD 裕量。",
    "MetricComparison.compare": "计算 predicted-measured、有符号/绝对误差以及以 measured 为分母的相对误差。",
    "_percentile": "对排序样本做线性插值分位数。",
    "_event_phase": "把具体峰值事件归一成 forward/backward 或原始阶段名。",
    "_event_matches": "精确事件相同或前后向阶段相同即算匹配；无实测事件返回 None。",
    "_stage_order_match": "按峰值从高到低比较预测与实测的 PP stage 排序，stage id 用于稳定处理并列。",
    "_normalize_stage_mapping": "把任意可转整数的 stage key 规范为 int key。",
    "CalibrationRunner.__init__": "设置并校验目标相对误差阈值。",
    "CalibrationRunner.run_case": "运行一个预测，逐 stage 对齐峰值/事件/桶并判断 OOM 与 stage 排序是否匹配。",
    "CalibrationRunner.run_cases": "聚合多 case 的误差分位数、事件命中率、OOM 混淆矩阵和建议 reserve。",
    "CalibrationRunner.run_profile": "读取批量 profiling 文件，选择 native/MindFormers adapter 并运行所有 case。",
    "parse_bytes": "把整数或十进制/二进制字节单位字符串转为字节。",
    "_parse_yaml_scalar": "解析通用配置 YAML 的布尔、空值、引号、列表、映射、数值和字符串标量。",
    "_simple_yaml_load": "解析项目支持的受限 YAML：嵌套映射、列表和标量。",
    "_simple_yaml_load.parse_block": "递归解析同一缩进层级的 YAML 映射或列表块。",
    "EvaluationInputs.evaluator": "把通用适配结果组装成 Evaluator。",
    "ConfigAdapter.from_dict": "校验通用配置，构造 dense/MoE 模型图以及并行、优化器、硬件、重计算、swap。",
    "ConfigAdapter.load": "读取 JSON/YAML 后要求根节点为映射，再调用 from_dict。",
    "load_data_file": "按扩展名读取 JSON 或受限 YAML。",
    "build_1f1b": "生成给定 PP stage 的 warmup 前向和随后一前一反的 rank-local 事件序列。",
    "build_interleaved_1f1b": "把每个 1F1B 事件按 virtual chunk 扩展；前向升序、反向逆序。",
    "_norm_weight": "创建一个 FP32、可训练的归一化权重 TensorRef。",
    "build_mla_attention": "构造 DeepSeek-V3 MLA 从 down projection、RoPE、KV up 到 FlashAttention 和输出投影的完整算子图。",
    "_moe_tail": "构造 DeepSeek 的 RMSNorm、router、dispatch、routed experts、可选 shared experts 和残差尾部。",
    "_dense_tail": "构造 DeepSeek dense SwiGLU FFN 尾部。",
    "build_mla_dense_decoder": "把 MLA attention、dense FFN 和可选 Hyper-Connections 组合成层图。",
    "build_mla_moe_decoder": "把 MLA attention、MoE/shared-expert 尾部和可选 Hyper-Connections 组合成层图。",
    "build_mtp_layer": "构造 embedding/上一隐藏态融合、一个完整 MLA-MoE 内层和最终归一化的 MTP 图。",
    "_with_hash_router": "给 router 注入每 token 到 expert 的 replicated int32 查表参数。",
    "build_v4_legacy_mla_moe_decoder": "构造显式选择 legacy MLA 的 V4 MoE 层，并按配置加入 mHC/hash router。",
    "_mhc_pre": "构造 V4 mHC 前置 FP32 mapping/Sinkhorn 参数、保存张量和 workspace。",
    "_mhc_post": "构造 mHC output-cell 混合，保存 streams/h_res/h_post/子层输出并计 FP32 workspace。",
    "_with_mhc": "用独立 attention mHC 和 FFN mHC 替换普通两个 residual add。",
    "_compressor_op": "按压缩比 4/128 构造 V4 KV compressor 参数、输出和保存张量。",
    "build_v4_hybrid_attention": "构造 V4 q/kv 低秩投影、可选 compressor/indexer、稀疏注意力和 grouped low-rank 输出。",
    "build_v4_hybrid_moe_decoder": "组合 V4 hybrid attention、MoE、mHC 和可选 hash router。",
    "_mtp_outer": "构造 V4 MTP 外层的 embedding/hidden 归一化、拼接和 2H->H 投影。",
    "build_v4_hybrid_mtp_layer": "构造一个不启用 hash router 的 hybrid V4 MTP 层，含 mHC expand/collapse。",
    "build_v4_legacy_mtp_layer": "构造 legacy MLA V4 MTP 层，含 mHC expand/collapse。",
    "hyper_connection_ops": "若启用，增加 H->hc_mult*H 投影和多流混合的参数、激活与保存张量。",
    "build_dense_decoder": "构造通用 GQA/FlashAttention/SwiGLU dense decoder 的张量、参数、saves 和 workspace。",
    "build_loss_ops": "按 fused/fallback 交叉熵和 loss parallel 设置构造 logits、FP32 中间量、saves 与反向工作区。",
    "build_moe_decoder": "构造通用 attention + router/dispatch/pure-EP expert/combine MoE 层。",
    "build_dispatch_tensors": "创建 FP32 top-k scores 和 int32 indices/counts/offsets/permute/inverse-map 张量合同。",
    "build_qwen3_decoder": "构造 Qwen3 Q/K RMSNorm、RoPE 辅助量、FlashAttention 和 SwiGLU 的精确 saved-tensor 图。",
    "_int_mapping": "把测量映射的 key/value 转为非负整数。",
    "HBMComponents.dynamic_total_bytes": "返回 max(model active, allocator pool) + untracked runtime。",
    "HBMComponents.nominal_total_bytes": "返回 device baseline + dynamic total。",
    "HBMComponents.validate_dynamic_total": "检查互斥组件合成值与实测动态峰值在容差内守恒。",
    "HBMMeasurement.from_case_result": "从 collector case_result 解析 stage/rank/组件/质量并立即检查分量守恒。",
    "HBMMeasurement.worst_dynamic_bytes": "返回最大的逐 stage 动态 HBM；无 stage 时为 0。",
    "HBMMeasurement.worst_total_bytes": "优先返回逐 rank 最大 peak_used；否则返回组件 nominal total。",
    "HBMMeasurement.eligibility_issues": "汇总质量状态、rank 数、采样数和缺失逐 rank 数据等不可用于校准的问题。",
    "HBMMeasurement.validate_stage_aggregation": "检查每个 stage 的 max(rank peak_delta) 与保存的 stage peak 一致。",
    "Buckets.total": "把当前七个物理桶相加。",
    "MemBreakdown.total": "把峰值快照的七个物理桶和 framework 综合开销相加。",
    "LayerMemoryPlan.resident": "返回前向结束后仍驻留在 HBM 的保存激活字节。",
    "LayerMemoryPlan.offloaded": "返回换出到 host、反向需预取的保存激活字节。",
    "LayerMemoryPlan.recomputed": "返回反向重算时同时物化的 scratch 字节。",
    "_unique_tensors": "按 storage_key 去重张量，防止共享权重或同一数据流值重复计费。",
    "_layer_memory_plan": "按 recompute/swap 选择把每层保存张量互斥划到 resident、offloaded、scratch 和通信重算。",
    "_layer_memory_plan.byte_sum": "把一组 storage key 对应的本地张量字节相加。",
    "_save_plan": "对外兼容包装，返回 _layer_memory_plan 的结果。",
    "_layer_fsdp_buffer_bytes": "汇总一层反分片/参数预取时需要的完整本地参数 buffer。",
    "_layer_grad_buffer_bytes": "汇总一层 reduce-scatter 前物化的完整梯度 buffer，尊重 FP32 main-grad。",
    "_layer_workspace": "返回层内单算子 workspace+最大 collective staging 的最大值。",
    "_op_fsdp_buffer_bytes": "计算一个 edge/普通算子的参数 all-gather buffer。",
    "_op_grad_buffer_bytes": "计算一个算子的完整梯度临时 buffer。",
    "_edge_saved_bytes": "按 storage_key 去重 embedding/final norm/lm head/loss 的非权重保存张量。",
    "MemTimeline.simulate": "逐 stage 回放交错 1F1B；在每次分配、gather、重算、换入、反向和 optimizer step 记录同时桶总和。",
    "MemoryAction.__post_init__": "校验动作类型和非负字节。",
    "forward_layer_actions": "按最后使用点生成单层前向输入/输出/saves、kernel workspace 和通信 staging 的分配释放动作。",
    "forward_layer_actions.ensure": "若非权重张量尚未存活则分配；保留值进 act_live，临时值进 workspace。",
    "MemoryLedger.__init__": "初始化空的真实分配、别名、分桶和局部峰值。",
    "MemoryLedger.live_bytes": "返回当前所有真实分配的字节和；alias 不重复计费。",
    "MemoryLedger.buckets": "返回当前分桶副本。",
    "MemoryLedger.live_keys": "返回当前真实分配 key 的只读集合。",
    "MemoryLedger._record": "若当前存活字节刷新最高值，则保存所有者和分桶快照。",
    "MemoryLedger.apply": "执行 ALLOC/FREE/ALIAS/MOVE，维护桶守恒并在动作后记录峰值。",
    "MemoryLedger.replay": "顺序执行动作并检查结束后除 allowed_live 外没有泄漏。",
    "DimTable.__post_init__": "要求核心维度和 dtype 为正、capacity_factor 为正，并拒绝负的 V4/HC 可选维度。",
    "DimTable.as_dict": "返回所有维度，并增加 T_routed=ceil(S*B*topk*capacity_factor)。",
    "DimTable.total_layers": "返回 decoder 层数加 MTP 层数。",
    "TensorRef.has_ep": "判断张量 placement 是否含 expert-parallel 分片轴。",
    "TensorRef.is_expert": "把 has_ep 暴露为专家张量属性。",
    "TensorRef.__post_init__": "复制 shard 映射以冻结外部修改，并检查每个分片维索引没有越界。",
    "ModelSpec.__post_init__": "要求 layer_pattern 覆盖 decoder+MTP 全部层、每个层型都有 LayerSpec，并冻结 capabilities 副本。",
    "ModelSpec.get_layer": "按层型名返回 LayerSpec。",
    "OffloadPolicy.from_parallel": "从 ParallelConfig 的三个 offload 开关构造策略。",
    "OffloadPolicy.split": "把 StaticBreakdown 拆成 HBM resident 与 host pinned 两份。",
    "state_bytes": "按本地权重元素数、dtype、trainable 和 AdamW/Muon 分区计算四类持久字节。",
    "optimizer_step_workspace_bytes": "AdamW 取最大梯度 chunk；Muon 取其 3 倍；optimizer offload 再加一个预取 chunk。",
    "ParallelModel.build": "用 DimTable.n_layers 构造 ParallelModel 的便捷入口。",
    "ParallelModel.__init__": "校验 world size/EP 整除和 PP 合法性，计算 eFSDP、层到 stage 映射，并检查 interleave 每 stage 层数足够。",
    "ParallelModel._StageLayers.__init__": "保存所属 ParallelModel，供 Mapping/可调用兼容接口转发。",
    "ParallelModel.degree": "返回 tp/cp/ep/dp/pp 度数；sp 仅在 sequence_parallel 时等于 tp。",
    "ParallelModel.fsdp_degree": "返回 dense FSDP shard 度数。",
    "ParallelModel.efsdp_degree": "返回 expert FSDP 度数；ep=1 时与 dense FSDP 一致。",
    "ParallelModel.stage_of": "查询 decoder layer 所属物理 PP stage。",
    "ParallelModel.stage_of_rank": "按 rank % pp 映射 rank 到物理 stage。",
    "ParallelModel._stage_layers": "返回指定 stage 的 layer-id 列表。",
    "ParallelModel._build_layer_to_stage": "解析显式层映射；否则按前部 stage 多分余数的策略自动均分。",
    "Evaluator.evaluate": "执行完整 HBM 流程，生成逐 stage/rank 点估计、上界、最紧事件与 OOM/能力状态。",
    "Evaluator.__init__": "保存六类输入，为缺省 recompute/swap 建空策略，并把旧 framework_reserve 映射告警加入报告。",
    "Evaluator._capability_status": "按源码合同、V4 probe/HBM 验证和硬件/runtime/source profile 决定 unsupported/experimental/validated。",
    "layout_key": "把 tp/cp/pp/ep/dp/interleave 编码为 runtime profile 布局键。",
    "RuntimeProfileRegistry.load": "读取 v1 JSON 档案并用文件内容 SHA-256 生成 profile id。",
    "RuntimeProfileRegistry.__init__": "按 (family, layout) 建立 profile 索引并保存可审计 profile id。",
    "RuntimeProfileRegistry.apply": "匹配家族+布局并写入点估计/上界；缺失时标 OOD 且至少增加 2 GiB。",
    "eval_expr": "用受限 AST 求值整数维度表达式，只允许已知符号和 +、-、*、整除。",
    "eval_expr.evaluate": "递归解释受限表达式 AST，并拒绝未知节点或非整除。",
    "Placement.__init__": "把 shard 映射排序并冻结为元组，同时保存 partial 轴。",
    "Placement.of": "从 TensorRef 的 shard/partial 构造 Placement。",
    "Placement.shards": "返回排序后的分片轴元组。",
    "CommSpec.kind": "兼容属性，返回 collective 类型。",
    "detect_reshards": "比较源/目标 placement：partial 先 reduce，去分片 all-gather，同轴换维 all-to-all，加分片本地完成。",
    "detect_reshard": "兼容旧调用方，只返回 detect_reshards 的第一条通信。",
    "ResolvedTensor.local_bytes": "返回 local_numel*dtype_bytes。",
    "ResolvedTensor.tid": "显示兼容标识，返回张量名；计费身份使用 storage_key。",
    "ResolvedTensor.storage_key": "按 storage_id、value_id、name 的优先级返回去重身份。",
    "_placement_value_id": "把名称、切分、partial 和 dtype 编码成稳定的数据流 value id。",
    "resolve_tensor": "代入全局 shape、CP/SP/TP/EP 切分和 dtype，生成 ResolvedTensor 的本地 shape/字节。",
    "ResolvedLayer.activation_bytes": "按 storage_key 去重并汇总该层所有反向保存激活。",
    "ResolvedLayer.checkpoint_bytes": "返回该层首个非权重输入的本地字节，供 PP 边界与 full recompute 使用。",
    "ShapeEval.__init__": "可选接收精确 kernel workspace registry。",
    "ShapeEval.resolve": "解析每层算子与 edge 模块，插入 placement/CP/EP 通信，计算 workspace，并按 PP stage 组图。",
    "deepseek_v4_attention_variant": "把 V4 experimental_attention_variant 归一化，缺省为 dsv4_hybrid。",
    "deepseek_v4_compress_ratios": "解析覆盖 decoder+MTP 的逐层压缩比；缺省 decoder=128、MTP=0。",
    "validate_deepseek_v4_contract": "拒绝未覆盖的 V4 变体、维度、比例、HC head、非融合 attention 和非法 hash/indexer 配置。",
    "python_source_tree_sha256": "按相对路径和文件内容稳定排序，计算整个 Python 源码树指纹。",
    "_normalize_op_map": "把 layer -> 算子名配置规范为 int -> frozenset[str]。",
    "ParallelConfig.__post_init__": "解析 reshard/interleave/cpu-offload 别名，校验所有并行度、CP 方法、dispatcher、FSDP 对齐和显式 PP 层配置。",
    "ParallelConfig.world_size": "返回 dp_replicate*dp_shard*cp*tp*pp；EP 是区域内重排，不额外乘 world size。",
    "ParallelConfig.fsdp_degree": "返回显式 dense_fsdp_shard_size 或 dp_shard*cp。",
    "ParallelConfig.expert_fsdp_degree": "ep=1 时等于 dense FSDP；否则为 dp_shard*cp*tp/ep。",
    "ParallelConfig.virtual_pipeline_size": "返回 interleave，即每物理 stage 的 virtual chunk 数。",
    "OptimizerSpec.name": "返回小写优化器类型。",
    "OptimizerSpec.__post_init__": "校验逐参数总字节和 Muon 参数，并从总量扣除参数/梯度/master 得到 optimizer-state 字节。",
    "OptimizerSpec.bytes_per_parameter": "返回配置的每参数总持久字节。",
    "OptimizerSpec.is_muon": "根据名称或 Newton-Schulz 步数识别 Muon 合同。",
    "OptimizerSpec.adamw": "构造 BF16/FP16 参数、可选 FP32 梯度的标准 AdamW 字节合同。",
    "HardwareSpec.__post_init__": "校验容量/运行时/裕量非负、allocator 粒度为正，并把可用容量规范到 (0,max]。",
    "RecomputeSpec.__post_init__": "规范 mode、逐层算子映射、全局/逐层排除和模块选择为不可变集合。",
    "RecomputeSpec.layers": "兼容属性，返回 full recompute layer 集合。",
    "RecomputeSpec.is_full": "判断指定层是否进入 full recompute。",
    "RecomputeSpec.recomputed_ops": "展开 full/select 模块组和别名，应用全局/逐层排除并返回精确算子集合。",
    "RecomputeSpec.recomputed_comm_ops": "把模块路径选择匹配到含 collective 的算子。",
    "SwapSpec.__post_init__": "解析 enable/layers/prefetch 兼容字段，规范逐层/逐模块选择，并把全局 op_names 合并到规则。",
    "SwapSpec.swaps": "判断整层 swap 是否覆盖指定层。",
    "SwapSpec.swapped_ops": "展开整层/算子/模块组 swap 选择并校验只含现有算子。",
    "StaticBreakdown.total": "返回参数+梯度+master weight+优化器状态。",
    "StageStaticMemory.total": "返回该 stage 的 HBM persistent_bytes。",
    "_unique_layer_params": "按 storage_key 去重一层全部算子参数。",
    "_state_bytes": "调用 optimizer state_bytes 并转换为 StaticBreakdown。",
    "_add_breakdown": "逐字段相加两个 StaticBreakdown。",
    "_sharded_group_state": "把同 dtype/训练属性的扁平参数组按 degree 和 alignment padding 后计算单 shard 持久态。",
    "StaticMem.compute": "逐 stage 汇总 decoder/edge 参数组，应用 FSDP/eFSDP、共享存储去重和 CPU offload。",
    "_metric_gate": "按 mean/worst APE 阈值生成 pass/fail/unavailable 数值门。",
    "assess_hbm_gates": "核对 offline/calibrated 标签一致性，评估 P0/P1 数值/覆盖率并显式保留生产证据缺口。",
    "render_gate_markdown": "把 GateResult 列表渲染为简洁 Markdown 表。",
    "WorkspaceProfile.__post_init__": "校验 point<=upper、正样本数和 64 位内容 SHA-256。",
    "WorkspaceRegistry.__init__": "把 profile 列表索引为完整 WorkspaceKey 到 profile 的精确映射。",
    "WorkspaceRegistry.lookup": "按完整 WorkspaceKey 精确查找；不做近似或最近邻匹配。",
    "HBMComponents.__post_init__": "要求四个互斥 HBM 分量全部非负。",
    "RankHBMMeasurement.__post_init__": "校验 rank/stage/sample 和字节非负，并要求 peak_used-baseline 与 peak_delta 在 1 MiB 内守恒。",
}


def source_sha256() -> str:
    digest = hashlib.sha256()
    for path in sorted(PACKAGE.rglob("*.py")):
        digest.update(path.relative_to(PACKAGE).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def signature(node: ast.AST, qualname: str) -> str:
    if isinstance(node, ast.ClassDef):
        fields = [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        ]
        return "字段: " + ", ".join(fields) if fields else "服务类/枚举"
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    result = f"{qualname.split('.')[-1]}({ast.unparse(node.args)})"
    if node.returns is not None:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def symbol_kind(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        return "类"
    decorators = {ast.unparse(item) for item in node.decorator_list}
    if "property" in decorators:
        return "属性"
    if any(item.endswith("classmethod") for item in decorators):
        return "类方法"
    return "方法" if "." in getattr(node, "_qualname", "") else "函数"


def fallback_description(node: ast.AST, qualname: str, module_path: str) -> str:
    name = qualname.split(".")[-1]
    owner = qualname.rsplit(".", 1)[0] if "." in qualname else "本模块"
    if isinstance(node, ast.ClassDef):
        return CLASS_DESCRIPTIONS.get(node.name, f"{owner} 中用于 {dict((p, r) for p, _, r in MODULES)[module_path]} 的类型。")
    if name == "__post_init__":
        return f"对象创建后校验并规范化 {owner} 字段；非法配置直接报错。"
    if name == "__init__":
        return f"初始化 {owner} 的内部状态并执行构造期校验。"
    protocol = {
        "__getitem__": "按 stage 读取层列表。",
        "__iter__": "按 stage id 顺序迭代。",
        "__len__": "返回物理 PP stage 数。",
        "__call__": "以可调用形式返回指定 stage 的层列表。",
        "__int__": "把对象转换为 persistent_bytes。",
        "__eq__": "支持与整数或同类对象比较。",
        "__floordiv__": "兼容旧接口，对 decoder persistent bytes 做整除。",
    }
    if name in protocol:
        return protocol[name]
    if name.startswith("build_"):
        return f"构造 {name.removeprefix('build_').replace('_', ' ')} 对应的声明式内存图或调度。"
    if name.startswith("parse_") or name.startswith("_parse_"):
        return f"解析并校验 {name.removeprefix('_parse_').removeprefix('parse_').replace('_', ' ')} 配置。"
    if name.startswith("validate_"):
        return f"校验 {name.removeprefix('validate_').replace('_', ' ')} 的完整性与守恒条件。"
    if name.startswith("_is_"):
        return f"判断输入是否满足 {name.removeprefix('_is_').replace('_', ' ')} 条件。"
    if name.startswith("_"):
        return f"{module_path} 的内部辅助逻辑：执行 {name.strip('_').replace('_', ' ')}。"
    return f"{owner} 的 {name.replace('_', ' ')} 操作；输入输出见签名。"


def collect_symbols() -> list[dict]:
    module_meta = {path: (category, responsibility) for path, category, responsibility in MODULES}
    rows: list[dict] = []

    def walk(body, parents, module_path):
        for node in body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            qualname = ".".join((*parents, node.name))
            node._qualname = qualname
            if isinstance(node, ast.ClassDef):
                description = CLASS_DESCRIPTIONS.get(
                    node.name,
                    fallback_description(node, qualname, module_path),
                )
            else:
                description = FUNCTION_DESCRIPTIONS.get(
                    qualname,
                    FUNCTION_DESCRIPTIONS.get(node.name, fallback_description(node, qualname, module_path)),
                )
            category, _ = module_meta[module_path]
            rows.append({
                "file": module_path,
                "line": node.lineno,
                "category": CATEGORIES[category],
                "kind": symbol_kind(node),
                "symbol": qualname,
                "signature": signature(node, qualname),
                "function": description,
            })
            walk(node.body, (*parents, node.name), module_path)

    for module_path, _, _ in MODULES:
        path = ROOT / module_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        walk(tree.body, (), module_path)
    return rows


def markdown_block(block_id: str, body: str, source_id: str | None = None) -> dict:
    result = {"id": block_id, "type": "markdown", "body": body.strip()}
    if source_id:
        result["sourceId"] = source_id
    return result


def sql_literal(value) -> str:
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _markdown_cell(value, format_name: str | None = None) -> str:
    """Render one artifact value safely inside a Markdown table cell."""
    if value is None:
        return ""
    if format_name == "percent" and isinstance(value, (int, float)):
        text = f"{value:.1%}"
    elif format_name == "number" and isinstance(value, int):
        text = f"{value:,}"
    else:
        text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(title: str, subtitle: str, columns: list[dict], rows: list[dict]) -> str:
    """Convert one canonical artifact table into portable Markdown."""
    lines = [f"### {title}"]
    if subtitle:
        lines.extend(["", f"_{subtitle}_"])
    labels = [_markdown_cell(column["label"]) for column in columns]
    lines.extend([
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ])
    for row in rows:
        values = [
            _markdown_cell(row.get(column["field"]), column.get("format"))
            for column in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def render_markdown_report(artifact: dict) -> str:
    """Render the canonical report artifact as one self-contained Markdown file."""
    manifest = artifact["manifest"]
    datasets = artifact["snapshot"]["datasets"]
    cards_by_id = {card["id"]: card for card in manifest.get("cards", [])}
    charts_by_id = {chart["id"]: chart for chart in manifest.get("charts", [])}
    tables_by_id = {table["id"]: table for table in manifest.get("tables", [])}
    parts: list[str] = []

    for block in manifest["blocks"]:
        block_type = block["type"]
        if block_type == "markdown":
            parts.append(block["body"].strip())
            continue

        if block_type == "metric-strip":
            columns = []
            values = {}
            for card_id in block["cardIds"]:
                card = cards_by_id[card_id]
                row = datasets[card["dataset"]][0]
                for metric in card["metrics"]:
                    field = metric["field"]
                    columns.append({
                        "field": field,
                        "label": metric["label"],
                        "format": metric.get("format"),
                    })
                    values[field] = row[field]
            parts.append(_markdown_table("报告范围", "当前 cost_eval 工作区快照。", columns, [values]))
            continue

        if block_type == "chart":
            chart = charts_by_id[block["chartId"]]
            # Markdown has no portable chart runtime, so preserve the exact chart
            # evidence as a compact lookup table rather than a raster screenshot.
            columns = [
                {"field": "category", "label": "职责域"},
                {"field": "lines", "label": "源码行", "format": "number"},
                {"field": "files", "label": "文件数", "format": "number"},
                {"field": "callables", "label": "类/函数数", "format": "number"},
                {"field": "share", "label": "行数占比", "format": "percent"},
            ]
            parts.append(_markdown_table(
                f"{chart['title']}（表格版）",
                chart.get("subtitle", ""),
                columns,
                datasets[chart["dataset"]],
            ))
            continue

        if block_type == "table":
            table = tables_by_id[block["tableId"]]
            parts.append(_markdown_table(
                table["title"],
                table.get("subtitle", ""),
                table["columns"],
                datasets[table["dataset"]],
            ))
            continue

        raise RuntimeError(f"unsupported Markdown block type: {block_type}")

    scope = datasets["scope"][0]
    parts.append(
        "<!-- "
        f"generated_at={manifest['generatedAt']}; "
        f"cost_eval_source_sha256={scope['source_sha256']}"
        " -->"
    )
    return "\n\n".join(parts).rstrip() + "\n"


def main() -> None:
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    module_rows = []
    for module_path, category, responsibility in MODULES:
        line_count = len((ROOT / module_path).read_text(encoding="utf-8").splitlines())
        module_rows.append({
            "file": module_path,
            "category": CATEGORIES[category],
            "lines": line_count,
            "responsibility": responsibility,
        })
    api_rows = collect_symbols()
    callables_by_file = Counter(row["file"] for row in api_rows)
    for row in module_rows:
        row["callables"] = callables_by_file[row["file"]
        ]

    category_rows = []
    total_lines = sum(row["lines"] for row in module_rows)
    for category in CATEGORIES.values():
        selected = [row for row in module_rows if row["category"] == category]
        category_rows.append({
            "category": category,
            "lines": sum(row["lines"] for row in selected),
            "files": len(selected),
            "callables": sum(row["callables"] for row in selected),
            "share": sum(row["lines"] for row in selected) / total_lines,
        })
    category_rows.sort(key=lambda row: row["lines"], reverse=True)
    for rank, row in enumerate(category_rows, 1):
        row["rank"] = rank

    api_datasets = {}
    category_ids = {}
    for index, category in enumerate(CATEGORIES.values(), 1):
        dataset_id = f"api_{index}"
        category_ids[category] = dataset_id
        api_datasets[dataset_id] = [row for row in api_rows if row["category"] == category]

    inventory_sql = (REPORT_DIR / "source_inventory.sql").read_text(encoding="utf-8")
    api_sql_rows = ",\n    ".join(
        "(" + ", ".join(sql_literal(row[field]) for field in (
            "file", "line", "category", "kind", "symbol", "signature", "function"
        )) + ")"
        for row in api_rows
    )
    api_inventory_sql = (
        "WITH api_inventory(file, line, category, kind, symbol, signature, function) AS (\n"
        "  VALUES\n    " + api_sql_rows + "\n)\n"
        "SELECT file, line, category, kind, symbol, signature, function\n"
        "FROM api_inventory\nORDER BY file, line, symbol;\n"
    )
    api_inventory_path = REPORT_DIR / "api_inventory.sql"
    api_inventory_path.write_text(api_inventory_sql, encoding="utf-8")
    scope_sql = (
        "SELECT "
        f"{len(module_rows)} AS files, {total_lines} AS lines, "
        f"{len(api_rows)} AS symbols, 7 AS buckets, "
        f"'{source_sha256()}' AS source_sha256;"
    )
    sources = [
        {
            "id": "cost_eval_tree",
            "label": "cost_eval Python 源码树",
            "path": "cost_eval",
        },
        {
            "id": "source_inventory",
            "label": "cost_eval 源码文件盘点",
            "path": "docs/cost_eval-technical-report/source_inventory.sql",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": inventory_sql,
                "description": "列出报告范围内的 Python 文件、分类、行数和职责。",
                "executed_at": generated_at,
                "metric_definitions": [
                    "源码行数按当前工作区文件的 splitlines() 计数。",
                    "callables 由 Python AST 递归统计类、方法、属性、函数和内部函数定义。",
                ],
            },
        },
        {
            "id": "api_inventory",
            "label": "cost_eval AST API 盘点",
            "path": "docs/cost_eval-technical-report/api_inventory.sql",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": api_inventory_sql,
                "description": "由 build_report.py 递归解析当前 Python AST 生成每个类、方法、属性、函数和内部函数。",
                "executed_at": generated_at,
                "metric_definitions": ["每个 AST ClassDef/FunctionDef/AsyncFunctionDef 计一行，包含嵌套定义。"],
            },
        },
        {
            "id": "scope_summary",
            "label": "报告范围汇总",
            "path": "docs/cost_eval-technical-report/build_report.py",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": scope_sql,
                "description": "汇总当前源码文件数、行数、AST 符号数、物理显存桶数和源码指纹。",
                "executed_at": generated_at,
                "metric_definitions": [
                    "files 为 MODULES 清单长度。",
                    "lines 为当前清单文件 splitlines() 之和。",
                    "symbols 为递归 AST 定义数。",
                    "buckets 为 MemTimeline 同时维护的物理显存桶数。",
                ],
            },
        },
    ]

    title = "cost_eval 峰值 HBM 计算技术报告"
    blocks = [
        markdown_block("title", f"# {title}"),
        markdown_block(
            "technical_summary",
            """
## 技术摘要

`cost_eval` 的核心不是“把模型各部分的最大显存相加”，而是先把模型写成张量与算子图，求出每张卡上的局部字节，再沿每个流水 stage 的 1F1B 时间线逐次执行分配与释放，找到**同一时刻**七个物理显存桶的最大总和。最后才加入 allocator 池、设备基线、未追踪运行时、碎片和安全裕量。

最终点估计可以压缩为：

`HBM_point = device_baseline + max(physical_active_peak, allocator_pool_peak) + untracked_runtime + fragmentation`

安全上界为：

`HBM_safe = HBM_point + calibrated_upper_margin + OOD_margin`

因此，参数、梯度、激活、FSDP gather、重计算、swap 和 kernel workspace 是否真正同时存在，比单个模块的孤立峰值更重要。当前源码覆盖通用 Dense/MoE、Qwen3、DeepSeek-V3 MLA/MTP、DeepSeek-V4 legacy/hybrid/mHC/hash-router，并保留明确的能力状态：未经 runtime profile 和真实 HBM 验证的配置最多只能报告 `risky`，不能报告 `definitely_safe`。
""",
            "cost_eval_tree",
        ),
        {"id": "scope_metrics", "type": "metric-strip", "cardIds": ["files_card", "lines_card", "symbols_card", "buckets_card"]},
        markdown_block(
            "scope_map",
            """
## 代码主体集中在形状解析、内存时间线和模型图

下图按职责汇总当前源码规模，只用于帮助定位阅读重点，**不代表代码质量或运行耗时**。形状/静态态/时间线是 HBM 算法主体；模型图决定“有哪些参数和反向保存张量”；适配器决定真实配置如何落到这些合同上。图后给出逐文件精确盘点。
""",
            "source_inventory",
        ),
        {"id": "category_chart_block", "type": "chart", "chartId": "category_chart", "layout": "full"},
        {"id": "module_table_block", "type": "table", "tableId": "module_table", "layout": "full"},
        markdown_block(
            "scope_definitions",
            """
## 范围和关键术语先统一

- **HBM**：设备显存。报告中的 host pinned memory 单独列出，不计入 HBM。
- **global shape / local shape**：前者是模型逻辑形状；后者已除以 CP/TP/EP/SP 等切分度数，是单 rank 真正持有的形状。
- **persistent**：跨整个训练 step 常驻的参数、梯度、master weight 和优化器状态。
- **saved activation**：前向为反向保存的张量；可按重计算或 swap 规则变成 checkpoint、recompute scratch 或 host-offloaded 数据。
- **workspace**：某个 kernel、collective、loss 或 optimizer step 在执行期间的临时内存。
- **physical active peak**：七个物理桶在一个真实模拟时刻的和；它不是各桶独立峰值之和。
- **point estimate / safe upper**：前者是最可能的总 HBM；后者再加校准分位数与 OOD 裕量，用于安全判断。

本报告审阅 `cost_eval/**/*.py` 的当前工作区快照，不把测试、工具脚本和 `validation/` 结果算进 API 范围。源码树指纹写入报告元数据，便于之后判断报告是否过期。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "pipeline_overview",
            """
## 最终 HBM 由九步流水计算出来

1. **配置适配**：`ConfigAdapter` 或 `MindFormersAdapter` 把配置转成 `DimTable`、`ModelSpec`、`ParallelConfig`、`OptimizerSpec`、`HardwareSpec`、`RecomputeSpec`、`SwapSpec`。
2. **声明模型图**：各 `layers/*.py` 用 `TensorRef` 和 `OpSpec` 写清参数、输出、反向保存张量、workspace 与模块来源。
3. **解析并行布局**：`ParallelModel` 校验 world size，计算 dense FSDP 与 expert eFSDP 度数，并把 decoder 层分给 PP stage。
4. **解析每卡张量**：`ShapeEval` 代入符号 shape 和 dtype，应用 CP/SP/TP/EP 切分，生成 `ResolvedTensor.local_bytes`，同时插入 placement、CP 和 EP collective。
5. **计算持久态**：`StaticMem` 对共享存储去重，把参数按 FSDP/eFSDP 和 flatten alignment 切分，计算参数、梯度、master weight、优化器状态及 CPU offload。
6. **规划保存激活**：`_layer_memory_plan` 把每层 saves 互斥划为 resident、offloaded、recompute scratch 和 communication recompute。
7. **回放 1F1B**：`MemTimeline` 按每个 stage 的 FWD/BWD/virtual chunk 顺序，逐动作维护七个桶，记录同时峰值与事件。
8. **组合 allocator/runtime**：`AllocatorModel` 加入 pool、基线、未追踪运行时、分桶粒度碎片及上界裕量。
9. **决定 OOM 和置信度**：`Evaluator` 与 `usable_device_memory` 比较，并结合模型合同、probe、HBM validation 与 profile 指纹输出 `predicted_oom`、`unsupported`、`risky` 或 `definitely_safe`。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "tensor_logic",
            """
## 第一步：一个张量怎样变成“每卡多少字节”

`eval_expr` 只允许整数、已知维度符号以及 `+ - * //`，并要求整除，避免任意代码执行或悄悄截断。`resolve_tensor` 先求 global shape，再按 placement 逐维切分：

- 标记为 `sp` 的第 0 维，在 CP>1 时先除以 CP；若启用 sequence parallel，再按 TP 除一次。
- 普通 `tp/cp/ep` shard 把对应维度除以 mesh degree；不能整除就直接失败。
- `alltoall_deredundency/zero_redundancy` dispatcher 会把 routed-token 第一维再按 TP 去冗余。
- 权重缺省使用 `param_dtype_bytes`，激活缺省使用 `dtype_bytes`，显式 FP32/int32 张量固定 4 字节。

最后：`local_bytes = product(local_shape) * dtype_bytes`。

去重不靠张量名猜测。`storage_key` 优先使用显式 `storage_id`，其次是 placement-aware `value_id`，最后才是名称。这样 tied embedding/output weight 在同一 stage 可共享，而不同 placement 的同名数据流不会误合并。

placement 改变也会产生显存：partial 先 `reduce_scatter` 或 `all_reduce`；去掉某个 shard 需要 `all_gather`；同一 mesh 轴换到另一张量维度需要 `all_to_all`；只增加 shard 可本地完成。Flash/Sparse Attention 的 CP 还会按 colossal、Ulysses 或 hybrid 追加 ring/all-to-all；MoE dispatch/combine 在 EP>1 时追加 all-to-all。collective 的最大通信量会在相应算子时刻进入 workspace。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "persistent_logic",
            """
## 第二步：参数、梯度和优化器持久态按真实 shard 计算

`StaticMem` 对每个 stage、每层执行以下逻辑：

1. 按 `storage_key` 去重权重。
2. 按 FSDP degree、dtype、是否可训练、是否专家、是否 replicated 分组。
3. 对每组先把元素数按 `degree * alignment_numel` 向上 padding，再除以 degree 得到单 rank shard。`alignment_numel = ceil(alignment_bytes / dtype_bytes)`。
4. 对单 shard 元素数 `N` 计算四类持久字节。

AdamW：

- parameter = `N * weight_dtype_bytes`
- gradient = `N * max(weight_dtype_bytes, optimizer.gradient_bytes)`；MindFormers adapter 固定 FP32 main-grad，即 4 字节。
- master weight = FP32 参数时为 0，否则 `N * master_weight_bytes`。
- optimizer state = `N * 8`，对应一阶和二阶 FP32 状态。

Muon：按 `muon_parameter_fraction` 把 N 分成 Muon 与 Adam 两部分；Muon 部分使用 main parameter + momentum，剩余部分仍使用 Adam 的 FP32 master + 8-byte states。Newton-Schulz 步数影响时间，不增加同时矩阵数量。

Dense 权重使用 `fsdp_degree = dense_fsdp_shard_size or dp_shard*cp`。专家权重在 EP>1 时使用 `efsdp_degree = dp_shard*cp*tp/ep`；EP=1 时仍跟普通 FSDP，不能错误地再除 TP。`fsdp_replicated=True` 的标量、查表等完全复制。

embedding、final norm、lm head 作为 edge ops 参与同一持久态计算；tied weight 用 storage id 去重。offload 只改变驻留位置：parameter/gradient/optimizer 可独立从 HBM 移到 host pinned，host 字节仍在 `StageStaticMemory` 单独报告。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "activation_plan",
            """
## 第三步：保存激活先做 resident、recompute 和 swap 的互斥划分

每个 `OpSpec.saves` 都表示反向真正需要的值。`_layer_memory_plan` 先按 `storage_key` 去重，再应用策略：

- 未重算算子的 saves 默认进入 **resident**。
- 每段连续重计算区域只保留第一个非权重输入作为 **checkpoint**；其余 saves 从前向 resident 中删除，字节计入反向 **recompute_scratch**。
- communication recompute 会删除被选 collective 的输出激活，反向前以 collective volume 计入 scratch。
- swap 只处理被选择、可换出的 saves。某张量若同时被未 swap 的算子需要，或它是重计算 checkpoint，就不能换出。
- 同一算子同时被 recompute 和 swap 选择时，recompute 优先，避免重复节省同一字节。

结果中的 resident/offloaded/scratch 是同一批 saves 的互斥视图。它们不能再相互相加为“原始激活总量”；只有放到具体时间线事件上才知道哪些会重叠。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "timeline_buckets",
            """
## 第四步：七个显存桶在 1F1B 时间线上同时记账

时间线维护七个物理桶：

| 桶 | 何时增加 | 何时释放 |
|---|---|---|
| `persistent` | stage 初始化时一次加入 | step 内不释放 |
| `act_live` | 前向保存 resident 激活、edge/loss saves | 对应 microbatch 反向结束后释放 |
| `gather_buf` | FSDP/eFSDP 参数 all-gather 与预取窗口 | `reshard_after_forward` 时算子/层后释放；否则持有到反向 |
| `grad_buf` | 反向形成完整梯度、reduce-scatter 之前 | 每层/edge 梯度处理后释放 |
| `recomp_scratch` | 反向激活或 collective 重算 | 重算完成后释放 |
| `swap_buf` | 反向按 `default_prefetch` 预取已换出 saves | 当前预取事件后释放 |
| `workspace` | kernel、collective、PP send/recv、loss、optimizer step | 对应操作后释放 |

`build_1f1b` 的 warmup 数为 `min(pp-1-stage, num_microbatches)`，之后一前一反；interleave>1 时，前向按 chunk 0→N-1，反向按 N-1→0 扩展。早期 stage 因 warmup 更长，通常会同时保留更多 microbatch 激活。

前向的精细存活由 `MemoryLedger` 管理。它为输入、输出和 saves 建立 ALLOC/FREE 动作，普通临时值在最后一次使用后释放，resident saves 留在 layer exit；kernel workspace 和最大 collective staging 显式分配，使早期保存张量可以与后续大 workspace 正确重叠。账本拒绝重复分配、非法释放、alias 引用未解除和意外泄漏。

FSDP gather 窗口包含当前层和 `prefetch_depth` 个未来层；反向则向相反方向取窗口。若 `reshard_after_forward=False`，已 gather 的层会累计保留。反向依次模拟 edge loss、参数 gather、swap 预取、激活/通信重算、逐算子 backward workspace、完整梯度 buffer，随后释放该 microbatch 的 resident saves。训练末尾再模拟 optimizer step workspace。

每个动作后执行 `record()`：`current = persistent + act_live + gather_buf + grad_buf + recomp_scratch + swap_buf + workspace`。只有刷新 current 最大值的那个事件才成为 `peak_event` 和最终 `MemBreakdown`。`BucketPeaks` 只是各桶各自历史最高值，不能相加。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "allocator_logic",
            """
## 第五步：物理峰值再组合成点估计、安全上界和 OOM 状态

`MemTimeline` 在 `Evaluator` 中以 framework reserve=0 运行，所以得到的是纯物理 active peak。`AllocatorModel` 随后处理：

1. 对峰值快照的七个物理桶分别向上对齐到 `allocator_granularity_bytes`，碎片为各桶 `round_up(bucket)-bucket` 的和。
2. `allocator_pool_peak = max(active, configured_pool_point, active + pool_slack_point)`。
3. runtime 优先使用 `untracked_runtime_point_bytes`；若为 0，才退回旧字段 `framework_reserve`。
4. `total_point = device_baseline + max(active, allocator_pool_peak) + runtime + fragmentation`。
5. `safe_upper = total_point + calibrated_upper_margin + ood_margin`。

最终报告把 `total_point-active` 统一放入 `breakdown.framework`，它实际包含设备基线、pool slack、未追踪 runtime 和分桶对齐碎片，不只是 Python 框架本身。

OOM 判定顺序很保守：点估计超过 `usable_device_memory` → `predicted_oom`；模型合同不支持 → `unsupported`；安全上界超过容量 → `risky`；能力未验证 → 仍是 `risky`；只有点估计和上界都不过线且能力状态为 validated 才是 `definitely_safe`。逐 rank 结果按 `rank % pp` 继承对应 stage，最紧 rank 取第一个最大值。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "dense_memory",
            """
## Dense 与 Qwen3：显存主要来自四组权重、保存激活和 logits

### 通用 Dense decoder

- QKV 权重：`H * (n_heads + 2*n_kv) * head_dim`，输出维按 TP 切。
- Attention output 权重：`(n_heads*head_dim) * H`，输入维按 TP 切，输出为 TP partial 后再 reduce-scatter/all-reduce。
- SwiGLU FC1：`H * 2F`；FC2：`F * H`。
- FlashAttention 前向保存 qkv、attn、LSE；公式 workspace 为 `S*B*n_heads*head_dim*dtype_bytes`。
- Norm/MatMul/SwiGLU 通过各自 `saves` 保留 x、LN 输出、gate、act 等；SP/CP/TP 会改变每卡字节。
- 通用 Hyper-Connections 若启用，额外增加 `H*(H*hc_mult)` 投影权重、`S*B*H*hc_mult` 投影激活和 source/projected saves。

### Qwen3 专用差异

Qwen3 在通用 GQA 上增加 FP32 input/post-attention RMSNorm 权重、FP32 Q/K head norm 权重、int32 position ids、FP32 RoPE cos/sin，以及 FP32 LSE。FlashAttention 保存的是 RoPE 后 qkv、attn 和 FP32 LSE；其后仍是 TP-sharded SwiGLU。也就是说，Qwen3 不能只套通用 Dense 的“权重总量”，因为额外 FP32 小参数和 RoPE/QK-Norm 保存张量会改变峰值拆解。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "moe_memory",
            """
## MoE：路由容量、专家权重、dispatcher 元数据和通信共同决定峰值

路由 token 容量为：`T_routed = ceil(S * B * topk * capacity_factor)`。

通用 MoE 层先复用 attention，再加入：

- router logits：`S*B*n_experts`，FP32；另保存 `topk_scores/topk_indices`、每专家 token count、offset。
- dispatched/expert_out：`T_routed*H`；expert gate：`T_routed*2*moe_F`；expert act：`T_routed*moe_F`。
- 专家权重：W1=`n_experts*H*2*moe_F`，W2=`n_experts*moe_F*H`；先按 EP 切专家维，静态态再按 eFSDP 规则处理。
- permute/inverse map：各 `T_routed` 个 int32；dispatch/combine 还各自增加一份 output-sized 本地 permute workspace，并在 EP>1 时叠加 all-to-all staging。
- 两个 expert GEMM 标记 `workspace_like_output`，因此工作区再包含相应 gate/expert_out 本地字节。

`alltoall_deredundency/zero_redundancy` 会让 dispatched、gate、act、expert_out 的 routed-token 维再按 TP 除一次。该规则只减本地 routed 激活，不改变全局 token 容量。

DeepSeek shared expert 是独立 dense 分支：FC1=`H*2*shared_F`，FC2=`shared_F*H`，可选 FP32 gate=`H*1`；它与 routed expert 输出同时存活到 merge，因此不能忽略。hash router 额外加入 replicated、non-trainable 的 int32 表 `vocab*topk`，但 learned router weight 仍存在。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "deepseek_memory",
            """
## DeepSeek-V3/V4：模型变体必须使用各自来源合同

### DeepSeek-V3 MLA 与 MTP

MLA 合并 down projection 权重为 `H*(q_lora_rank+kv_lora_rank+qk_rope_head_dim)`；Q up 为 `q_lora_rank*n_heads*(qk_nope_head_dim+qk_rope_head_dim)`；KV up 为 `kv_lora_rank*n_heads*(qk_nope_head_dim+v_head_dim)`；输出投影为 `n_heads*v_head_dim*H`。FlashAttention 保存 query_rope、key、value、attn、LSE，workspace=`S*B*n_heads*v_head_dim*dtype_bytes`。MTP 外层再加两个 FP32 norm、2H→H 投影、一个完整 MLA-MoE 内层和最终 norm，并固定放在最后 PP stage。

### DeepSeek-V4 hybrid CSA

V4 hybrid 不复用 V3 MLA 公式。它包含 Q down/up、Q head RMS、KV projection/norm；按 ratio 4/128 可加入 compressor；ratio=4 且非 dense mode 时再加入 indexer；attention 使用 sparse/sliding-window 合同；输出由 grouped low-rank 两段投影完成。关键参数公式包括：

- Q：`H*q_lora_rank + q_lora_rank*n_heads*v_head_dim`
- KV：`H*v_head_dim`
- grouped output：`(o_groups*o_lora_rank)*(n_heads*v_head_dim/o_groups) + (o_groups*o_lora_rank)*H`
- compressor：`2*H*(coff*head_dim) + ratio*(coff*head_dim) + head_dim`，ratio=4 时 coff=2，否则为 1
- attention sink：`n_heads` 个 FP32 可训练值；Q RMS gamma：`v_head_dim` 个 FP32 固定值

### V4 mHC

每层 attention 和 FFN 各有一个独立 mHC cell。pre mapping 是 FP32 `(hc_mult*H)*(2*hc_mult+hc_mult^2)`，另有三个 FP32 alpha、FP32 bias 和固定 FP32 RMS gamma；保存 packed streams、`h_res[S,B,n,n]`、`h_post[S,B,n,1]`。pre workspace=`S*B*(n^2+2n)*4`，post workspace=`S*B*n*H*4`。MTP 还需要 H→nH expand 和 mean collapse。

源码合同固定到具体 MindFormers commit 与文件 SHA-256。V4 只接受已覆盖的 ratio 0/4/128、fused hybrid attention、sliding_window=128 等；否则立即拒绝，不能退回 V3 近似。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "edge_loss_memory",
            """
## Edge 与 Loss：大词表 logits 往往是最后 stage 的关键峰值

stage 0 的 embedding 权重为 `vocab*H` 并按 TP 切；最后 stage 有 FP32 final norm、lm head 和 loss。lm head 权重同样是 `vocab*H` 按 TP 切；若 `tie_word_embeddings=True` 且处于可共享的同一 stage，storage id 防止重复计费。

logits 全局形状为 `S*B*vocab`。不开 loss parallel 时 logits 不按词表 TP 切，并在 lm head 后产生 all-gather staging；开启时词表维保留 TP shard，显著降低每卡 logits 与 loss 工作集。

- fused cross entropy：保存 logits、labels、mask、FP32 stats，前向 workspace=`S*B*2*4`；反向再预留一份最大输入大小。
- fallback cross entropy：先物化 FP32 logits，再物化同形状 FP32 probabilities；softmax 保存 probabilities/stats，reduce 保存 probabilities/labels/mask；反向按最大输入的 2 倍建 workspace，表示 probability gradient 与 dlogits 同时存在。

这些 edge ops 参与与 decoder 相同的 FSDP gather、前向 ledger 和反向 grad buffer。报告中的 `loss_logits`、`loss_bwd_dlogits` 峰值事件就是专门为识别大词表风险保留的标签。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "profiles_validation",
            """
## 校准与验证只修正运行时开销，不重写物理公式

`WorkspaceRegistry` 可按模型家族、算子类型、融合变体、dtype、本地 shape、TP/CP/EP、MindSpore/CANN 版本和硬件精确替换 kernel workspace 点值/上界；没有精确键就保留公式值，不做最近邻猜测。

`RuntimeProfileRegistry` 按 `family + tp/cp/pp/ep/dp/interleave` 选择档案，把实测 runtime、allocator pool slack 和 baseline 点值写入 HardwareSpec。上界由这些分量的 P90-点值差、物理 shortfall P90 和 singleton 2 GiB 裕量相加。完全缺档时不借用别的家族/布局，只标记 missing 并把 OOD margin 至少扩大 2 GiB。

真实测量合同要求分量互斥：`dynamic = max(model_active, allocator_pool) + runtime`，`nominal_total = baseline + dynamic`；逐 stage peak 应等于该 stage 所有 rank 的最大 peak_delta。质量状态、rank 数和每 rank 样本不足会使标签不适合校准。

`CalibrationRunner` 比较逐 stage 峰值、峰值事件阶段和可选分桶，汇总 mean/P50/P90/max relative error、signed bias、事件命中率和 OOM 混淆矩阵。`validation_gate` 把历史 replay 与生产验证严格分开：缺 blind holdout、真实 OOM、stage/rank 排序或策略 paired delta 证据时状态是 `unavailable`，绝不当作 pass。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "api_reference",
            """
## 全量 API 索引覆盖每个类、方法、属性、函数和内部函数

以下表由当前 `cost_eval/**/*.py` 的 Python AST 递归生成，并为每个符号附上职责说明。这样既覆盖公开 API，也覆盖 `_` 前缀辅助函数、property、classmethod、嵌套兼容类和内部函数。表按职责拆分，便于筛选；`line` 是定义起始行。
""",
            "cost_eval_tree",
        ),
    ]

    table_ids = []
    tables = [
        {
            "id": "module_table",
            "title": "逐文件职责与规模",
            "subtitle": "当前工作区快照；行数为物理源码行，callables 含类/方法/属性/函数。",
            "dataset": "modules",
            "sourceId": "source_inventory",
            "defaultSort": {"field": "file", "direction": "asc"},
            "columns": [
                {"field": "file", "label": "文件", "type": "text"},
                {"field": "category", "label": "职责域", "type": "text"},
                {"field": "lines", "label": "行数", "format": "number"},
                {"field": "callables", "label": "类/函数数", "format": "number"},
                {"field": "responsibility", "label": "主要功能", "type": "text"},
            ],
        }
    ]
    for index, category in enumerate(CATEGORIES.values(), 1):
        table_id = f"api_table_{index}"
        table_ids.append(table_id)
        tables.append({
            "id": table_id,
            "title": f"{category}：类与函数",
            "subtitle": "签名来自 AST；说明按当前实现语义编写。",
            "dataset": category_ids[category],
            "sourceId": "api_inventory",
            "defaultSort": {"field": "file", "direction": "asc"},
            "columns": [
                {"field": "file", "label": "文件", "type": "text"},
                {"field": "line", "label": "行", "format": "number"},
                {"field": "kind", "label": "类型", "type": "text"},
                {"field": "symbol", "label": "符号", "type": "text"},
                {"field": "signature", "label": "签名/字段", "type": "text"},
                {"field": "function", "label": "功能", "type": "text"},
            ],
        })
        blocks.append({"id": f"api_block_{index}", "type": "table", "tableId": table_id, "layout": "full"})

    blocks.extend([
        markdown_block(
            "limitations",
            """
## 当前实现的边界会直接影响结论可信度

1. **顶层 Evaluator 尚未注入 WorkspaceRegistry。** `ShapeEval` 支持 profile registry，但 `Evaluator.evaluate()` 当前调用的是 `ShapeEval()`，所以正常顶层路径仍使用算子公式 workspace；只有自定义解析路径显式传 registry 才会命中精确档案。
2. **调度只支持 1F1B。** `ParallelConfig` 会拒绝其他 pipeline schedule；interleave 是在 rank-local 1F1B 上按 virtual chunk 展开，不模拟算子耗时、通信延迟或跨 rank 亚事件同步。
3. **峰值是合同模型，不是 allocator trace 重放。** 碎片按峰值快照的七个桶分别做粒度对齐，不是逐次真实 malloc/free 的地址级碎片。
4. **workspace 公式可能随 kernel、融合、MindSpore/CANN 和硬件改变。** 精确 profile 缺失时使用公式 fallback；这就是 runtime/source fingerprint 和安全上界存在的原因。
5. **能力状态刻意保守。** DeepSeek-V4 在 probe 未通过或 HBM 未 validated 时直接 unsupported；Qwen3/V3 的历史 holdout 也不能自动升级为生产 validated。
6. **只计算单训练 step 的显存峰值。** 不计算执行时间、带宽瓶颈、host RAM 容量、NVMe swap 时延或长期内存泄漏。
7. **microbatch 字段要区分。** 时间线使用 `num_microbatches`；`microbatch` 配置字段本身不参与当前事件数计算。

这些限制不是可忽略的注脚：在新模型家族、新 kernel 版本、未知并行布局或接近容量边界时，应把结果视为诊断性上界，而不是部署承诺。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "next_steps",
            """
## 建议优先补齐三件事

1. 把 `WorkspaceRegistry` 正式接入 `Evaluator` 和两类 adapter，并在报告 fingerprints 中记录 workspace profile id。
2. 用新的冻结标签补齐每个声明支持的模型家族/布局：至少包括逐 rank 样本、真实 OOM、峰值 stage/rank 排序和策略 paired delta。
3. 给调度模型增加与真实 runtime trace 对齐的事件版本，特别验证 PP interleave、FSDP prefetch 和 `reshard_after_forward=False` 的并发窗口。

完成后再按 `validation_gate` 升级能力状态；不要通过放大一个无界 residual 或复用别的模型家族档案来“通过”验收。
""",
            "cost_eval_tree",
        ),
        markdown_block(
            "further_questions",
            """
## 后续需要持续回答的问题

- 不同 MindSpore/CANN 版本下，哪些算子的 workspace 公式变化足以移动 peak_event？
- 真实 interleaved 1F1B 中，参数预取、通信和 backward kernel 的重叠是否比当前 rank-local 模型更高？
- tied embedding/output weight 在跨 PP stage 时的运行时持有方式，是否需要显式共享/复制合同？
- CPU offload 与 activation swap 的 host pinned 峰值和带宽约束，应如何与 HBM 安全判断联合报告？
- 当 allocator pool 已包含 rounded active allocations 时，当前额外分桶 fragmentation 是否会在某些 runtime profile 上重复计入？
""",
            "cost_eval_tree",
        ),
    ])

    scope_dataset = [{
        "files": len(module_rows),
        "lines": total_lines,
        "symbols": len(api_rows),
        "buckets": 7,
        "source_sha256": source_sha256(),
    }]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "从声明式模型图到每 stage/rank 峰值、allocator 组合与 OOM 状态的完整实现说明。",
        "generatedAt": generated_at,
        "cards": [
            {"id": "files_card", "description": "报告范围内的 Python 模块。", "dataset": "scope", "sourceId": "scope_summary", "metrics": [{"label": "源码文件", "field": "files", "format": "number"}]},
            {"id": "lines_card", "description": "当前快照物理源码行。", "dataset": "scope", "sourceId": "scope_summary", "metrics": [{"label": "源码行", "field": "lines", "format": "number"}]},
            {"id": "symbols_card", "description": "递归 AST 统计类、方法、属性、函数和内部函数。", "dataset": "scope", "sourceId": "scope_summary", "metrics": [{"label": "类/函数符号", "field": "symbols", "format": "number"}]},
            {"id": "buckets_card", "description": "时间线同时维护的物理显存桶。", "dataset": "scope", "sourceId": "scope_summary", "metrics": [{"label": "物理显存桶", "field": "buckets", "format": "number"}]},
        ],
        "charts": [{
            "id": "category_chart",
            "title": "源码行数按职责域分布",
            "subtitle": "当前 cost_eval 工作区快照，共 6,349 行；仅作阅读导航。",
            "type": "bar",
            "intent": "comparison",
            "question": "cost_eval 的实现规模主要集中在哪些职责域？",
            "rationale": "八个离散职责域适合用排序条形图比较源码行数；横向布局容纳较长中文标签。",
            "comparisonContext": {"unit": "Python 源码行", "grain": "职责域"},
            "dataset": "categories",
            "sourceId": "source_inventory",
            "options": {"orientation": "horizontal"},
            "valueFormat": "number",
            "encodings": {
                "x": {"field": "category", "type": "nominal", "label": "职责域"},
                "y": {"field": "lines", "type": "quantitative", "label": "源码行"},
                "tooltip": [
                    {"field": "files", "type": "quantitative", "label": "文件数", "format": "number"},
                    {"field": "callables", "type": "quantitative", "label": "类/函数数", "format": "number"},
                    {"field": "share", "type": "quantitative", "label": "行数占比", "format": "percent"},
                ],
            },
        }],
        "tables": tables,
        "sources": [{"id": item["id"], "label": item["label"], "path": item["path"]} for item in sources],
        "blocks": blocks,
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "scope": scope_dataset,
                "categories": category_rows,
                "modules": module_rows,
                **api_datasets,
            },
        },
        "sources": sources,
    }

    # The report promise is explicit: every current class/function definition
    # must appear once and every row must have a useful Chinese description.
    expected_files = {path for path, _, _ in MODULES}
    actual_files = {path.relative_to(ROOT).as_posix() for path in PACKAGE.rglob("*.py")}
    if expected_files != actual_files:
        raise RuntimeError(f"module inventory drift: missing={actual_files-expected_files}, stale={expected_files-actual_files}")
    if len(api_rows) != len({(row["file"], row["line"], row["symbol"]) for row in api_rows}):
        raise RuntimeError("duplicate API inventory rows")
    if any(not row["function"].strip() for row in api_rows):
        raise RuntimeError("empty API description")

    output = REPORT_DIR / "artifact.json"
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_output = REPORT_DIR / "report.md"
    markdown_output.write_text(render_markdown_report(artifact), encoding="utf-8")
    print(json.dumps({
        "artifact": str(output),
        "markdown": str(markdown_output),
        "files": len(module_rows),
        "lines": total_lines,
        "symbols": len(api_rows),
        "source_sha256": scope_dataset[0]["source_sha256"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
