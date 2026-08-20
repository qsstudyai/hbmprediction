# cost_eval 峰值 HBM 计算技术报告

## 技术摘要

`cost_eval` 的核心不是“把模型各部分的最大显存相加”，而是先把模型写成张量与算子图，求出每张卡上的局部字节，再沿每个流水 stage 的 1F1B 时间线逐次执行分配与释放，找到**同一时刻**七个物理显存桶的最大总和。最后才加入 allocator 池、设备基线、未追踪运行时、碎片和安全裕量。

最终点估计可以压缩为：

`HBM_point = device_baseline + max(physical_active_peak, allocator_pool_peak) + untracked_runtime + fragmentation`

安全上界为：

`HBM_safe = HBM_point + calibrated_upper_margin + OOD_margin`

因此，参数、梯度、激活、FSDP gather、重计算、swap 和 kernel workspace 是否真正同时存在，比单个模块的孤立峰值更重要。当前源码覆盖通用 Dense/MoE、Qwen3、DeepSeek-V3 MLA/MTP、DeepSeek-V4 legacy/hybrid/mHC/hash-router，并保留明确的能力状态：未经 runtime profile 和真实 HBM 验证的配置最多只能报告 `risky`，不能报告 `definitely_safe`。

### 报告范围

_当前 cost_eval 工作区快照。_

| 源码文件 | 源码行 | 类/函数符号 | 物理显存桶 |
| --- | --- | --- | --- |
| 36 | 6,349 | 263 | 7 |

## 代码主体集中在形状解析、内存时间线和模型图

下图按职责汇总当前源码规模，只用于帮助定位阅读重点，**不代表代码质量或运行耗时**。形状/静态态/时间线是 HBM 算法主体；模型图决定“有哪些参数和反向保存张量”；适配器决定真实配置如何落到这些合同上。图后给出逐文件精确盘点。

### 源码行数按职责域分布（表格版）

_当前 cost_eval 工作区快照，共 6,349 行；仅作阅读导航。_

| 职责域 | 源码行 | 文件数 | 类/函数数 | 行数占比 |
| --- | --- | --- | --- | --- |
| 形状、静态显存与时间线 | 1,706 | 9 | 85 | 26.9% |
| 模型与算子图 | 1,124 | 8 | 27 | 17.7% |
| 配置与核心数据模型 | 1,088 | 4 | 66 | 17.1% |
| 报告、测量与校准 | 1,052 | 4 | 42 | 16.6% |
| MindFormers 配置适配 | 859 | 2 | 26 | 13.5% |
| 运行时源码合同 | 311 | 4 | 3 | 4.9% |
| 运行时校准与溯源 | 153 | 3 | 13 | 2.4% |
| 入口与公开 API | 56 | 2 | 1 | 0.9% |

### 逐文件职责与规模

_当前工作区快照；行数为物理源码行，callables 含类/方法/属性/函数。_

| 文件 | 职责域 | 行数 | 类/函数数 | 主要功能 |
| --- | --- | --- | --- | --- |
| cost_eval/__init__.py | 入口与公开 API | 32 | 0 | 公开 API 重导出 |
| cost_eval/__main__.py | 入口与公开 API | 24 | 1 | 命令行评估入口 |
| cost_eval/adapters/__init__.py | MindFormers 配置适配 | 5 | 0 | 适配器公开 API |
| cost_eval/adapters/mindformers.py | MindFormers 配置适配 | 854 | 26 | 把 MindFormers TrainConfig 转为模型图、并行、优化器、硬件、重计算和 swap 合同 |
| cost_eval/allocator_model.py | 形状、静态显存与时间线 | 67 | 4 | 把物理存活峰值与 allocator、设备基线、运行时及安全裕量组合成总 HBM |
| cost_eval/calibration.py | 报告、测量与校准 | 471 | 17 | 预测与真实 profiling 的逐 stage、逐桶、OOM 和峰值事件对账 |
| cost_eval/config_adapter.py | 配置与核心数据模型 | 347 | 11 | 通用 JSON/YAML 配置适配 |
| cost_eval/event_schedule.py | 形状、静态显存与时间线 | 53 | 3 | 生成 1F1B 和交错流水事件 |
| cost_eval/layers/__init__.py | 模型与算子图 | 22 | 0 | 内置层图公开 API |
| cost_eval/layers/deepseek.py | 模型与算子图 | 194 | 7 | DeepSeek-V3 MLA、MoE、共享专家和 MTP 图 |
| cost_eval/layers/deepseek_v4.py | 模型与算子图 | 501 | 12 | DeepSeek-V4 legacy MLA、hybrid CSA、mHC、hash router 和 MTP 图 |
| cost_eval/layers/dense.py | 模型与算子图 | 130 | 2 | Dense GQA、FlashAttention、SwiGLU 与通用 Hyper-Connections 图 |
| cost_eval/layers/loss.py | 模型与算子图 | 79 | 1 | 融合与回退交叉熵的 logits、保存张量和反向工作区 |
| cost_eval/layers/moe.py | 模型与算子图 | 81 | 1 | 通用 attention + pure-EP routed experts 图 |
| cost_eval/layers/moe_dispatch.py | 模型与算子图 | 29 | 2 | MoE 路由、permute 和 inverse-map 辅助张量 |
| cost_eval/layers/qwen3.py | 模型与算子图 | 88 | 2 | Qwen3 GQA、Q/K RMSNorm、RoPE、FlashAttention 和 SwiGLU 图 |
| cost_eval/measurement_contract.py | 报告、测量与校准 | 228 | 14 | 真实 NPU HBM 的互斥分量、逐 rank 样本和质量合同 |
| cost_eval/mem_timeline.py | 形状、静态显存与时间线 | 588 | 22 | 沿逐 stage 1F1B 事件追踪七个物理显存桶并捕获同时峰值 |
| cost_eval/memory_actions.py | 形状、静态显存与时间线 | 106 | 4 | 把单层前向解析为显式 ALLOC/FREE/通信工作区动作 |
| cost_eval/memory_ledger.py | 形状、静态显存与时间线 | 75 | 9 | 执行分配/释放账本，检查别名、泄漏和桶守恒 |
| cost_eval/model_spec.py | 配置与核心数据模型 | 188 | 14 | 声明符号维度、张量 placement、算子和模型层型 |
| cost_eval/offload_model.py | 形状、静态显存与时间线 | 37 | 3 | 把参数、梯度、master weight 和优化器状态拆到 HBM 或 host pinned |
| cost_eval/optimizers.py | 形状、静态显存与时间线 | 63 | 3 | AdamW/Muon 持久状态字节和 optimizer step 临时工作区 |
| cost_eval/parallel_model.py | 配置与核心数据模型 | 145 | 16 | 校验并行 mesh，计算 FSDP/eFSDP 度数和 PP 层分配 |
| cost_eval/report.py | 报告、测量与校准 | 195 | 7 | 统一评估门面、OOM 状态、逐 stage/逐 rank 输出 |
| cost_eval/runtime_profiles.py | 运行时校准与溯源 | 91 | 6 | 按模型家族和并行布局加载 allocator/runtime 校准档案 |
| cost_eval/shape_eval.py | 形状、静态显存与时间线 | 562 | 24 | 求值符号 shape、代入切分、检测通信并解析成每卡字节图 |
| cost_eval/source_contracts/__init__.py | 运行时源码合同 | 19 | 0 | 源码合同公开 API |
| cost_eval/source_contracts/deepseek_v3.py | 运行时源码合同 | 46 | 0 | DeepSeek-V3 运行时来源、参数公式与并行依据 |
| cost_eval/source_contracts/deepseek_v4.py | 运行时源码合同 | 229 | 3 | DeepSeek-V4 固定 commit/hash、变体公式和覆盖校验 |
| cost_eval/source_contracts/qwen3.py | 运行时源码合同 | 17 | 0 | Qwen3 固定 commit、数据类型和 saved-tensor 合同 |
| cost_eval/source_fingerprint.py | 运行时校准与溯源 | 17 | 1 | 生成稳定的 Python 源码树 SHA-256 |
| cost_eval/specs.py | 配置与核心数据模型 | 408 | 25 | 并行、优化器、硬件、重计算和 swap 配置合同 |
| cost_eval/static_mem.py | 形状、静态显存与时间线 | 155 | 13 | 计算每个 PP stage 的参数、梯度、master weight 和优化器持久态 |
| cost_eval/validation_gate.py | 报告、测量与校准 | 158 | 4 | P0/P1 数值门、覆盖率门和生产证据缺口 |
| cost_eval/workspace_registry.py | 运行时校准与溯源 | 45 | 6 | 按算子、形状、版本和硬件精确匹配 kernel workspace 档案 |

## 范围和关键术语先统一

- **HBM**：设备显存。报告中的 host pinned memory 单独列出，不计入 HBM。
- **global shape / local shape**：前者是模型逻辑形状；后者已除以 CP/TP/EP/SP 等切分度数，是单 rank 真正持有的形状。
- **persistent**：跨整个训练 step 常驻的参数、梯度、master weight 和优化器状态。
- **saved activation**：前向为反向保存的张量；可按重计算或 swap 规则变成 checkpoint、recompute scratch 或 host-offloaded 数据。
- **workspace**：某个 kernel、collective、loss 或 optimizer step 在执行期间的临时内存。
- **physical active peak**：七个物理桶在一个真实模拟时刻的和；它不是各桶独立峰值之和。
- **point estimate / safe upper**：前者是最可能的总 HBM；后者再加校准分位数与 OOD 裕量，用于安全判断。

本报告审阅 `cost_eval/**/*.py` 的当前工作区快照，不把测试、工具脚本和 `validation/` 结果算进 API 范围。源码树指纹写入报告元数据，便于之后判断报告是否过期。

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

## 第一步：一个张量怎样变成“每卡多少字节”

`eval_expr` 只允许整数、已知维度符号以及 `+ - * //`，并要求整除，避免任意代码执行或悄悄截断。`resolve_tensor` 先求 global shape，再按 placement 逐维切分：

- 标记为 `sp` 的第 0 维，在 CP>1 时先除以 CP；若启用 sequence parallel，再按 TP 除一次。
- 普通 `tp/cp/ep` shard 把对应维度除以 mesh degree；不能整除就直接失败。
- `alltoall_deredundency/zero_redundancy` dispatcher 会把 routed-token 第一维再按 TP 去冗余。
- 权重缺省使用 `param_dtype_bytes`，激活缺省使用 `dtype_bytes`，显式 FP32/int32 张量固定 4 字节。

最后：`local_bytes = product(local_shape) * dtype_bytes`。

去重不靠张量名猜测。`storage_key` 优先使用显式 `storage_id`，其次是 placement-aware `value_id`，最后才是名称。这样 tied embedding/output weight 在同一 stage 可共享，而不同 placement 的同名数据流不会误合并。

placement 改变也会产生显存：partial 先 `reduce_scatter` 或 `all_reduce`；去掉某个 shard 需要 `all_gather`；同一 mesh 轴换到另一张量维度需要 `all_to_all`；只增加 shard 可本地完成。Flash/Sparse Attention 的 CP 还会按 colossal、Ulysses 或 hybrid 追加 ring/all-to-all；MoE dispatch/combine 在 EP>1 时追加 all-to-all。collective 的最大通信量会在相应算子时刻进入 workspace。

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

## 第三步：保存激活先做 resident、recompute 和 swap 的互斥划分

每个 `OpSpec.saves` 都表示反向真正需要的值。`_layer_memory_plan` 先按 `storage_key` 去重，再应用策略：

- 未重算算子的 saves 默认进入 **resident**。
- 每段连续重计算区域只保留第一个非权重输入作为 **checkpoint**；其余 saves 从前向 resident 中删除，字节计入反向 **recompute_scratch**。
- communication recompute 会删除被选 collective 的输出激活，反向前以 collective volume 计入 scratch。
- swap 只处理被选择、可换出的 saves。某张量若同时被未 swap 的算子需要，或它是重计算 checkpoint，就不能换出。
- 同一算子同时被 recompute 和 swap 选择时，recompute 优先，避免重复节省同一字节。

结果中的 resident/offloaded/scratch 是同一批 saves 的互斥视图。它们不能再相互相加为“原始激活总量”；只有放到具体时间线事件上才知道哪些会重叠。

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

## 第五步：物理峰值再组合成点估计、安全上界和 OOM 状态

`MemTimeline` 在 `Evaluator` 中以 framework reserve=0 运行，所以得到的是纯物理 active peak。`AllocatorModel` 随后处理：

1. 对峰值快照的七个物理桶分别向上对齐到 `allocator_granularity_bytes`，碎片为各桶 `round_up(bucket)-bucket` 的和。
2. `allocator_pool_peak = max(active, configured_pool_point, active + pool_slack_point)`。
3. runtime 优先使用 `untracked_runtime_point_bytes`；若为 0，才退回旧字段 `framework_reserve`。
4. `total_point = device_baseline + max(active, allocator_pool_peak) + runtime + fragmentation`。
5. `safe_upper = total_point + calibrated_upper_margin + ood_margin`。

最终报告把 `total_point-active` 统一放入 `breakdown.framework`，它实际包含设备基线、pool slack、未追踪 runtime 和分桶对齐碎片，不只是 Python 框架本身。

OOM 判定顺序很保守：点估计超过 `usable_device_memory` → `predicted_oom`；模型合同不支持 → `unsupported`；安全上界超过容量 → `risky`；能力未验证 → 仍是 `risky`；只有点估计和上界都不过线且能力状态为 validated 才是 `definitely_safe`。逐 rank 结果按 `rank % pp` 继承对应 stage，最紧 rank 取第一个最大值。

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

## Edge 与 Loss：大词表 logits 往往是最后 stage 的关键峰值

stage 0 的 embedding 权重为 `vocab*H` 并按 TP 切；最后 stage 有 FP32 final norm、lm head 和 loss。lm head 权重同样是 `vocab*H` 按 TP 切；若 `tie_word_embeddings=True` 且处于可共享的同一 stage，storage id 防止重复计费。

logits 全局形状为 `S*B*vocab`。不开 loss parallel 时 logits 不按词表 TP 切，并在 lm head 后产生 all-gather staging；开启时词表维保留 TP shard，显著降低每卡 logits 与 loss 工作集。

- fused cross entropy：保存 logits、labels、mask、FP32 stats，前向 workspace=`S*B*2*4`；反向再预留一份最大输入大小。
- fallback cross entropy：先物化 FP32 logits，再物化同形状 FP32 probabilities；softmax 保存 probabilities/stats，reduce 保存 probabilities/labels/mask；反向按最大输入的 2 倍建 workspace，表示 probability gradient 与 dlogits 同时存在。

这些 edge ops 参与与 decoder 相同的 FSDP gather、前向 ledger 和反向 grad buffer。报告中的 `loss_logits`、`loss_bwd_dlogits` 峰值事件就是专门为识别大词表风险保留的标签。

## 校准与验证只修正运行时开销，不重写物理公式

`WorkspaceRegistry` 可按模型家族、算子类型、融合变体、dtype、本地 shape、TP/CP/EP、MindSpore/CANN 版本和硬件精确替换 kernel workspace 点值/上界；没有精确键就保留公式值，不做最近邻猜测。

`RuntimeProfileRegistry` 按 `family + tp/cp/pp/ep/dp/interleave` 选择档案，把实测 runtime、allocator pool slack 和 baseline 点值写入 HardwareSpec。上界由这些分量的 P90-点值差、物理 shortfall P90 和 singleton 2 GiB 裕量相加。完全缺档时不借用别的家族/布局，只标记 missing 并把 OOD margin 至少扩大 2 GiB。

真实测量合同要求分量互斥：`dynamic = max(model_active, allocator_pool) + runtime`，`nominal_total = baseline + dynamic`；逐 stage peak 应等于该 stage 所有 rank 的最大 peak_delta。质量状态、rank 数和每 rank 样本不足会使标签不适合校准。

`CalibrationRunner` 比较逐 stage 峰值、峰值事件阶段和可选分桶，汇总 mean/P50/P90/max relative error、signed bias、事件命中率和 OOM 混淆矩阵。`validation_gate` 把历史 replay 与生产验证严格分开：缺 blind holdout、真实 OOM、stage/rank 排序或策略 paired delta 证据时状态是 `unavailable`，绝不当作 pass。

## 全量 API 索引覆盖每个类、方法、属性、函数和内部函数

以下表由当前 `cost_eval/**/*.py` 的 Python AST 递归生成，并为每个符号附上职责说明。这样既覆盖公开 API，也覆盖 `_` 前缀辅助函数、property、classmethod、嵌套兼容类和内部函数。表按职责拆分，便于筛选；`line` 是定义起始行。

### 入口与公开 API：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/__main__.py | 12 | 函数 | main | main(argv=None) -> int | 解析命令行参数，运行相应评估或校准，并输出 JSON；阈值失败时返回非零退出码。 |

### MindFormers 配置适配：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/adapters/mindformers.py | 34 | 类 | MindFormersInputs | 字段: model_spec, parallel_config, optimizer, hardware, recompute, swap, warnings, source | 不可变输入包，保存 MindFormers 适配后的模型图、并行、优化器、硬件、重计算、swap、告警和来源。 |
| cost_eval/adapters/mindformers.py | 44 | 方法 | MindFormersInputs.evaluator | evaluator(self) -> Evaluator | 把已适配的全部输入和告警组装成 Evaluator。 |
| cost_eval/adapters/mindformers.py | 56 | 类 | MindFormersAdapter | 服务类/枚举 | 无须导入 MindSpore 的适配门面，把 TrainConfig YAML/字典转换为可执行的离线评估输入。 |
| cost_eval/adapters/mindformers.py | 60 | 类方法 | MindFormersAdapter.from_yaml | from_yaml(cls, path: str \| Path, world_size: int, framework_reserve: int=0) -> MindFormersInputs | 读取 MindFormers YAML，并以文件绝对路径作为来源交给 from_mapping。 |
| cost_eval/adapters/mindformers.py | 71 | 类方法 | MindFormersAdapter.from_mapping | from_mapping(cls, data: Mapping[str, Any], world_size: int, framework_reserve: int=0, source: str='<mapping>') -> MindFormersInputs | 完成 world-size/批量/并行校验，识别模型家族，构图并生成所有评估合同。 |
| cost_eval/adapters/mindformers.py | 406 | 函数 | load_mindformers_yaml | load_mindformers_yaml(path: Path) -> dict[str, Any] | 优先用 PyYAML 安全加载；没有依赖时退回到受限顶层 YAML 解析器。 |
| cost_eval/adapters/mindformers.py | 424 | 函数 | parse_memory_bytes | parse_memory_bytes(value: Any) -> int | 把整数或 KB/MB/GB/TB 字符串按二进制单位换算为字节。 |
| cost_eval/adapters/mindformers.py | 434 | 函数 | _build_model_shape | _build_model_shape(model: Mapping[str, Any], training: Mapping[str, Any]) -> tuple[DimTable, tuple[str, ...], list[str]] | 从 MindFormers model/training 字段解析 DimTable、dense/MoE 层模式和缺失能力告警。 |
| cost_eval/adapters/mindformers.py | 560 | 函数 | _is_deepseek_v4 | _is_deepseek_v4(model: Mapping[str, Any]) -> bool | 根据 model_type/architectures 名称识别 DeepSeek-V4。 |
| cost_eval/adapters/mindformers.py | 568 | 函数 | _is_qwen3 | _is_qwen3(model: Mapping[str, Any]) -> bool | 根据 model_type/architectures 名称识别 Qwen3。 |
| cost_eval/adapters/mindformers.py | 576 | 函数 | _int_alias_default | _int_alias_default(data: Mapping[str, Any], names: tuple[str, ...], default: int) -> int | 依次读取多个整数别名，均缺失时返回默认值。 |
| cost_eval/adapters/mindformers.py | 585 | 函数 | _moe_pattern | _moe_pattern(model: Mapping[str, Any], layers: int, experts: int) -> tuple[str, ...] | 按专家数、前置 dense 层、频率或逐层列表生成 dense/MoE 层模式。 |
| cost_eval/adapters/mindformers.py | 600 | 函数 | _build_optimizer | _build_optimizer(config: Mapping[str, Any], model: Mapping[str, Any], warnings: list[str]) -> OptimizerSpec | 按 MindFormers FP32 梯度累积合同构造 AdamW 或 Muon 的逐参数字节配置。 |
| cost_eval/adapters/mindformers.py | 642 | 函数 | _build_recompute | _build_recompute(config: Mapping[str, Any], recompute_comm: Mapping[str, Any], n_layers: int, warnings: list[str]) -> RecomputeSpec | 把 full/select、排除算子和 communication recompute 模块选择转成 RecomputeSpec。 |
| cost_eval/adapters/mindformers.py | 681 | 函数 | _build_swap | _build_swap(config: Mapping[str, Any], n_layers: int, warnings: list[str]) -> SwapSpec | 把 layer_swap/op_swap 和 prefetch 配置转成 SwapSpec，并拒绝越界预取。 |
| cost_eval/adapters/mindformers.py | 707 | 函数 | _parse_stage_layers | _parse_stage_layers(value: Any, n_layers: int) -> tuple[tuple[int, ...], ...] \| None | 解析显式 PP stage 层范围；auto 返回 None。 |
| cost_eval/adapters/mindformers.py | 716 | 函数 | _parse_layer_selection | _parse_layer_selection(value: Any, n_layers: int) -> frozenset[int] | 把整数、列表、映射、区间和逗号区间统一成合法 layer-id 集合。 |
| cost_eval/adapters/mindformers.py | 750 | 函数 | _parse_module_selection | _parse_module_selection(value: Any, n_layers: int) -> dict[str, frozenset[int]] | 把模块名与层选择配置统一成 module -> layer-id 集合。 |
| cost_eval/adapters/mindformers.py | 770 | 函数 | _load_yaml_subset | _load_yaml_subset(text: str) -> dict[str, Any] | 无 PyYAML 时只读取评估器需要的顶层 section 和标量字段。 |
| cost_eval/adapters/mindformers.py | 790 | 函数 | _strip_comment | _strip_comment(line: str) -> str | 删除引号外的 YAML # 注释。 |
| cost_eval/adapters/mindformers.py | 800 | 函数 | _parse_scalar | _parse_scalar(value: str) -> Any | 把受限 YAML 标量解析为 null、布尔、字面量、数值、列表或字符串。 |
| cost_eval/adapters/mindformers.py | 822 | 函数 | _section | _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any] | 读取并校验一个配置 section 必须是映射。 |
| cost_eval/adapters/mindformers.py | 831 | 函数 | _required_alias | _required_alias(data: Mapping[str, Any], *names: str) -> int | 从候选字段中返回第一个存在的必填整数，否则报错。 |
| cost_eval/adapters/mindformers.py | 838 | 函数 | _positive_int | _positive_int(value: Any, name: str) -> int | 转为整数并要求至少为 1。 |
| cost_eval/adapters/mindformers.py | 845 | 函数 | _optional_int | _optional_int(value: Any) -> int \| None | 把可空值转为 int 或 None。 |
| cost_eval/adapters/mindformers.py | 849 | 函数 | _dtype_bytes | _dtype_bytes(value: Any) -> int | 把 fp16/bf16/fp32 名称映射为 2/4 字节。 |

### 配置与核心数据模型：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/config_adapter.py | 44 | 函数 | parse_bytes | parse_bytes(value: Any) -> int | 把整数或十进制/二进制字节单位字符串转为字节。 |
| cost_eval/config_adapter.py | 58 | 函数 | _parse_yaml_scalar | _parse_yaml_scalar(text: str) -> Any | 解析通用配置 YAML 的布尔、空值、引号、列表、映射、数值和字符串标量。 |
| cost_eval/config_adapter.py | 93 | 函数 | _simple_yaml_load | _simple_yaml_load(text: str) -> Any | 解析项目支持的受限 YAML：嵌套映射、列表和标量。 |
| cost_eval/config_adapter.py | 108 | 方法 | _simple_yaml_load.parse_block | parse_block(index: int, indent: int) | 递归解析同一缩进层级的 YAML 映射或列表块。 |
| cost_eval/config_adapter.py | 156 | 函数 | _section | _section(config: Mapping[str, Any], *names: str) -> dict | 读取并校验一个配置 section 必须是映射。 |
| cost_eval/config_adapter.py | 167 | 类 | EvaluationInputs | 字段: model_spec, parallel, optimizer, hardware, recompute, swap | 通用配置适配后的六类评估输入，并能直接构造 Evaluator。 |
| cost_eval/config_adapter.py | 175 | 方法 | EvaluationInputs.evaluator | evaluator(self) -> Evaluator | 把通用适配结果组装成 Evaluator。 |
| cost_eval/config_adapter.py | 186 | 类 | ConfigAdapter | 服务类/枚举 | 通用 JSON/YAML 配置适配器，适合项目自有而非 MindFormers 原生格式。 |
| cost_eval/config_adapter.py | 188 | 类方法 | ConfigAdapter.from_dict | from_dict(cls, config: Mapping[str, Any]) -> EvaluationInputs | 校验通用配置，构造 dense/MoE 模型图以及并行、优化器、硬件、重计算、swap。 |
| cost_eval/config_adapter.py | 331 | 类方法 | ConfigAdapter.load | load(cls, path) -> EvaluationInputs | 读取 JSON/YAML 后要求根节点为映射，再调用 from_dict。 |
| cost_eval/config_adapter.py | 338 | 函数 | load_data_file | load_data_file(path) -> Any | 按扩展名读取 JSON 或受限 YAML。 |
| cost_eval/model_spec.py | 13 | 类 | OpType | 服务类/枚举 | 评估器识别的算子类别枚举，用于通信、workspace 和模型能力分支。 |
| cost_eval/model_spec.py | 29 | 类 | DimTable | 字段: H, F, n_heads, n_kv, head_dim, S, B, vocab, n_layers, n_experts, topk, n_shared, moe_F, capacity_factor, dtype_bytes, param_dtype_bytes, q_lora_rank, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, shared_F, n_mtp_layers, use_shared_expert_gating, tie_word_embeddings, hc_mult, enable_hyper_connections, o_lora_rank, o_groups, index_n_heads, index_head_dim, index_topk, num_hash_layers, sliding_window | 模型全局维度表，并派生 routed-token 容量与 decoder+MTP 总层数。 |
| cost_eval/model_spec.py | 65 | 方法 | DimTable.__post_init__ | __post_init__(self) -> None | 要求核心维度和 dtype 为正、capacity_factor 为正，并拒绝负的 V4/HC 可选维度。 |
| cost_eval/model_spec.py | 103 | 方法 | DimTable.as_dict | as_dict(self) -> dict[str, Union[int, float]] | 返回所有维度，并增加 T_routed=ceil(S*B*topk*capacity_factor)。 |
| cost_eval/model_spec.py | 111 | 属性 | DimTable.total_layers | total_layers(self) -> int | 返回 decoder 层数加 MTP 层数。 |
| cost_eval/model_spec.py | 116 | 类 | TensorRef | 字段: name, shape, shard, is_weight, partial, dtype_bytes, storage_id, trainable, swappable, recomputable, fsdp_replicated, value_id | 符号张量合同：shape、placement、dtype、权重/训练/换出/重算及显式存储别名。 |
| cost_eval/model_spec.py | 135 | 方法 | TensorRef.__post_init__ | __post_init__(self) -> None | 复制 shard 映射以冻结外部修改，并检查每个分片维索引没有越界。 |
| cost_eval/model_spec.py | 141 | 方法 | TensorRef.has_ep | has_ep(self) -> bool | 判断张量 placement 是否含 expert-parallel 分片轴。 |
| cost_eval/model_spec.py | 145 | 属性 | TensorRef.is_expert | is_expert(self) -> bool | 把 has_ep 暴露为专家张量属性。 |
| cost_eval/model_spec.py | 150 | 类 | OpSpec | 字段: name, type, inputs, output, params, saves, workspace, attrs, module_paths, backward_workspace | 声明式算子合同：输入、输出、参数、反向保存张量、workspace、模块路径和属性。 |
| cost_eval/model_spec.py | 164 | 类 | LayerSpec | 字段: ops | 一个层型的有序 OpSpec 序列。 |
| cost_eval/model_spec.py | 169 | 类 | ModelSpec | 字段: name, dims, layer_pattern, layer_specs, capabilities | 模型名称、维度、逐层类型模式、层型图和能力/来源元数据。 |
| cost_eval/model_spec.py | 176 | 方法 | ModelSpec.__post_init__ | __post_init__(self) -> None | 要求 layer_pattern 覆盖 decoder+MTP 全部层、每个层型都有 LayerSpec，并冻结 capabilities 副本。 |
| cost_eval/model_spec.py | 187 | 方法 | ModelSpec.get_layer | get_layer(self, layer_type: str) -> LayerSpec | 按层型名返回 LayerSpec。 |
| cost_eval/parallel_model.py | 11 | 类 | ParallelModel | 服务类/枚举 | 并行 mesh 的已解析视图，提供 FSDP/eFSDP 度数、层到 stage、rank 到 stage 映射。 |
| cost_eval/parallel_model.py | 12 | 类 | ParallelModel._StageLayers | 服务类/枚举 | ParallelModel 中用于 校验并行 mesh，计算 FSDP/eFSDP 度数和 PP 层分配 的类型。 |
| cost_eval/parallel_model.py | 13 | 方法 | ParallelModel._StageLayers.__init__ | __init__(self, owner) | 保存所属 ParallelModel，供 Mapping/可调用兼容接口转发。 |
| cost_eval/parallel_model.py | 15 | 方法 | ParallelModel._StageLayers.__getitem__ | __getitem__(self, stage) | 按 stage 读取层列表。 |
| cost_eval/parallel_model.py | 17 | 方法 | ParallelModel._StageLayers.__iter__ | __iter__(self) | 按 stage id 顺序迭代。 |
| cost_eval/parallel_model.py | 19 | 方法 | ParallelModel._StageLayers.__len__ | __len__(self) | 返回物理 PP stage 数。 |
| cost_eval/parallel_model.py | 21 | 方法 | ParallelModel._StageLayers.__call__ | __call__(self, stage) | 以可调用形式返回指定 stage 的层列表。 |
| cost_eval/parallel_model.py | 25 | 类方法 | ParallelModel.build | build(cls, pc, dims, world_size=None) | 用 DimTable.n_layers 构造 ParallelModel 的便捷入口。 |
| cost_eval/parallel_model.py | 28 | 方法 | ParallelModel.__init__ | __init__(self, pc: ParallelConfig, n_layers: int, world_size: Optional[int]=None) | 校验 world size/EP 整除和 PP 合法性，计算 eFSDP、层到 stage 映射，并检查 interleave 每 stage 层数足够。 |
| cost_eval/parallel_model.py | 74 | 方法 | ParallelModel.degree | degree(self, axis: str) -> int | 返回 tp/cp/ep/dp/pp 度数；sp 仅在 sequence_parallel 时等于 tp。 |
| cost_eval/parallel_model.py | 90 | 方法 | ParallelModel.fsdp_degree | fsdp_degree(self) -> int | 返回 dense FSDP shard 度数。 |
| cost_eval/parallel_model.py | 93 | 方法 | ParallelModel.efsdp_degree | efsdp_degree(self) -> int | 返回 expert FSDP 度数；ep=1 时与 dense FSDP 一致。 |
| cost_eval/parallel_model.py | 96 | 方法 | ParallelModel.stage_of | stage_of(self, layer_id: int) -> int | 查询 decoder layer 所属物理 PP stage。 |
| cost_eval/parallel_model.py | 99 | 方法 | ParallelModel.stage_of_rank | stage_of_rank(self, rank: int) -> int | 按 rank % pp 映射 rank 到物理 stage。 |
| cost_eval/parallel_model.py | 106 | 方法 | ParallelModel._stage_layers | _stage_layers(self, stage: int) -> list[int] | 返回指定 stage 的 layer-id 列表。 |
| cost_eval/parallel_model.py | 115 | 方法 | ParallelModel._build_layer_to_stage | _build_layer_to_stage(self) -> list[int] | 解析显式层映射；否则按前部 stage 多分余数的策略自动均分。 |
| cost_eval/specs.py | 9 | 函数 | _normalize_op_map | _normalize_op_map(value) -> dict[int, frozenset[str]] | 把 layer -> 算子名配置规范为 int -> frozenset[str]。 |
| cost_eval/specs.py | 19 | 类 | ParallelConfig | 字段: dp_replicate, dp_shard, cp, tp, pp, ep, sequence_parallel, reshard_after_forward_policy, reshard_after_forward, cpu_offload, microbatch, interleave, pipeline_schedule, pipeline_interleave, layers_per_stage, prefetch_depth, num_microbatches, dense_fsdp_shard_size, context_parallel_method, ulysses_degree_in_cp, moe_token_dispatcher, enable_loss_parallel, parameter_offload, gradient_offload, optimizer_offload, fsdp_flatten_alignment_bytes | 并行与训练调度合同，校验 DP/CP/TP/PP/EP、SP、offload、prefetch 和 interleave。 |
| cost_eval/specs.py | 49 | 方法 | ParallelConfig.__post_init__ | __post_init__(self) -> None | 解析 reshard/interleave/cpu-offload 别名，校验所有并行度、CP 方法、dispatcher、FSDP 对齐和显式 PP 层配置。 |
| cost_eval/specs.py | 120 | 属性 | ParallelConfig.world_size | world_size(self) -> int | 返回 dp_replicate*dp_shard*cp*tp*pp；EP 是区域内重排，不额外乘 world size。 |
| cost_eval/specs.py | 124 | 属性 | ParallelConfig.fsdp_degree | fsdp_degree(self) -> int | 返回显式 dense_fsdp_shard_size 或 dp_shard*cp。 |
| cost_eval/specs.py | 128 | 属性 | ParallelConfig.expert_fsdp_degree | expert_fsdp_degree(self) -> int | ep=1 时等于 dense FSDP；否则为 dp_shard*cp*tp/ep。 |
| cost_eval/specs.py | 139 | 属性 | ParallelConfig.virtual_pipeline_size | virtual_pipeline_size(self) -> int | 返回 interleave，即每物理 stage 的 virtual chunk 数。 |
| cost_eval/specs.py | 146 | 类 | OptimizerSpec | 字段: type, state_bytes_per_param, parameter_bytes, gradient_bytes, master_weight_bytes, optimizer_state_bytes, muon_ns_steps, muon_parameter_fraction, muon_momentum_bytes, muon_main_parameter_bytes | 优化器持久字节合同，支持 AdamW 与 Muon/混合参数分区。 |
| cost_eval/specs.py | 158 | 方法 | OptimizerSpec.__post_init__ | __post_init__(self) -> None | 校验逐参数总字节和 Muon 参数，并从总量扣除参数/梯度/master 得到 optimizer-state 字节。 |
| cost_eval/specs.py | 173 | 属性 | OptimizerSpec.name | name(self) -> str | 返回小写优化器类型。 |
| cost_eval/specs.py | 177 | 属性 | OptimizerSpec.bytes_per_parameter | bytes_per_parameter(self) -> int | 返回配置的每参数总持久字节。 |
| cost_eval/specs.py | 181 | 属性 | OptimizerSpec.is_muon | is_muon(self) -> bool | 根据名称或 Newton-Schulz 步数识别 Muon 合同。 |
| cost_eval/specs.py | 185 | 类方法 | OptimizerSpec.adamw | adamw(cls, fp32_grad: bool=False) -> 'OptimizerSpec' | 构造 BF16/FP16 参数、可选 FP32 梯度的标准 AdamW 字节合同。 |
| cost_eval/specs.py | 190 | 类 | HardwareSpec | 字段: max_device_memory, framework_reserve, usable_device_memory, device_baseline_bytes, allocator_pool_point_bytes, allocator_pool_slack_point_bytes, untracked_runtime_point_bytes, allocator_granularity_bytes, calibrated_upper_margin_bytes, ood_margin_bytes, hardware_profile, runtime_profile, source_profile | 设备容量、可用容量、allocator、runtime、校准裕量和来源指纹。 |
| cost_eval/specs.py | 205 | 方法 | HardwareSpec.__post_init__ | __post_init__(self) -> None | 校验容量/运行时/裕量非负、allocator 粒度为正，并把可用容量规范到 (0,max]。 |
| cost_eval/specs.py | 226 | 类 | RecomputeSpec | 字段: mode, full_layers, select_ops, exclude_ops, exclude_ops_by_layer, select_modules, recompute_comm, comm_select_modules | full/select 激活重计算及 communication recompute 的选择与排除规则。 |
| cost_eval/specs.py | 236 | 方法 | RecomputeSpec.__post_init__ | __post_init__(self) -> None | 规范 mode、逐层算子映射、全局/逐层排除和模块选择为不可变集合。 |
| cost_eval/specs.py | 263 | 属性 | RecomputeSpec.layers | layers(self) -> frozenset[int] | 兼容属性，返回 full recompute layer 集合。 |
| cost_eval/specs.py | 266 | 方法 | RecomputeSpec.is_full | is_full(self, layer_id: int) -> bool | 判断指定层是否进入 full recompute。 |
| cost_eval/specs.py | 271 | 方法 | RecomputeSpec.recomputed_ops | recomputed_ops(self, layer_id: int, available_ops: Sequence[str]) -> frozenset[str] | 展开 full/select 模块组和别名，应用全局/逐层排除并返回精确算子集合。 |
| cost_eval/specs.py | 313 | 方法 | RecomputeSpec.recomputed_comm_ops | recomputed_comm_ops(self, layer_id: int, ops) -> frozenset[str] | 把模块路径选择匹配到含 collective 的算子。 |
| cost_eval/specs.py | 335 | 类 | SwapSpec | 字段: enable, default_prefetch, swap_layers, swap_ops, enabled, layers, op_layers, prefetch_depth, op_names | 按层/算子选择保存激活换出及反向预取深度。 |
| cost_eval/specs.py | 346 | 方法 | SwapSpec.__post_init__ | __post_init__(self) -> None | 解析 enable/layers/prefetch 兼容字段，规范逐层/逐模块选择，并把全局 op_names 合并到规则。 |
| cost_eval/specs.py | 369 | 方法 | SwapSpec.swaps | swaps(self, layer_id: int) -> bool | 判断整层 swap 是否覆盖指定层。 |
| cost_eval/specs.py | 375 | 方法 | SwapSpec.swapped_ops | swapped_ops(self, layer_id: int, available_ops: Sequence[str]) -> frozenset[str] | 展开整层/算子/模块组 swap 选择并校验只含现有算子。 |

### 模型与算子图：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/layers/deepseek.py | 10 | 函数 | _norm_weight | _norm_weight(name: str, dim='H') -> TensorRef | 创建一个 FP32、可训练的归一化权重 TensorRef。 |
| cost_eval/layers/deepseek.py | 14 | 函数 | build_mla_attention | build_mla_attention(dims: DimTable) -> tuple[OpSpec, ...] | 构造 DeepSeek-V3 MLA 从 down projection、RoPE、KV up 到 FlashAttention 和输出投影的完整算子图。 |
| cost_eval/layers/deepseek.py | 84 | 函数 | _moe_tail | _moe_tail(dims: DimTable) -> tuple[OpSpec, ...] | 构造 DeepSeek 的 RMSNorm、router、dispatch、routed experts、可选 shared experts 和残差尾部。 |
| cost_eval/layers/deepseek.py | 136 | 函数 | _dense_tail | _dense_tail() -> tuple[OpSpec, ...] | 构造 DeepSeek dense SwiGLU FFN 尾部。 |
| cost_eval/layers/deepseek.py | 160 | 函数 | build_mla_dense_decoder | build_mla_dense_decoder(dims: DimTable, include_hc: bool=True) -> LayerSpec | 把 MLA attention、dense FFN 和可选 Hyper-Connections 组合成层图。 |
| cost_eval/layers/deepseek.py | 165 | 函数 | build_mla_moe_decoder | build_mla_moe_decoder(dims: DimTable, include_hc: bool=True) -> LayerSpec | 把 MLA attention、MoE/shared-expert 尾部和可选 Hyper-Connections 组合成层图。 |
| cost_eval/layers/deepseek.py | 170 | 函数 | build_mtp_layer | build_mtp_layer(dims: DimTable) -> LayerSpec | 构造 embedding/上一隐藏态融合、一个完整 MLA-MoE 内层和最终归一化的 MTP 图。 |
| cost_eval/layers/deepseek_v4.py | 11 | 函数 | _norm_weight | _norm_weight(name: str, dim: str='H') -> TensorRef | 创建一个 FP32、可训练的归一化权重 TensorRef。 |
| cost_eval/layers/deepseek_v4.py | 15 | 函数 | _with_hash_router | _with_hash_router(layer: LayerSpec, enabled: bool) -> LayerSpec | 给 router 注入每 token 到 expert 的 replicated int32 查表参数。 |
| cost_eval/layers/deepseek_v4.py | 45 | 函数 | build_v4_legacy_mla_moe_decoder | build_v4_legacy_mla_moe_decoder(dims: DimTable, hash_router: bool=False) -> LayerSpec | 构造显式选择 legacy MLA 的 V4 MoE 层，并按配置加入 mHC/hash router。 |
| cost_eval/layers/deepseek_v4.py | 56 | 函数 | _mhc_pre | _mhc_pre(dims: DimTable, prefix: str, stream_name: str, output_name: str) -> tuple[OpSpec, TensorRef, TensorRef, TensorRef] | 构造 V4 mHC 前置 FP32 mapping/Sinkhorn 参数、保存张量和 workspace。 |
| cost_eval/layers/deepseek_v4.py | 119 | 函数 | _mhc_post | _mhc_post(prefix: str, streams: TensorRef, h_res: TensorRef, h_post: TensorRef, sublayer_output: TensorRef, output_name: str) -> OpSpec | 构造 mHC output-cell 混合，保存 streams/h_res/h_post/子层输出并计 FP32 workspace。 |
| cost_eval/layers/deepseek_v4.py | 142 | 函数 | _with_mhc | _with_mhc(dims: DimTable, attention: tuple[OpSpec, ...], tail: tuple[OpSpec, ...]) -> tuple[OpSpec, ...] | 用独立 attention mHC 和 FFN mHC 替换普通两个 residual add。 |
| cost_eval/layers/deepseek_v4.py | 179 | 函数 | _compressor_op | _compressor_op(dims: DimTable, ratio: int, head_dim: str, prefix: str) -> tuple[OpSpec, TensorRef] | 按压缩比 4/128 构造 V4 KV compressor 参数、输出和保存张量。 |
| cost_eval/layers/deepseek_v4.py | 213 | 函数 | build_v4_hybrid_attention | build_v4_hybrid_attention(dims: DimTable, compress_ratio: int, dense_mode: bool=False) -> tuple[OpSpec, ...] | 构造 V4 q/kv 低秩投影、可选 compressor/indexer、稀疏注意力和 grouped low-rank 输出。 |
| cost_eval/layers/deepseek_v4.py | 392 | 函数 | build_v4_hybrid_moe_decoder | build_v4_hybrid_moe_decoder(dims: DimTable, compress_ratio: int, hash_router: bool=False, dense_mode: bool=False, include_hc: bool=True) -> LayerSpec | 组合 V4 hybrid attention、MoE、mHC 和可选 hash router。 |
| cost_eval/layers/deepseek_v4.py | 408 | 函数 | _mtp_outer | _mtp_outer(dims: DimTable) -> tuple[OpSpec, ...] | 构造 V4 MTP 外层的 embedding/hidden 归一化、拼接和 2H->H 投影。 |
| cost_eval/layers/deepseek_v4.py | 429 | 函数 | build_v4_hybrid_mtp_layer | build_v4_hybrid_mtp_layer(dims: DimTable, compress_ratio: int=0, dense_mode: bool=False) -> LayerSpec | 构造一个不启用 hash router 的 hybrid V4 MTP 层，含 mHC expand/collapse。 |
| cost_eval/layers/deepseek_v4.py | 470 | 函数 | build_v4_legacy_mtp_layer | build_v4_legacy_mtp_layer(dims: DimTable) -> LayerSpec | 构造 legacy MLA V4 MTP 层，含 mHC expand/collapse。 |
| cost_eval/layers/dense.py | 11 | 函数 | hyper_connection_ops | hyper_connection_ops(dims: DimTable, source_name: str) -> tuple[OpSpec, ...] | 若启用，增加 H->hc_mult*H 投影和多流混合的参数、激活与保存张量。 |
| cost_eval/layers/dense.py | 45 | 函数 | build_dense_decoder | build_dense_decoder(dims: DimTable) -> LayerSpec | 构造通用 GQA/FlashAttention/SwiGLU dense decoder 的张量、参数、saves 和 workspace。 |
| cost_eval/layers/loss.py | 13 | 函数 | build_loss_ops | build_loss_ops(dims: DimTable, variant: str, loss_parallel: bool) -> tuple[OpSpec, ...] | 按 fused/fallback 交叉熵和 loss parallel 设置构造 logits、FP32 中间量、saves 与反向工作区。 |
| cost_eval/layers/moe.py | 10 | 函数 | build_moe_decoder | build_moe_decoder(dims: DimTable) -> LayerSpec | 构造通用 attention + router/dispatch/pure-EP expert/combine MoE 层。 |
| cost_eval/layers/moe_dispatch.py | 11 | 类 | MoEDispatchTensors | 字段: topk_scores, topk_indices, token_counts, expert_offsets, permute_map, inverse_map | MoE dispatcher 必须保存的 top-k、计数、偏移、置换和逆置换张量集合。 |
| cost_eval/layers/moe_dispatch.py | 20 | 函数 | build_dispatch_tensors | build_dispatch_tensors(prefix: str='router') -> MoEDispatchTensors | 创建 FP32 top-k scores 和 int32 indices/counts/offsets/permute/inverse-map 张量合同。 |
| cost_eval/layers/qwen3.py | 9 | 函数 | _norm_weight | _norm_weight(name: str) -> TensorRef | 创建一个 FP32、可训练的归一化权重 TensorRef。 |
| cost_eval/layers/qwen3.py | 13 | 函数 | build_qwen3_decoder | build_qwen3_decoder(dims: DimTable) -> LayerSpec | 构造 Qwen3 Q/K RMSNorm、RoPE 辅助量、FlashAttention 和 SwiGLU 的精确 saved-tensor 图。 |

### 形状、静态显存与时间线：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/allocator_model.py | 8 | 函数 | _round_up | _round_up(value: int, granularity: int) -> int | 把非零字节向上对齐到 allocator 粒度。 |
| cost_eval/allocator_model.py | 13 | 类 | AllocatorEstimate | 字段: physical_active_peak_bytes, allocator_pool_peak_bytes, device_baseline_bytes, untracked_runtime_point_bytes, fragmentation_point_bytes, total_point_bytes, safe_upper_bytes | 总 HBM 组合结果，分别保留物理存活峰值、allocator 池、基线、运行时、碎片、点估计和安全上界。 |
| cost_eval/allocator_model.py | 23 | 类 | AllocatorModel | 服务类/枚举 | allocator/runtime 组合器；使用可审计的分桶公式，不使用全局拟合倍率。 |
| cost_eval/allocator_model.py | 26 | 方法 | AllocatorModel.estimate | estimate(self, active_bytes: int, breakdown, hardware) -> AllocatorEstimate | 逐桶计算对齐碎片，再组合 pool、设备基线、运行时、校准和 OOD 裕量。 |
| cost_eval/event_schedule.py | 9 | 类 | Event | 字段: kind, mb, layer, chunk | rank-local 流水事件；记录前向/反向、microbatch、层和 virtual chunk。 |
| cost_eval/event_schedule.py | 16 | 函数 | build_1f1b | build_1f1b(stage: int, pp: int, m: int \| None=None, microbatches: int \| None=None) -> list[Event] | 生成给定 PP stage 的 warmup 前向和随后一前一反的 rank-local 事件序列。 |
| cost_eval/event_schedule.py | 35 | 函数 | build_interleaved_1f1b | build_interleaved_1f1b(stage: int, pp: int, m: int, interleave: int) -> list[Event] | 把每个 1F1B 事件按 virtual chunk 扩展；前向升序、反向逆序。 |
| cost_eval/mem_timeline.py | 14 | 类 | Buckets | 字段: persistent, act_live, gather_buf, grad_buf, recomp_scratch, swap_buf, workspace | 时间线当前时刻的七个物理显存桶，可原地增减。 |
| cost_eval/mem_timeline.py | 23 | 方法 | Buckets.total | total(self) -> int | 把当前七个物理桶相加。 |
| cost_eval/mem_timeline.py | 28 | 类 | MemBreakdown | 字段: persistent, act_live, gather_buf, grad_buf, recomp_scratch, swap_buf, workspace, framework | 峰值时刻的只读显存快照；七个物理桶外加 framework/运行时综合开销。 |
| cost_eval/mem_timeline.py | 39 | 属性 | MemBreakdown.total | total(self) -> int | 把峰值快照的七个物理桶和 framework 综合开销相加。 |
| cost_eval/mem_timeline.py | 44 | 类 | StagePeak | 字段: stage, peak_bytes, breakdown, peak_event, oom, bucket_peaks, physical_dynamic_peak_bytes, allocator_pool_peak_bytes, device_baseline_bytes, untracked_runtime_point_bytes, fragmentation_point_bytes, total_point_bytes, safe_upper_bytes, oom_status | 一个 stage 的峰值字节、峰值事件、分桶、点估计、安全上界和 OOM 状态。 |
| cost_eval/mem_timeline.py | 62 | 类 | BucketPeaks | 字段: persistent, act_live, gather_buf, grad_buf, recomp_scratch, swap_buf, workspace | 各物理桶在整个时间线上的独立最大值；不能相加当成同时峰值。 |
| cost_eval/mem_timeline.py | 73 | 类 | LayerMemoryPlan | 字段: resident_bytes, offloaded_bytes, recompute_scratch_bytes, recomputed_ops, recompute_comm_bytes, recomputed_comm_ops, resident_keys, offloaded_keys, recompute_keys | 一层保存张量在 resident、swap、recompute scratch 和通信重算之间的互斥划分。 |
| cost_eval/mem_timeline.py | 85 | 属性 | LayerMemoryPlan.resident | resident(self) | 返回前向结束后仍驻留在 HBM 的保存激活字节。 |
| cost_eval/mem_timeline.py | 87 | 属性 | LayerMemoryPlan.offloaded | offloaded(self) | 返回换出到 host、反向需预取的保存激活字节。 |
| cost_eval/mem_timeline.py | 89 | 属性 | LayerMemoryPlan.recomputed | recomputed(self) | 返回反向重算时同时物化的 scratch 字节。 |
| cost_eval/mem_timeline.py | 92 | 函数 | _unique_tensors | _unique_tensors(tensors) | 按 storage_key 去重张量，防止共享权重或同一数据流值重复计费。 |
| cost_eval/mem_timeline.py | 99 | 函数 | _layer_memory_plan | _layer_memory_plan(layer, recompute, swap) -> LayerMemoryPlan | 按 recompute/swap 选择把每层保存张量互斥划到 resident、offloaded、scratch 和通信重算。 |
| cost_eval/mem_timeline.py | 169 | 方法 | _layer_memory_plan.byte_sum | byte_sum(names) -> int | 把一组 storage key 对应的本地张量字节相加。 |
| cost_eval/mem_timeline.py | 185 | 函数 | _save_plan | _save_plan(layer, recompute, swap) | 对外兼容包装，返回 _layer_memory_plan 的结果。 |
| cost_eval/mem_timeline.py | 190 | 函数 | _layer_fsdp_buffer_bytes | _layer_fsdp_buffer_bytes(layer, pm) -> int | 汇总一层反分片/参数预取时需要的完整本地参数 buffer。 |
| cost_eval/mem_timeline.py | 203 | 函数 | _layer_grad_buffer_bytes | _layer_grad_buffer_bytes(layer, pm, gradient_bytes: int \| None=None) -> int | 汇总一层 reduce-scatter 前物化的完整梯度 buffer，尊重 FP32 main-grad。 |
| cost_eval/mem_timeline.py | 222 | 函数 | _layer_workspace | _layer_workspace(layer) -> int | 返回层内单算子 workspace+最大 collective staging 的最大值。 |
| cost_eval/mem_timeline.py | 231 | 函数 | _op_fsdp_buffer_bytes | _op_fsdp_buffer_bytes(op, pm) -> int | 计算一个 edge/普通算子的参数 all-gather buffer。 |
| cost_eval/mem_timeline.py | 240 | 函数 | _op_grad_buffer_bytes | _op_grad_buffer_bytes(op, pm, gradient_bytes: int \| None=None) -> int | 计算一个算子的完整梯度临时 buffer。 |
| cost_eval/mem_timeline.py | 251 | 函数 | _edge_saved_bytes | _edge_saved_bytes(ops) -> int | 按 storage_key 去重 embedding/final norm/lm head/loss 的非权重保存张量。 |
| cost_eval/mem_timeline.py | 260 | 类 | MemTimeline | 服务类/枚举 | 核心逐事件仿真器，沿每个 stage 的交错 1F1B 时间线寻找同时存活峰值。 |
| cost_eval/mem_timeline.py | 261 | 方法 | MemTimeline.simulate | simulate(self, graph, recompute, swap, pm, static_persistent, framework_reserve: int, max_device_memory: int, gradient_bytes: int \| None=None, optimizer=None) -> dict[int, StagePeak] | 逐 stage 回放交错 1F1B；在每次分配、gather、重算、换入、反向和 optimizer step 记录同时桶总和。 |
| cost_eval/memory_actions.py | 10 | 类 | MemoryAction | 字段: kind, key, size_bytes, bucket, owner, alias_of, op_name | 一条带所有者、字节、桶和别名信息的分配/释放/迁移动作。 |
| cost_eval/memory_actions.py | 19 | 方法 | MemoryAction.__post_init__ | __post_init__(self) -> None | 校验动作类型和非负字节。 |
| cost_eval/memory_actions.py | 26 | 函数 | forward_layer_actions | forward_layer_actions(layer, retained_keys: Iterable[str]) -> tuple[MemoryAction, ...] | 按最后使用点生成单层前向输入/输出/saves、kernel workspace 和通信 staging 的分配释放动作。 |
| cost_eval/memory_actions.py | 45 | 方法 | forward_layer_actions.ensure | ensure(tensor, index: int, role: str) -> None | 若非权重张量尚未存活则分配；保留值进 act_live，临时值进 workspace。 |
| cost_eval/memory_ledger.py | 11 | 类 | LedgerPeak | 字段: bytes, owner, buckets | MemoryLedger 回放期间观察到的局部最高存活字节及分桶快照。 |
| cost_eval/memory_ledger.py | 17 | 类 | MemoryLedger | 服务类/枚举 | 严格的存活值账本，拒绝重复分配、非法释放、悬空别名和未授权泄漏。 |
| cost_eval/memory_ledger.py | 18 | 方法 | MemoryLedger.__init__ | __init__(self) -> None | 初始化空的真实分配、别名、分桶和局部峰值。 |
| cost_eval/memory_ledger.py | 25 | 属性 | MemoryLedger.live_bytes | live_bytes(self) -> int | 返回当前所有真实分配的字节和；alias 不重复计费。 |
| cost_eval/memory_ledger.py | 29 | 属性 | MemoryLedger.buckets | buckets(self) -> dict[str, int] | 返回当前分桶副本。 |
| cost_eval/memory_ledger.py | 33 | 属性 | MemoryLedger.live_keys | live_keys(self) -> frozenset[str] | 返回当前真实分配 key 的只读集合。 |
| cost_eval/memory_ledger.py | 36 | 方法 | MemoryLedger._record | _record(self, owner: str) -> None | 若当前存活字节刷新最高值，则保存所有者和分桶快照。 |
| cost_eval/memory_ledger.py | 41 | 方法 | MemoryLedger.apply | apply(self, action: MemoryAction) -> None | 执行 ALLOC/FREE/ALIAS/MOVE，维护桶守恒并在动作后记录峰值。 |
| cost_eval/memory_ledger.py | 68 | 方法 | MemoryLedger.replay | replay(self, actions, allowed_live=frozenset()) -> LedgerPeak | 顺序执行动作并检查结束后除 allowed_live 外没有泄漏。 |
| cost_eval/offload_model.py | 9 | 类 | OffloadPolicy | 字段: parameter, gradient, optimizer | 参数、梯度和优化器三类状态的独立 CPU offload 开关。 |
| cost_eval/offload_model.py | 15 | 类方法 | OffloadPolicy.from_parallel | from_parallel(cls, pc) -> 'OffloadPolicy' | 从 ParallelConfig 的三个 offload 开关构造策略。 |
| cost_eval/offload_model.py | 22 | 方法 | OffloadPolicy.split | split(self, breakdown) | 把 StaticBreakdown 拆成 HBM resident 与 host pinned 两份。 |
| cost_eval/optimizers.py | 12 | 类 | OptimizerStateBytes | 字段: parameter, gradient, master_weight, optimizer_state | 一个本地权重 shard 的参数、梯度、master weight 和优化器状态字节。 |
| cost_eval/optimizers.py | 19 | 函数 | state_bytes | state_bytes(weight, optimizer) -> OptimizerStateBytes | 按本地权重元素数、dtype、trainable 和 AdamW/Muon 分区计算四类持久字节。 |
| cost_eval/optimizers.py | 46 | 函数 | optimizer_step_workspace_bytes | optimizer_step_workspace_bytes(optimizer, largest_gradient_bytes: int, optimizer_offload: bool=False) -> int | AdamW 取最大梯度 chunk；Muon 取其 3 倍；optimizer offload 再加一个预取 chunk。 |
| cost_eval/shape_eval.py | 22 | 函数 | eval_expr | eval_expr(expr: DimExpr, dims: DimTable) -> int | 用受限 AST 求值整数维度表达式，只允许已知符号和 +、-、*、整除。 |
| cost_eval/shape_eval.py | 31 | 方法 | eval_expr.evaluate | evaluate(node: ast.AST) -> Union[int, float] | 递归解释受限表达式 AST，并拒绝未知节点或非整除。 |
| cost_eval/shape_eval.py | 65 | 类 | Placement | 字段: shard, partial | 张量在各 mesh 轴上的分片位置以及 partial 状态。 |
| cost_eval/shape_eval.py | 69 | 方法 | Placement.__init__ | __init__(self, shard=(), partial=None) | 把 shard 映射排序并冻结为元组，同时保存 partial 轴。 |
| cost_eval/shape_eval.py | 75 | 方法 | Placement.of | of(tensor: TensorRef) -> 'Placement' | 从 TensorRef 的 shard/partial 构造 Placement。 |
| cost_eval/shape_eval.py | 79 | 属性 | Placement.shards | shards(self) | 返回排序后的分片轴元组。 |
| cost_eval/shape_eval.py | 84 | 类 | CommSpec | 字段: ctype, volume_bytes, group_axis, phase, output_tensor, rank_group, topology_class, algorithm, output_value_id | 一次 collective 的类型、字节量、通信轴、阶段和输出值身份。 |
| cost_eval/shape_eval.py | 98 | 属性 | CommSpec.kind | kind(self) | 兼容属性，返回 collective 类型。 |
| cost_eval/shape_eval.py | 102 | 函数 | detect_reshards | detect_reshards(src: Optional[Placement], dst: Placement, numel: Optional[int]=None, dtype_bytes: Optional[int]=None) -> tuple[CommSpec, ...] | 比较源/目标 placement：partial 先 reduce，去分片 all-gather，同轴换维 all-to-all，加分片本地完成。 |
| cost_eval/shape_eval.py | 147 | 函数 | detect_reshard | detect_reshard(src: Optional[Placement], dst: Placement, numel: Optional[int]=None, dtype_bytes: Optional[int]=None) -> Optional[CommSpec] | 兼容旧调用方，只返回 detect_reshards 的第一条通信。 |
| cost_eval/shape_eval.py | 160 | 类 | ResolvedTensor | 字段: name, global_shape, local_shape, local_numel, dtype_bytes, is_weight, is_expert, placement, storage_id, trainable, swappable, recomputable, fsdp_replicated, logical_name, value_id | 已代入真实整数和并行切分后的本地张量，可直接给出每卡字节。 |
| cost_eval/shape_eval.py | 178 | 属性 | ResolvedTensor.local_bytes | local_bytes(self) -> int | 返回 local_numel*dtype_bytes。 |
| cost_eval/shape_eval.py | 182 | 属性 | ResolvedTensor.tid | tid(self) | 显示兼容标识，返回张量名；计费身份使用 storage_key。 |
| cost_eval/shape_eval.py | 187 | 属性 | ResolvedTensor.storage_key | storage_key(self) -> str | 按 storage_id、value_id、name 的优先级返回去重身份。 |
| cost_eval/shape_eval.py | 196 | 函数 | _placement_value_id | _placement_value_id(tensor: TensorRef, placement: Placement) -> str | 把名称、切分、partial 和 dtype 编码成稳定的数据流 value id。 |
| cost_eval/shape_eval.py | 205 | 函数 | resolve_tensor | resolve_tensor(tensor: TensorRef, dims: DimTable, pm) -> ResolvedTensor | 代入全局 shape、CP/SP/TP/EP 切分和 dtype，生成 ResolvedTensor 的本地 shape/字节。 |
| cost_eval/shape_eval.py | 265 | 类 | ResolvedOp | 字段: name, type, inputs, output, params, saves, workspace_bytes, collectives, module_paths, attrs, backward_workspace_bytes, workspace_source, workspace_upper_bytes | 已解析算子，包含本地输入/输出/参数/saves、collective 和前后向 workspace。 |
| cost_eval/shape_eval.py | 282 | 类 | ResolvedLayer | 字段: layer_id, layer_type, ops | 带全局 layer_id 的已解析算子序列，并能汇总保存激活和 checkpoint。 |
| cost_eval/shape_eval.py | 288 | 属性 | ResolvedLayer.activation_bytes | activation_bytes(self) | 按 storage_key 去重并汇总该层所有反向保存激活。 |
| cost_eval/shape_eval.py | 293 | 属性 | ResolvedLayer.checkpoint_bytes | checkpoint_bytes(self) | 返回该层首个非权重输入的本地字节，供 PP 边界与 full recompute 使用。 |
| cost_eval/shape_eval.py | 299 | 类 | ResolvedGraph | 字段: stages, stage_params, stage_output_bytes, edge_ops | 按 PP stage 组织的 decoder 层、edge 模块、参数和输出字节图。 |
| cost_eval/shape_eval.py | 306 | 类 | ShapeEval | 服务类/枚举 | 把声明式 ModelSpec 解析成每卡 ResolvedGraph。 |
| cost_eval/shape_eval.py | 307 | 方法 | ShapeEval.__init__ | __init__(self, workspace_registry=None) | 可选接收精确 kernel workspace registry。 |
| cost_eval/shape_eval.py | 310 | 方法 | ShapeEval.resolve | resolve(self, spec: ModelSpec, pm) -> ResolvedGraph | 解析每层算子与 edge 模块，插入 placement/CP/EP 通信，计算 workspace，并按 PP stage 组图。 |
| cost_eval/static_mem.py | 9 | 类 | StaticBreakdown | 字段: parameter, gradient, master_weight, optimizer_state | 持久态四桶：参数、梯度、master weight、优化器状态。 |
| cost_eval/static_mem.py | 16 | 属性 | StaticBreakdown.total | total(self) | 返回参数+梯度+master weight+优化器状态。 |
| cost_eval/static_mem.py | 21 | 类 | StageStaticMemory | 字段: stage, persistent_bytes, breakdown, edge_parameter_bytes, decoder_persistent_bytes, host_pinned_bytes, host_breakdown | 单 stage 的 HBM 持久态、edge 参数、decoder 持久态和 host pinned 拆分。 |
| cost_eval/static_mem.py | 31 | 属性 | StageStaticMemory.total | total(self) | 返回该 stage 的 HBM persistent_bytes。 |
| cost_eval/static_mem.py | 32 | 方法 | StageStaticMemory.__int__ | __int__(self) | 把对象转换为 persistent_bytes。 |
| cost_eval/static_mem.py | 33 | 方法 | StageStaticMemory.__eq__ | __eq__(self, other) | 支持与整数或同类对象比较。 |
| cost_eval/static_mem.py | 37 | 方法 | StageStaticMemory.__floordiv__ | __floordiv__(self, other) | 兼容旧接口，对 decoder persistent bytes 做整除。 |
| cost_eval/static_mem.py | 41 | 函数 | _unique_layer_params | _unique_layer_params(layer) | 按 storage_key 去重一层全部算子参数。 |
| cost_eval/static_mem.py | 51 | 函数 | _state_bytes | _state_bytes(weight, optimizer) -> StaticBreakdown | 调用 optimizer state_bytes 并转换为 StaticBreakdown。 |
| cost_eval/static_mem.py | 61 | 函数 | _add_breakdown | _add_breakdown(left, right) | 逐字段相加两个 StaticBreakdown。 |
| cost_eval/static_mem.py | 68 | 函数 | _sharded_group_state | _sharded_group_state(weights, degree, alignment_bytes, optimizer) | 把同 dtype/训练属性的扁平参数组按 degree 和 alignment padding 后计算单 shard 持久态。 |
| cost_eval/static_mem.py | 85 | 类 | StaticMem | 服务类/枚举 | 按 stage、权重组和 FSDP 对齐规则计算持久态。 |
| cost_eval/static_mem.py | 86 | 方法 | StaticMem.compute | compute(self, graph, optimizer, pm, cpu_offload: bool=None) | 逐 stage 汇总 decoder/edge 参数组，应用 FSDP/eFSDP、共享存储去重和 CPU offload。 |

### 报告、测量与校准：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/calibration.py | 16 | 类 | MetricComparison | 字段: predicted, measured, signed_error, absolute_error, relative_error | 一对预测值和实测值的有符号误差、绝对误差与相对误差。 |
| cost_eval/calibration.py | 24 | 类方法 | MetricComparison.compare | compare(cls, predicted: int, measured: int) -> 'MetricComparison' | 计算 predicted-measured、有符号/绝对误差以及以 measured 为分母的相对误差。 |
| cost_eval/calibration.py | 33 | 类 | StageCalibration | 字段: stage, peak, predicted_event, measured_event, event_match, predicted_breakdown, bucket_errors | 一个 stage 的峰值、事件、分桶误差对账结果。 |
| cost_eval/calibration.py | 44 | 类 | CaseCalibration | 字段: name, stages, predicted_oom, measured_oom, oom_match, framework_reserve, metadata, peak_stage_order_match | 一个 profiling case 的逐 stage 对账、OOM 混淆结果和元数据。 |
| cost_eval/calibration.py | 56 | 类 | CalibrationSummary | 字段: case_count, stage_count, mean_relative_error, p50_relative_error, p90_relative_error, max_relative_error, mean_signed_bias, within_target_rate, target_relative_error, peak_event_match_rate, oom_true_positives, oom_true_negatives, oom_false_positives, oom_false_negatives, suggested_framework_reserve, peak_stage_order_match_rate | 多个 case 的均值/P50/P90/最大误差、事件命中率和 OOM 混淆矩阵汇总。 |
| cost_eval/calibration.py | 76 | 类 | CalibrationReport | 字段: cases, summary | 完整校准报告，由逐 case 明细和总体摘要组成。 |
| cost_eval/calibration.py | 81 | 函数 | _percentile | _percentile(values: Sequence[float], quantile: float) -> Optional[float] | 对排序样本做线性插值分位数。 |
| cost_eval/calibration.py | 92 | 函数 | _event_phase | _event_phase(event: Optional[str]) -> Optional[str] | 把具体峰值事件归一成 forward/backward 或原始阶段名。 |
| cost_eval/calibration.py | 103 | 函数 | _event_matches | _event_matches(predicted: str, measured: Optional[str]) -> Optional[bool] | 精确事件相同或前后向阶段相同即算匹配；无实测事件返回 None。 |
| cost_eval/calibration.py | 109 | 函数 | _stage_order_match | _stage_order_match(predicted_stages, measured_peaks: Mapping[int, int]) -> bool | 按峰值从高到低比较预测与实测的 PP stage 排序，stage id 用于稳定处理并列。 |
| cost_eval/calibration.py | 132 | 函数 | _normalize_stage_mapping | _normalize_stage_mapping(value: Mapping[Any, Any], field_name: str) -> dict[int, Any] | 把任意可转整数的 stage key 规范为 int key。 |
| cost_eval/calibration.py | 147 | 类 | CalibrationRunner | 服务类/枚举 | 执行预测与实测对账、聚合误差统计并读取批量 profiling 配置。 |
| cost_eval/calibration.py | 148 | 方法 | CalibrationRunner.__init__ | __init__(self, target_relative_error: float=0.1) | 设置并校验目标相对误差阈值。 |
| cost_eval/calibration.py | 153 | 方法 | CalibrationRunner.run_case | run_case(self, name: str, inputs, measured: Mapping[str, Any], metadata: Optional[Mapping[str, Any]]=None) -> CaseCalibration | 运行一个预测，逐 stage 对齐峰值/事件/桶并判断 OOM 与 stage 排序是否匹配。 |
| cost_eval/calibration.py | 287 | 方法 | CalibrationRunner.run_cases | run_cases(self, cases: Sequence[CaseCalibration]) -> CalibrationReport | 聚合多 case 的误差分位数、事件命中率、OOM 混淆矩阵和建议 reserve。 |
| cost_eval/calibration.py | 362 | 方法 | CalibrationRunner.run_profile | run_profile(self, path) -> CalibrationReport | 读取批量 profiling 文件，选择 native/MindFormers adapter 并运行所有 case。 |
| cost_eval/calibration.py | 434 | 函数 | main | main(argv=None) -> int | 解析命令行参数，运行相应评估或校准，并输出 JSON；阈值失败时返回非零退出码。 |
| cost_eval/measurement_contract.py | 12 | 函数 | _int_mapping | _int_mapping(value: Mapping[Any, Any], field: str) -> dict[int, int] | 把测量映射的 key/value 转为非负整数。 |
| cost_eval/measurement_contract.py | 29 | 类 | HBMComponents | 字段: device_baseline_bytes, model_active_peak_bytes, allocator_pool_peak_bytes, untracked_runtime_bytes | 同一采集窗口内互斥的设备基线、模型 active、allocator pool 和未追踪运行时分量。 |
| cost_eval/measurement_contract.py | 37 | 方法 | HBMComponents.__post_init__ | __post_init__(self) -> None | 要求四个互斥 HBM 分量全部非负。 |
| cost_eval/measurement_contract.py | 42 | 属性 | HBMComponents.dynamic_total_bytes | dynamic_total_bytes(self) -> int | 返回 max(model active, allocator pool) + untracked runtime。 |
| cost_eval/measurement_contract.py | 48 | 属性 | HBMComponents.nominal_total_bytes | nominal_total_bytes(self) -> int | 返回 device baseline + dynamic total。 |
| cost_eval/measurement_contract.py | 51 | 方法 | HBMComponents.validate_dynamic_total | validate_dynamic_total(self, measured: int, tolerance_bytes: int=2 * 2 ** 20) -> None | 检查互斥组件合成值与实测动态峰值在容差内守恒。 |
| cost_eval/measurement_contract.py | 66 | 类 | RankHBMMeasurement | 字段: rank, stage, baseline_bytes, peak_used_bytes, peak_delta_bytes, sample_count | 单个 rank 的 stage、空闲基线、峰值、增量和采样数量。 |
| cost_eval/measurement_contract.py | 74 | 方法 | RankHBMMeasurement.__post_init__ | __post_init__(self) -> None | 校验 rank/stage/sample 和字节非负，并要求 peak_used-baseline 与 peak_delta 在 1 MiB 内守恒。 |
| cost_eval/measurement_contract.py | 91 | 类 | HBMMeasurement | 字段: name, per_stage_peak_bytes, per_rank, components, oom, quality_status, quality_issues, collection_attempt_id | 一个真实 NPU case 的逐 stage/逐 rank HBM、组件、OOM 与测量质量。 |
| cost_eval/measurement_contract.py | 102 | 类方法 | HBMMeasurement.from_case_result | from_case_result(cls, root: Mapping[str, Any]) -> 'HBMMeasurement' | 从 collector case_result 解析 stage/rank/组件/质量并立即检查分量守恒。 |
| cost_eval/measurement_contract.py | 181 | 属性 | HBMMeasurement.worst_dynamic_bytes | worst_dynamic_bytes(self) -> int | 返回最大的逐 stage 动态 HBM；无 stage 时为 0。 |
| cost_eval/measurement_contract.py | 187 | 属性 | HBMMeasurement.worst_total_bytes | worst_total_bytes(self) -> int | 优先返回逐 rank 最大 peak_used；否则返回组件 nominal total。 |
| cost_eval/measurement_contract.py | 192 | 方法 | HBMMeasurement.eligibility_issues | eligibility_issues(self, expected_rank_count: Optional[int]=None, min_samples: int=20) -> tuple[str, ...] | 汇总质量状态、rank 数、采样数和缺失逐 rank 数据等不可用于校准的问题。 |
| cost_eval/measurement_contract.py | 211 | 方法 | HBMMeasurement.validate_stage_aggregation | validate_stage_aggregation(self, tolerance_bytes: int=2 ** 20) -> None | 检查每个 stage 的 max(rank peak_delta) 与保存的 stage peak 一致。 |
| cost_eval/report.py | 15 | 类 | PeakMemoryReport | 字段: per_stage, tightest_stage, oom, static_per_stage, summary, warnings, schema_version, tightest_rank, tightest_event, oom_status, capability_status, fingerprints, per_rank | 评估器最终报告：逐 stage/rank 峰值、最紧位置、OOM、能力状态、告警和指纹。 |
| cost_eval/report.py | 32 | 类 | ModelSummary | 字段: name, num_layers, world_size | 模型名、总层数和 world size 的简要信息。 |
| cost_eval/report.py | 39 | 类 | RankPeak | 字段: rank, stage, peak_bytes, safe_upper_bytes, peak_event, oom_status | 单 rank 继承其物理 stage 的峰值、上界、事件和 OOM 状态。 |
| cost_eval/report.py | 48 | 类 | Evaluator | 服务类/枚举 | 顶层评估门面，串联并行解析、形状解析、静态态、时间线和 allocator 组合。 |
| cost_eval/report.py | 51 | 方法 | Evaluator.__init__ | __init__(self, model_spec, parallel_config, optimizer, hardware, recompute=None, swap=None, warnings=()) | 保存六类输入，为缺省 recompute/swap 建空策略，并把旧 framework_reserve 映射告警加入报告。 |
| cost_eval/report.py | 77 | 方法 | Evaluator.evaluate | evaluate(self) -> PeakMemoryReport | 执行完整 HBM 流程，生成逐 stage/rank 点估计、上界、最紧事件与 OOM/能力状态。 |
| cost_eval/report.py | 174 | 方法 | Evaluator._capability_status | _capability_status(self) -> str | 按源码合同、V4 probe/HBM 验证和硬件/runtime/source profile 决定 unsupported/experimental/validated。 |
| cost_eval/validation_gate.py | 15 | 类 | GateResult | 字段: name, status, observed, requirement, scope | 一个验收门的名称、状态、观测值、阈值和证据范围。 |
| cost_eval/validation_gate.py | 23 | 函数 | _metric_gate | _metric_gate(name: str, values: Mapping[str, Any], mean_limit: float, worst_limit: float) -> GateResult | 按 mean/worst APE 阈值生成 pass/fail/unavailable 数值门。 |
| cost_eval/validation_gate.py | 47 | 函数 | assess_hbm_gates | assess_hbm_gates(offline: Mapping[str, Any], calibrated: Mapping[str, Any]) -> Mapping[str, Any] | 核对 offline/calibrated 标签一致性，评估 P0/P1 数值/覆盖率并显式保留生产证据缺口。 |
| cost_eval/validation_gate.py | 123 | 函数 | render_gate_markdown | render_gate_markdown(result: Mapping[str, Any]) -> str | 把 GateResult 列表渲染为简洁 Markdown 表。 |

### 运行时源码合同：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/source_contracts/deepseek_v4.py | 131 | 函数 | deepseek_v4_attention_variant | deepseek_v4_attention_variant(model: Mapping[str, Any]) -> str | 把 V4 experimental_attention_variant 归一化，缺省为 dsv4_hybrid。 |
| cost_eval/source_contracts/deepseek_v4.py | 137 | 函数 | deepseek_v4_compress_ratios | deepseek_v4_compress_ratios(model: Mapping[str, Any], n_layers: int, n_mtp_layers: int) -> tuple[int, ...] | 解析覆盖 decoder+MTP 的逐层压缩比；缺省 decoder=128、MTP=0。 |
| cost_eval/source_contracts/deepseek_v4.py | 156 | 函数 | validate_deepseek_v4_contract | validate_deepseek_v4_contract(model: Mapping[str, Any], dims: Any, ratios: Sequence[int]) -> None | 拒绝未覆盖的 V4 变体、维度、比例、HC head、非融合 attention 和非法 hash/indexer 配置。 |

### 运行时校准与溯源：类与函数

_签名来自 AST；说明按当前实现语义编写。_

| 文件 | 行 | 类型 | 符号 | 签名/字段 | 功能 |
| --- | --- | --- | --- | --- | --- |
| cost_eval/runtime_profiles.py | 11 | 函数 | layout_key | layout_key(pc) -> str | 把 tp/cp/pp/ep/dp/interleave 编码为 runtime profile 布局键。 |
| cost_eval/runtime_profiles.py | 19 | 类 | RuntimeProfile | 字段: family, layout, sample_count, runtime_point, runtime_p90, pool_slack_point, pool_slack_p90, baseline_point, baseline_p90, physical_shortfall_p90, label_sha256 | 一个模型家族+并行布局的实测 runtime/pool/baseline 分位数档案。 |
| cost_eval/runtime_profiles.py | 33 | 类 | RuntimeProfileRegistry | 服务类/枚举 | 加载并选择 runtime profile；缺档时显式标 OOD 并扩大上界。 |
| cost_eval/runtime_profiles.py | 34 | 方法 | RuntimeProfileRegistry.__init__ | __init__(self, profiles, profile_id: str) | 按 (family, layout) 建立 profile 索引并保存可审计 profile id。 |
| cost_eval/runtime_profiles.py | 39 | 类方法 | RuntimeProfileRegistry.load | load(cls, path) -> 'RuntimeProfileRegistry' | 读取 v1 JSON 档案并用文件内容 SHA-256 生成 profile id。 |
| cost_eval/runtime_profiles.py | 61 | 方法 | RuntimeProfileRegistry.apply | apply(self, hardware, family: str, pc) | 匹配家族+布局并写入点估计/上界；缺失时标 OOD 且至少增加 2 GiB。 |
| cost_eval/source_fingerprint.py | 9 | 函数 | python_source_tree_sha256 | python_source_tree_sha256(root) -> str | 按相对路径和文件内容稳定排序，计算整个 Python 源码树指纹。 |
| cost_eval/workspace_registry.py | 9 | 类 | WorkspaceKey | 字段: family, op_type, fusion_variant, dtype_bytes, local_shape_bucket, tp, cp, ep, mindspore_version, cann_version, hardware | kernel workspace 的精确匹配键：家族、算子、融合、shape、并行、版本和硬件。 |
| cost_eval/workspace_registry.py | 24 | 类 | WorkspaceProfile | 字段: key, point_bytes, upper_bytes, source, sample_count, content_sha256 | workspace 点估计/上界及样本数、来源和内容 hash。 |
| cost_eval/workspace_registry.py | 32 | 方法 | WorkspaceProfile.__post_init__ | __post_init__(self) -> None | 校验 point<=upper、正样本数和 64 位内容 SHA-256。 |
| cost_eval/workspace_registry.py | 39 | 类 | WorkspaceRegistry | 服务类/枚举 | WorkspaceKey 到 WorkspaceProfile 的精确查找表。 |
| cost_eval/workspace_registry.py | 40 | 方法 | WorkspaceRegistry.__init__ | __init__(self, profiles=()) | 把 profile 列表索引为完整 WorkspaceKey 到 profile 的精确映射。 |
| cost_eval/workspace_registry.py | 43 | 方法 | WorkspaceRegistry.lookup | lookup(self, key: WorkspaceKey) -> WorkspaceProfile \| None | 按完整 WorkspaceKey 精确查找；不做近似或最近邻匹配。 |

## 当前实现的边界会直接影响结论可信度

1. **顶层 Evaluator 尚未注入 WorkspaceRegistry。** `ShapeEval` 支持 profile registry，但 `Evaluator.evaluate()` 当前调用的是 `ShapeEval()`，所以正常顶层路径仍使用算子公式 workspace；只有自定义解析路径显式传 registry 才会命中精确档案。
2. **调度只支持 1F1B。** `ParallelConfig` 会拒绝其他 pipeline schedule；interleave 是在 rank-local 1F1B 上按 virtual chunk 展开，不模拟算子耗时、通信延迟或跨 rank 亚事件同步。
3. **峰值是合同模型，不是 allocator trace 重放。** 碎片按峰值快照的七个桶分别做粒度对齐，不是逐次真实 malloc/free 的地址级碎片。
4. **workspace 公式可能随 kernel、融合、MindSpore/CANN 和硬件改变。** 精确 profile 缺失时使用公式 fallback；这就是 runtime/source fingerprint 和安全上界存在的原因。
5. **能力状态刻意保守。** DeepSeek-V4 在 probe 未通过或 HBM 未 validated 时直接 unsupported；Qwen3/V3 的历史 holdout 也不能自动升级为生产 validated。
6. **只计算单训练 step 的显存峰值。** 不计算执行时间、带宽瓶颈、host RAM 容量、NVMe swap 时延或长期内存泄漏。
7. **microbatch 字段要区分。** 时间线使用 `num_microbatches`；`microbatch` 配置字段本身不参与当前事件数计算。

这些限制不是可忽略的注脚：在新模型家族、新 kernel 版本、未知并行布局或接近容量边界时，应把结果视为诊断性上界，而不是部署承诺。

## 建议优先补齐三件事

1. 把 `WorkspaceRegistry` 正式接入 `Evaluator` 和两类 adapter，并在报告 fingerprints 中记录 workspace profile id。
2. 用新的冻结标签补齐每个声明支持的模型家族/布局：至少包括逐 rank 样本、真实 OOM、峰值 stage/rank 排序和策略 paired delta。
3. 给调度模型增加与真实 runtime trace 对齐的事件版本，特别验证 PP interleave、FSDP prefetch 和 `reshard_after_forward=False` 的并发窗口。

完成后再按 `validation_gate` 升级能力状态；不要通过放大一个无界 residual 或复用别的模型家族档案来“通过”验收。

## 后续需要持续回答的问题

- 不同 MindSpore/CANN 版本下，哪些算子的 workspace 公式变化足以移动 peak_event？
- 真实 interleaved 1F1B 中，参数预取、通信和 backward kernel 的重叠是否比当前 rank-local 模型更高？
- tied embedding/output weight 在跨 PP stage 时的运行时持有方式，是否需要显式共享/复制合同？
- CPU offload 与 activation swap 的 host pinned 峰值和带宽约束，应如何与 HBM 安全判断联合报告？
- 当 allocator pool 已包含 rounded active allocations 时，当前额外分桶 fragmentation 是否会在某些 runtime profile 上重复计入？

<!-- generated_at=2026-08-20T09:27:53+08:00; cost_eval_source_sha256=fab1f04f724aea984b15ecc8a84c68930d1a1eb33986e6d7448202a840261840 -->
