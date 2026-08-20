# P0/P1 HBM 物理模型修正实施计划

> 日期：2026-08-19
> 范围：MindFormers PyNative；Qwen3、DeepSeek V3、DeepSeek V4；Ascend 910B2 × 8 首期验收
> 目标：修正当前物理模型的系统性低估、布局错误和 OOM 不安全输出，并完成 P0/P1 真机验收
> 约束：P0/P1 不使用无界学习残差掩盖物理模型错误；残差模型留到物理基线过门后单独实施

> 2026-08-19 执行状态：M0–M4 的离线代码、合同测试和历史校准回放已完成；
> M5 的新 frozen holdout、实际 OOM、策略配对和 DeepSeek-V4 真机门因当前禁止
> 使用 NPU 而未执行。当前不能标记 P0/P1 production-validated，详见
> `validation/real_npu/MODEL_CARD.md` 与 `validation_gates_v1.{json,md}`。

## 1. 完成定义

本计划不是以“代码合入”作为完成，而是同时满足以下四类条件：

1. 新物理账本能逐 rank、逐事件解释 peak，所有 allocation 有对应 free；
2. loss、allocator/runtime、placement、PP/interleave/FSDP 等关键分量不再由单一常数替代；
3. Qwen3、DeepSeek V3、DeepSeek V4 和关键策略具有源码合同、probe 或明确的 unsupported 状态；
4. 新冻结的 real-NPU holdout 和 OOM blind 达到本文验收门。

P0 完成门：

- physical dynamic HBM mean APE ≤ 25%，worst APE ≤ 35%；
- 所有已支持配置无 OOM false-safe；
- Qwen3 loss 主导配置、DeepSeek V3 TP+CP+EP、PP+interleave 至少各有一个有效新 holdout；
- 输出区分 physical active、allocator pool、runtime、total point 和 safe upper。

P1 完成门：

- total HBM mean APE ≤ 15%，worst APE ≤ 20%；
- dynamic HBM mean APE ≤ 20%，worst APE ≤ 30%；
- safe upper 覆盖率 ≥ 80%，stage/rank 峰值排序正确率 ≥ 80%；
- OOM 分类 100%，false-safe 为 0，至少 1 个正式实际 OOM；
- recompute、offload、FSDP/HSDP、PP/interleave、EP/dispatcher 的配对节省量误差 ≤ 20%；
- 每个模型家族至少 2 个有效 frozen holdout，DeepSeek V4 至少一个包含 PP+EP+interleave。

## 2. 不包含的工作

- 训练时间、通信耗时或自动并行搜索；
- 在物理基线未通过 P0 门之前拟合 feature residual；
- 未经 source contract/probe 的任意自定义算子和优化器；
- 在未验收硬件、MindSpore/CANN 大版本上复用高置信度结论。

## 3. 目标架构

```text
MindFormers YAML + runtime/source/hardware fingerprint
                         │
                         ▼
Family Source Contract ──┬── Tensor value/storage identity
                         ├── Multi-axis placement + reshard plan
                         ├── Op forward/backward memory actions
                         └── Loss/optimizer/offload contracts
                                      │
                                      ▼
                        Rank-aware event schedule
                                      │
                                      ▼
                     Active allocation/free ledger
                                      │
                    ┌─────────────────┴────────────────┐
                    ▼                                  ▼
             physical active peak              allocator/runtime model
                    └─────────────────┬────────────────┘
                                      ▼
                    point / safe upper / OOM / confidence
```

旧字段兼容原则：

- `StagePeak.peak_bytes` 暂保留，定义为 `total_point_bytes` 的兼容别名；
- `framework_reserve` 暂接受，但转换为 legacy `untracked_runtime_point` 并输出弃用 warning；
- 报告新增 schema version；旧配置可运行，但不能静默报告为高置信度；
- 每次数据结构迁移都先加 compatibility adapter，再切换生产路径，最后删除旧内部逻辑。

## 4. 里程碑与依赖

| 里程碑 | 内容 | 依赖 | 退出条件 |
|---|---|---|---|
| M0 | 冻结当前基线和测量合同 | 无 | 当前代码可对全部有效标签生成可审计报告 |
| M1 / P0-core | identity、placement、ledger、loss | M0 | Qwen/DS 缩层 golden 和账本守恒通过 |
| M2 / P0-safe | allocator、rank、PP/interleave/FSDP、OOM 输出 | M1 | 新 P0 holdout 达到 physical gate，零 false-safe |
| M3 / P1-family | Qwen/DS3/DS4、MoE、dtype/workspace 合同 | M2 | 三家族 source/probe 软件门通过 |
| M4 / P1-strategy | offload、Swap、optimizer、策略配对 | M3 | 所有策略 delta 门通过 |
| M5 / P1-validation | frozen holdout、OOM blind、model card | M4 | P1 全部门通过 |

不允许跳过的依赖：

- loss 必须运行在新 ledger 上，避免再增加不可解释的静态 workspace；
- allocator model 必须消费 active ledger，避免 active 与 reserved 重复相加；
- family/MoE/offload 只在 identity 和 placement 语义稳定后迁移；
- residual 只能在 M5 完成后开始。

## 5. M0：冻结基线和证据

### Task M0.1：建立“当前代码”统一基线命令

**文件：**

- Create: `tools/evaluate_current_hbm.py`
- Create: `cost_eval/measurement_contract.py`
- Modify: `tools/analyze_real_npu.py`
- Test: `tests/test_cost_eval/test_measurement_contract.py`

**工作项：**

- [ ] 从 `validation/real_npu/manifest.json` 读取配置、标签、split 和 SHA-256；
- [ ] 按 HBM label hash 去重，禁止 calibration/历史 holdout 别名重复计分；
- [ ] 使用当前 `MindFormersAdapter + Evaluator` 重新预测，而不是只读取相邻仓库的 `physical_base`；
- [ ] 同时输出 raw active、加 legacy reserve 后结果、measured active/pool/runtime/total；
- [ ] 保存 family/layout/split、signed bias、APE、stage/rank 排序和数据质量状态；
- [ ] diagnostic-only、中断、污染或无 `case_result.json` 的记录不得进入正式指标。

**验收：**

- [ ] 一条命令可重建基线 JSON/Markdown；
- [ ] 结果包含配置、代码、合同和标签 hash；
- [ ] 当前 pytest、P0 conformance、functional test 继续通过；
- [ ] 基线报告作为后续每个 task 的 before/after 对照，禁止覆盖历史版本。

### Task M0.2：建立组件级 golden fixture

**文件：**

- Create: `tests/fixtures/hbm_components/`
- Create: `tests/test_cost_eval/test_component_goldens.py`

**工作项：**

- [ ] 从现有 Qwen3 DP8、Qwen3 TP4+DP2、DeepSeek V3 TP2+CP2+EP2、PP2+interleave 中选择最小代表集；
- [ ] 固定 model active、allocator active/reserved、untracked runtime、per-rank total；
- [ ] fixture 只保存小型互斥摘要和来源 hash，不复制大 profiler；
- [ ] golden 区分“标签对账测试”和“模型正确性测试”，不得要求 calibration case 误差为零。

## 6. M1：P0-core 修正

### Task P0.1：重构 tensor identity 和多轴 placement

**文件：**

- Modify: `cost_eval/model_spec.py`
- Modify: `cost_eval/shape_eval.py`
- Modify: `cost_eval/layers/*.py`
- Test: `tests/test_cost_eval/test_tensor_identity.py`
- Test: `tests/test_cost_eval/test_placement_algebra.py`

**目标数据结构：**

```text
TensorValueRef:
  value_id                 # 数据流中的唯一值/版本
  logical_name             # 可读名称
  storage_id               # 仅用于真实 alias/shared storage
  shape / dtype
  placement                # 每个 mesh axis 独立表达 shard/replicate/partial

ResolvedValue:
  value_id + placement_version
  local_shape / local_bytes
  alias_of / owner_event
```

**工作项：**

- [ ] 禁止以 `tensor.name` 作为 activation 去重键；
- [ ] 同名的 SP、CP-only、TP gathered、partial 和 replicated 值必须拥有不同 `value_id`；
- [ ] `storage_id` 只在 profiler/source 已证明 alias 时去重；
- [ ] placement 支持同一张量同时在 CP 和 TP/SP 等不同 mesh 轴上分布；
- [ ] reshard 根据 source/destination 的轴差集生成 0..N 个 collective，而不是统一 shard→shard=all-to-all；
- [ ] 正确覆盖 `SP(TP)+CP → CP-only`、`Partial(TP) → SP(TP)+CP`、EP dispatch/combine、hybrid CP；
- [ ] collective 明确 input/output 是否 alias、分配字节和释放事件；
- [ ] 对 `S//ratio` 等表达式增加整除或 runtime padding 合同，禁止静默截断。

**回归案例：**

- [ ] `mla_q_a_norm` gather 前后不能互相吞掉；
- [ ] `mla_compressed_kv_norm`、`mla_k_pe`、V4 `v4_query` 同名多 placement 正确；
- [ ] CP 增大时保留 CP shard，不因 TP all-gather 变成全局 S；
- [ ] placement 转换的全局 numel/本地 numel/collective 输出守恒；
- [ ] alias storage 只计一份，非 alias 同名 value 各自计数。

**验收：**

- [ ] 所有 placement golden 通过；
- [ ] 禁止依赖 Python set 迭代顺序选择通信轴；
- [ ] 当前 Qwen3/DS3 参数总量和静态切分守恒不退化。

### Task P0.2：建立逐算子 allocation/free ledger

**文件：**

- Create: `cost_eval/memory_actions.py`
- Create: `cost_eval/memory_ledger.py`
- Create: `cost_eval/event_schedule.py`
- Modify: `cost_eval/model_spec.py`
- Modify: `cost_eval/mem_timeline.py`
- Test: `tests/test_cost_eval/test_memory_ledger.py`
- Test: `tests/test_cost_eval/test_event_schedule.py`

**工作项：**

- [ ] Op contract 分别声明 forward/backward 的 output、saved、workspace、communication 和临时张量；
- [ ] 生成 `ALLOC/FREE/ALIAS/MOVE_IN/MOVE_OUT` 事件；
- [ ] 当前层已保存 activation 必须与后续 op workspace 同时存活；
- [ ] backward 逐 op 模拟 recompute、saved consumption、dgrad/wgrad、grad reduce-scatter 和 workspace；
- [ ] PP boundary activation、send/recv staging 和 edge module 生命周期进入同一 ledger；
- [ ] optimizer step 先预留事件接口，P1 再填具体优化器合同；
- [ ] 每 rank ledger 结束时 transient bytes 回零，persistent 单独核对；
- [ ] breakdown 取真实 peak event 的同时值，不允许把各 bucket 独立峰值相加伪造总峰值。

**验收：**

- [ ] 每个 allocation 有且只有一个 owner/free 规则；
- [ ] 无负 bucket、重复 free 或 leaked transient；
- [ ] 当前层 saved/workspace overlap、backward workspace、PP send/recv 有独立测试；
- [ ] recompute/resident/offloaded 三态保持互斥；
- [ ] 旧 `MemTimeline` 对外入口保留兼容，但内部由新 schedule+ledger 驱动。

### Task P0.3：实现完整 loss/logits 合同

**文件：**

- Create: `cost_eval/layers/loss.py`
- Modify: `cost_eval/shape_eval.py`
- Modify: `cost_eval/adapters/mindformers.py`
- Modify: `cost_eval/model_spec.py`
- Test: `tests/test_cost_eval/test_loss_contract.py`

**工作项：**

- [ ] 显式建模 lm_head output 到 loss forward/backward，而不是使用一个 logits workspace 代替；
- [ ] 覆盖 logits BF16/FP16→FP32 cast、max/logsumexp/softmax、labels、loss mask、dlogits；
- [ ] 区分 fused cross entropy 与 fallback 路径；
- [ ] 区分 loss parallel 开关、vocab TP shard、sequence parallel 和 logits all-gather；
- [ ] 为各 buffer 标注 alias/in-place/allocator pool 复用规则；
- [ ] 未识别 loss 实现时输出 unsupported 或宽安全区间，不能默认一个 logits buffer；
- [ ] peak event 使用真实名称，例如 `loss_fwd_softmax`、`loss_bwd_dlogits`。

**离线验收：**

- [ ] S、B、vocab 翻倍时对应 loss buffer 精确按合同缩放；
- [ ] TP/loss-parallel 配置的本地 vocab 与通信字节守恒；
- [ ] Qwen3 46L DP8 不再出现 raw ledger 仅覆盖 profiler active 小部分的系统性缺口；
- [ ] loss 各 dtype 和融合路径有 source-derived golden。

**真机验收：**

- [ ] Qwen3 固定模型，至少改变一次 S、一次 TP/loss-parallel；
- [ ] loss 峰值事件、active bytes 和预测方向匹配 profiler；
- [ ] loss 配对 case 的 dynamic HBM delta 误差 ≤ 20%。

## 7. M2：P0-safe 修正

### Task P0.4：实现 allocator-aware total HBM

**文件：**

- Create: `cost_eval/allocator_model.py`
- Modify: `cost_eval/specs.py`
- Modify: `cost_eval/report.py`
- Modify: `cost_eval/calibration.py`
- Test: `tests/test_cost_eval/test_allocator_model.py`
- Test: `tests/test_cost_eval/test_total_memory_contract.py`

**输出分量：**

```text
physical_active_peak
allocator_pool_peak
device_baseline
untracked_runtime_point
fragmentation_point
total_point = baseline + max(active, pool) + runtime + fragmentation
safe_upper = total_point + calibrated_upper_margin + ood_margin
```

**工作项：**

- [ ] `HardwareSpec` 增加 hardware/runtime/source profile identity；
- [ ] baseline、active、reserved/pool、untracked runtime 使用互斥测量合同；
- [ ] allocator pool 首版使用按 runtime/family/layout/shape bucket 的可审计规则，不使用任意全局乘数；
- [ ] fragmentation 基于分配粒度和 size histogram，缺数据时进入安全上界而不是点预测；
- [ ] legacy `framework_reserve` 只作为兼容输入并产生 warning；
- [ ] point 和 safe upper 分离；
- [ ] usable device memory 与物理容量分离，保留系统/驱动安全余量；
- [ ] calibration 不再只建议 reserve 中位数，改为分别校准 pool/runtime/upper margin。

**验收：**

- [ ] 分量守恒且无重复计算；
- [ ] `pool < active` 和 `pool > active` 两种路径均有测试；
- [ ] 相同配置重复运行的 baseline/runtime 漂移进入区间，不污染模型结构残差；
- [ ] total point 与 safe upper 的含义在 JSON schema 中固定。

### Task P0.5：修复 rank、PP/interleave 和 FSDP 生命周期

**文件：**

- Modify: `cost_eval/event_schedule.py`
- Modify: `cost_eval/parallel_model.py`
- Modify: `cost_eval/memory_ledger.py`
- Modify: `cost_eval/static_mem.py`
- Test: `tests/test_cost_eval/test_rank_schedule.py`
- Test: `tests/test_cost_eval/test_fsdp_lifecycle.py`
- Test: `tests/test_cost_eval/test_interleaved_pipeline.py`

**工作项：**

- [ ] 立即修复 interleave gather 使用完整 `layer_ids` 而不是当前 `active_layer_ids` 的问题；
- [ ] 从 MindFormers runtime/source 提取 1F1B 与 interleave 的真实 virtual-stage 发射序列；
- [ ] 每个 physical rank/virtual chunk 分别维护 activation、gather 和通信状态；
- [ ] embedding、decoder、final norm、lm_head、loss、MTP 进入同一 prefetch chain；
- [ ] FSDP gather window 使用参数 storage union，处理 alias、prefetch depth 和 unfinished modules；
- [ ] 区分 reshard-after-forward always/never/default；
- [ ] full parameter gather 是否复用 shard storage 由 source/probe 合同决定；
- [ ] PP send/recv、stage edge 和 rank 非对称进入最坏 rank 判断；
- [ ] 检查 FSDP flatten/padding/alignment，禁止简单逐参数整除假设与实际 runtime 不一致。

**验收：**

- [ ] interleave=1 与 runtime 1F1B golden 完全一致；
- [ ] interleave=2 的 event order、warmup 数、最大 pinned microbatch、gather high-water mark 与 probe 一致；
- [ ] PP stage/rank 峰值排序与新真机 case 一致；
- [ ] 所有 rank 独立输出，job peak 取真实 max-rank，而不是 stage 平均。

### Task P0.6：升级报告、OOM 和能力状态

**文件：**

- Modify: `cost_eval/report.py`
- Modify: `cost_eval/__main__.py`
- Modify: `cost_eval/calibration.py`
- Test: `tests/test_cost_eval/test_report_schema_v2.py`

**工作项：**

- [ ] 输出 `physical_dynamic_peak_bytes`、`dynamic_point_bytes`、`total_point_bytes`、`safe_upper_bytes`；
- [ ] 输出 `tightest_rank/stage/event` 和 peak 同时分量；
- [ ] 输出 source/runtime/hardware/capability fingerprint；
- [ ] OOM 状态改为 `definitely_safe/risky/predicted_oom/unsupported`；
- [ ] capability 未 probe、V4 HBM 未验证、未知 dispatcher/optimizer 时不能返回高置信度 definitely-safe；
- [ ] `PeakMemoryReport.oom` 暂作为兼容字段，并明确其映射；
- [ ] calibration 增加 dynamic/total/interval/false-safe/stage-rank-order 指标。

### Task P0.7：P0 冻结验收

**真机矩阵：**

- [ ] Qwen3 DP8：两个层数，固定 S/vocab，验证 loss 主导峰；
- [ ] Qwen3 TP4+DP2：loss parallel off/on 或等价 vocab-shard 对照；
- [ ] DeepSeek V3 TP2+CP2+EP2：至少两个层数；
- [ ] DeepSeek V3 TP2+PP2+DP2：interleave 1/2 配对；
- [ ] 至少一个 near-boundary 成功 case 和一个实际 OOM blind；
- [ ] 关键配置至少重复两次，用于估计 allocator/runtime 漂移。

**退出条件：**

- [ ] 达到 P0 physical gate；
- [ ] 所有 false-safe 为 0；
- [ ] 每个 prediction/config/source/profile/label 都有冻结 hash；
- [ ] 不合格 archive 只保留 diagnostic，不进入验收。

## 8. M3：P1-family 精度修正

### Task P1.1：建立 Qwen3 独立 family contract

**文件：**

- Create: `cost_eval/layers/qwen3.py`
- Create: `cost_eval/source_contracts/qwen3.py`
- Modify: `cost_eval/adapters/mindformers.py`
- Test: `tests/test_cost_eval/test_qwen3_contract.py`

**工作项：**

- [ ] 固定 MindFormers commit 和 Qwen3 相关源码 hash；
- [ ] 补 RMSNorm 参数/统计量、q/k layernorm、RoPE cache/position、attention mask；
- [ ] 正确处理 layernorm、softmax、rotary、LSE 的 compute/storage dtype；
- [ ] 区分 qkv concat、GQA、FlashAttention 融合路径；
- [ ] 建立缩层 saved-tensor probe，逐 op 核对 shape/dtype/alias；
- [ ] 通用 dense builder 仅保留通用 API，不再冒充 Qwen3 高置信度合同。

**验收：**

- [ ] 参数全局 numel 与 runtime/profiler 一致；
- [ ] saved tensors 按类别与 probe 对齐；
- [ ] Qwen3 DP8 和 TP+DP 两种布局达到 family P1 指标。

### Task P1.2：完善 DeepSeek V3/V4 和 MoE dispatcher

**文件：**

- Modify: `cost_eval/layers/deepseek.py`
- Modify: `cost_eval/layers/deepseek_v4.py`
- Modify: `cost_eval/source_contracts/deepseek_v3.py`
- Modify: `cost_eval/source_contracts/deepseek_v4.py`
- Create: `cost_eval/layers/moe_dispatch.py`
- Test: `tests/test_cost_eval/test_moe_dispatch_contract.py`
- Test: `tests/test_cost_eval/test_deepseek_saved_tensors.py`

**工作项：**

- [ ] router logits 保留正确 SP/CP placement 和 FP32 dtype；
- [ ] 建模 top-k values/indices、token count、sort/permute map、expert offsets 和 auxiliary-loss 临时量；
- [ ] capacity 采用 local dispatch group、expert padding/alignment 和最坏 rank 规则；
- [ ] alltoall、deredundancy、zero-redundancy 建立独立 buffer/alias 合同；
- [ ] grouped GEMM workspace 按 local routed-token/expert shape 建模；
- [ ] shared expert、MTP、embedding/output sharing 做 storage identity golden；
- [ ] DeepSeek V4 compressor/indexer/CSA/mHC 补 saved-tensor probe；
- [ ] V4 probe 或 HBM blind 未通过前保持 unsupported/risky，不能只靠源码 hash 升级置信度。

**验收：**

- [ ] EP/dispatcher 的参数和临时量切分守恒；
- [ ] route imbalance 的 safe upper 覆盖最坏 rank；
- [ ] dispatcher 配对 delta 误差 ≤ 20%；
- [ ] V4 至少一个 calibration 和两个新 frozen holdout，其中一个 PP+EP+interleave。

### Task P1.3：workspace profile registry

**文件：**

- Create: `cost_eval/workspace_registry.py`
- Create: `validation/workspace_profiles/`
- Modify: `cost_eval/model_spec.py`
- Test: `tests/test_cost_eval/test_workspace_registry.py`

**key：**

```text
family, op_type, fusion_variant, dtype,
local_shape_bucket, tp, cp, ep,
mindspore_version, cann_version, hardware
```

**工作项：**

- [ ] profiler 值优先，解析公式作为 fallback；
- [ ] fallback 必须标记 source 和不确定区间；
- [ ] 通信 staging 与 kernel workspace 分开，只有实际并发时才相加；
- [ ] profile lookup 采用版本和覆盖距离，禁止跨 runtime major 静默复用；
- [ ] profile 数据 content-addressed，并包含采集方法和样本数。

## 9. M4：P1-strategy 精度修正

### Task P1.4：CPU offload 与 activation Swap

**文件：**

- Create: `cost_eval/offload_model.py`
- Modify: `cost_eval/specs.py`
- Modify: `cost_eval/memory_ledger.py`
- Modify: `cost_eval/adapters/mindformers.py`
- Test: `tests/test_cost_eval/test_offload_model.py`

**工作项：**

- [ ] 参数、梯度、master、optimizer state 分别声明 offload policy；
- [ ] 不再用一个 bool 将全部 persistent 归零；
- [ ] 建模当前 layer full parameter、H2D/D2H staging、双缓冲和 prefetch depth；
- [ ] activation Swap 建模 MOVE_OUT/MOVE_IN、DMA buffer 和与 gather/workspace 的重叠；
- [ ] host pinned memory 独立报告，不计入 HBM；
- [ ] prefetch target 越过 PP/virtual chunk 边界时按 runtime 规则处理。

**验收：**

- [ ] no-offload/offload、no-swap/swap 配对方向正确；
- [ ] persistent 下降和 prefetch 峰值均可解释；
- [ ] 两类策略的节省量误差均 ≤ 20%。

### Task P1.5：optimizer step 与 Muon/非 AdamW 合同

**文件：**

- Create: `cost_eval/optimizers.py`
- Modify: `cost_eval/static_mem.py`
- Modify: `cost_eval/event_schedule.py`
- Modify: `cost_eval/specs.py`
- Test: `tests/test_cost_eval/test_optimizer_contracts.py`

**工作项：**

- [ ] AdamW step 进入事件图，覆盖 grad、master、m/v 和 fused workspace；
- [ ] 明确 gradient set-to-none、累积、clip/global norm 和 step 后生命周期；
- [ ] Muon 按 parameter group 应用 `muon_parameter_fraction`；
- [ ] 建模 momentum/main parameter、Newton–Schulz matrix/workspace 和 Adam/Muon 混合组；
- [ ] 未知优化器必须提供版本化合同，否则 unsupported，不能只警告后使用 AdamW；
- [ ] CPU-offloaded optimizer state 与 step H2D staging 联动。

**验收：**

- [ ] optimizer state 参数量与 parameter groups 守恒；
- [ ] optimizer peak event 可在 profiler 中定位；
- [ ] AdamW/Muon 缩层 golden 和至少一个真实配对 case 通过。

## 10. M5：P1 验证和发布

### Task P1.6：配对实验矩阵

除绝对峰值外，每组必须计算预测 delta 与实测 delta：

| 组别 | 固定项 | 变量 |
|---|---|---|
| loss | family/layers/B | S、TP、loss parallel |
| FSDP/HSDP | family/S/B | dp shard、dp replicate、dense shard size |
| recompute | model/layout/batch | none、full、select |
| PP | model/batch | PP、interleave、reshard policy |
| MoE | model/batch | EP、dispatcher、route capacity |
| offload | model/layout/batch | parameter/state offload、prefetch |
| Swap | model/layout/batch | layer/op swap、prefetch |
| optimizer | model/layout/batch | AdamW、Muon/混合组 |

每个正式 case 必须：

- [ ] HBM/timing 分离采集；
- [ ] 所有 rank 至少 20 个有效稳定样本；
- [ ] 保存 baseline、active、reserved/pool、runtime、total 和 peak phase；
- [ ] 保存 rank/stage map、失败分配信息和环境版本；
- [ ] blind 实跑前冻结 config/source/profile/prediction hash。

### Task P1.7：验收、兼容迁移和 model card

**文件：**

- Create: `validation/real_npu/MODEL_CARD.md`
- Modify: `README.md`
- Modify: `validation/real_npu/ANALYSIS.md`
- Test: all test suites and validation gates

**工作项：**

- [ ] calibration、validation、frozen holdout、OOM blind 严格分离；
- [ ] 已揭晓 holdout 只能进入下一版 calibration；
- [ ] 分 family、layout、runtime 报告指标，禁止总体均值掩盖新家族；
- [ ] 输出支持矩阵、已知边界、OOD 行为和复现命令；
- [ ] 旧 schema 提供迁移说明和至少一个版本的兼容读取；
- [ ] 未通过 family/strategy 的 capability 标记为 unsupported 或 experimental；
- [ ] 达到 P1 完成门后才允许标记 production-validated。

## 11. 测试策略

每个 task 遵循以下顺序：

1. 添加能复现当前缺陷的失败测试；
2. 实现最小正确合同；
3. 跑目标模块测试；
4. 跑 `python -m pytest`；
5. 跑 `python run_p0_tests.py` 和 `python run_functional_tests.py`；
6. 生成 before/after 基线报告；
7. 需要真机的 task 在标签揭晓前冻结 prediction；
8. 提交代码和测试，禁止只改阈值或 reserve 让用例通过。

必须新增的负向测试：

- [ ] 未识别 loss/dispatcher/optimizer 返回 unsupported；
- [ ] V4 probe pending 不得 definitely-safe；
- [ ] placement 轴不守恒、同名非 alias、重复 free、ledger leak 必须报错；
- [ ] allocator 分量重复或出现负 runtime 必须拒绝；
- [ ] OOD runtime/hardware 不得沿用窄区间；
- [ ] 缺 rank、样本不足、baseline 污染的 archive 不得进入正式指标。

## 12. 建议执行批次

在 1 名主开发、可周期性获得 8 卡 NPU 的前提下：

| 批次 | Tasks | 预计开发量 | NPU 依赖 |
|---|---|---:|---|
| A | M0.1–M0.2、P0.1 | 5–8 人日 | 无 |
| B | P0.2–P0.3 | 6–10 人日 | loss probe/小规模验证 |
| C | P0.4–P0.6 | 8–12 人日 | allocator、PP/interleave probe |
| D | P0.7 | 3–5 人日 + 采集时间 | 必需 |
| E | P1.1–P1.3 | 8–12 人日 | family/workspace probe |
| F | P1.4–P1.5 | 6–10 人日 | 策略配对 |
| G | P1.6–P1.7 | 4–7 人日 + 采集时间 | 必需 |

估算不包含 NPU 排队、失败 archive 重跑和外部 runtime bug 修复。若 NPU 不可用，最多只能推进到离线软件门，不能宣称 P0/P1 完成。

## 13. 风险控制

| 风险 | 处理原则 |
|---|---|
| loss 修正后仍低估 | 对齐 profiler active 和 peak event，检查 dtype/alias，不先扩大 runtime reserve |
| 修 placement 后误差暂时变大 | 优先保证物理语义正确，记录补偿性旧误差，不回退错误 placement |
| allocator active/reserved 重复 | 统一使用 `baseline + max(active,pool) + runtime + fragmentation` 守恒测试 |
| PP/interleave probe 与解析调度不一致 | source/runtime 优先，调度标记版本，不用近似结果报告高置信度 |
| 路由随机导致单点不稳定 | 固定 seed、保存 per-rank token counts，并输出最坏 rank/分位安全上界 |
| calibration 好但 blind 退化 | 禁止复用已揭晓 holdout；按 family/layout/runtime 报告，不调大无界 residual |
| 兼容 API 掩盖新旧语义 | schema version、弃用 warning 和字段映射测试必须同时存在 |

## 14. 最终交付物

- [ ] rank-aware、逐算子 allocation/free 物理 ledger；
- [ ] 正确的 value/storage identity 和多轴 placement/reshard；
- [ ] loss、allocator/runtime、PP/interleave/FSDP P0 合同；
- [ ] Qwen3、DeepSeek V3/V4、MoE P1 source/probe 合同；
- [ ] offload、Swap、AdamW/Muon 和 workspace profile；
- [ ] point/safe upper/OOM/confidence schema v2；
- [ ] 可审计 baseline/calibration/holdout/OOM 报告；
- [ ] 通过 P0/P1 门的 model card，或明确记录仍未通过的 capability。
