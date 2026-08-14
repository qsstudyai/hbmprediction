# 训练显存预测生产化设计方案

> 日期：2026-08-14  
> 范围：MindFormers PyNative 动态图训练的单机多卡/多机扩展显存预测  
> 目标模型：DeepSeek V3、DeepSeek V4、Qwen3  
> 目标硬件首期：Ascend 910B2 × 8；更多卡数和其他硬件需独立验收  
> 本文只规定显存、OOM 和置信度；训练耗时预测不是本方案的验收前置条件

## 1. 结论与设计原则

本项目的目标不是给出一个“参数量乘系数”的经验值，而是构建一个以真实训练执行语义为基础、能够被 NPU 实测校准和证伪的显存预测器。给定模型配置、并行切分、重计算、CPU offload、Swap、优化器和设备容量，系统应输出：

1. 每个 rank / Pipeline stage 的动态显存峰值及峰值事件；
2. 参数、梯度、优化器、激活、通信、FSDP gather、重计算、Swap、workspace 和 allocator/runtime 的互斥拆解；
3. 设备总显存点预测、安全上界、OOM 结论和置信度；
4. 当前配置中未被精确建模的能力和超出校准域的原因；
5. 对 DeepSeek V3、DeepSeek V4、Qwen3 PyNative 动态图训练，在不同 DP/HSDP、TP、CP、PP、EP、interleave、重计算和 offload 组合上的泛化预测。

设计遵循以下原则：

- **物理模型优先**：参数切分、张量生命周期、调度和 allocator 均显式建模，学习残差只能做有界修正。
- **点预测与安全判断分离**：点预测用于误差评价，安全上界用于 OOM 风险，二者不能混为一个数。
- **动态峰值而非静态求和**：显存峰值必须沿真实前反向事件和 rank/stage 生命周期取最大值。
- **配置和源码双重对齐**：配置适配以 MindFormers PyNative schema 为入口，算子和切分语义以实际运行源码为合同。
- **最坏 rank 原则**：PP、EP、edge module 等造成 rank 非对称时，以真实最坏 rank 判断 OOM，不能只看 rank 0 或平均 stage。
- **不确定性显式化**：未知算子、未知 dispatcher、新布局或超出校准域时，必须降低置信度、放宽安全区间或拒绝给出安全结论。
- **盲测防泄漏**：正式 holdout 必须在实跑前冻结配置、模型/profile 和预测哈希，标签揭晓后不得回填拟合。

## 2. 项目边界与术语

### 2.1 本方案包含

- PyNative 动态图训练的峰值 HBM 和 OOM 预测；
- Dense、GQA/SwiGLU、MLA、MoE、shared expert、MTP 等模型结构；
- DP replicate、FSDP/DP shard、HSDP、TP、CP、PP、EP、sequence parallel；
- 1F1B 和 MindFormers 支持的 interleaved pipeline；
- full/select/exclude recompute、通信重计算；
- activation Swap、参数/优化器 CPU offload；
- AdamW，以及具备明确状态合同后的 Muon/其他优化器；
- 真实 NPU 采集、标签质量门、校准、冻结盲测和 OOM 验收；
- 解析物理基线之上的有界泛化残差与置信区间。

### 2.2 本方案暂不承诺

- 在没有对应真机标签的硬件、卡数或运行时版本上宣称精度已经通过；
- 任意自定义算子或任意第三方优化器的零配置精确预测；
- 用显存模型替代训练时间、通信性能或自动并行搜索模型；
- 对驱动故障、网络故障、非显存资源不足等非 HBM OOM 给出容量预测；
- 对动态 shape、随机路由极端倾斜给出无条件的单点精确值。此类场景输出区间或风险分位数。

### 2.3 三类输出必须区分

```text
physical_dynamic_peak
    由模型张量、调度和运行时内存事件解析得到的动态峰值

total_point_prediction
    用于 MAPE/APE 的设备总峰值点预测

safe_upper_bound
    加入经校准的不确定性和碎片安全裕量，用于 OOM 风险判断
```

OOM 结论定义为：

```text
definitely_safe : safe_upper_bound <= usable_device_memory
risky           : point <= usable_device_memory < safe_upper_bound
predicted_oom   : total_point_prediction > usable_device_memory
unsupported     : 关键语义未知，不能给出安全结论
```

## 3. 当前模型已经完成的能力

### 3.1 独立 `hbmprediction` 仓库已完成

当前仓库已经实现 P0/P0.5 显存主链：

| 能力 | 当前状态 | 说明 |
|---|---|---|
| 声明式 ModelSpec | 已完成 | 逐 op 定义符号 shape、placement、参数、saved tensor、workspace |
| Dense/GQA/SwiGLU | 已完成 | 包含 embedding、decoder、final norm、lm_head |
| 标准 MoE/EP | 已完成 | router、dispatch、grouped expert GEMM、combine |
| DeepSeek MLA | 已完成 | q/kv 低秩压缩、up projection、RoPE、attention 和输出投影 |
| shared expert / gate | 已完成 | 包含共享专家参数、激活和可选 FP32 gate |
| MTP | 已完成 | MTP 层位于最后 PP stage，边缘权重共享按 storage identity 去重 |
| Hyper-Connections | 已完成 | 已有 projection/mix 内存合同 |
| TP/CP/EP/SP placement | 已完成 | 可派生 all-reduce、reduce-scatter、all-gather、all-to-all |
| DP shard/FSDP/HSDP | 已完成 | dense FSDP 和 expert eFSDP 分开建模 |
| PP 1F1B | 已完成 | warmup/steady/cooldown 激活生命周期 |
| interleave | 已实现解析模型 | 需要更多真实动态图库验证其发射顺序和 gather 生命周期 |
| full/select/exclude recompute | 已完成 | saved tensor 三态互斥，支持模块和 op 选择 |
| 通信重计算 | 已完成 | 前向释放、反向重新物化通信输出 |
| activation Swap | 已完成 | 支持 layer/op 选择和反向预取 buffer |
| CPU offload | 已完成基础模型 | 常驻设备状态归零，执行时 gather/prefetch 仍计入；allocator/DMA 细化待补 |
| 静态显存拆解 | 已完成 | parameter/gradient/master/optimizer state |
| 事件驱动峰值 | 已完成 | persistent/activation/gather/grad/recompute/swap/workspace/framework |
| OOM 和最紧 stage | 已完成 | 依据给定容量输出逐 stage OOM |
| 通用 JSON/YAML Adapter | 已完成 | 可直接作为离线 API/CLI 使用 |
| MindFormers PyNative Adapter | 已完成 | 支持主要并行、重计算、Swap 和 DeepSeek 配置字段 |


### 3.2 已有真实 NPU 资产

真机采集、动态 allocator 模型、残差拟合和归档目前主要保留在相邻 `parallelsearch` 仓库，尚未完整迁入独立显存库。与本项目目标直接相关的资产包括：

- Ascend 910B2 × 8 的真实 PyNative 训练采集链；
- `npu-smi` per-rank HBM 时间序列和 MindSpore memory profiler；
- timing 与 HBM 分离采集，避免 profiler stop/flush 污染标签；
- device baseline、model active、allocator pool、untracked runtime 的测量合同；
- frozen prediction、profile/config/label hash、calibration/holdout 分离；
- total/dynamic HBM APE、区间覆盖、OOM false-safe 等验收脚本；
- Qwen3、DeepSeek V3 的多个版本化 calibration/blind 实验；
- DeepSeek V4 的 PP+EP+interleave 成功训练归档和 OOM 诊断归档。

### 3.3 三个目标模型的真机预测结论

#### DeepSeek V3

- 修正 `EP=1` 时 expert 使用普通 FSDP mesh 的语义后，参数分片与 profiler 更一致；
- 单个 DeepSeek V3 校准案例在加入实测 runtime reserve 后曾达到约 0.707% P90 HBM 误差；
- v14 DeepSeek V3 38L TP2+CP2+EP2 的 total/dynamic HBM 点误差约为 6.11%/7.08%，但 total HBM 实测值落在冻结预测区间之外，说明点预测通过不等于安全区间通过；
- v16 DeepSeek V3 42L TP2+CP2+EP2 的 HBM 训练与采样已经结束，但 profiler 后处理被中断，没有形成合法 `case_result.json`，因此不能进入 calibration 或 blind 验收，必须新建 archive 重跑；
- 结论：DeepSeek V3 已有较强的结构和真机基础，但 CP/EP 布局的区间覆盖及新 blind 尚未最终通过。

#### Qwen3

- v16 Qwen3 46L DP8 的 total/dynamic HBM 实测约为 16.086/12.503 GB，冻结预测误差约为 3.70%/4.77%，每卡 53 个有效样本，baseline 一致且预测区间覆盖实测；
- 早期 Qwen3 TP4+DP2 blind 曾出现 total/dynamic HBM 明显超门的情况，证明 DP8 上的低误差不能直接外推到新的 TP/DP 布局；
- 结论：Qwen3 DP8/相邻层数方向已有有效泛化证据，但 TP、CP、DP shard、重计算和 CPU offload 的跨布局泛化仍需独立 frozen holdout。

#### DeepSeek V4

- 已有 Ascend 910B2 × 8 上的 DeepSeek V4 PyNative 动态图成功训练归档，覆盖 TP2+PP2+DP shard2+EP2+interleave2；另有真实 OOM 诊断归档；
- 既有离线评估器曾因该配置超出当时安全事件图能力而跳过正式预测，说明“训练归档存在”不代表“显存预测已支持”；
- 当前独立库已有 MLA、MoE、shared expert、MTP 等结构基础，但尚未建立经过目标 V4 源码版本验证的独立 source contract，也没有完成 V4 frozen HBM blind；
- 结论：DeepSeek V4 目前只有实现基础和真机诊断证据，不能宣称泛化预测已经完成。

#### 总体限制

- allocator baseline 漂移、pool 复用和碎片尚未在独立库形成完整合同；
- 当前独立仓库只有 `framework_reserve`，不能充分表达设备总 HBM 的互斥分量和安全区间；
- 尚缺按新测量合同执行并纳入正式门禁的 OOM blind；
- 当前解析模型能表达 TP 增大、重计算、Swap 等显存变化方向，但策略节省量仍需三家族的配对真机实验验收。

因此，当前状态应定义为：**解析显存主链完成，生产级跨模型泛化和 OOM 安全验收未完成**。

## 4. 目标支持矩阵

### 4.1 模型家族

| 家族 | 必须支持的结构 | 动态图要求 | 泛化目标 |
|---|---|---|---|
| Qwen3 | embedding、GQA、RoPE、FlashAttention、SwiGLU、RMSNorm、lm_head | MindFormers PyNative eager/autograd saved tensor 与融合路径 | 层数、S、B、TP/CP/DP/FSDP、重计算、offload 间泛化 |
| DeepSeek V3 | MLA、dense first layers、routed MoE、shared expert/gate、MTP | 动态 router/dispatch、MLA saved tensors、MTP、eFSDP | TP/CP/EP/PP/DP/interleave 和 MoE 路由容量间泛化 |
| DeepSeek V4 | 以实际 MindFormers V4 PyNative 源码和配置为准，不从 V3 名称推断结构 | 为 V4 建独立 source contract、layer builders 和 adapter capability；禁止静默复用不一致的 V3 合同 | 至少覆盖一种非 PP 和一种 PP+EP+interleave 布局的冻结盲测 |

DeepSeek V4 的实现必须遵循“源码合同先行”：

1. 固定目标 MindFormers commit/version；
2. 从 V4 PyNative forward、parallelize、checkpoint 和 optimizer 路径提取参数、saved tensor、workspace、placement；
3. 与 V3 相同的部分显式复用 builder，不同部分建立 V4 专用 builder；
4. 在 `capabilities` 中记录版本和已建模特性；
5. 未识别结构必须输出 `unsupported` 或 warning，不能以 V3 近似后仍报告高置信度。

### 4.2 并行切分

必须支持以下数据并行方案：

| 方案 | 配置语义 | 持久状态 | 临时显存 |
|---|---|---|---|
| 纯 DDP / `dp_replicate>1` | 每个 DP replica 拥有完整本地模型分片 | 不按 replicate 进一步切参数 | all-reduce/reduce-scatter staging |
| Full FSDP / `dp_shard>1` | parameter、gradient、optimizer state 分片 | 按 FSDP degree 切分 | forward/backward all-gather、grad reduce-scatter、prefetch |
| HSDP | replica 组 × shard 组 | shard 组内分片、组间复制 | 组内 gather + 组间同步 buffer |
| Dense limited FSDP | `dense_fsdp_shard_size` 小于完整 DP×CP mesh | dense 权重按指定 shard size | 与完整 FSDP 不同的 gather window |
| Expert eFSDP | EP>1 时专家使用独立 mesh | routed expert 按 EP 和 eFSDP 语义切分 | expert gather、dispatch/combine buffer |
| EP=1 expert | 专家属于普通 decoder FSDP wrapper | 只能按普通 FSDP 切，不能额外按 TP 假想 eFSDP 切 | 与 dense decoder 同类 gather |
| CPU offload | 参数/梯度/优化器可从 HBM 移出 | 设备持久状态下降 | H2D/D2H staging、prefetch、当前 layer full parameter、allocator pool |

DP/HSDP 必须与 TP、CP、PP、EP 组合验证，并满足：

```text
world_size = dp_replicate * dp_shard * cp * tp * pp
```

专家 mesh、dense mesh 和 replicate group 的关系应来自运行时构造，不允许用一个统一除数代替。

## 5. 物理显存模型

### 5.1 声明式算子与张量合同

每个 op 至少声明：

```python
OpMemoryContract(
    params,                 # 持久权重
    saves,                  # backward 需要的张量
    workspace,              # kernel 临时空间公式或 profile key
    outputs,
    collectives,
    placement,
    recomputable,
    swappable,
    dtype,
    storage_identity,       # 共享权重/别名去重
    source_contract_version,
)
```

动态图支持不依赖 trace 一次具体执行来猜全模型，而采用“源码验证的声明式图 + profiler 校准”。trace/profiler 用于核对生命周期、融合和 workspace，不作为唯一模型来源，以避免一次输入路径遗漏分支。

### 5.2 本卡 shape 与 placement

符号全局 shape 先代入模型维度，再逐维应用 TP/CP/EP/SP placement：

```text
local_numel = product(global_shape[d] / mesh_degree[placement[d]])
local_bytes = local_numel * dtype_bytes
```

必须校验整除关系。生产端和消费端 placement 不同时派生 reshard：

| 转换 | 通信/临时内存 |
|---|---|
| Partial(TP) → Replicate | all-reduce |
| Partial(TP) → Shard(SP) | reduce-scatter |
| Shard → Replicate | all-gather |
| Replicate → Shard | local slice，通常无通信 |
| EP dispatch/combine | all-to-all 或 dispatcher 专用合同 |
| CP colossal/ulysses/hybrid | ring P2P、all-to-all 或内外两级通信 |

### 5.3 持久训练状态

对每个唯一 parameter storage 计算：

```text
persistent_param   = local_param_numel / fsdp_degree * parameter_bytes
persistent_grad    = local_param_numel / grad_shard_degree * gradient_bytes
master_weight      = local_param_numel / state_shard_degree * master_bytes
optimizer_state    = local_param_numel / state_shard_degree * optimizer_bytes
```

AdamW BF16 基线通常为 16 B/已分片参数；FP32 grad 通常为 18 B。Muon 和其他优化器必须单独声明：

- 哪部分参数使用该优化器；
- momentum/main parameter 字节数；
- optimizer step 临时矩阵和 Newton-Schulz workspace；
- 是否和 Adam 分组共存。

不能仅修改一个总 `bytes_per_parameter` 而遗漏 optimizer-step 峰值。

### 5.4 激活、重计算和 Swap

正常前向为每个 microbatch pin 住去重后的 `saves`，对应反向消费后释放。

对每个 saved storage 强制三态：

```text
resident   : 前向后留在 HBM，反向后释放
recomputed : 前向只保留 checkpoint，反向重物化为 scratch
offloaded  : 前向 D2H 后离开 HBM，反向前 H2D 预取
```

重计算需要建模：

- full layer recompute；
- select module/op recompute；
- exclude op；
- 多段连续重计算区域各自的 checkpoint；
- communication recompute；
- 反向重物化的瞬时峰值和额外 workspace。

CPU offload/Swap 需要建模：

- 从 `act_live` 或 persistent 中移出的字节；
- D2H/H2D staging buffer；
- `prefetch_depth` 和并发 DMA buffer；
- 回迁对象与当前 gather/workspace 的重叠；
- pinned host memory 不计入 HBM，但进入报告的 host-memory breakdown；
- 带宽只影响时间，不影响最终字节，但并发窗口会改变 HBM 峰值。

### 5.5 FSDP/HSDP 生命周期

每层完整参数 buffer：

```text
G_layer = sum(unique full local-TP parameter bytes before FSDP sharding)
```

`reshard_after_forward=always` 时，用完释放，反向再次 gather；`never` 时参数可跨前反向保留。prefetch window 为当前层和未来若干层之和，不能简单使用固定倍数：

```text
gather_window(event, rank)
    = union(parameters required by all prefetched but unfinished modules)
```

PP/interleave 下每个 virtual chunk 单独维护 gather state。embedding、final norm、lm_head、MTP 等 edge module 进入同一事件图和 prefetch chain。

### 5.6 通信与 kernel workspace

通信临时量由实际 collective 合同决定，而不是无条件把 message volume 加到峰值：

- 输入、输出是否别名；
- 是否有额外 staging；
- chunk 数和流水；
- dispatcher 是否去冗余；
- 通信 buffer 是否来自 allocator pool 并可复用；
- 通信与算子 workspace 是否互斥或并发。

kernel workspace 优先按以下 key 查询真实 profile：

```text
(model_family, op_type, fusion_variant, dtype,
 local_shape_bucket, tp, cp, ep, runtime_version)
```

没有 profile 时使用解析 fallback，同时标记较宽区间。

### 5.7 事件时间线与 rank/stage 峰值

事件图至少包含：

```text
FWD op alloc/free
save activation
FSDP gather/prefetch/reshard
collective staging
Swap D2H/H2D
BWD recompute
BWD gradient materialization/reduce-scatter
optimizer step
edge modules and loss
```

每个 rank 维护独立 ledger：

```text
active_tensor_bytes(rank, event)
gathered_parameter_bytes(rank, event)
communication_bytes(rank, event)
workspace_bytes(rank, event)
allocator_pool_bytes(rank, event)
```

最终：

```text
dynamic_peak(rank) = max_event(dynamic_bytes(rank, event))
job_dynamic_peak   = max_rank(dynamic_peak(rank))
```

### 5.8 allocator-aware 设备总显存

各分量必须互斥：

```text
total_point_prediction
    = device_baseline
    + max(model_active_peak, allocator_pool_peak)
    + untracked_runtime_point
    + fragmentation_point

safe_upper_bound
    = total_point_prediction
    + residual_upper_quantile
    + ood_margin
```

说明：

- `device_baseline` 是训练加载前设备/进程基础占用；
- `model_active_peak` 是物理 ledger 的活动张量与临时 buffer；
- `allocator_pool_peak` 已包含 pool 内未活跃但未归还的块，不能再作为 framework reserve 重复叠加；
- `untracked_runtime` 是 runtime、图执行器、通信库等无法归入 allocator/model 的占用；
- `fragmentation` 与分配粒度、峰值 active、对象尺寸分布相关；
- 无法辨认的 reserve 优先进入安全区间，不直接污染点预测。

## 6. DeepSeek V3/V4 与 Qwen3 动态图实现

### 6.1 适配分层

```text
MindFormers YAML
  └─ Config Adapter
      ├─ 通用训练/并行/优化器配置
      └─ Family Adapter
          ├─ qwen3
          ├─ deepseek_v3
          └─ deepseek_v4
               ↓
        versioned source contract
               ↓
        ModelSpec + capabilities + warnings
```

Family Adapter 不应只通过模型名称选择图，还应检查必要字段和运行时版本。例如 MLA rank、head dimension、expert/shared expert、MTP、dispatcher、Hyper-Connections 和 tie embedding 必须进入 capability fingerprint。

### 6.2 PyNative saved-tensor 对齐

对每个模型家族建立最小动态图探针，在缩小模型上记录：

- forward/backward op 与 module path；
- autograd saved tensor 的 shape、dtype、storage alias；
- FSDP wrapper 边界和 gather/reshard 顺序；
- recompute 前后实际保留对象；
- CPU offload/Swap 前后的 HBM 对象；
- kernel workspace 和 allocator active/reserved 曲线。

探针结果用于验证声明式合同，而不是在生产预测时强制执行模型。每个 source contract 必须保存：

```text
model_family
mindformers_commit
mindspore/runtime_version
contract_version
supported_capabilities
probe_artifact_hash
```

### 6.3 DeepSeek V4 专项要求

DeepSeek V4 是本阶段新增的正式目标，完成标准不是“V3 图能够跑 V4 YAML”，而是：

1. V4 配置可被显式识别并生成 V4 capability fingerprint；
2. V4 所有新增/变化模块都有 source contract 或明确 unsupported；
3. 参数全局守恒、每类权重 placement 和 storage sharing 有单测；
4. 缩层 V4 动态图 probe 的 saved tensor 与解析合同逐类对齐；
5. 至少一个 V4 calibration 和两个配置不同的 frozen holdout，其中至少一个包含 PP+EP+interleave；
6. V4 OOD 查询不得借用 V3 残差并报告高置信度。

## 7. 泛化预测模型

### 7.1 两层结构

```text
prediction = physical_model(config)
           × bounded_residual(features, coverage)
```

物理模型始终可独立输出。残差模型的目标是修正尚未完全解析的 allocator/runtime/fusion 差异，不能取代参数和生命周期公式。

HBM residual 拟合目标使用去 baseline 后的 dynamic HBM：

```text
target = log(measured_dynamic_hbm / physical_dynamic_hbm)
```

total HBM 由 dynamic prediction 和同次测量合同中的 baseline/runtime 分量重建，避免设备 baseline 漂移被错误学习成模型结构残差。

### 7.2 特征

特征至少包括：

- family/capability fingerprint；
- layers、H、F、S、B、vocab、heads、kv heads；
- MLA ranks/head dims；
- expert 数、top-k、shared expert、capacity factor；
- TP/CP/PP/EP、DP replicate/shard、FSDP mesh、interleave；
- sequence parallel、loss parallel、dispatcher；
- full/select recompute 比例和模块类别；
- CPU offload、Swap 比例、prefetch depth；
- physical persistent/activation/gather/workspace/communication 各桶和比例；
- runtime/source contract/hardware profile 版本。

### 7.3 泛化和 OOD 规则

- p50 timing、p90 timing、dynamic HBM、total HBM 使用独立 head；本项目只消费 HBM heads。
- 模型家族、运行时大版本或 capability 不兼容时禁止跨域残差。
- 特征距离超出 calibration convex/coverage domain 时，残差回退到 0，置信度下降，安全区间扩大。
- residual 修正幅度必须有上限；上限由独立 validation 确定，不能为通过单一 holdout 任意放大。
- calibration anchor 可以精确回放，但 calibration 误差 0 不算泛化证据。
- 每轮已揭晓的 holdout 只能在下一版本转为 calibration；下一轮必须使用新配置作为 blind。

### 7.4 输出合同

```json
{
  "model_family": "deepseek_v4",
  "source_contract_version": "...",
  "per_rank": [],
  "tightest_rank": 0,
  "tightest_stage": 0,
  "peak_event": "bwd_grad@layer_...",
  "physical_dynamic_peak_bytes": 0,
  "dynamic_point_bytes": 0,
  "total_point_bytes": 0,
  "safe_upper_bytes": 0,
  "usable_device_memory_bytes": 0,
  "oom_status": "definitely_safe|risky|predicted_oom|unsupported",
  "confidence": 0.0,
  "coverage_distance": 0.0,
  "breakdown": {
    "parameter": 0,
    "gradient": 0,
    "optimizer": 0,
    "activation": 0,
    "gathered_parameter": 0,
    "communication": 0,
    "recompute": 0,
    "offload_prefetch": 0,
    "workspace": 0,
    "allocator_pool": 0,
    "device_baseline": 0,
    "untracked_runtime": 0,
    "fragmentation": 0
  },
  "warnings": []
}
```

## 8. 真实 NPU 测试设计

### 8.1 测量原则

每个正式 case 必须采用固定：

- global batch、micro batch 和 gradient accumulation；
- sequence length、dtype、optimizer；
- 模型和 MindFormers commit；
- 并行拓扑和 rank mapping；
- 固定训练步数，关闭 ramp-up/dynamic batch；
- 设备空闲基线和采集环境。

HBM 与 timing 分开运行。显存 run 开启所需 profiler 和 HBM 采样；即使本项目不验收时间，也要避免 profiler stop/flush 或采样线程改变标签语义。

### 8.2 HBM 采集合同

每个 rank 至少采集：

```text
device_baseline_bytes
model_active_peak_bytes
allocator_pool_peak_bytes
untracked_runtime_bytes
total_peak_used_bytes
peak_timestamp / phase / step
sample_count
```

要求：

- 取所有 rank 的最坏峰值，并保存 stage/rank 映射；
- 每卡稳定窗口至少 20 个有效 HBM 样本；低频 `npu-smi` 轮询不足时必须延长窗口；
- profiler active/reserved 和 device total 的采样窗口必须可对齐；
- baseline 在运行前后不一致或其他进程污染时降为 diagnostic-only；
- 各 HBM 分量必须互斥，不允许出现无法解释的负值；
- 归档必须含 `collection_state=completed`、attempt id、配置 hash 和原始日志；
- 采集中断、样本不足、rank 缺失或 profiler 解析失败的 archive 不得进入拟合/验收。

### 8.3 OOM 采集合同

正式 OOM case 必须保存：

- 运行前冻结的 point/safe upper/OOM 预测；
- 实际失败 rank 和 stage；
- 失败发生在初始化、forward、backward、optimizer 或通信阶段；
- 失败分配大小；
- 失败时设备 used/free、allocator active/reserved；
- 错误类型和完整日志；
- 没有 steady-state HBM 峰值时仍作为分类标签保留，但不得伪造连续 HBM 数值。

### 8.4 Calibration、validation 和 blind holdout

首期 8 卡矩阵至少应包括：

| 集合 | 最低要求 |
|---|---|
| Calibration | 每个家族至少 2 个有效 HBM case；至少覆盖 DP/FSDP 基线和一种模型特有能力 |
| Validation | 至少 3 个 case，用于 residual 上限、allocator 参数和安全区间选择 |
| Frozen holdout | 至少 6 个普通 case；Qwen3、DeepSeek V3、DeepSeek V4 每个家族至少 2 个 |
| OOM blind | 至少 2 个，其中至少 1 个实际 OOM、至少 1 个近边界但成功 |

普通 holdout 的覆盖要求：

- 至少一个纯 DP replicate 或 DP8；
- 至少一个 TP+DP shard；
- 至少一个 TP+CP+EP；
- 至少一个 PP+DP shard+interleave；
- 至少一个 full/select recompute 对照；
- 至少一个 CPU offload 对照；
- Dense 与 MoE 均覆盖；
- DeepSeek V4 至少一个 PP+EP+interleave。

重计算和 CPU offload 必须做成配对实验：同一模型、batch 和并行拓扑，仅改变待验证开关。除绝对峰值外还要验收预测的差值：

```text
delta_recompute = peak(no_recompute) - peak(recompute)
delta_offload   = peak(no_offload)   - peak(cpu_offload)
```

### 8.5 冻结和数据治理

正式 blind 实跑前保存：

```text
config SHA256
source contract SHA256
hardware/residual profile SHA256
prediction JSON SHA256
collector version
MindFormers/MindSpore/driver/CANN version
```

实跑后原始 archive 只追加，不覆盖。短窗口、污染、中断和失败尝试保留作审计证据，但必须标明 `diagnostic_only`。

## 9. 验收条件

### 9.1 软件正确性门

必须全部满足：

- 全量单测、P0 conformance 和 functional tests 通过；
- 参数全局守恒和每类权重切分守恒；
- 每个 allocation 都有对应 free，结束时动态 ledger 回零；
- recompute/resident/offloaded 三态互斥；
- baseline、allocator pool、runtime reserve 不重复计算；
- stage/rank 峰值拆解之和等于报告峰值；
- TP/DP shard/EP 增大等受约束场景满足预期单调性；
- 未支持配置显式拒绝，不得静默退化成高置信度结果；
- DeepSeek V3/V4/Qwen3 各自的 source-contract golden tests 通过。

### 9.2 物理基线门

在不使用 holdout 标签残差的 frozen holdout 上：

| 指标 | 平均 | 最坏 |
|---|---:|---:|
| physical dynamic HBM APE | ≤25% | ≤35% |
| OOM false-safe | 0 | 0 |

该门用于证明物理模型没有被残差完全掩盖。若不通过，不能只扩大 residual 修正幅度。

### 9.3 最终泛化门

在所有有效 frozen holdout 上同时满足：

| 指标 | 验收条件 |
|---|---:|
| total HBM mean APE | ≤15% |
| total HBM worst APE | ≤20% |
| dynamic HBM mean APE | ≤20% |
| dynamic HBM worst APE | ≤30% |
| safe HBM interval coverage | ≥80% |
| stage/rank 峰值排序正确率 | ≥80% |
| OOM 分类 | 100% |
| false-safe | 0 |
| 正式实际 OOM 数 | ≥1，目标 ≥2 |

除总门外，每个模型家族单独要求：

- 至少 2 个有效 blind holdout；
- total HBM mean APE ≤15%；
- 单例 worst APE ≤20%；
- 不允许某一新家族由其他家族的低误差掩盖。

### 9.4 策略效果门

配对实验要求：

| 策略 | 要求 |
|---|---|
| full/select recompute | 峰值方向正确；预测节省量相对实测节省量误差 ≤20% |
| CPU offload | 峰值方向正确；持久下降和 prefetch 峰值均可解释；节省量误差 ≤20% |
| FSDP/HSDP 切分 | 不同 shard/replicate 方案的峰值排序正确；差值误差 ≤20% |
| PP/interleave | 最坏 stage/rank 正确；峰值排序正确；单例 total APE ≤20% |
| EP/dispatcher | expert persistent 和 dispatch temporary 的方向、切分守恒正确；单例 total APE ≤20% |

### 9.5 置信度门

- calibration 域内同类配置的区间应窄于 OOD 配置；
- coverage distance 增大时 confidence 不得增大；
- 未见过的 family/runtime major/capability 必须回退物理基线并降低置信度；
- `definitely_safe` 的 blind case 不允许实际 OOM；否则本版本直接验收失败。

## 10. 实现架构调整

在现有模块基础上新增/调整：

```text
cost_eval/
├── model_spec.py                 # 现有：声明式张量/op 合同
├── layers/
│   ├── qwen3.py                  # Qwen3 显式 family builder
│   ├── deepseek_v3.py            # 从通用 deepseek builder 拆出版本合同
│   └── deepseek_v4.py            # 新增 V4 专用 builder
├── source_contracts/
│   ├── qwen3.py
│   ├── deepseek_v3.py
│   └── deepseek_v4.py
├── adapters/
│   └── mindformers.py            # family/version/capability 路由
├── shape_eval.py                 # 现有：placement 和本卡 shape
├── static_mem.py                 # 现有：持久状态
├── event_schedule.py             # 从 mem_timeline 拆出 rank/stage/VP 调度
├── memory_ledger.py              # allocation/free 和物理动态峰值
├── allocator_model.py            # pool、baseline、runtime、fragmentation
├── offload_model.py              # 参数/激活 offload 和 prefetch 窗口
├── uncertainty.py                # point/safe upper/confidence/OOD
├── measurement_contract.py       # 从 parallelsearch 迁入并裁剪为 HBM
├── npu_profile_collector.py      # 可选验证依赖，不进入纯离线核心依赖
├── calibration.py                # 扩展 dynamic/total/interval/OOM 指标
└── report.py                     # 新输出合同
```

核心离线库仍不依赖 MindSpore/NPU；collector 和动态图 probe 放入可选 `validation` extra。

## 11. 分阶段实施路线

### Phase 0：冻结现状和证据迁移

- 为独立仓库建立可追踪版本基线；
- 将真实 HBM measurement contract、归档索引和验收脚本从 `parallelsearch` 迁入；
- 只引用 Qwen3、DeepSeek V3、DeepSeek V4 的原始 archive，不复制大文件；使用 manifest/hash 定位；
- 生成当前解析模型对全部有效 archive 的统一 baseline report。

完成条件：同一命令可从 manifest 重建现状报告，所有标签来源可审计。

### Phase 1：allocator-aware HBM

- 替换单一 `framework_reserve`；
- 实现互斥 total/dynamic HBM 合同；
- 建立 point/safe upper/confidence；
- 加入 per-rank pool 和 fragmentation 模型。

完成条件：物理基线达到第 9.2 节门限，且无已知 OOM false-safe。

### Phase 2：动态图 family 合同

- 完成 Qwen3、DeepSeek V3 显式 family contract；
- 从目标 MindFormers 版本提取 DeepSeek V4 contract；
- 建立缩层 PyNative saved-tensor probes 和 golden tests；
- 修正 PP/interleave/FSDP/edge module 生命周期。

完成条件：三家族软件门全部通过，V4 至少一个真实 calibration case 可解释。

### Phase 3：策略专项

- DP replicate/FSDP/HSDP 配对；
- full/select recompute 配对；
- CPU offload/Swap 配对；
- CP/EP/dispatcher 和 PP/interleave 配对；
- optimizer-step workspace。

完成条件：第 9.4 节所有策略效果门通过。

### Phase 4：冻结泛化和 OOM 验收

- 建立三家族 calibration/validation/blind 矩阵；
- 每轮实跑前冻结全部哈希；
- 普通 holdout 通过后运行正式 OOM blind；
- 输出最终 model card、支持矩阵、已知边界和复现实验命令。

完成条件：第 9 节所有门同时通过，最终报告 `passed: true`。在此之前不得声明“DeepSeek V3/V4/Qwen3 泛化预测已经完成”。

## 12. 风险与决策

| 风险 | 后果 | 应对 |
|---|---|---|
| allocator 与 runtime 分量重复 | 系统性高估 | 强制互斥测量合同和分量守恒测试 |
| 低频 HBM 采样漏峰 | 标签偏低 | 每卡最少样本门、延长稳定窗口、profiler 交叉验证 |
| 只看 rank 0 | PP/EP 峰值漏报 | 所有 rank 逐分量取峰并保留 stage mapping |
| DeepSeek V4 直接复用 V3 | 参数和激活合同漂移 | 独立 capability/source contract 和 V4 blind |
| residual 记住配置 | calibration 好、泛化差 | 冻结盲测、family gate、距离回退、修正幅度上限 |
| offload 只扣常驻不加预取 | 严重低估峰值 | 显式 DMA/prefetch/当前 full parameter 生命周期 |
| interleave 调度过度简化 | stage 峰值和排序错误 | 运行时事件 probe、per-chunk ledger、PP blind |
| OOM 没有连续 HBM 标签 | 错误丢弃重要样本 | 分离数值回归标签和 OOM 分类标签 |

## 13. 最终交付物

1. 可安装的纯离线 `hbmprediction` Python 包；
2. Qwen3、DeepSeek V3、DeepSeek V4 版本化 source contracts；
3. allocator-aware 逐 rank/stage 显存事件模拟器；
4. DP/HSDP、重计算、CPU offload、Swap 等策略报告；
5. point prediction、safe upper、OOM 和 confidence 输出；
6. HBM collector、measurement contract 和 archive manifest；
7. calibration/validation/frozen holdout/OOM blind 自动报告；
8. 通过第 9 节门限的最终 model card；
9. 未支持能力、适用硬件/运行时版本和 OOD 行为说明。

本方案的最终成功标准不是“某个校准案例误差接近 0”，而是：在冻结后的真实 Ascend 910B2 × 8 PyNative blind 测试中，DeepSeek V3、DeepSeek V4、Qwen3 及关键并行/offload/recompute 策略同时达到误差、区间和 OOM 安全门，并且每个预测都能由物理内存事件和互斥分量解释。
