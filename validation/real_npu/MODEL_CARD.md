# HBM physical model card — 2026-08-19 offline correction

## Status

`offline-software-complete / calibration-gates-partial / blind-validation-blocked`

The P0/P1 correction code is implemented and passes the repository's offline
software gates. It is **not production-validated**: no new NPU run is counted
in this release, current v17 has no eligible blind holdout, there is no valid
actual-OOM label, and DeepSeek-V4 probe/HBM gates remain pending. At the user's
direction, NPU execution was stopped; the attempted collector preflight failed
before training started and is not evidence.

## Implemented contracts

- value identity is separate from explicit shared-storage identity;
- multi-axis CP/TP/SP/EP placement and deterministic 0..N reshard plans;
- checked `ALLOC/FREE/ALIAS/MOVE_IN/MOVE_OUT` ledger with simultaneous peaks;
- forward saved/workspace overlap, backward workspace, PP boundary staging,
  per-rank output and interleave gather fix;
- explicit fallback/fused cross-entropy, FP32 softmax/probability/dlogits;
- device baseline, allocator pool/slack, runtime, fragmentation, point and safe
  upper as mutually named components;
- Qwen3 family graph, DeepSeek V3/V4 contracts, MoE routing metadata and
  grouped-GEMM workspace;
- version-aware workspace registry, FSDP flatten padding, independent offload,
  AdamW and mixed Muon state/step contracts;
- schema-v2 OOM states: `definitely_safe`, `risky`, `predicted_oom`,
  `unsupported`.

## Historical calibration replay

Source: 25 unique, formally eligible imported Qwen3/DeepSeek-V3 HBM labels,
deduplicated by label SHA-256. Runtime profiles use calibration rows only and
are keyed by family and exact parallel layout.

| Metric | Mean APE | Worst APE |
|---|---:|---:|
| physical active | 9.334% | 29.721% |
| dynamic point | 8.961% | 24.212% |
| total point | 7.118% | 20.478% |

Calibration safe-upper coverage is 23/25 (92%). The P0 physical gate, P1
dynamic gate, P1 total mean gate and safe-upper coverage gate pass on this
replay. The P1 total worst gate misses its 20% limit by 0.478 percentage points.
None of these figures may be cited as blind generalization results.

Artifacts:

- `current_model_baseline_v1.{json,md}` — pre-correction baseline;
- `p0_core_after.{json,md}` — identity/ledger/loss intermediate result;
- `p0_p1_offline_after.{json,md}` — physical-only corrected replay;
- `runtime_profiles_v1.json` — content-addressed calibration profiles;
- `p0_p1_calibrated_after.{json,md}` — calibrated point/interval replay;
- `validation_gates_v1.{json,md}` — explicit pass/fail/unavailable gate audit.

## Capability state

| Capability | State |
|---|---|
| Qwen3 DP8 / TP2 / TP4 historical layouts | experimental, calibration-backed |
| DeepSeek-V3 TP+CP+EP and PP+interleave historical layouts | experimental, calibration-backed |
| loss parallel | software-contract tested; paired NPU delta pending |
| FSDP/HSDP, recompute, Swap, offload, Muon | software-contract tested; paired NPU delta pending |
| DeepSeek-V4 | unsupported for safe decisions until probe and HBM blind pass |
| unknown loss/dispatcher/optimizer/runtime | rejected or risky/unsupported |

## Remaining release gates

1. Freeze and collect new Qwen3, DeepSeek-V3 and DeepSeek-V4 blind holdouts.
2. Collect at least one actual OOM plus a near-boundary success case.
3. Run paired loss, FSDP, recompute, PP/interleave, dispatcher, offload, Swap
   and optimizer experiments; require delta error at or below 20%.
4. Verify stage/rank ordering and safe-upper coverage on new labels.
5. Only then change status to `production-validated`.
