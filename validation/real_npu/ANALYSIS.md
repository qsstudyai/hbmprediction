# Qwen3 / DeepSeek-V3 real-NPU HBM evidence analysis

> 2026-08-19 correction note: this file preserves the imported historical
> evidence analysis. The newly implemented physical ledger, loss contract and
> calibration-only runtime profiles are evaluated in `p0_p1_calibrated_after.md`
> and summarized in `MODEL_CARD.md`. Those calibration metrics do not replace
> the failed current blind/OOM gate described below.

## Bottom line

All 136 copied evidence files pass SHA-256 verification. The set contains 40 rows but only 25 independent, formally eligible HBM labels after content-hash deduplication ({'deepseek_v3': 14, 'qwen3': 11}).

The physical model is a useful baseline, not an accurate final predictor: overall total-HBM MAPE is 31.919% (max 51.999%, n=25), and dynamic-HBM MAPE is 41.759% (max 69.070%, n=25).

The 13 historical frozen holdouts improve this to total-HBM MAPE 10.112% (max 28.446%, n=13) and dynamic-HBM MAPE 13.425% (max 55.542%, n=13). These are versioned historical predictions, not a current blind test set.

The current v17 acceptance gate is **FAIL**: it has 0/4 eligible holdouts and 0/1 actual OOM labels. Therefore the evidence does not yet validate generalization or the OOM classification boundary.

## Error by family

| Model | Physical total APE | Physical dynamic APE | Frozen total APE | Frozen dynamic APE |
|---|---:|---:|---:|---:|
| qwen3 | 44.331% (max 51.999%, n=11) | 61.468% (max 69.070%, n=11) | 8.060% (max 28.446%, n=7) | 13.248% (max 55.542%, n=7) |
| deepseek_v3 | 22.167% (max 40.037%, n=14) | 26.273% (max 47.338%, n=14) | 12.506% (max 22.447%, n=6) | 13.632% (max 26.347%, n=6) |

## Measured scaling at fixed layouts

| Layout | Total GiB/layer | Total R² | Dynamic GiB/layer | Dynamic R² | n |
|---|---:|---:|---:|---:|---:|
| deepseek_v3:tp2-cp2-ep2 | 0.2242 | 0.4449 | 0.2147 | 0.5407 | 8 |
| deepseek_v3:tp2-pp2-dp2-interleave | 0.4880 | 1.0000 | 0.4880 | 1.0000 | 5 |
| qwen3:dp8 | 0.0511 | 0.4343 | 0.0511 | 0.4340 | 8 |

The slopes are descriptive within each fixed layout; they are not interchangeable across TP/CP/PP/EP strategies.

## Data-quality limits

- One calibration row is diagnostic-only because of a contaminated cross-rank baseline.
- Minimum valid per-rank HBM sample count is 21.
- Valid baseline range is 3.336–4.850 GiB.
- The four intended v17 holdouts are diagnostic-only because their timing archives lack `case_result.json`.
- Raw profiler traces, checkpoints, logs, and datasets remain in the source repository and are intentionally not copied.

Reproduce with `python tools/import_parallelsearch_hbm.py` followed by `python tools/analyze_real_npu.py`.
