# Imported real-NPU evidence

This directory is a small, auditable evidence layer imported from the sibling
`parallelsearch` repository at the commit recorded in `manifest.json`.

It contains Qwen3 and DeepSeek-V3 comparison summaries, collector
`case_result.json` labels, exact profile configs, available frozen predictions,
and the v17 calibration controls. Each copied file has a source path and SHA-256
digest in the manifest. The 245 GiB raw profiler tree, checkpoints, logs, and
datasets are deliberately not duplicated.

Rebuild and analyze from the repository root:

```bash
python tools/import_parallelsearch_hbm.py
python tools/analyze_real_npu.py
```

`ANALYSIS.md` is the human-readable result and `analysis.json` is the
machine-readable result. Formal metrics deduplicate HBM labels by their source
content hash, because calibration and historical comparison names sometimes
refer to the same physical run.

The current evidence is suitable for calibration diagnostics and historical
frozen-prediction error analysis. It is not a passing current blind-holdout
validation: the v17 matrix has no eligible holdout and contains no actual OOM
label.

Current correction status and the machine-readable gate audit are in
`MODEL_CARD.md` and `validation_gates_v1.{json,md}`. Rebuild the offline audit
without running NPU code:

```bash
python tools/evaluate_current_hbm.py --root validation/real_npu \
  -o validation/real_npu/p0_p1_offline_after.json \
  --markdown validation/real_npu/p0_p1_offline_after.md
python tools/build_runtime_profiles.py --root validation/real_npu \
  -o validation/real_npu/runtime_profiles_v1.json
python tools/evaluate_current_hbm.py --root validation/real_npu \
  --runtime-profiles validation/real_npu/runtime_profiles_v1.json \
  -o validation/real_npu/p0_p1_calibrated_after.json \
  --markdown validation/real_npu/p0_p1_calibrated_after.md
python tools/check_hbm_gates.py \
  -o validation/real_npu/validation_gates_v1.json \
  --markdown validation/real_npu/validation_gates_v1.md
```
