# P0/P1 HBM validation gates

Evidence scope: `historical_calibration_replay`; unique labels: 25.

P0 numeric: **pass**; P1 numeric: **fail**; production: **validation_unavailable**.

| Gate | Status | Observed | Requirement | Scope |
|---|---|---|---|---|
| p0_physical_active | pass | mean_ape_pct=9.334, worst_ape_pct=29.721 | mean <= 25%, worst <= 35% | historical_calibration_replay |
| p1_dynamic_point | pass | mean_ape_pct=8.961, worst_ape_pct=24.212 | mean <= 20%, worst <= 30% | historical_calibration_replay |
| p1_total_point | fail | mean_ape_pct=7.118, worst_ape_pct=20.478 | mean <= 15%, worst <= 20% | historical_calibration_replay |
| p1_safe_upper_coverage | pass | covered=23, total=25, coverage=92.000% | coverage >= 80% | historical_calibration_replay |
| blind_holdout | unavailable | — | new frozen holdouts for every supported family/layout | production_validation |
| actual_oom_false_safe | unavailable | — | at least one actual OOM and zero false-safe predictions | production_validation |
| stage_rank_peak_order | unavailable | — | peak ordering accuracy >= 80% on new labels | production_validation |
| strategy_paired_delta | unavailable | — | paired strategy delta error <= 20% | production_validation |

`unavailable` is not a pass. Historical calibration replay cannot establish blind generalization, OOM safety, or paired-strategy accuracy.
