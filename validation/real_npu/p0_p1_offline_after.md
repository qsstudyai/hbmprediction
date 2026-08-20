# Current hbmprediction historical-label replay

Model source SHA-256: `fab1f04f724aea984b15ecc8a84c68930d1a1eb33986e6d7448202a840261840`

Eligible unique labels: 25; diagnostics: 15.

| Family | Metric | Mean APE | Max APE | Mean signed error GiB | n |
|---|---|---:|---:|---:|---:|
| deepseek_v3 | active | 6.480% | 29.721% | 0.159 | 14 |
| deepseek_v3 | dynamic | 26.386% | 48.229% | -4.697 | 14 |
| deepseek_v3 | total | 38.082% | 56.203% | -8.142 | 14 |
| deepseek_v3 | legacy_total | 15.504% | 26.515% | -3.287 | 14 |
| qwen3 | active | 12.967% | 13.226% | -0.875 | 11 |
| qwen3 | dynamic | 35.925% | 50.486% | -3.104 | 11 |
| qwen3 | total | 53.972% | 74.629% | -6.442 | 11 |
| qwen3 | legacy_total | 35.684% | 52.444% | -4.213 | 11 |
| overall | active | 9.334% | 29.721% | -0.296 | 25 |
| overall | dynamic | 30.584% | 50.486% | -3.996 | 25 |
| overall | total | 45.074% | 74.629% | -7.394 | 25 |
| overall | legacy_total | 24.383% | 52.444% | -3.694 | 25 |

`active` compares the raw evaluator ledger with profiler model-active bytes. `dynamic` compares it with max(active, allocator pool) plus untracked runtime. `legacy_total` adds the archived legacy reserve and compares device total HBM.
