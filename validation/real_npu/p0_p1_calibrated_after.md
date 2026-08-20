# Current hbmprediction historical-label replay

Model source SHA-256: `fab1f04f724aea984b15ecc8a84c68930d1a1eb33986e6d7448202a840261840`

Eligible unique labels: 25; diagnostics: 15.

| Family | Metric | Mean APE | Max APE | Mean signed error GiB | n |
|---|---|---:|---:|---:|---:|
| deepseek_v3 | active | 6.480% | 29.721% | 0.159 | 14 |
| deepseek_v3 | dynamic | 7.686% | 24.212% | -0.362 | 14 |
| deepseek_v3 | total | 6.818% | 20.478% | -0.470 | 14 |
| deepseek_v3 | legacy_total | 15.504% | 26.515% | -3.287 | 14 |
| qwen3 | active | 12.967% | 13.226% | -0.875 | 11 |
| qwen3 | dynamic | 10.584% | 23.628% | -0.631 | 11 |
| qwen3 | total | 7.500% | 15.388% | -0.631 | 11 |
| qwen3 | legacy_total | 35.684% | 52.444% | -4.213 | 11 |
| overall | active | 9.334% | 29.721% | -0.296 | 25 |
| overall | dynamic | 8.961% | 24.212% | -0.480 | 25 |
| overall | total | 7.118% | 20.478% | -0.541 | 25 |
| overall | legacy_total | 24.383% | 52.444% | -3.694 | 25 |

`active` compares the raw evaluator ledger with profiler model-active bytes. `dynamic` compares it with max(active, allocator pool) plus untracked runtime. `legacy_total` adds the archived legacy reserve and compares device total HBM.
