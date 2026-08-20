# Current hbmprediction real-NPU baseline

Model source SHA-256: `900500886de8da20c4f4abcb56764254cacddfa469b272ec91ede3e0c52b6cae`

Eligible unique labels: 25; diagnostics: 15.

| Family | Metric | Mean APE | Max APE | Mean signed error GiB | n |
|---|---|---:|---:|---:|---:|
| deepseek_v3 | active | 17.653% | 46.835% | -1.539 | 14 |
| deepseek_v3 | dynamic | 36.656% | 59.976% | -6.395 | 14 |
| deepseek_v3 | legacy_total | 23.993% | 48.621% | -4.985 | 14 |
| qwen3 | active | 72.770% | 80.679% | -5.008 | 11 |
| qwen3 | dynamic | 80.294% | 85.266% | -7.237 | 11 |
| qwen3 | legacy_total | 67.757% | 78.898% | -8.347 | 11 |
| overall | active | 41.904% | 80.679% | -3.066 | 25 |
| overall | dynamic | 55.857% | 85.266% | -6.765 | 25 |
| overall | legacy_total | 43.249% | 78.898% | -6.464 | 25 |

`active` compares the raw evaluator ledger with profiler model-active bytes. `dynamic` compares it with max(active, allocator pool) plus untracked runtime. `legacy_total` adds the archived legacy reserve and compares device total HBM.
