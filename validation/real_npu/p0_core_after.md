# Current hbmprediction real-NPU baseline

Model source SHA-256: `20b891f65b8bde26ebb4e5ea668d07adc97ccaba4c217fda09bdcc5326f30fcd`

Eligible unique labels: 25; diagnostics: 15.

| Family | Metric | Mean APE | Max APE | Mean signed error GiB | n |
|---|---|---:|---:|---:|---:|
| deepseek_v3 | active | 11.523% | 30.264% | -0.847 | 14 |
| deepseek_v3 | dynamic | 32.400% | 51.605% | -5.703 | 14 |
| deepseek_v3 | legacy_total | 20.480% | 38.721% | -4.293 | 14 |
| qwen3 | active | 40.342% | 41.271% | -2.728 | 11 |
| qwen3 | dynamic | 56.098% | 65.386% | -4.957 | 11 |
| qwen3 | legacy_total | 50.184% | 60.079% | -6.066 | 11 |
| overall | active | 24.203% | 41.271% | -1.675 | 25 |
| overall | dynamic | 42.827% | 65.386% | -5.375 | 25 |
| overall | legacy_total | 33.550% | 60.079% | -5.073 | 25 |

`active` compares the raw evaluator ledger with profiler model-active bytes. `dynamic` compares it with max(active, allocator pool) plus untracked runtime. `legacy_total` adds the archived legacy reserve and compares device total HBM.
