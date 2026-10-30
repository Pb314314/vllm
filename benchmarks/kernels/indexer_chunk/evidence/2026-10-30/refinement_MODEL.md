# Model forward comparison

Real weights and routing; synthetic tokens and zero historical KV. Fixed KV allocation across budgets. GPU timings are max-rank one-shot graph forward; not serving throughput. No hidden/token bit-equivalence claim when baseline or no-op paths are nondeterministic. Numerical exceptions are preserved in raw outputs. The refinement adds production-capacity indexer host bounds, actual-bound eager wall time and graph-private reservation.

|Case|Budget MiB|Graph ms|Change|Peak saved GiB/GPU|Max sample spread|
|---|---:|---:|---:|---:|---:|
|q8_b256_l8k|0|40.156|+0.000%|0.000|0.14%|
|q8_b256_l8k|2048|40.175|+0.048%|2.501|0.12%|
|q8_b256_l8k|4096|40.167|+0.027%|0.000|0.15%|
|q8_b512_l8k|0|71.544|+0.000%|0.000|0.36%|
|q8_b512_l8k|2048|71.689|+0.204%|7.503|0.28%|
|q8_b512_l8k|4096|71.578|+0.048%|5.002|0.32%|
|q8_b1024_l4k|0|118.817|+0.000%|0.000|0.05%|
|q8_b1024_l4k|2048|119.070|+0.213%|17.507|0.06%|
|q8_b1024_l4k|4096|118.938|+0.101%|15.006|0.04%|
|q8_mixed|0|46.635|+0.000%|0.000|0.12%|
|q8_mixed|2048|46.774|+0.296%|4.142|0.13%|
|q8_mixed|4096|46.663|+0.059%|3.142|0.12%|
