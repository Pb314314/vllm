# Model forward comparison

Real weights and routing; synthetic tokens and zero historical KV. Fixed KV allocation across budgets. GPU timings are max-rank one-shot graph forward; not serving throughput. No hidden/token bit-equivalence claim when baseline or no-op paths are nondeterministic. Numerical exceptions are preserved in raw outputs. The refinement adds production-capacity indexer host bounds, actual-bound eager wall time and graph-private reservation.

|Case|Budget MiB|Graph ms|Change|Peak saved GiB/GPU|Max sample spread|
|---|---:|---:|---:|---:|---:|
|q8_b512_l8k|0|71.757|+0.000%|0.000|0.48%|
|q8_b512_l8k|1024|71.955|+0.275%|8.573|0.31%|
|q8_b512_l8k|1536|71.883|+0.175%|8.323|0.09%|
|q8_b512_l8k|2048|71.768|+0.015%|7.503|0.17%|
|q8_b512_l8k|4096|71.662|-0.133%|5.002|0.29%|
|q8_b1024_l4k|0|118.534|+0.000%|0.000|0.11%|
|q8_b1024_l4k|1024|119.048|+0.434%|18.008|0.10%|
|q8_b1024_l4k|1536|118.890|+0.300%|18.008|0.04%|
|q8_b1024_l4k|2048|118.795|+0.220%|17.507|0.07%|
|q8_b1024_l4k|4096|118.665|+0.111%|15.006|11.94%|
