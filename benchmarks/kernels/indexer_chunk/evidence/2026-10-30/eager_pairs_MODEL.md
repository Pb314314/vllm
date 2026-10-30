# Interleaved same-batch eager comparison

All visits retained, including the initial 512-request baseline transient. Each nonzero visit is compared to the arithmetic mean of its adjacent unchunked visit medians. Bracket disagreement is uncertainty, not a proven effect of chunking. GPU events include CPU launch gaps.

|Case|Budget MiB|Host medians ms (two visits)|Adjacent-baseline changes|Baseline bracket disagreement|
|---|---:|---|---|---|
|q8_b1024_l4k|1024|184.25 / 179.04|6.44 / 6.42%|2.93 / 0.59%|
|q8_b1024_l4k|1536|176.33 / 175.30|4.08 / 3.70%|1.36 / 0.37%|
|q8_b1024_l4k|2048|173.97 / 174.83|3.73 / 3.34%|0.65 / 0.21%|
|q8_b1024_l4k|4096|171.97 / 171.59|2.16 / 1.38%|1.38 / 0.29%|
|q8_b512_l8k|1024|175.27 / 178.19|-6.30 / 3.91%|19.08 / 0.02%|
|q8_b512_l8k|1536|175.25 / 175.51|3.22 / 0.09%|0.67 / 4.37%|
|q8_b512_l8k|2048|175.44 / 175.88|2.35 / 0.22%|1.23 / 4.20%|
|q8_b512_l8k|4096|173.21 / 173.80|0.81 / 1.35%|0.75 / 0.37%|
