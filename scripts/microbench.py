#!/usr/bin/env python3
"""Processor micro-benchmark + label dump for the MiMo video-timestamp fix.

Times the multimodal processor on a real 150 s clip (N reps), prints
p50/p99 and the injected labels + prompt token count. Runs on CPU, before
the server starts (no GPU contention).
"""
import json
import os
import re
import statistics
import sys
import time

MODEL = os.environ.get("MICROBENCH_MODEL", "XiaomiMiMo/MiMo-V2.6-Flash-RL")
CLIP = "$TASKDIR/clips/flash_060.mp4"

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO

mc = ModelConfig(
    MODEL,
    trust_remote_code=True,
    dtype="bfloat16",
    seed=0,
    limit_mm_per_prompt={"image": 2, "video": 2},
    enforce_eager=True,
)
proc = MULTIMODAL_REGISTRY.create_processor(mc)
tok = proc.info.get_tokenizer()

from pathlib import Path

video = VideoMediaIO(ImageMediaIO()).load_file(Path(CLIP))
mm_items = proc.info.parse_mm_data({"video": [video]})
prompt = "<|vision_start|><|video_pad|><|vision_end|>"

# warmup
processed = proc(prompt, mm_items=mm_items)
ids = processed["prompt_token_ids"]
labels = re.findall(r"\d\d:\d\d", tok.decode(ids))
data = processed["mm_kwargs"]["video"][0].get_data()
print("labels:", labels)
print("second_per_grid_ts:", float(data["second_per_grid_ts"]))
print("video_start_times:", float(data["video_start_times"]))
print("prompt_tokens:", len(ids))

# NOTE: the video is decoded once above (loader); the timed region is the
# processor call itself, which is what this PR changes.
times = []
for _ in range(30):
    t0 = time.perf_counter()
    proc(prompt, mm_items=mm_items)
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(
    "processor_ms: p50=%.2f p99=%.2f min=%.2f max=%.2f"
    % (times[len(times) // 2], times[int(len(times) * 0.99) - 1], times[0], times[-1])
)
print("MICRO_DONE")
