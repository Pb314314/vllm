# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import cProfile
import functools
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.session_scheduler import SessionKVCacheManager, SessionScheduler

TAG = None
TIMES = defaultdict(list)
SHAPES = []
ROOT = Path(os.environ["SESSION_PROFILE_DIR"])
ROOT.mkdir(parents=True, exist_ok=True)


def instrument(cls, name):
    original = getattr(cls, name)
    label = cls.__name__ + "." + name

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        if TAG is None:
            return original(*args, **kwargs)
        start = time.perf_counter_ns()
        try:
            result = original(*args, **kwargs)
            if name == "schedule":
                SHAPES.append(
                    {
                        "tokens": dict(result.num_scheduled_tokens),
                        "new_requests": [
                            {
                                "prompt_tokens": x.prompt_len,
                                "prefill_tokens": len(x.prefill_token_ids),
                                "block_ids": sum(map(len, x.block_ids)),
                                "computed_tokens": x.num_computed_tokens,
                            }
                            for x in result.scheduled_new_reqs
                        ],
                    }
                )
            return result
        finally:
            TIMES[label].append((time.perf_counter_ns() - start) / 1e6)

    setattr(cls, name, wrapped)


class ProfileMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        from vllm.v1.engine.core import EngineCore
        from vllm.v1.worker.gpu.block_table import BlockTables
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
        from vllm.v1.worker.gpu.states import RequestState

        original = EngineCore.preprocess_add_request

        def preprocess(engine, request):
            extra = request.sampling_params.extra_args or {}
            tag = extra.get("profile_tag")
            if tag != "cpu":
                return original(engine, request)
            profiler = cProfile.Profile()
            result = profiler.runcall(original, engine, request)
            profiler.dump_stats(str(ROOT / "engine-input.pstats"))
            return result

        EngineCore.preprocess_add_request = preprocess
        for cls, names in [
            (
                Scheduler,
                [
                    "add_request",
                    "schedule",
                    "update_from_output",
                    "_free_request_blocks",
                ],
            ),
            (
                KVCacheManager,
                [
                    "get_computed_blocks",
                    "allocate_slots",
                    "cache_blocks",
                    "free",
                    "get_blocks",
                ],
            ),
            (
                GPUModelRunner,
                [
                    "add_requests",
                    "update_requests",
                    "finish_requests",
                    "prepare_inputs",
                    "prepare_attn",
                    "execute_model",
                    "sample_tokens",
                ],
            ),
            (RequestState, ["add_request", "apply_staged_writes", "remove_request"]),
            (BlockTables, ["append_block_ids", "apply_staged_writes"]),
            (SessionScheduler, ["_bind_session"]),
            (SessionKVCacheManager, ["resume", "retain", "evict"]),
        ]:
            for name in names:
                instrument(cls, name)
        self._profile_req = None
        self._cpu_profile = None

    def add_request(self, request):
        global TAG
        tag = (request.sampling_params.extra_args or {}).get("profile_tag")
        if tag:
            TAG = tag
            TIMES.clear()
            SHAPES.clear()
            self._profile_req = request.request_id
            if tag == "cpu":
                self._cpu_profile = cProfile.Profile()
                self._cpu_profile.enable()
        return super().add_request(request)

    def update_from_output(self, *args, **kwargs):
        global TAG
        result = super().update_from_output(*args, **kwargs)
        if self._profile_req is not None and self._profile_req not in self.requests:
            if self._cpu_profile is not None:
                self._cpu_profile.disable()
                self._cpu_profile.dump_stats(str(ROOT / "engine-loop.pstats"))
                self._cpu_profile = None
            report = {
                "tag": TAG,
                "host_inclusive_ms": dict(TIMES),
                "schedule_shapes": SHAPES,
                "note": (
                    "Nested inclusive spans; do not sum. CPU profile perturbs "
                    "timings. GPU calls may be asynchronous."
                ),
            }
            (ROOT / f"{TAG}-host-spans.json").write_text(json.dumps(report, indent=2))
            TAG = None
            self._profile_req = None
        return result


class APCProfileScheduler(ProfileMixin, Scheduler):
    pass


class SessionProfileScheduler(ProfileMixin, SessionScheduler):
    pass
