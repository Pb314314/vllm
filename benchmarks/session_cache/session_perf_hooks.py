# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.session_scheduler import SessionScheduler


class MetricsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.perf_steps = []
        self.perf_evictions = 0
        pool = self.kv_cache_manager.block_pool
        original = pool._maybe_evict_cached_block

        def count_eviction(block):
            result = original(block)
            self.perf_evictions += int(result)
            return result

        pool._maybe_evict_cached_block = count_eviction
        from vllm.v1.engine.core import EngineCore

        def snapshot(engine, reset=False):
            scheduler = engine.scheduler
            pool = scheduler.kv_cache_manager.block_pool
            result = {
                "steps": scheduler.perf_steps,
                "block_evictions": scheduler.perf_evictions,
                "session_evictions": getattr(
                    scheduler.kv_cache_manager, "num_session_evictions", 0
                ),
                "session_trims": getattr(
                    scheduler.kv_cache_manager, "num_session_trims", 0
                ),
                "session_evicted_blocks": getattr(
                    scheduler.kv_cache_manager, "num_session_evicted_blocks", 0
                ),
                "resident_blocks": sum(
                    not b.is_null and (b.ref_cnt > 0 or b.block_hash is not None)
                    for b in pool.blocks
                ),
                "capacity_blocks": len(pool.blocks) - 1,
            }
            if reset:
                scheduler.perf_steps = []
                scheduler.perf_evictions = 0
                if hasattr(scheduler.kv_cache_manager, "num_session_evictions"):
                    scheduler.kv_cache_manager.num_session_evictions = 0
                    scheduler.kv_cache_manager.num_session_trims = 0
                    scheduler.kv_cache_manager.num_session_evicted_blocks = 0
            return result

        EngineCore.session_perf_snapshot = snapshot

    def schedule(self, *args, **kwargs):
        result = super().schedule(*args, **kwargs)
        self.perf_steps.append(
            {
                "tokens": sum(result.num_scheduled_tokens.values()),
                "batch": len(result.num_scheduled_tokens),
                "new": [
                    {
                        "prompt_tokens": r.prompt_len,
                        "cached_tokens": r.num_computed_tokens,
                    }
                    for r in result.scheduled_new_reqs
                ],
            }
        )
        return result


class APCMetricsScheduler(MetricsMixin, Scheduler):
    pass


class SessionMetricsScheduler(MetricsMixin, SessionScheduler):
    pass
