# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the parked-async-KV-load deadlock (#65938).

Async KV loads (e.g. OffloadingConnector) allocate their loaded prefix's GPU
blocks at admission and park in `kv_holding_waiting`. Their un-cached tails
are reserved only against other async-load admissions, so blocks held by
holders the reservation does not cover (paused streaming sessions holding
their KV between turns, connector-pinned saves, synchronous admissions) can
leave the completed loads unable to allocate their tails. Parked requests
are not preemption candidates, so once `running` drains the engine deadlocks:
every step schedules nothing and no block is ever freed.

The scheduler must instead free the youngest completed load's blocks and
requeue it, which strictly grows the free pool until the head fits.
"""

from collections.abc import Callable

from vllm.v1.request import Request, RequestStatus

from .utils import (
    assert_scheduler_empty,
    create_mock_connector,
    create_model_runner_output,
    create_request,
    create_scheduler,
    create_vllm_config,
)

BLOCK_SIZE = 16
# 26 GPU blocks -> 25 allocatable (block 0 is the reserved null block).
NUM_BLOCKS = 26
# Each async load: 12-block full sequence; its 8-block prefix parks at
# admission, leaving a 4-block un-cached tail.
PROMPT_TOKENS = 12 * BLOCK_SIZE
EXTERNAL_TOKENS = 8 * BLOCK_SIZE
# Blocks pinned by holders the async admission reservation does not cover
# (e.g. paused streaming sessions holding KV between turns). Chosen so the
# completed loads' 4-block tails no longer fit: 25 - 8 - 8 - 6 = 3 < 4.
PINNED_BLOCKS = 6


def _make_hits(
    req_hits: dict[str, int], async_load: bool = True
) -> Callable[[Request, int], tuple[int, bool]]:
    def get_num_new_matched_tokens(request: Request, _: int) -> tuple[int, bool]:
        hit = req_hits.get(request.request_id, 0)
        return hit, async_load and hit > 0

    return get_num_new_matched_tokens


def _make_scheduler_with_two_loads():
    scheduler = create_scheduler(
        create_vllm_config(
            max_num_batched_tokens=1024,
            block_size=BLOCK_SIZE,
        ),
        num_blocks=NUM_BLOCKS,
    )
    load1 = create_request(num_tokens=PROMPT_TOKENS)
    load2 = create_request(num_tokens=PROMPT_TOKENS)
    scheduler.add_request(load1)
    scheduler.add_request(load2)

    hits = {
        load1.request_id: EXTERNAL_TOKENS,
        load2.request_id: EXTERNAL_TOKENS,
    }
    scheduler.connector = create_mock_connector()
    scheduler.connector.get_loaded_kv_cache_group_ids.return_value = (0,)
    scheduler.connector.get_num_new_matched_tokens.side_effect = _make_hits(hits)
    scheduler.connector.request_finished.return_value = (False, None)
    scheduler.connector.take_events.return_value = ()
    scheduler.connector.has_pending_block_frees.return_value = False
    return scheduler, load1, load2


def test_parked_async_loads_cannot_deadlock_engine():
    scheduler, load1, load2 = _make_scheduler_with_two_loads()
    block_pool = scheduler.kv_cache_manager.block_pool

    # Step 1: both async loads are admitted and park holding their 8-block
    # prefixes (25 - 8 - 8 = 9 free).
    scheduler.schedule()
    assert load1.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert load2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert block_pool.get_num_free_blocks() == 9

    # Blocks get pinned by holders outside the async-load reservation (e.g.
    # paused streaming sessions), leaving less than a tail free.
    pinned = block_pool.get_new_blocks(PINNED_BLOCKS)
    assert block_pool.get_num_free_blocks() == 3

    # Both loads complete with no request running.
    runner_out = create_model_runner_output(
        [], finished_recving={load1.request_id, load2.request_id}
    )
    scheduler.update_from_output(scheduler.schedule(), runner_out)
    assert not scheduler.running

    # Deadlock state: 3 free blocks fit neither completed load's 4-block
    # tail, and nothing is running to ever free more. On unfixed code every
    # step schedules nothing forever; the stall breaker must free the
    # youngest completed load's blocks and requeue it as a fresh request.
    sched_out = scheduler.schedule()
    assert sched_out.total_num_scheduled_tokens == 0
    assert load2.status == RequestStatus.WAITING
    assert load2.num_computed_tokens == 0
    assert load2 in list(scheduler.waiting)
    assert load2 not in list(scheduler.kv_holding_waiting)
    assert not scheduler.finished_recving_kv_req_ids
    # 25 usable - 8 (load1 prefix) - 6 (pinned).
    assert block_pool.get_num_free_blocks() == 11

    # Drive to completion: load1's tail now fits, so it promotes and runs;
    # load2 is eventually re-admitted as a new async load (its prefix is
    # still in the CPU tier), completes, runs, and finishes. Everything
    # drains.
    for _ in range(8):
        sched_out = scheduler.schedule()
        scheduled = set(sched_out.num_scheduled_tokens)
        still_loading = {
            r.request_id
            for r in (load1, load2)
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        }
        runner_out = create_model_runner_output(
            [r for r in (load1, load2) if r.request_id in scheduled],
            finished_recving=still_loading or None,
            use_eos=True,
        )
        scheduler.update_from_output(sched_out, runner_out)

    block_pool.free_blocks(pinned)
    assert_scheduler_empty(scheduler)


def test_stall_breaker_does_not_fire_while_running():
    """No victim is freed while any request can still run or be scheduled."""
    scheduler, load1, load2 = _make_scheduler_with_two_loads()
    block_pool = scheduler.kv_cache_manager.block_pool

    scheduler.schedule()
    assert block_pool.get_num_free_blocks() == 9
    pinned = block_pool.get_new_blocks(PINNED_BLOCKS)

    # The loads complete, but a request is still running (it will free its
    # blocks when it finishes): the breaker must not fire.
    consumer = create_request(num_tokens=BLOCK_SIZE)
    scheduler.add_request(consumer)
    sched_out = scheduler.schedule()
    assert consumer.status == RequestStatus.RUNNING

    runner_out = create_model_runner_output(
        [consumer], finished_recving={load1.request_id, load2.request_id}
    )
    scheduler.update_from_output(sched_out, runner_out)

    sched_out = scheduler.schedule()
    # The consumer keeps running; the completed loads still cannot allocate.
    assert consumer.status == RequestStatus.RUNNING
    assert load1.status == RequestStatus.WAITING
    assert load2.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    # No stall-breaking victim was freed: both loads keep their blocks.
    assert load1.num_computed_tokens == EXTERNAL_TOKENS
    assert load2.num_computed_tokens == EXTERNAL_TOKENS
    assert len(scheduler.kv_holding_waiting) == 2

    # Cleanup.
    block_pool.free_blocks(pinned)
