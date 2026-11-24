# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Session KV reuse through the real scheduler, KV manager and resolver.

Requests carry no session id unless stated: the resolver must find reuse from
token content alone, and the store must never hand out KV whose content or
ownership is wrong.
"""

import random
from collections import Counter

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.session.protocol import Candidate, SessionBinding
from vllm.v1.session.resolver import SessionResolver, SessionState

from .utils import EOS_TOKEN_ID, create_scheduler, new_mamba_spec

BLOCK = 16
SAMPLED = 7


class Harness:
    def __init__(self, num_blocks: int = 1000, hybrid: bool = False, **kwargs):
        groups = None
        if hybrid:
            groups = [
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32
                ),
                new_mamba_spec(block_size=2048, num_speculative_blocks=0),
            ]
        self.sched: Scheduler = create_scheduler(
            session_cache=True,
            num_blocks=num_blocks,
            block_size=BLOCK,
            max_model_len=2048,
            kv_cache_groups=groups,
            scheduler_block_size=2048 if hybrid else None,
            **kwargs,
        )
        self.store = self.sched.kv_cache_manager
        self.resolver = SessionResolver()
        self.store.connect(self.resolver.apply, self.resolver.resolve)
        self._next = 0
        # Tokens each request reused when it was first scheduled.
        self.cached: dict[str, int] = {}
        # Simulated KV content: a slot (or Mamba state) holds the key of the
        # token prefix it was computed from.
        self.slots: dict[int, dict[int, int]] = {}
        self.states: dict[int, int] = {}

    def add(
        self, tokens: list[int], max_tokens: int = 3, session_id: str | None = None
    ) -> Request:
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, EOS_TOKEN_ID)
        request = Request(
            request_id=f"r{self._next}",
            prompt_token_ids=list(tokens),
            sampling_params=params,
            pooling_params=None,
            session_id=session_id,
        )
        self._next += 1
        request.session_binding = self.resolver.resolve(request)
        self.sched.add_request(request)
        return request

    def step(self):
        output = self.sched.schedule()
        for new in output.scheduled_new_reqs:
            self.cached.setdefault(new.req_id, new.num_computed_tokens)
        for copy in output.kv_cache_block_copies or ():
            self.slots[copy.dst_block_id] = dict(self.slots.get(copy.src_block_id, {}))
            if copy.src_block_id in self.states:
                self.states[copy.dst_block_id] = self.states[copy.src_block_id]
        for req_id, num_tokens in output.num_scheduled_tokens.items():
            request = self.sched.requests[req_id]
            start = request.num_computed_tokens - num_tokens
            self._verify(request, start)
            self._write(request, start, start + num_tokens)
        req_ids = list(output.num_scheduled_tokens)
        sampled = []
        for req_id in req_ids:
            request = self.sched.requests[req_id]
            done = (
                request.num_computed_tokens + output.num_scheduled_tokens[req_id]
                >= request.num_tokens
            )
            sampled.append([SAMPLED] if done else [])
        self.sched.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=req_ids,
                sampled_token_ids=sampled,
            ),
        )
        check_invariants(self)
        return output

    def _groups(self, request: Request):
        for manager in self.store.coordinator.single_type_managers:
            yield manager, manager.req_to_blocks[request.request_id]

    def _verify(self, request: Request, end: int) -> None:
        """The KV the request reads must come from its own token prefix."""
        keys = prefix_keys(request.all_token_ids, end)
        for manager, blocks in self._groups(request):
            if isinstance(manager, MambaManager):
                if end:
                    assert self.states.get(blocks[0].block_id) == keys[end]
                continue
            size = manager.block_size
            for pos in range(end):
                slot = self.slots.get(blocks[pos // size].block_id, {})
                assert slot.get(pos % size) == keys[pos + 1], (request, pos)

    def _write(self, request: Request, start: int, end: int) -> None:
        end = min(end, request.num_tokens)
        keys = prefix_keys(request.all_token_ids, end)
        for manager, blocks in self._groups(request):
            if isinstance(manager, MambaManager):
                self.states[blocks[0].block_id] = keys[end]
                continue
            size = manager.block_size
            for pos in range(start, end):
                block_id = blocks[pos // size].block_id
                self.slots.setdefault(block_id, {})[pos % size] = keys[pos + 1]

    def run(self, *requests: Request, max_steps: int = 100):
        for _ in range(max_steps):
            if all(r.is_finished() for r in requests):
                return
            self.step()
        raise AssertionError("requests did not finish")

    def session(self, request: Request):
        return self.resolver.sessions.get(request.session_binding.session_id)


def check_invariants(h: Harness) -> None:
    store = h.store
    expected: Counter[int] = Counter()
    for node in store.nodes.values():
        assert node.end > node.start
        if node.parent is not None:
            assert node.parent.node_id in store.nodes
            assert node.start <= node.parent.end
        for group in node.blocks:
            expected.update(b.block_id for b in group)
    for manager in store.coordinator.single_type_managers:
        for blocks in manager.req_to_blocks.values():
            expected.update(b.block_id for b in blocks if not b.is_null)
    for block in store.block_pool.blocks:
        if not block.is_null:
            assert block.ref_cnt == expected[block.block_id], block.block_id
            assert block.block_hash is None
    num_free = sum(
        b.ref_cnt == 0 for b in store.block_pool.blocks if not b.is_null
    )
    assert store.block_pool.get_num_free_blocks() == num_free
    # Same process, no lag: the resolver mirrors the store exactly.
    assert h.resolver.seq == store.seq
    assert h.resolver.nodes.keys() == store.nodes.keys()
    for node_id, node in store.nodes.items():
        assert h.resolver.nodes[node_id].end == node.end
    for request in h.sched.requests.values():
        assert not request.block_hashes


def prefix_keys(tokens, end: int) -> list[int]:
    keys = [0]
    for token in tokens[:end]:
        keys.append(hash((keys[-1], token)))
    return keys


def history(request: Request) -> list[int]:
    return list(request.all_token_ids)


def test_anonymous_continuation_resumes_at_block_boundary():
    h = Harness()
    first = h.add(list(range(100, 140)))
    h.run(first)
    assert first.num_computed_tokens == 42  # last sampled token not computed

    second = h.add(history(first) + list(range(200, 210)))
    output = h.step()
    # Attention-only: recompute the partial block rather than copy it.
    assert h.cached[second.request_id] == 32
    assert output.num_scheduled_tokens[second.request_id] == 43 + 10 - 32
    assert not output.kv_cache_block_copies
    assert h.session(second) is h.session(first)
    h.run(second)
    assert h.session(first).turns == 2
    assert h.session(first).state == SessionState.IDLE


def test_concurrent_requests_share_a_running_prompt():
    h = Harness()
    system = list(range(300, 348))
    first = h.add(system + [1, 2, 3], max_tokens=20)
    h.step()
    h.step()  # first's prompt is published once its decode is allocated
    second = h.add(system + [4, 5, 6], max_tokens=2)
    h.step()
    assert h.cached[second.request_id] == 48
    assert h.session(second) is not h.session(first)
    h.run(first, second)


def test_fork_of_active_session_gets_its_own_session():
    h = Harness()
    root = h.add(list(range(40)))
    h.run(root)
    base = history(root)
    child = h.add(base + [1, 1], max_tokens=10)
    sibling = h.add(base + [2, 2], max_tokens=2)
    h.step()
    assert h.cached[child.request_id] == h.cached[sibling.request_id] == 32
    assert h.session(child) is h.session(root)
    assert h.session(sibling) is not h.session(root)
    h.run(child, sibling)


def test_edited_history_does_not_reuse_stale_kv():
    h = Harness()
    first = h.add(list(range(64)))
    h.run(first)
    edited = list(range(20)) + [999] * 50
    second = h.add(edited)
    h.step()
    assert h.cached[second.request_id] == 16
    assert h.session(second) is not h.session(first)
    assert h.session(first).cursor is not None
    h.run(second)


def test_namespaces_are_isolated():
    h = Harness()
    first = h.add(list(range(64)))
    h.run(first)
    params = SamplingParams(max_tokens=2, ignore_eos=True)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    salted = Request("salted", history(first) + [5], params, None, cache_salt="x")
    salted.session_binding = h.resolver.resolve(salted)
    h.sched.add_request(salted)
    h.step()
    assert h.cached[salted.request_id] == 0
    h.run(salted)


def test_evicted_candidate_falls_back_to_surviving_ancestor():
    h = Harness()
    root = h.add(list(range(48)))
    h.run(root)
    leaf = h.add(history(root) + list(range(500, 532)))
    h.run(leaf)
    late = h.add(history(leaf) + [9])
    # Evict leaf's own blocks before `late` is scheduled.
    leaf_node = h.store.nodes[h.session(leaf).cursor]
    h.store._remove(leaf_node)
    h.step()
    assert h.cached[late.request_id] == leaf_node.start // BLOCK * BLOCK
    h.run(late)


def test_memory_pressure_trims_idle_leaves_not_running_requests():
    h = Harness(num_blocks=24)
    finished = []
    for i in range(4):
        request = h.add([1000 * (i + 1) + t for t in range(60)], max_tokens=2)
        h.run(request)
        finished.append(request)
    assert h.store.nodes
    big = h.add(list(range(5000, 5000 + 300)), max_tokens=2)
    h.run(big)
    assert big.num_preemptions == 0
    assert h.store.num_evicted_blocks > 0


def test_abort_returns_session_to_idle_and_frees_nothing_shared():
    h = Harness()
    first = h.add(list(range(40)))
    h.run(first)
    second = h.add(history(first) + [3], max_tokens=50)
    h.step()
    assert h.session(first).state == SessionState.ACTIVE
    h.sched.finish_requests([second.request_id], RequestStatus.FINISHED_ABORTED)
    check_invariants(h)
    assert h.session(first).state == SessionState.IDLE
    third = h.add(history(first) + [4])
    h.step()
    assert h.cached[third.request_id] == 32
    assert h.session(third) is h.session(first)
    h.run(third)


def test_reset_invalidates_bindings():
    h = Harness()
    first = h.add(list(range(40)))
    h.run(first)
    pending = h.add(history(first) + [1])
    assert h.sched.reset_prefix_cache()
    assert not h.store.nodes
    h.step()
    assert h.cached[pending.request_id] == 0
    h.run(pending)


def test_hybrid_resumes_only_at_commit_points():
    h = Harness(hybrid=True)
    first = h.add(list(range(40)))
    h.run(first)
    second = h.add(history(first) + [5, 6])
    output = h.step()
    assert h.cached[second.request_id] == 42
    # Both the attention tail block and the Mamba state are copied.
    assert len(output.kv_cache_block_copies) == 2
    h.run(second)
    # Diverging inside first's span has no state to resume from.
    third = h.add(history(first)[:30] + [8] * 20)
    h.step()
    assert h.cached[third.request_id] == 0
    h.run(third)


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("seed", range(8))
def test_fuzz_invariants(seed: int, hybrid: bool):
    rng = random.Random(seed)
    h = Harness(num_blocks=60, hybrid=hybrid)
    done: list[Request] = []
    live: list[Request] = []
    for _ in range(400):
        op = rng.random()
        if op < 0.35 or not live:
            if done and rng.random() < 0.7:
                base = history(rng.choice(done))
                cut = rng.choice([len(base), len(base), rng.randrange(len(base))])
                tokens = base[:cut] + [rng.randrange(50) for _ in range(rng.randrange(1, 20))]
            else:
                tokens = [rng.randrange(50) for _ in range(rng.randrange(1, 80))]
            live.append(h.add(tokens[:300], max_tokens=rng.randrange(1, 8)))
        elif op < 0.42:
            victim = rng.choice(live)
            h.sched.finish_requests([victim.request_id], RequestStatus.FINISHED_ABORTED)
            check_invariants(h)
        elif op < 0.44:
            h.sched.reset_prefix_cache()
            check_invariants(h)
        else:
            h.step()
        for request in [r for r in live if r.is_finished()]:
            live.remove(request)
            done.append(request)
    while live:
        h.step()
        live = [r for r in live if not r.is_finished()]
    for state in h.resolver.sessions.values():
        assert state.state != SessionState.ACTIVE
    assert h.store.num_attached_requests > 0


def test_store_rejects_non_commit_point_candidates():
    """The store, not only the resolver, enforces resumable positions."""
    h = Harness(hybrid=True)
    first = h.add(list(range(40)))
    h.run(first)
    second = h.add(history(first) + [5])
    node_id = h.session(first).cursor
    binding = second.session_binding
    second.session_binding = SessionBinding(
        binding.epoch, binding.session_id, (Candidate(node_id, 30),), h.store.seq
    )
    h.step()
    assert h.cached[second.request_id] == 0
    h.run(second)


def test_failed_admission_keeps_idle_states():
    """A request that cannot fit even after eviction must not destroy the
    cache on the way to failing."""
    h = Harness(num_blocks=24)
    first = h.add(list(range(60)), max_tokens=2)
    h.run(first)
    nodes = dict(h.store.nodes)
    blocker = h.add(list(range(1000, 1000 + 16 * 15)), max_tokens=40)
    h.step()
    too_big = h.add(list(range(2000, 2000 + 16 * 12)), max_tokens=2)
    h.step()
    assert too_big.status == RequestStatus.WAITING
    assert nodes.keys() <= h.store.nodes.keys()
    h.run(blocker, too_big)


def test_aborted_request_publishes_without_moving_session():
    h = Harness()
    first = h.add(list(range(40)))
    h.run(first)
    cursor = h.session(first).cursor
    second = h.add(history(first) + list(range(300, 340)), max_tokens=50)
    h.step()
    h.step()
    h.sched.finish_requests([second.request_id], RequestStatus.FINISHED_ABORTED)
    check_invariants(h)
    assert h.session(first).cursor == cursor
    third = h.add(history(second) + [1])
    h.step()
    assert h.cached[third.request_id] >= len(history(first)) + 32
    h.run(third)
