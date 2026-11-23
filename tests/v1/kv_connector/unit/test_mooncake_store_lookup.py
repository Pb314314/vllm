# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
import zmq

from vllm.config import DeviceConfig, KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import CircuitBreaker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store import worker
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.data import (  # noqa: E501
    MooncakeLookupResult,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.protocol import (  # noqa: E501
    encode_lookup_response,
)

pytestmark = pytest.mark.cpu_test

LOOKUP_TIMEOUT_S = 0.5


@pytest.fixture
def lookup_service(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[tuple[worker.LookupKeyClient, zmq.Socket]]:
    monkeypatch.setattr(worker, "LOOKUP_BREAKER_COOLDOWN_S", 0.5)
    monkeypatch.setattr(
        worker, "LOOKUP_BREAKER_FAILURE_THRESHOLD", getattr(request, "param", 1)
    )
    # Keep Unix socket paths below the platform limit.
    with tempfile.TemporaryDirectory(prefix="mc-", dir="/tmp") as rpc_dir:
        monkeypatch.setenv("VLLM_RPC_BASE_PATH", rpc_dir)
        config = VllmConfig(
            device_config=DeviceConfig("cpu"),
            kv_transfer_config=KVTransferConfig(
                kv_connector="MooncakeStoreConnector",
                kv_role="kv_both",
                kv_connector_extra_config={
                    "lookup_prefetch_limit": 1,
                    "lookup_timeout_s": LOOKUP_TIMEOUT_S,
                },
            ),
        )
        context = zmq.Context()
        peer = context.socket(zmq.REP)
        peer.setsockopt(zmq.RCVTIMEO, 2000)
        peer.bind(worker.get_zmq_rpc_path_lookup(config))
        client = worker.LookupKeyClient(config)
        try:
            yield client, peer
        finally:
            client.executor.shutdown(wait=True, cancel_futures=True)
            client.close()
            client.ctx.term()
            peer.close(linger=0)
            context.term()


def _resolve_lookup(
    client: worker.LookupKeyClient, req_id: str, mode: str
) -> MooncakeLookupResult:
    if mode == "prefetch":
        assert client.prefetch(req_id, 128, [])
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = client.lookup(req_id, 128, [], non_block=mode != "blocking")
        if result is not None:
            return result
        threading.Event().wait(0.001)
    raise AssertionError("lookup remained pending")


@pytest.mark.parametrize("mode", ["blocking", "async", "prefetch"])
def test_slow_lookup_returns_miss_and_recovers_without_stale_reply(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket], mode: str
) -> None:
    client, peer = lookup_service
    release_reply = threading.Event()
    errors: list[Exception] = []

    def serve() -> None:
        try:
            peer.recv_multipart()
            assert release_reply.wait(10)
            peer.send(encode_lookup_response(MooncakeLookupResult(32)))
            peer.recv_multipart()
            peer.send(encode_lookup_response(MooncakeLookupResult(16)))
        except Exception as exc:
            errors.append(exc)

    server = threading.Thread(target=serve)
    server.start()
    with ThreadPoolExecutor(max_workers=1) as requests:
        try:
            # The outer timeout also makes this test fail promptly on the base.
            result = requests.submit(_resolve_lookup, client, "slow", mode).result(5)
            assert result.hit_length == 0
            assert client.circuit_breaker.is_open
            assert client.lookup("skip", 128, []).hit_length == 0

            # Recovery must ignore the abandoned request's 32-token reply.
            release_reply.set()
            deadline = time.monotonic() + 5
            while client.circuit_breaker.is_open:
                assert time.monotonic() < deadline
                threading.Event().wait(0.001)
            assert client.lookup("recovered", 128, []).hit_length == 16
            assert not client.circuit_breaker.is_open
        finally:
            release_reply.set()
            server.join(10)
    assert not server.is_alive()
    assert not errors


@pytest.mark.parametrize("mode", ["blocking", "async", "prefetch"])
def test_lookup_expiring_in_queue_returns_miss_without_failure(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket], mode: str
) -> None:
    client, _ = lookup_service
    release_executor = threading.Event()
    client.executor.submit(release_executor.wait)
    with ThreadPoolExecutor(max_workers=1) as requests:
        try:
            result = requests.submit(_resolve_lookup, client, "queued", mode).result(5)
            assert result.hit_length == 0
            # The lookup never reached the store, so the breaker stays closed.
            assert not client.circuit_breaker.is_open
        finally:
            client.discard("queued")
            release_executor.set()


def test_circuit_breaker_reopens_after_failed_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    breaker = CircuitBreaker("Test", 30, failure_threshold=3, probe_window_s=5)
    assert not breaker.is_open
    for _ in range(2):
        breaker.record_failure()
        assert not breaker.is_open
    breaker.record_success()
    for _ in range(2):
        breaker.record_failure()
        assert not breaker.is_open
    breaker.record_failure()
    assert breaker.is_open
    assert "3 consecutive failures; skipping requests for 30 seconds" in caplog.text
    clock[0] = 130
    assert not breaker.is_open
    # After the cooldown a single probe is admitted; others keep skipping.
    assert breaker.allow()
    assert breaker.is_open
    assert not breaker.allow()
    breaker.record_failure()
    clock[0] = 159
    assert breaker.is_open
    clock[0] = 160
    assert breaker.allow()
    # A probe that never reports back frees the slot after the probe window.
    clock[0] = 164
    assert not breaker.allow()
    clock[0] = 165
    assert breaker.allow()
    breaker.record_success()
    assert not breaker.is_open
    assert breaker.allow()
    assert breaker.allow()


@pytest.mark.parametrize("lookup_service", [3], indirect=True)
def test_failed_lookup_counts_once_toward_threshold(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket],
) -> None:
    client, _ = lookup_service
    # The bound peer sends no replies; each request times out once.
    for index in range(3):
        assert client.lookup(f"failed-{index}", 128, []).hit_length == 0
        assert client.circuit_breaker.is_open == (index == 2)


@pytest.mark.parametrize("lookup_service", [2], indirect=True)
def test_lookup_started_after_deadline_does_not_update_breaker(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket],
) -> None:
    client, _ = lookup_service
    client.circuit_breaker.record_failure()
    release_executor = threading.Event()
    client.executor.submit(release_executor.wait)
    try:
        assert client.lookup("queued", 128, [], non_block=True) is None
        # Skip polling past the deadline so the lookup starts late, not cancelled.
        threading.Event().wait(LOOKUP_TIMEOUT_S + 0.1)
    finally:
        release_executor.set()
    client.executor.submit(lambda: None).result(timeout=5)
    assert client.lookup("queued", 128, [], non_block=True).hit_length == 0
    assert not client.circuit_breaker.is_open
    # Neither a failure nor a success: the earlier failure still counts.
    assert client.lookup("failed", 128, []).hit_length == 0
    assert client.circuit_breaker.is_open


@pytest.mark.parametrize("lookup_service", [3], indirect=True)
def test_cached_hit_does_not_reset_newer_failures(
    lookup_service: tuple[worker.LookupKeyClient, zmq.Socket],
) -> None:
    client, peer = lookup_service

    def serve() -> None:
        peer.recv_multipart()
        peer.send(encode_lookup_response(MooncakeLookupResult(16)))

    server = threading.Thread(target=serve)
    server.start()
    try:
        assert _resolve_lookup(client, "cached", "async").hit_length == 16
        for index in range(3):
            assert client.lookup(f"failed-{index}", 128, []).hit_length == 0
            assert client.circuit_breaker.is_open == (index == 2)
            if index < 2:
                assert client.lookup("cached", 128, [], non_block=True).hit_length == 16
    finally:
        server.join(10)
    assert not server.is_alive()


@pytest.mark.parametrize("mode", ["blocking", "prefetch"])
@pytest.mark.parametrize("positions", [None, (), (64, 96)])
def test_lookup_hints_preserve_the_remote_result(lookup_service, mode, positions):
    """Optional hints survive IPC without changing the Store response."""
    import struct

    from vllm.v1.core.kv_cache_utils import BlockHash

    client, peer = lookup_service
    hashes = [BlockHash(b"a"), BlockHash(b"b")]
    received = []

    def serve():
        received.extend(peer.recv_multipart())
        peer.send(encode_lookup_response(MooncakeLookupResult(32)))

    server = threading.Thread(target=serve)
    server.start()
    try:
        if mode == "prefetch":
            assert client.prefetch("hinted", 128, hashes, positions)
        result = client.lookup("hinted", 128, hashes, prefix_match_positions=positions)
        assert result == MooncakeLookupResult(32)
        server.join(5)
        assert not server.is_alive()
        assert len(received) == (4 if positions is None else 5)
        if positions is not None:
            assert received[-1] == struct.pack(f">{len(positions)}I", *positions)
    finally:
        server.join(5)


def test_lookup_plan_reuses_only_the_matching_group_prefix():
    """An append can reuse keys; edited history and other groups cannot."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.lookup_plan_cache import (  # noqa: E501
        LookupPlanCache,
    )
    from vllm.v1.core.kv_cache_utils import BlockHash

    cache = LookupPlanCache(1 << 20)
    hashes = [BlockHash(bytes([i]) * 32) for i in range(7)]
    meta = [(0, bytes(h), (i + 1) * 16) for i, h in enumerate(hashes[:4])]
    keys = [f"rank{rank}:{h.hex()}" for h in hashes[:4] for rank in range(2)]
    cache.put(0, keys, meta, 2)
    plan = cache.get(0, 16, 16, hashes, (64, 80), 7)
    assert plan is not None
    assert plan.keys == tuple(keys[:6])
    assert plan.meta == tuple(meta[:3])
    assert cache.reused_keys == 6
    assert cache.get(1, 16, 16, hashes, (64,), 7) is None
    changed = [BlockHash(b"changed" + bytes([i])) for i in range(7)]
    assert cache.get(0, 16, 16, changed, (64,), 7) is None
    assert cache.get(0, 16, 16, hashes, (64,), 2) is None


def test_lookup_plan_bounds_metadata_and_reset():
    """Old metadata can be evicted independently of physical KV storage."""
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.lookup_plan_cache import (  # noqa: E501
        LookupPlanCache,
    )
    from vllm.v1.core.kv_cache_utils import BlockHash

    cache = LookupPlanCache(1 << 20, max_entries=1)
    for suffix in [b"first", b"second"]:
        hashes = [suffix + bytes([i]) for i in range(3)]
        cache.put(
            0,
            [h.hex() for h in hashes],
            [(0, h, 16 * (i + 1)) for i, h in enumerate(hashes)],
            1,
        )
        assert 0 < cache.metadata_bytes <= cache.max_bytes
    old = [BlockHash(b"first" + bytes([i])) for i in range(3)]
    new = [BlockHash(b"second" + bytes([i])) for i in range(3)]
    assert cache.get(0, 16, 16, old, (48,), 3) is None
    assert cache.get(0, 16, 16, new, (48,), 3) is not None
    cache.clear()
    assert cache.metadata_bytes == 0
    assert cache.get(0, 16, 16, new, (48,), 3) is None
    too_small = LookupPlanCache(1)
    too_small.put(
        0,
        [h.hex() for h in new],
        [(0, bytes(h), 16 * (i + 1)) for i, h in enumerate(new)],
        1,
    )
    assert too_small.metadata_bytes == 0
    assert too_small.get(0, 16, 16, new, (48,), 3) is None
