"""Standalone checks for the ShmRingBuffer attach-race fix (no pytest needed).

T1: bare ShmRingBuffer attach to a missing segment stays silent (pickle contract).
T2: MessageQueue.create_from_handle for a confirmed local reader fails fast with
    a clear RuntimeError when the segment is missing.
T3: writer+reader attached before an external unlink keep passing messages
    (POSIX shm mmap survives unlink); a NEW attach after the unlink fails fast.
"""
import os
import sys
from unittest import mock

from vllm.distributed.device_communicators import shm_broadcast
from vllm.distributed.device_communicators.shm_broadcast import (
    Handle,
    MessageQueue,
    ShmRingBuffer,
)

failures = 0

# T1
with mock.patch.object(
    shm_broadcast.shared_memory, "SharedMemory", side_effect=FileNotFoundError()
):
    buf = ShmRingBuffer(n_reader=1, max_chunk_bytes=24 * 1024 * 1024, max_chunks=10,
                        name="vllm-test-x")
if not hasattr(buf, "shared_memory"):
    print("T1 PASS (silent deferred failure for bare buffer)")
else:
    print("T1 FAIL: buffer unexpectedly attached"); failures += 1

# T2
handle = Handle(
    local_reader_ranks=[0],
    buffer_handle=(1, 24 * 1024 * 1024, 10, "vllm-test-x"),
    local_subscribe_addr="unused",
    local_notify_addr="unused",
)
try:
    with mock.patch.object(
        shm_broadcast.shared_memory, "SharedMemory",
        side_effect=FileNotFoundError(),
    ):
        MessageQueue.create_from_handle(handle, rank=0)
    print("T2 FAIL: create_from_handle did not raise"); failures += 1
except RuntimeError as e:
    if "vllm-test-x" in str(e) and "could not attach" in str(e):
        print("T2 PASS (fail-fast RuntimeError):", str(e)[:120], "...")
    else:
        print("T2 FAIL: wrong error:", e); failures += 1

# T3
writer = MessageQueue(n_reader=1, n_local_reader=1, max_chunk_bytes=1024,
                      max_chunks=4, connect_ip="127.0.0.1")
reader = MessageQueue.create_from_handle(writer.export_handle(), rank=0)
writer.wait_until_ready()
reader.wait_until_ready()
shm_path = os.path.join("/dev/shm", writer.buffer.shared_memory.name)
os.unlink(shm_path)  # the external purge (RemoveIPC simulation)
writer.enqueue(b"payload")
got = reader.dequeue(timeout=5)
if got == b"payload":
    print("T3a PASS (roundtrip survives external unlink)")
else:
    print("T3a FAIL: got", got); failures += 1
try:
    MessageQueue.create_from_handle(writer.export_handle(), rank=0)
    print("T3b FAIL: late attach did not raise"); failures += 1
except RuntimeError as e:
    print("T3b PASS (late attach fails fast)")
writer.shutdown()
reader.shutdown()

print("RESULT-1:", "ALL PASS" if failures == 0 else f"{failures} FAILURES")
if failures:
    sys.exit(1)

# T4: process-level ordering — attach before vs after a mid-load purge
import multiprocessing as _mp


def _order_child(handle, attach_after_load, child_ready, purge_done, result_q):
    from vllm.distributed.device_communicators.shm_broadcast import MessageQueue as MQ
    if attach_after_load:
        child_ready.set()
        purge_done.wait(timeout=30)
    try:
        mq = MQ.create_from_handle(handle, rank=0)
    except RuntimeError as e:
        result_q.put(("raise", str(e)))
        return
    if not attach_after_load:
        child_ready.set()
        purge_done.wait(timeout=30)
    try:
        result_q.put(("dequeued", mq.dequeue(timeout=10)))
    except Exception as e:
        result_q.put(("error", f"{type(e).__name__}: {e}"))


def t4():
    ok = True
    for after_load in (False, True):
        w = MessageQueue(n_reader=1, n_local_reader=1, max_chunk_bytes=1024,
                         max_chunks=4, connect_ip="127.0.0.1")
        result_q = _mp.Queue()
        child_ready = _mp.Event()
        purge_done = _mp.Event()
        p = _mp.Process(target=_order_child,
                        args=(w.export_handle(), after_load, child_ready, purge_done, result_q))
        p.start()
        assert child_ready.wait(timeout=60)
        os.unlink(os.path.join("/dev/shm", w.buffer.shared_memory.name))
        purge_done.set()
        w.enqueue(b"payload")
        kind, detail = result_q.get(timeout=60)
        p.join(timeout=30)
        w.shutdown()
        if after_load:
            good = kind == "raise" and "could not attach" in detail
        else:
            good = (kind, detail) == ("dequeued", b"payload")
        print(f"T4({'old-order' if after_load else 'new-order'})",
              "PASS" if good else f"FAIL: {kind} {detail}")
        ok = ok and good
    return ok


# T5: real WorkerProc.__init__ ordering; the stub's load_model performs the purge
def t5():
    from types import SimpleNamespace
    from vllm.v1.executor import multiproc_executor
    from vllm.v1.executor.multiproc_executor import WorkerProc

    events = []
    w = MessageQueue(n_reader=1, n_local_reader=1, max_chunk_bytes=1024,
                     max_chunks=4, connect_ip="127.0.0.1")
    shm_path = os.path.join("/dev/shm", w.buffer.shared_memory.name)

    class _FakeWorker:
        def __init__(self, rpc_rank, global_rank):
            self.rank = rpc_rank

        def init_worker(self, all_kwargs):
            pass

        def init_device(self):
            events.append("init_device")

        def load_model(self):
            events.append("load_model")
            os.unlink(shm_path)

    orig_attach = WorkerProc._attach_input_mq

    def _spy(self, *a, **k):
        events.append("attach_input_mq")
        return orig_attach(self, *a, **k)

    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(world_size=1, nnodes_within_dp=1,
                                        enable_expert_parallel=False),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    with mock.patch.object(multiproc_executor, "WorkerWrapperBase", _FakeWorker), \
         mock.patch.object(WorkerProc, "_attach_input_mq", _spy), \
         mock.patch.object(multiproc_executor.current_platform,
                           "update_block_size_for_backend", lambda c: None):
        proc = WorkerProc(vllm_config=cfg, local_rank=0, rank=0,
                          distributed_init_method="file:///dev/null",
                          input_shm_handle=w.export_handle(),
                          shared_worker_lock=None, is_driver_worker=True)
    try:
        assert events == ["attach_input_mq", "init_device", "load_model"], events
        w.enqueue(b"payload")
        assert proc.rpc_broadcast_mq.dequeue(timeout=5) == b"payload"
        print("T5 PASS (real __init__ attaches before load; survives mid-load purge)")
        return True
    finally:
        proc.rpc_broadcast_mq.shutdown()
        proc.worker_response_mq.shutdown()
        w.shutdown()


def t6():
    try:
        import ray  # noqa
    except ImportError:
        print("T6 SKIP (ray not installed)")
        return True
    from vllm.v1.executor.ray_executor_v2 import RayWorkerProc
    assert "_attach_input_mq" in RayWorkerProc.__dict__
    bare = RayWorkerProc.__new__(RayWorkerProc)
    bare._attach_input_mq(None, None)
    assert "rpc_broadcast_mq" not in bare.__dict__
    print("T6 PASS (Ray override is a no-op)")
    return True


for fn in (t4, t5, t6):
    try:
        if not fn():
            failures += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"{fn.__name__} ERROR: {e}")
        failures += 1

print("RESULT-2:", "ALL PASS" if failures == 0 else f"{failures} FAILURES")
sys.exit(1 if failures else 0)
