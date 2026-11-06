"""sitecustomize shim: log every POSIX shm segment create/attach/unlink with
pid/ppid/timestamp to $SHM_TRACE (jsonl). Pure stdlib; no vllm changes."""
import json
import os
import time
from multiprocessing import shared_memory as _sm

_TRACE = os.environ.get("SHM_TRACE")
_orig = _sm.SharedMemory


class _TracedSharedMemory(_orig):
    def __init__(self, name=None, create=False, size=0, **kw):
        super().__init__(name=name, create=create, size=size, **kw)
        if _TRACE:
            try:
                with open(_TRACE, "a") as f:
                    json.dump({"ts": time.time(), "event": "create" if create else "attach",
                               "name": self.name, "pid": os.getpid(),
                               "ppid": os.getppid()}, f)
                    f.write("\n")
            except Exception:
                pass

    def unlink(self):
        if _TRACE:
            try:
                with open(_TRACE, "a") as f:
                    json.dump({"ts": time.time(), "event": "unlink",
                               "name": self.name, "pid": os.getpid(),
                               "ppid": os.getppid()}, f)
                    f.write("\n")
            except Exception:
                pass
        super().unlink()


_sm.SharedMemory = _TracedSharedMemory
