# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read static CUDA device limits without creating a GPU context."""

import argparse
import ctypes
import json
import socket
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    driver = ctypes.CDLL("libcuda.so.1")
    assert driver.cuInit(0) == 0
    attributes = {}
    for name, number in {
        "multiprocessor_count": 16,
        "max_threads_per_multiprocessor": 39,
        "shared_memory_per_multiprocessor": 81,
        "registers_per_multiprocessor": 82,
        "max_blocks_per_multiprocessor": 106,
    }.items():
        value = ctypes.c_int()
        assert (
            driver.cuDeviceGetAttribute(ctypes.byref(value), number, args.device) == 0
        )
        attributes[name] = value.value
    context = ctypes.c_void_p()
    assert driver.cuCtxGetCurrent(ctypes.byref(context)) == 0
    record = dict(
        hostname=socket.gethostname(),
        device_index=args.device,
        current_context=context.value,
        attributes=attributes,
        source="CUDA driver enum values from local cuda.h",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as output:
        output.write(json.dumps(record) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
