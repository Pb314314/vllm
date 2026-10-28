# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare real runtime per-indexer peaks with transient dtype-buffer storage."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    events = [json.loads(line) for line in raw.splitlines()]
    overlaps = {}
    for event in events:
        if event["event"] == "upcast_overlap" and event["phase"] == "runtime":
            overlaps[event["layer"], event["source"]["shape"][0]] = event
    with args.out.open("w") as target:
        target.write(
            json.dumps(
                dict(
                    kind="provenance",
                    trace=str(args.trace),
                    trace_sha256=hashlib.sha256(raw).hexdigest(),
                    hooks=[e for e in events if e["event"] == "hooks_loaded"],
                )
            )
            + "\n"
        )
        for event in events:
            if event["event"] != "runtime_indexer_peak":
                continue
            row = dict(
                kind="runtime_indexer_peak",
                **{
                    key: event.get(key)
                    for key in [
                        "time",
                        "layer",
                        "rows",
                        "num_decode_tokens",
                        "num_prefill_tokens",
                        "max_seq_len_host",
                        "peak_increment_bytes",
                    ]
                },
            )
            row["before_allocated"] = event["before"]["allocated"]
            row["after_allocated"] = event["after"]["allocated"]
            row["allocator_retries"] = event["after"]["retries"]
            overlap = overlaps.get((event["layer"], event.get("num_decode_tokens")))
            if overlap:
                storage = (
                    overlap["source"]["storage_bytes"]
                    + overlap["target"]["storage_bytes"]
                )
                row.update(
                    bf16=overlap["source"],
                    fp32=overlap["target"],
                    dtype_buffer_storage_bytes=storage,
                    peak_minus_dtype_buffers=event["peak_increment_bytes"] - storage,
                )
            target.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
