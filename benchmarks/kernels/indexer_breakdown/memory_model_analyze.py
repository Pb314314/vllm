# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attribute model allocator segments using event history and Python call sites."""

import argparse
import bisect
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from memory_snapshot import load_snapshot


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def site(event):
    frames = [
        f for f in event.get("frames", []) if f.get("filename", "").endswith(".py")
    ]
    if any(f["name"] == "as_fp32_logits" for f in frames):
        owner = "indexer_fp32"
    elif any(f["name"] == "fp8_fp4_paged_mqa_logits" for f in frames):
        owner = "indexer_bf16"
    elif any("flash_mla" in f["name"] for f in frames):
        owner = "attention_scratch"
    elif any("logits_processor.py" in f["filename"] for f in frames):
        owner = "vocab_logits"
    else:
        owner = "other"
    return owner, frames[:6]


def analyze(trace_path, snapshot_path):
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    anchors = [
        event["time"]
        for event in events
        if event["event"] == "snapshot" and event.get("path") == str(snapshot_path)
    ]
    cutoff = max(anchors, default=snapshot_path.stat().st_mtime)
    events = [event for event in events if event["time"] <= cutoff]
    archive = snapshot_path.with_suffix(".trace.jsonl")
    archive.write_text("".join(json.dumps(event) + "\n" for event in events))
    snapshot = load_snapshot(snapshot_path)
    allocations = []
    for device, traces in enumerate(snapshot["device_traces"]):
        for sequence, event in enumerate(traces):
            if event["action"] == "alloc":
                allocations.append((device, sequence, event))
    pools = defaultdict(lambda: dict(reserved_bytes=0, allocated_bytes=0))
    indexer_private = defaultdict(lambda: dict(events=0, max_requested_bytes=0))
    for device, traces in enumerate(snapshot["device_traces"]):
        current = sorted(
            (s for s in snapshot["segments"] if s["device"] == device),
            key=lambda s: s["address"],
        )
        starts = [s["address"] for s in current]
        created = {
            e["addr"]: i for i, e in enumerate(traces) if e["action"] == "segment_alloc"
        }
        for sequence, event in enumerate(traces):
            if event["action"] != "alloc":
                continue
            position = bisect.bisect_right(starts, event["addr"]) - 1
            if position < 0:
                continue
            segment = current[position]
            if (
                event["addr"] >= segment["address"] + segment["total_size"]
                or sequence < created.get(segment["address"], -1)
                or segment.get("segment_pool_id") == (0, 0)
            ):
                continue
            names = {frame["name"] for frame in event.get("frames", [])}
            owner = (
                "fp32"
                if "as_fp32_logits" in names
                else "bf16"
                if "fp8_fp4_paged_mqa_logits" in names
                else None
            )
            if owner:
                item = indexer_private[owner]
                item["events"] += 1
                item["max_requested_bytes"] = max(
                    item["max_requested_bytes"], event["size"]
                )
    segments = []
    for segment in snapshot["segments"]:
        pool = str(segment.get("segment_pool_id"))
        pools[pool]["reserved_bytes"] += segment["total_size"]
        pools[pool]["allocated_bytes"] += segment["allocated_size"]
        if segment["total_size"] < 256 * 1024**2 or pool == "(0, 0)":
            continue
        owners = {}
        start, end = segment["address"], segment["address"] + segment["total_size"]
        creation = max(
            (
                sequence
                for sequence, event in enumerate(
                    snapshot["device_traces"][segment["device"]]
                )
                if event["action"] == "segment_alloc" and event.get("addr") == start
            ),
            default=-1,
        )
        for device, sequence, event in allocations:
            if (
                device != segment["device"]
                or sequence < creation
                or not start <= event["addr"] < end
            ):
                continue
            owner, frames = site(event)
            key = (owner, event["size"], event["addr"], event.get("stream"))
            if key not in owners:
                owners[key] = dict(
                    owner=owner,
                    requested_bytes=event["size"],
                    address=event["addr"],
                    stream=event.get("stream"),
                    count=0,
                    first_event=sequence,
                    last_event=sequence,
                    python_frames=frames,
                )
            owners[key]["count"] += 1
            owners[key]["last_event"] = sequence
        segments.append(
            dict(
                address=start,
                reserved_bytes=segment["total_size"],
                allocated_bytes=segment["allocated_size"],
                pool=pool,
                segment_created_event=creation,
                allocations=list(owners.values()),
            )
        )
    topk = {}
    for event in events:
        if event["event"] != "topk_dispatch":
            continue
        key = (
            event["phase"],
            event["memory"]["capturing"],
            tuple(event["logits"]["shape"]),
            event["backend"],
            event["max_seq_len_host"],
        )
        if key not in topk:
            topk[key] = dict(
                phase=event["phase"],
                capturing=event["memory"]["capturing"],
                shape=event["logits"]["shape"],
                dtype=event["logits"]["dtype"],
                stride=event["logits"]["stride"],
                backend=event["backend"],
                max_seq_len_host=event["max_seq_len_host"],
                next_n=event["next_n"],
                layers=[],
                pointers=[],
            )
        item = topk[key]
        if event["layer"] not in item["layers"]:
            item["layers"].append(event["layer"])
        if event["logits"]["ptr"] not in item["pointers"]:
            item["pointers"].append(event["logits"]["ptr"])
    return dict(
        trace=str(archive),
        trace_sha256=digest(archive),
        source_trace=str(trace_path),
        snapshot=str(snapshot_path),
        snapshot_sha256=digest(snapshot_path),
        pools=dict(pools),
        private_indexer_allocations_all_segment_sizes=dict(indexer_private),
        large_private_segments=segments,
        topk_dispatch=list(topk.values()),
        capture_events=[
            e for e in events if e["event"] in ("capture_start", "capture_end")
        ],
        hooks_provenance=[e for e in events if e["event"] == "hooks_loaded"],
        hook_errors=[e for e in events if e["event"] == "hook_error"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.out.open("a") as stream:
        stream.write(json.dumps(analyze(args.trace, args.snapshot)) + "\n")


if __name__ == "__main__":
    main()
