# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize allocation experiments without equating live bytes with reservation."""

import argparse
import hashlib
import json
from pathlib import Path

from memory_snapshot import load_snapshot


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ownership(snapshot, pointer):
    data = load_snapshot(snapshot)
    for segment in data["segments"]:
        address = segment["address"]
        for block in segment["blocks"]:
            address = block.get("address", address)
            if address <= pointer < address + block["size"]:
                return dict(
                    segment_address=segment["address"],
                    segment_bytes=segment["total_size"],
                    pool=segment.get("segment_pool_id"),
                    block_address=address,
                    block_bytes=block["size"],
                    requested_bytes=block.get("requested_size"),
                    state=block["state"],
                    frames=block.get("frames", []),
                )
            address += block["size"]
    return None


def summarize(row):
    before, capture, replay = (row["before"], row["after_capture"], row["after_replay"])
    result = dict(
        case_key=row["case_key"],
        source_commit=row["source_commit"],
        input_layout=row.get("input_layout", "obsolete_unvalidated_scales"),
        rows=row["rows"],
        width=row["width"],
        mode=row["mode"],
        variant=row["variant"],
        chunk_rows=row["chunk_rows"],
        chunks=row["chunks"],
        live_peak_increment_bytes=capture["peak_allocated"] - before["allocated"],
        capture_reserved_increment_bytes=capture["reserved"] - before["reserved"],
        replay_reserved_increment_bytes=replay["reserved"] - before["reserved"],
        capture_private_pool_bytes=sum(
            value["reserved"]
            for key, value in capture["pools"].items()
            if key != "(0, 0)"
        ),
        valid_score_nonfinite_count=row.get("valid_score_nonfinite_count"),
        selected_values_replay_equal=row["replay_selected_values_equal"],
    )
    first = row["observations"][0]
    result["first_chunk_bf16_storage_bytes"] = first["bf16_storage_bytes"]
    result["first_chunk_fp32_storage_bytes"] = (
        first["consumer_storage_bytes"] if row["backend"] != "deep_select" else 0
    )
    result["snapshots"] = [
        dict(path=path, sha256=sha256(path)) for path in row["snapshots"]
    ]
    overlap = next(
        (
            p
            for p in row["snapshots"]
            if p.endswith("capture_conversion_overlap.pickle")
            or p.endswith("eager_conversion_overlap.pickle")
        ),
        None,
    )
    if overlap:
        result["first_bf16_allocation"] = ownership(overlap, first["bf16_ptr"])
        result["first_consumer_allocation"] = ownership(overlap, first["consumer_ptr"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("w") as target:
        for line in args.input.read_text().splitlines():
            row = json.loads(line)
            if "error" not in row:
                target.write(json.dumps(summarize(row)) + "\n")


if __name__ == "__main__":
    main()
