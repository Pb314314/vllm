# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve 1 Hz NVML samples and join them to pipeline measurement windows."""

import argparse
import bisect
import csv
import datetime
import json
from pathlib import Path


def main():
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag", choices=("pipeline", "pipeline_newnative"), default="pipeline"
    )
    args = parser.parse_args()
    samples = []
    for row in csv.reader(
        (directory / f"{args.tag}_telemetry.log").read_text().splitlines(),
        skipinitialspace=True,
    ):
        if len(row) != 10:
            continue
        try:
            timestamp = datetime.datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S.%f")
            timestamp = timestamp.replace(tzinfo=datetime.UTC)
            samples.append(
                dict(
                    record_kind="telemetry",
                    timestamp_utc=timestamp.isoformat(),
                    unix_time=timestamp.timestamp(),
                    uuid=row[1],
                    sm_clock_mhz=int(row[2]),
                    temperature_c=float(row[3]),
                    power_w=float(row[4]),
                    power_limit_w=float(row[5]),
                    pstate=row[6],
                    clock_reasons=int(row[7], 16),
                    software_power_cap=row[8] == "Active",
                    hardware_thermal_slowdown=row[9] == "Active",
                )
            )
        except ValueError:
            continue
    samples.sort(key=lambda r: r["unix_time"])
    results = directory / "results"
    with (results / f"{args.tag}_telemetry.jsonl").open("w") as output:
        for sample in samples:
            output.write(json.dumps(sample) + "\n")
    times = [sample["unix_time"] for sample in samples]
    count = 0
    with (results / f"{args.tag}_power_case_summary.jsonl").open("w") as output:
        for source in sorted(results.glob("pipeline_*.jsonl")):
            if "power_case_summary" in source.name or "telemetry.jsonl" in source.name:
                continue
            for line in source.read_text().splitlines():
                record = json.loads(line)
                if "started" not in record or "key" not in record:
                    continue
                uuid = record.get("gpu_uuid")
                if uuid is not None:
                    expected = samples[0]["uuid"].removeprefix("GPU-").lower()
                    if uuid.removeprefix("GPU-").lower() != expected:
                        continue
                elif args.tag == "pipeline_newnative":
                    if "newnative" not in source.name:
                        continue
                elif "newnative" in source.name or "gpu3" in source.name:
                    continue
                start = record["started"]
                end = start + record.get("elapsed_s", 0)
                selected = samples[
                    bisect.bisect_left(times, start) : bisect.bisect_right(times, end)
                ]
                if not selected:
                    continue
                active = [s for s in selected if s["clock_reasons"] & 1 == 0]
                result = dict(
                    record_kind="telemetry_join",
                    source_file=source.name,
                    source_key=record["key"],
                    case=record["case"],
                    variant=record["variant"],
                    sample_count=len(selected),
                    min_active_clock_mhz=min(
                        (s["sm_clock_mhz"] for s in active), default=None
                    ),
                    max_power_w=max(s["power_w"] for s in selected),
                    max_temperature_c=max(s["temperature_c"] for s in selected),
                    software_power_cap_samples=sum(
                        s["software_power_cap"] for s in selected
                    ),
                    thermal_slowdown_samples=sum(
                        s["hardware_thermal_slowdown"] for s in selected
                    ),
                    caveat=(
                        "1 Hz sampled observations; not continuous interval monitoring"
                    ),
                )
                output.write(json.dumps(result) + "\n")
                count += 1
    print(
        "telemetry samples",
        len(samples),
        "joined measurements",
        count,
        "power-cap samples",
        sum(s["software_power_cap"] for s in samples),
    )


if __name__ == "__main__":
    main()
