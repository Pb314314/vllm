# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test graph-replayed per-step chunk schedules shared by three c2 layers."""

import hashlib
import json
import os
import random
import socket
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from flashinfer.testing import bench_gpu_time_with_cupti
from pipeline_bench import (
    correctness,
    distribution,
    get_indexer_topk,
    get_num_sms,
    get_paged_mqa_logits_metadata,
    indexer_module,
    init_workspace_manager,
    logits_for_rows,
    make_inputs,
    profile_kernels,
    telemetry,
)

import vllm


def save(path, record):
    with path.open("a") as output:
        output.write(json.dumps(record) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main():
    destination = Path(os.environ["OPTIMIZATION_RESULTS"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    init_workspace_manager(torch.device("cuda"))
    rows, width, compression, heads = 4096, 524288, 2, 32
    backend, chunks = indexer_module.varlen_decode_plan(rows, width, "auto", 512 << 20)
    assert backend == "deep_select" and len(chunks) == 8
    layers = [
        make_inputs(rows, heads, width, compression, 32768, "mxfp4", 8, 512)
        for _ in range(3)
    ]
    for index, x in enumerate(layers):
        x["weights"] = x["weights"].roll(index * 3, dims=0)
        x["ctx"] = layers[0]["ctx"]
        x["indices"] = layers[0]["indices"]
    decode = SimpleNamespace(
        seq_lens=layers[0]["ctx"],
        indices=layers[0]["indices"],
        block_table=layers[0]["bt"],
        schedule_metadata=layers[0]["schedule"],
    )

    def baseline():
        for x in layers:
            decode.block_table = x["bt"]
            for start, end in chunks:
                indexer_module._varlen_decode_rows(
                    x["q"],
                    x["qs"],
                    x["weights"],
                    x["kv"],
                    decode,
                    width,
                    1048576,
                    backend,
                    512,
                    x["out"],
                    None,
                    8,
                    False,
                    start,
                    end,
                )

    def shared_per_step():
        schedules = [
            get_paged_mqa_logits_metadata(
                layers[0]["ctx"][start:end],
                64,
                get_num_sms(),
                indices=layers[0]["indices"][start:end],
            )
            for start, end in chunks
        ]
        for x in layers:
            for (start, end), schedule in zip(chunks, schedules):
                logits = logits_for_rows(x, start, end, schedule)
                get_indexer_topk(backend)(
                    logits,
                    x["ctx"][start:end],
                    1,
                    x["out"][start:end],
                    512,
                    1048576,
                )

    source_root = Path(vllm.__file__).parent
    metadata = dict(
        experiment="shared_per_step_c2_metadata",
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        vllm_root=str(source_root),
        executable=os.sys.executable,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for pattern in ("_C_stable_libtorch*.so", "_deepselect_C*.so")
            for p in source_root.glob(pattern)
        },
        hostname=socket.gethostname(),
        gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        rows=rows,
        width=width,
        heads=heads,
        head_dim=128,
        topk=512,
        compression=compression,
        q_len=8,
        precision="mxfp4",
        layers=3,
        chunks=chunks,
        host_max_seq_len=1048576,
        policy="both graphs rebuild GPU schedules every replay; no frozen oracle",
    )
    graphs = {}
    for name, fn in (("baseline", baseline), ("shared_per_step", shared_per_step)):
        for _ in range(3):
            fn()
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graph.replay()
        torch.accelerator.synchronize()
        graphs[name] = graph
        save(
            destination,
            dict(
                metadata,
                record_kind="kernel_profile",
                arm=name,
                kernels=profile_kernels(fn, 3),
                status="ok",
            ),
        )
    for raw_context in (32768, 8192, 1536):
        lengths = torch.tensor(
            [(raw_context - 7 + i % 8) // 2 for i in range(rows)],
            device="cuda",
            dtype=torch.int32,
        )[:, None]
        layers[0]["ctx"].copy_(lengths)
        checks = {}
        for name, graph in graphs.items():
            checks[name] = []
            for x in layers:
                check = correctness(x, graph.replay, chunks, 512, "none")
                assert not any(
                    check.get(k, 0)
                    for k in (
                        "chunked_logit_mismatch_rows",
                        "nonfinite_valid_logits",
                        "invalid_index_rows",
                        "duplicate_index_rows",
                        "exact_topk_value_mismatch_rows",
                    )
                ), check
                checks[name].append(check)
        samples = {}
        for trial in range(3):
            order = [(name, cache) for name in graphs for cache in ("warm", "cold")]
            random.Random(128 + trial).shuffle(order)
            for name, cache in order:
                started = time.time()
                before = telemetry()
                values = bench_gpu_time_with_cupti(
                    graphs[name].replay,
                    use_cuda_graph=False,
                    dry_run_iters=5,
                    repeat_iters=100,
                    cold_l2_cache=cache == "cold",
                )
                timing = distribution([v * 1000 for v in values])
                samples.setdefault((name, cache), []).append(timing["median_us"])
                save(
                    destination,
                    dict(
                        metadata,
                        record_kind="trial",
                        arm=name,
                        raw_context=raw_context,
                        cache=cache,
                        trial=trial,
                        started=started,
                        elapsed_s=time.time() - started,
                        telemetry_before=before,
                        telemetry_after=telemetry(),
                        correctness=checks[name],
                        timing=timing,
                        status="ok",
                    ),
                )
        result = {
            f"{name}/{cache}": dict(
                median_us=statistics.median(values),
                medians_us=values,
                cv=statistics.pstdev(values) / statistics.mean(values),
            )
            for (name, cache), values in samples.items()
        }
        save(
            destination,
            dict(
                metadata,
                record_kind="summary",
                raw_context=raw_context,
                timings=result,
                status="ok",
            ),
        )
        print(raw_context, result, flush=True)


if __name__ == "__main__":
    main()
