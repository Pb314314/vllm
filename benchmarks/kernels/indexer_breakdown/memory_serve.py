# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run an instrumented real TP4 model for the indexer allocation audit.

This runner owns and stops only its server process group. It changes no model
or kernel implementation. The hook logs allocation metadata and snapshots;
latency from this instrumented run must not be used as a performance claim.
"""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import aiohttp


def emit(path, row):
    with path.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


async def requests(port, count, max_tokens, prompt_repeats=32):
    endpoint = f"http://127.0.0.1:{port}/v1/completions"
    timeout = aiohttp.ClientTimeout(total=1200)
    connector = aiohttp.TCPConnector(limit=1024)

    async def one(session, i):
        body = dict(
            prompt=(
                "Read this short note about numbers and answer clearly. "
                * prompt_repeats
            )
            + f" Question {i}: Explain why two plus two is four.",
            max_tokens=max_tokens,
            temperature=0,
            ignore_eos=True,
        )
        try:
            async with session.post(endpoint, json=body) as response:
                value = await response.json()
                return dict(
                    status=response.status,
                    usage=value.get("usage"),
                    error=value.get("error"),
                )
        except Exception as exc:
            return dict(status=0, error=repr(exc))

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        return await asyncio.gather(*(one(session, i) for i in range(count)))


def run_case(args, mode, max_len):
    root = args.out_dir
    key = f"{args.tag}_{mode.lower()}_m{max_len}"
    case = root / key
    case.mkdir(parents=True, exist_ok=True)
    hook_dir = case / "hooks"
    hook_dir.mkdir(exist_ok=True)
    source = Path(__file__).with_name("memory_hooks.py")
    startup = hook_dir / "sitecustomize.py"
    if startup.is_symlink():
        startup.unlink()
    startup.write_bytes(source.read_bytes())
    compilation = dict(
        cudagraph_mode=mode,
        prefill_cudagraph_capture_token_sizes=[256, 1024, 4096, 8192],
        decode_cudagraph_capture_request_sizes=[8, 16, 32, 64, 128, 256, 512],
        max_decode_cudagraph_capture_tokens=4096,
    )
    command = [
        str(args.venv / "bin/vllm"),
        "serve",
        args.model,
        "--port",
        str(args.port),
        "--trust-remote-code",
        "--cudagraph-metrics",
        "--kv-cache-dtype",
        "fp8",
        "--block-size",
        "256",
        "--enable-expert-parallel",
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(max_len),
        "--max-num-seqs",
        "1024",
        "--max-num-batched-tokens",
        "8192",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--attention-config",
        json.dumps(dict(indexer_kv_dtype="mxfp4")),
        "--moe-backend",
        "deep_gemm_mega_moe",
        "--tokenizer-mode",
        "deepseek_v4",
        "--speculative-config",
        json.dumps(
            dict(
                method="dspark",
                num_speculative_tokens=7,
                draft_sample_method="probabilistic",
            )
        ),
        "--compilation-config",
        json.dumps(compilation),
    ]
    if args.extra_args:
        command.extend(json.loads(args.extra_args))
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=args.gpus,
        HF_HOME="/mnt/lustre/hf-models",
        HF_HUB_OFFLINE="1",
        VLLM_USE_V2_MODEL_RUNNER="1",
        VLLM_USE_BREAKABLE_CUDAGRAPH="1",
        VLLM_USE_RUST_FRONTEND="0",
        PYTHONDONTWRITEBYTECODE="1",
        INDEXER_MEMORY_DIR=str(case),
        INDEXER_MEMORY_SNAPSHOTS=str(args.snapshot_dir / key),
        INDEXER_MEMORY_MIN_ROWS=str(args.snapshot_min_rows),
        VLLM_CACHE_ROOT=str(args.cache_dir),
        TORCHINDUCTOR_CACHE_DIR=str(args.cache_dir / "inductor"),
        TRITON_CACHE_DIR=str(args.cache_dir / "triton"),
        PYTHONPATH=str(hook_dir) + os.pathsep + str(args.tree),
        VLLM_ENGINE_READY_TIMEOUT_S="3600",
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="3600",
    )
    if args.sample_logits:
        environment["INDEXER_MEMORY_SAMPLES_DIR"] = str(args.out_dir / "real_logits")
    if args.runtime_peaks:
        environment["INDEXER_MEMORY_RUNTIME_PEAKS"] = "1"
    commit = subprocess.check_output(
        ["git", "-C", str(args.tree), "rev-parse", "HEAD"], text=True
    ).strip()
    meta = dict(
        case_key=key,
        kind="model_memory",
        phase="start",
        time=time.time(),
        source_commit=commit,
        model=args.model,
        graph_mode=mode,
        max_len=max_len,
        command=command,
        host=os.uname().nodename,
        gpus=args.gpus,
        tensor_parallel_size=args.tp,
        hooks_sha256=hashlib.sha256(startup.read_bytes()).hexdigest(),
        snapshots_dir=str(args.snapshot_dir / key),
    )
    emit(args.results, meta)
    if healthy(args.port):
        raise RuntimeError(f"port {args.port} already serves a process")
    logfile = case / "server.log"
    process = None
    successful = False
    try:
        with logfile.open("w") as log:
            log.write(json.dumps(meta) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd="/tmp",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited {process.returncode}")
                if healthy(args.port):
                    break
                time.sleep(3)
            else:
                raise TimeoutError("model startup timed out")
            emit(
                args.results,
                dict(
                    case_key=key,
                    phase="ready",
                    time=time.time(),
                    startup_seconds=time.time() - meta["time"],
                ),
            )
            for burst in map(int, args.bursts.split(",")):
                started = time.time()
                result = asyncio.run(requests(args.port, burst, args.max_tokens))
                failures = [r for r in result if r["status"] != 200]
                event = dict(
                    case_key=key,
                    phase="burst_complete",
                    burst=burst,
                    max_tokens=args.max_tokens,
                    time=time.time(),
                    wall_seconds=time.time() - started,
                    succeeded=burst - len(failures),
                    errors=failures[:8],
                )
                emit(args.results, event)
                if process.poll() is not None:
                    raise RuntimeError("server died during request burst")
                if failures:
                    raise RuntimeError(f"{len(failures)}/{burst} requests failed")
            if args.sample_logits:
                result = asyncio.run(requests(args.port, 1, 16, prompt_repeats=1024))
                emit(
                    args.results,
                    dict(
                        case_key=key,
                        phase="sample_workload_complete",
                        results=result,
                        memory_perturbed_by_copy=True,
                    ),
                )
            emit(args.results, dict(case_key=key, phase="complete", time=time.time()))
            successful = True
    except Exception as exc:
        emit(
            args.results,
            dict(
                case_key=key,
                phase="failed",
                error=repr(exc),
                time=time.time(),
                logfile=str(logfile),
            ),
        )
        print(f"{key}: {exc!r}", flush=True)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
        time.sleep(3)
    return successful


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-DSpark")
    parser.add_argument("--tag", default="dsv4")
    parser.add_argument(
        "--modes", default="PIECEWISE,FULL_DECODE_ONLY,FULL_AND_PIECEWISE"
    )
    parser.add_argument("--max-lens", default="262144,1048576")
    parser.add_argument("--bursts", default="8,128,1024")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--snapshot-min-rows", type=int, default=1024)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--sample-logits", action="store_true")
    parser.add_argument("--runtime-peaks", action="store_true")
    parser.add_argument("--port", type=int, default=19987)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--extra-args", help="JSON list appended to the serve command")
    args = parser.parse_args()
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    for max_len in map(int, args.max_lens.split(",")):
        for mode in args.modes.split(","):
            completed.append(run_case(args, mode, max_len))
    raise SystemExit(0 if all(completed) else 1)


if __name__ == "__main__":
    main()
