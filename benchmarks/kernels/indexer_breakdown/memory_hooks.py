# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only allocation hooks for a real vLLM model process.

Enable by making sitecustomize.py load this file and setting
INDEXER_MEMORY_DIR. Only CUDA device 0 is recorded by default. The allocation
audit reads metadata without retaining tensors. Optional real-logits sampling
is a separate diagnostic workload and explicitly logs its copy perturbation.
"""

import functools
import hashlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import time
from pathlib import Path

ROOT = os.environ.get("INDEXER_MEMORY_DIR")
PHASE = "initialization"
SEEN = set()
SNAPSHOTS = set()
MIN_ROWS = int(os.environ.get("INDEXER_MEMORY_MIN_ROWS", "512"))
SAMPLED_LAYERS = set()
PEAK_SHAPES = set()


def maybe_sample(output, args, family, layer):
    directory = os.environ.get("INDEXER_MEMORY_SAMPLES_DIR")
    control = Path(ROOT) / "sample_directory"
    if not directory and control.exists():
        directory = control.read_text().strip()
    if not directory or PHASE != "runtime" or family != "dense":
        return
    import torch

    if torch.cuda.is_current_stream_capturing() or layer in SAMPLED_LAYERS:
        return
    if len(SAMPLED_LAYERS) >= 3:
        return
    # Separate diagnostic workload: these copies deliberately perturb memory.
    # Only retain valid columns of a small number of real decode rows.
    lengths = args[3].flatten()[:64].cpu()
    width = int(lengths.max())
    if width < 1024:
        return
    SAMPLED_LAYERS.add(layer)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    path = path / f"logits_{os.getpid()}_{len(SAMPLED_LAYERS)}.pt"
    values = output[: len(lengths), :width].cpu()
    torch.save(
        dict(
            logits=values,
            seq_lens=lengths,
            family=family,
            layer=layer,
            full_shape=list(output.shape),
            full_stride=list(output.stride()),
        ),
        path,
    )
    emit(
        "logits_sample",
        path=str(path),
        layer=layer,
        rows=len(lengths),
        valid_width=width,
        memory_perturbed_by_copy=True,
    )


def enabled():
    import torch

    return (
        ROOT
        and torch.cuda.is_initialized()
        and torch.accelerator.current_device_index() == 0
    )


def emit(event, **data):
    if not ROOT:
        return
    row = dict(event=event, pid=os.getpid(), time=time.time(), phase=PHASE, **data)
    try:
        Path(ROOT).mkdir(parents=True, exist_ok=True)
        with open(f"{ROOT}/memory_trace.{os.getpid()}.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
    except Exception as exc:
        print(f"indexer memory emit failed: {exc!r}", file=sys.stderr)


def tensor_info(tensor):
    return dict(
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        dtype=str(tensor.dtype),
        ptr=tensor.data_ptr(),
        storage_bytes=tensor.untyped_storage().nbytes(),
    )


def light_memory():
    import torch

    stats = torch.accelerator.memory_stats()
    return dict(
        device=torch.accelerator.current_device_index(),
        allocated=torch.accelerator.memory_allocated(),
        reserved=torch.accelerator.memory_reserved(),
        peak_allocated=torch.accelerator.max_memory_allocated(),
        retries=stats.get("num_alloc_retries", 0),
        capturing=torch.cuda.is_current_stream_capturing(),
        stream=torch.cuda.current_stream().cuda_stream,
    )


def caller_layer():
    frame = inspect.currentframe()
    try:
        for _ in range(14):
            frame = frame.f_back
            if frame is None:
                break
            if "k_cache_prefix" in frame.f_locals:
                return str(frame.f_locals["k_cache_prefix"])
            obj = frame.f_locals.get("self")
            if obj is not None and "Indexer" in type(obj).__name__:
                cache = getattr(obj, "k_cache", None)
                if cache is not None:
                    return getattr(cache, "prefix", type(obj).__name__)
    finally:
        del frame
    return "unknown"


def dump_snapshot(label):
    if not enabled() or label in SNAPSHOTS:
        return
    import torch

    SNAPSHOTS.add(label)
    try:
        directory = Path(os.environ.get("INDEXER_MEMORY_SNAPSHOTS", ROOT))
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"memory_{os.getpid()}_{label}.pickle"
        torch.cuda.memory._dump_snapshot(str(path))
        emit("snapshot", label=label, path=str(path), memory=light_memory())
    except Exception as exc:
        emit("hook_error", where="snapshot", error=repr(exc))


def patch_logits(module, name, family):
    original = getattr(module, name, None)
    if original is None or getattr(original, "_indexer_memory_hook", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        output = original(*args, **kwargs)
        if enabled():
            import torch

            layer = caller_layer()
            maybe_sample(output, args, family, layer)
            key = (
                PHASE,
                family,
                layer,
                tuple(output.shape),
                str(output.dtype),
                torch.cuda.is_current_stream_capturing(),
            )
            if key not in SEEN:
                SEEN.add(key)
                emit(
                    "logits",
                    family=family,
                    layer=layer,
                    tensor=tensor_info(output),
                    memory=light_memory(),
                )
        return output

    wrapped._indexer_memory_hook = True
    setattr(module, name, wrapped)


def patch_dense(module):
    patch_logits(module, "fp8_fp4_paged_mqa_logits", "dense")
    original = module.as_fp32_logits
    if getattr(original, "_indexer_memory_hook", False):
        return

    @functools.wraps(original)
    def wrapped(logits, *args, **kwargs):
        output = original(logits, *args, **kwargs)
        if enabled():
            layer = caller_layer()
            import torch

            key = (
                PHASE,
                "upcast",
                layer,
                tuple(logits.shape),
                torch.cuda.is_current_stream_capturing(),
            )
            if key not in SEEN:
                SEEN.add(key)
                emit(
                    "upcast_overlap",
                    layer=layer,
                    source=tensor_info(logits),
                    target=tensor_info(output),
                    memory=light_memory(),
                )
                if logits.shape[0] >= MIN_ROWS:
                    kind = (
                        "captured"
                        if torch.cuda.is_current_stream_capturing()
                        else "eager"
                    )
                    dump_snapshot(PHASE + "_" + kind + "_upcast_overlap")
        return output

    wrapped._indexer_memory_hook = True
    module.as_fp32_logits = wrapped
    original_forward = module.SparseAttnIndexer.forward_cuda

    @functools.wraps(original_forward)
    def forward(self, hidden_states, *args, **kwargs):
        import torch

        layer = self.k_cache.prefix
        rows = hidden_states.shape[0]
        metadata = None
        if os.environ.get("INDEXER_MEMORY_RUNTIME_PEAKS") == "1" and PHASE == "runtime":
            from vllm.forward_context import get_forward_context

            attention = get_forward_context().attn_metadata
            if isinstance(attention, dict):
                metadata = attention.get(layer)
        decode_rows = getattr(metadata, "num_decode_tokens", None)
        prefill_rows = getattr(metadata, "num_prefill_tokens", None)
        key = (layer, rows, decode_rows, prefill_rows)
        measure = (
            os.environ.get("INDEXER_MEMORY_RUNTIME_PEAKS") == "1"
            and PHASE == "runtime"
            and enabled()
            and not torch.cuda.is_current_stream_capturing()
            and ".layers.2." in layer
            and key not in PEAK_SHAPES
        )
        if not measure:
            return original_forward(self, hidden_states, *args, **kwargs)
        PEAK_SHAPES.add(key)
        before = light_memory()
        torch.accelerator.reset_peak_memory_stats()
        try:
            return original_forward(self, hidden_states, *args, **kwargs)
        finally:
            after = light_memory()
            emit(
                "runtime_indexer_peak",
                layer=layer,
                rows=rows,
                num_decode_tokens=decode_rows,
                num_prefill_tokens=prefill_rows,
                max_seq_len_host=getattr(metadata, "max_seq_len", None),
                before=before,
                after=after,
                peak_increment_bytes=after["peak_allocated"] - before["allocated"],
                counters_reset_for_diagnostic=True,
            )
            if rows >= 4096:
                dump_snapshot(f"runtime_peak_r{rows}")

    module.SparseAttnIndexer.forward_cuda = forward
    emit("hook_installed", module=module.__name__)


def patch_sparse(module):
    patch_logits(module, "fp8_fp4_paged_sparse_mqa_logits", "candidate_sparse")
    patch_logits(module, "fp8_fp4_sparse_mqa_logits", "candidate_sparse_prefill")
    emit("hook_installed", module=module.__name__)


def patch_topk(module):
    original = module.SparseIndexerTopk.forward

    @functools.wraps(original)
    def forward(self, logits, seq_lens, next_n, output, topk_tokens, max_seq_len):
        if enabled():
            import torch

            layer = caller_layer()
            backend = self.resolve_backend(logits, topk_tokens, logits.shape[0])
            key = (
                PHASE,
                "topk",
                layer,
                tuple(logits.shape),
                backend,
                max_seq_len,
                torch.cuda.is_current_stream_capturing(),
            )
            if key not in SEEN:
                SEEN.add(key)
                emit(
                    "topk_dispatch",
                    layer=layer,
                    backend=backend,
                    configured_backend=self._backend,
                    topk=topk_tokens,
                    max_seq_len_host=max_seq_len,
                    next_n=next_n,
                    seq_lens_shape=list(seq_lens.shape),
                    logits=tensor_info(logits),
                    memory=light_memory(),
                )
        return original(
            self, logits, seq_lens, next_n, output, topk_tokens, max_seq_len
        )

    module.SparseIndexerTopk.forward = forward
    emit("hook_installed", module=module.__name__)


def patch_runner(module):
    cls = module.GPUModelRunner
    original_load = cls.load_model

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        if enabled():
            layers = []
            for name, obj in self.model.named_modules():
                if "Indexer" in type(obj).__name__ and hasattr(obj, "indexer_op"):
                    op = obj.indexer_op
                    layers.append(
                        dict(
                            name=name,
                            cls=type(obj).__name__,
                            op=type(op).__name__,
                            width=getattr(op, "max_model_len", None),
                            topk=getattr(op, "topk_tokens", None),
                            candidate_write=getattr(op, "candidate_write", None),
                        )
                    )
            emit(
                "model_loaded",
                layers=layers,
                cudagraph_mode=str(self.compilation_config.cudagraph_mode),
                sparse_logits=self.vllm_config.attention_config.indexer_sparse_logits,
                splitting_ops=self.compilation_config.splitting_ops,
                use_breakable=os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH"),
                memory=light_memory(),
            )
        return result

    cls.load_model = load
    original_capture = cls.capture_model

    @functools.wraps(original_capture)
    def capture(self, *args, **kwargs):
        global PHASE
        previous = PHASE
        profile = bool(kwargs.get("profile_only", args[0] if args else False))
        PHASE = "profile_capture" if profile else "final_capture"
        if enabled():
            import torch

            torch.cuda.memory._record_memory_history(max_entries=500000)
            emit("capture_start", profile_only=profile, memory=light_memory())
        try:
            result = original_capture(self, *args, **kwargs)
            if enabled():
                emit(
                    "capture_end",
                    profile_only=profile,
                    result=result,
                    memory=light_memory(),
                )
                dump_snapshot(PHASE + "_end")
            return result
        finally:
            PHASE = "runtime" if not profile else previous

    cls.capture_model = capture
    emit("hook_installed", module=module.__name__)


PATCHES = {
    "vllm.model_executor.layers.indexer_topk": patch_topk,
    "vllm.model_executor.layers.sparse_attn_indexer": patch_dense,
    "vllm.model_executor.kernels.attention.dsa.sparse_mqa_logits": patch_sparse,
    "vllm.v1.worker.gpu.model_runner": patch_runner,
}


class Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec and spec.loader:
            original_exec = spec.loader.exec_module

            def execute(module):
                original_exec(module)
                try:
                    PATCHES[fullname](module)
                except Exception as exc:
                    emit("hook_error", where=fullname, error=repr(exc))

            spec.loader.exec_module = execute
        return spec


if ROOT:
    emit(
        "hooks_loaded",
        path=__file__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    sys.meta_path.insert(0, Hook())
