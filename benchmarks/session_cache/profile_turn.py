# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import cProfile
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["apc", "session"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["SESSION_PROFILE_DIR"] = str(root)
    from vllm import LLM, SamplingParams

    scheduler = "session_profile_hooks." + (
        "APCProfileScheduler" if args.mode == "apc" else "SessionProfileScheduler"
    )
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        scheduler_cls=scheduler,
        enable_prefix_caching=args.mode == "apc",
        async_scheduling=False,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_model_len=17024,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=3072 * 1024 * 1024,
        gpu_memory_utilization=0.04,
        block_size=16,
        seed=0,
        enable_trace_replay=True,
        attention_config={"backend": "TRITON_ATTN"},
        profiler_config={
            "profiler": "torch",
            "torch_profiler_dir": str(root / "torch"),
            "torch_profiler_with_stack": False,
            "torch_profiler_record_shapes": True,
            "torch_profiler_with_memory": False,
        },
    )
    unit = llm.get_tokenizer().encode(
        "Inspect the repository, run the tests, and summarize the results. ",
        add_special_tokens=False,
    )
    source = (unit * (17024 // len(unit) + 1))[:17024]
    completion = source[:16]
    history = []
    outputs = []
    for turn in range(4):
        delta = source[:16384] if turn == 0 else source[:64]
        history.extend(delta)
        extra = (
            {"session_cache": {"id": "agent", "version": turn}}
            if args.mode == "session"
            else {}
        )
        tag = "cpu" if turn == 2 else "torch" if turn == 3 else None
        if tag:
            extra["profile_tag"] = tag
        params = SamplingParams(
            temperature=0,
            max_tokens=16,
            ignore_eos=True,
            trace_decode_token_ids=completion,
            extra_args=extra,
        )
        if tag == "torch":
            llm.start_profile(profile_prefix=args.mode)
        front_profile = cProfile.Profile() if tag == "cpu" else None
        if front_profile:
            front_profile.enable()
        output = llm.generate(
            {"prompt_token_ids": history if args.mode == "apc" else delta},
            params,
            use_tqdm=False,
        )[0]
        if front_profile:
            front_profile.disable()
            front_profile.dump_stats(str(root / "frontend.pstats"))
        if tag == "torch":
            llm.stop_profile()
        assert output.outputs[0].token_ids == completion
        outputs.append(
            {
                "turn": turn,
                "context": len(history),
                "cached": output.num_cached_tokens,
                "profile": tag,
            }
        )
        history.extend(completion)
    if args.mode == "session":
        assert llm.close_session("agent")
    (root / "turns.json").write_text(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
