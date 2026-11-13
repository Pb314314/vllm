# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.utils import multi_gpu_test
from tests.v1.e2e.spec_decode.utils import (
    Messages,
    assert_request_outputs_match,
    compute_acceptance_len,
    get_test_prompts,
)
from vllm import LLM, RequestOutput, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
DRAFT = "nm-testing/Llama3_2_1B_speculator.eagle3"
# Enough tokens that one greedy path changing under compilation can't move the mean
TOPICS = ["the ocean", "a robot", "Rome", "volcanoes"]
TEMPLATES = [
    "Write a short story about {}.",
    "Explain {} to a child.",
    "List five facts about {}:",
    "Q: What is the history of {}?\nA:",
]
PROMPTS = [template.format(topic) for topic in TOPICS for template in TEMPLATES]

ACCEPTANCE_TOLERANCE = 0.95


def _run(
    pp_size: int,
    model: str = MODEL,
    draft: str = DRAFT,
    max_model_len: int = 512,
    max_tokens: int = 64,
    chat_prompts: list[Messages] | None = None,
    model_impl: str = "auto",
    adaptive_verification: bool | None = False,
) -> tuple[float, list[RequestOutput]]:
    """Greedy outputs and acceptance length; `adaptive_verification=None` runs
    without speculative decoding."""
    speculative_config = None
    if adaptive_verification is not None:
        speculative_config = {
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": 3,
            "enable_adaptive_verification": adaptive_verification,
        }
    llm = LLM(
        model=model,
        model_impl=model_impl,
        tensor_parallel_size=1,
        pipeline_parallel_size=pp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=0.45,
        disable_log_stats=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        speculative_config=speculative_config,
    )
    try:
        sampling_params = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
        )
        if chat_prompts is None:
            outputs = llm.generate(PROMPTS, sampling_params)
        else:
            outputs = llm.chat(chat_prompts, sampling_params)
        if speculative_config is None:
            return 1.0, outputs
        acceptance = compute_acceptance_len(llm.get_metrics())
        assert acceptance > 1
        return acceptance, outputs
    finally:
        del llm
        torch.accelerator.empty_cache()
        cleanup_dist_env_and_memory()


@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("model_impl", ["vllm", "transformers"])
def test_eagle3_pipeline_parallel_acceptance(model_impl: str):
    baseline, _ = _run(1, model_impl=model_impl)
    for pp_size in (2, 4):
        parallel, _ = _run(pp_size, model_impl=model_impl)
        assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
            f"PP={pp_size} acceptance regressed: {parallel:.3f} < "
            f"{baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
        )


@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize(
    "model,draft",
    [
        # The text model inherits PP support from Qwen2Model but has its own
        # forward, which must send this stage's aux hidden states.
        ("Qwen/Qwen3-VL-4B-Instruct", "AngelSlim/Qwen3-VL-4B-Instruct_eagle3"),
        ("openai/gpt-oss-20b", "RedHatAI/gpt-oss-20b-speculator.eagle3"),
    ],
)
def test_eagle3_pipeline_parallel_targets(model, draft):
    """Targets whose own forward passes aux hidden states between PP stages."""
    # Greedy outputs at PP=1 and PP=2 can diverge after a few tokens, so the
    # acceptance lengths are compared over many prompts.
    prompts = get_test_prompts(mm_enabled=False, num_prompts=64)
    baseline, _ = _run(1, model, draft, 2048, 128, prompts)
    parallel, _ = _run(2, model, draft, 2048, 128, prompts)
    assert parallel >= baseline * ACCEPTANCE_TOLERANCE, (
        f"PP=2 acceptance regressed: {parallel:.3f} < "
        f"{baseline:.3f} * {ACCEPTANCE_TOLERANCE}"
    )


@multi_gpu_test(num_gpus=2)
def test_eagle3_pipeline_parallel_adaptive_verification():
    """Every PP stage must trim each step's drafts identically: stages that
    disagree run different token counts and their hidden states no longer line
    up, which fails the step or corrupts the outputs."""
    prompts = get_test_prompts(mm_enabled=False, num_prompts=64)
    # Unlike the Llama-3.2-1B pair, this pair is costly enough per token that
    # adaptive verification trims drafts on two GPUs.
    model, draft = "Qwen/Qwen3-8B", "RedHatAI/Qwen3-8B-speculator.eagle3"
    _, ref_outputs = _run(
        2, model, draft, 2048, 128, prompts, adaptive_verification=None
    )
    full, fixed_outputs = _run(2, model, draft, 2048, 128, prompts)
    trimmed, outputs = _run(
        2, model, draft, 2048, 128, prompts, adaptive_verification=True
    )
    # Trimming drops drafts that fixed drafts would have had accepted.
    assert trimmed < full, (
        f"Adaptive verification kept every draft ({trimmed:.3f} vs {full:.3f} "
        "accepted per step), so trimming is not exercised."
    )
    # Greedy outputs drift from the non-speculative ones even with fixed drafts,
    # so adaptive verification must match them about as often as fixed drafts do.
    fixed_matches = sum(
        ref.outputs[0].text == fixed.outputs[0].text
        for ref, fixed in zip(ref_outputs, fixed_outputs)
    )
    assert fixed_matches >= len(prompts) // 4, (
        f"Only {fixed_matches}/{len(prompts)} fixed-draft outputs match the "
        "non-speculative ones, too few to compare against."
    )
    assert_request_outputs_match(
        ref_outputs,
        outputs,
        required_matches=int(0.8 * fixed_matches),
        context=f"PP=2 EAGLE3 with adaptive verification ({fixed_matches} matches "
        "with fixed drafts)",
    )
