# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP helpers of the V2 model runner: which rows the sampled-token broadcast
must carry, and which received tensors do not match their buffers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tests.utils import multi_gpu_test
from vllm.distributed.parallel_state import get_pp_group
from vllm.utils.network_utils import get_open_port
from vllm.v1.worker.gpu import pp_utils
from vllm.v1.worker.gpu.spec_decode.adaptive_verification import (
    AdaptiveVerificationManager,
)


def _batch(
    num_computed,
    prefill_len,
    num_scheduled,
    is_prefilling=None,
    max_seq_len=None,
):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
        is_prefilling_np=(
            np.array(is_prefilling, dtype=np.bool_)
            if is_prefilling is not None
            else None
        ),
        max_seq_len_np=(
            np.array(max_seq_len, dtype=np.int32) if max_seq_len is not None else None
        ),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


def test_excludes_final_prefill_chunk_reaching_its_length_cap():
    """A max_tokens=1 final chunk hands its token off and leaves the engine.

    Disaggregated prefill submits max_tokens=1, so the request finishes with
    the final chunk's single sample and no follow-up step on this engine
    reads the broadcast payload.
    """
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1005],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_final_prefill_chunk_with_generation_budget_left():
    """The same final chunk, but the request keeps decoding here."""
    batch = _batch(
        num_computed=[1000],
        prefill_len=[1004],
        num_scheduled=[4],
        is_prefilling=[True],
        max_seq_len=[1004 + 256],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_length_cap_overrun_decode_row_is_kept():
    """Decoding rows are never excluded, even past their length cap.

    Speculative decoding can transiently overrun prompt_len + max_tokens
    while the scheduler still runs the request; dropping the row would
    desynchronize the PP stages.
    """
    batch = _batch(
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
        is_prefilling=[False],
        max_seq_len=[12176],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_finishing_prefill_row_does_not_drop_a_sibling_row():
    """Only finishing rows are excluded; a decoding sibling keeps its row."""
    batch = _batch(
        num_computed=[1000, 10],
        prefill_len=[1004, 8],
        num_scheduled=[4, 1],
        is_prefilling=[True, False],
        max_seq_len=[1005, 1024],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def _relay_steps(device: torch.device) -> tuple[list, list[int]]:
    """Run decode steps through a real PPHandler and adaptive verification
    manager. Returns each step's batch layout and, on the last stage, its
    draft budgets."""
    pp_group = get_pp_group()
    num_steps, num_drafts = 12, 2
    handler = pp_utils.PPHandler(
        max_num_reqs=8, num_speculative_steps=num_drafts, device=device
    )
    handler.configure_draft_confidence_relay()
    req_ids = ["a", "b", "c", "d"]
    req_states = SimpleNamespace(
        num_speculative_steps=num_drafts,
        device=device,
        max_num_reqs=8,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        num_computed_tokens_np=np.full(8, 10, dtype=np.int32),
        prefill_len=SimpleNamespace(np=np.ones(8, dtype=np.int32)),
    )
    manager = AdaptiveVerificationManager(
        req_states,  # type: ignore[arg-type]
        torch.zeros(9, dtype=torch.int32, device=device),
        num_bonus_tokens=1,
        max_total_logits=1 << 20,
    )
    # A fixed step overhead plus a per-token cost: only drafts that are likely
    # enough to survive pay for themselves, so the budget binds.
    manager.cost_tables = (np.zeros(9), 8.0 + np.arange(64, dtype=np.float64))
    generator = torch.Generator(device=device).manual_seed(0)
    idx_mapping_np = np.arange(4, dtype=np.int32)
    idx_mapping = torch.from_numpy(idx_mapping_np).to(device)
    layouts, budgets = [], []
    for step in range(num_steps):
        outputs = handler.get_prev_sampled_outputs()
        if outputs is not None and outputs["draft_confidences"] is not None:
            manager.record_confidences(*outputs["draft_confidences"])
        if step == 5:
            # Request "c" finishes and a new request reuses its slot while its
            # last confidences are still in flight.
            handler.on_req_idx_freed(2)
            manager.add_request(2)
        num_tokens = manager.get_num_tokens(
            dict.fromkeys(req_ids, 1 + num_drafts),
            dict.fromkeys(req_ids, [0] * num_drafts),
        )
        cu_num_logits, query_start_loc, budget, _ = manager.reallocate_drafts(
            req_ids, idx_mapping
        )
        layouts.append(
            (num_tokens, cu_num_logits.tolist(), query_start_loc[:5].tolist())
        )
        budgets.append(budget)

        batch = _batch([10] * 4, [1] * 4, [1 + num_drafts] * 4)
        batch.idx_mapping, batch.idx_mapping_np = idx_mapping, idx_mapping_np
        batch.num_draft_tokens = 4 * num_drafts
        if pp_group.is_last_rank:
            sampled = torch.zeros(4, 1 + num_drafts, dtype=torch.int64, device=device)
            counts = torch.zeros(4, dtype=torch.int32, device=device)
            handler.broadcast(sampled, counts, counts, batch)
            confidences = torch.rand(4, num_drafts, generator=generator, device=device)
            handler.broadcast_drafts(sampled[:, :num_drafts].clone(), confidences)
        else:
            handler.receive(batch, recv_drafts=True)
    torch.accelerator.synchronize()
    return layouts, budgets


def _relay_worker(rank: int, world_size: int, port: str) -> None:
    from tests.utils import init_test_distributed_environment

    device = torch.device(f"cuda:{rank}")
    torch.accelerator.set_device_index(device)
    init_test_distributed_environment(1, world_size, rank, port)
    layouts, budgets = _relay_steps(device)
    gathered: list = [None] * world_size
    torch.distributed.all_gather_object(
        gathered, (layouts, budgets), group=get_pp_group().cpu_group
    )
    stage_layouts = [layouts for layouts, _ in gathered]
    assert all(layouts == stage_layouts[-1] for layouts in stage_layouts)
    # The budget must bind on some steps, or the layouts agree trivially.
    last_budgets = gathered[-1][1]
    assert any(0 < budget < 8 for budget in last_budgets), last_budgets


@multi_gpu_test(num_gpus=2)
def test_pp_stages_trim_drafts_alike():
    """Adaptive verification trims each step's drafts from the drafts'
    confidences, which only the last stage computes. Every stage must pick the
    same draft budget and request boundaries at every step, including when a
    request slot is reused while its confidences are in flight."""
    torch.multiprocessing.spawn(_relay_worker, args=(2, str(get_open_port())), nprocs=2)
