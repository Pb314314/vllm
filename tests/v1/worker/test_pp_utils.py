# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

from unittest.mock import Mock

import numpy as np

from vllm.v1.worker.gpu import pp_utils


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_last_rank_records_draft_confidences_with_the_other_ranks(monkeypatch):
    """Adaptive verification trims each step from the draft confidences, so the
    last rank must apply its own at the step the other ranks receive them,
    without the rows of requests freed in between."""
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    pp_size = 2
    handler = pp_utils.PPHandler.__new__(pp_utils.PPHandler)
    handler.is_last_rank = True
    handler.last_rank = pp_size - 1
    handler.ring_depth = pp_size
    handler.queue = pp_utils.deque()
    handler.relay_draft_confidences = False
    handler.broadcast_slot = None
    handler.max_sample_len = 3
    handler.device = torch.device("cuda")
    handler.main_stream = torch.cuda.current_stream()
    handler.broadcast_stream = torch.cuda.Stream()
    handler.broadcast_group = None
    handler.req_idx_gen_np = np.zeros(4, dtype=np.int32)
    handler.configure_draft_confidence_relay()

    def step(idx_mapping: list[int], confidence: float) -> None:
        batch = _batch([5] * len(idx_mapping), [1] * len(idx_mapping), [3, 3])
        batch.num_draft_tokens = 2 * len(idx_mapping)
        batch.idx_mapping_np = np.array(idx_mapping, dtype=np.int32)
        batch.idx_mapping = torch.tensor(idx_mapping, device="cuda")
        num = torch.zeros(len(idx_mapping), dtype=torch.int32, device="cuda")
        sampled = torch.zeros(len(idx_mapping), 3, dtype=torch.int64, device="cuda")
        handler.broadcast(sampled, num, num, batch)
        confidences = torch.full((len(idx_mapping), 2), confidence, device="cuda")
        handler.broadcast_drafts(sampled[:, :2], confidences)

    assert handler.get_prev_sampled_outputs() is None
    step([0, 1], 0.5)
    assert handler.get_prev_sampled_outputs() is None
    step([2, 3], 0.25)
    outputs = handler.get_prev_sampled_outputs()
    assert outputs is not None and outputs.keys() == {"draft_confidences"}
    confidences, idx_mapping = outputs["draft_confidences"]
    assert idx_mapping.tolist() == [0, 1]
    assert confidences.tolist() == [[0.5, 0.5], [0.5, 0.5]]

    handler.on_req_idx_freed(2)
    confidences, idx_mapping = handler.get_prev_sampled_outputs()["draft_confidences"]
    assert idx_mapping.tolist() == [3]
    assert confidences.tolist() == [[0.25, 0.25]]
