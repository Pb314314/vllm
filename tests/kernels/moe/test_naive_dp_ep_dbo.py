# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
    make_moe_prepare_and_finalize_naive_dp_ep,
)


class _RegionProbe(MoEPrepareAndFinalizeNaiveDPEPModular):
    def __init__(self) -> None:
        super().__init__()
        self.body_ran = False

    def run_comm(self) -> None:
        with self._comm_region():
            self.body_ran = True


@pytest.fixture
def dbo_mocks():
    yield_switch = MagicMock()
    switch_back = MagicMock()
    with (
        patch.object(
            naive_dp_ep, "dbo_yield_and_switch_from_compute_to_comm", yield_switch
        ),
        patch.object(naive_dp_ep, "dbo_switch_to_compute_sync", switch_back),
    ):
        yield {"yield_switch": yield_switch, "switch_back": switch_back}


def _dbo_state(dbo_active: bool, capturing: bool):
    return (
        patch.object(naive_dp_ep, "dbo_enabled", return_value=dbo_active),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=capturing),
    )


def test_comm_region_noop_without_dbo(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=False, capturing=False)
    with dbo, capturing:
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_not_called()
    dbo_mocks["switch_back"].assert_not_called()


def test_comm_region_handoff_when_dbo_active(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=True, capturing=False)
    with dbo, capturing:
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_called_once_with()
    dbo_mocks["switch_back"].assert_called_once_with()


def test_comm_region_skipped_during_graph_capture(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=True, capturing=True)
    with dbo, capturing:
        probe.run_comm()
    assert probe.body_ran
    dbo_mocks["yield_switch"].assert_not_called()
    dbo_mocks["switch_back"].assert_not_called()


def test_comm_region_restores_compute_stream_on_error(dbo_mocks):
    probe = _RegionProbe()
    dbo, capturing = _dbo_state(dbo_active=True, capturing=False)
    with (
        dbo,
        capturing,
        pytest.raises(RuntimeError, match="collective failed"),
        probe._comm_region(),
    ):
        raise RuntimeError("collective failed")
    dbo_mocks["yield_switch"].assert_called_once_with()
    dbo_mocks["switch_back"].assert_called_once_with()


@pytest.mark.parametrize("is_rocm", [True, False])
def test_supports_dbo_only_on_rocm(is_rocm):
    inst = MoEPrepareAndFinalizeNaiveDPEPModular()
    platform = SimpleNamespace(is_rocm=lambda: is_rocm)
    with patch.object(naive_dp_ep, "current_platform", platform):
        assert inst.supports_dbo() is is_rocm
        # supports_async() also drives MK-internal shared-expert overlap, so
        # it must not change with DBO support.
        assert not inst.supports_async()


@pytest.mark.parametrize("is_rocm", [True, False])
def test_factory_returns_base_classes(is_rocm):
    platform = SimpleNamespace(is_rocm=lambda: is_rocm)
    with patch.object(naive_dp_ep, "current_platform", platform):
        mono = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=True)
        modular = make_moe_prepare_and_finalize_naive_dp_ep(use_monolithic=False)
    assert type(mono) is naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPMonolithic
    assert type(modular) is MoEPrepareAndFinalizeNaiveDPEPModular


def test_default_supports_dbo_follows_supports_async():
    # Classes that do not override supports_dbo() keep the old contract:
    # DBO only with async prepare/finalize.
    mono = naive_dp_ep.MoEPrepareAndFinalizeNaiveDPEPMonolithic()
    assert not mono.supports_async()
    assert not mono.supports_dbo()
    with patch.object(type(mono), "supports_async", return_value=True):
        assert mono.supports_dbo()


def test_monolithic_prepare_routed_dispatches_topk_not_logits():
    """DP padding skip routes locally and gathers the -1 ids of padded rows, so
    no rank's experts see them; logits must not be dispatched."""
    from vllm.model_executor.layers.fused_moe.config import (
        FUSED_MOE_UNQUANTIZED_CONFIG,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic,
    )

    def gather_two_ranks(hidden, weights, ids, **kwargs):
        return tuple(torch.cat([t, t]) for t in (hidden, weights, ids))

    ep_group = MagicMock()
    ep_group.dispatch.side_effect = gather_two_ranks
    pf = MoEPrepareAndFinalizeNaiveDPEPMonolithic()
    hidden = torch.randn(3, 8)
    weights = torch.rand(3, 2)
    ids = torch.tensor([[0, 1], [2, 3], [-1, -1]], dtype=torch.int32)
    with patch.object(naive_dp_ep, "get_ep_group", return_value=ep_group):
        a1q, a1q_scale, out_weights, out_ids = pf.prepare_routed(
            hidden, weights, ids, quant_config=FUSED_MOE_UNQUANTIZED_CONFIG
        )

    assert pf.supports_precomputed_routing()
    ep_group.dispatch_router_logits.assert_not_called()
    assert a1q.shape == (6, 8) and a1q_scale is None
    torch.testing.assert_close(out_ids, torch.cat([ids, ids]))
    torch.testing.assert_close(out_weights, torch.cat([weights, weights]))


@pytest.mark.parametrize(
    ("num_tokens", "dp_size", "expected"),
    [
        (8, 1, False),  # no DP: padding is a few rows, never worth a top-k
        (8, 4, True),  # 320 picks over 512 experts: padded rows add experts
        (24, 4, True),  # 960 picks: still under 2x the experts
        (48, 4, False),  # 1920 picks: real rows already cover the experts
    ],
)
def test_padding_skip_only_when_padded_rows_can_add_experts(
    num_tokens, dp_size, expected
):
    from vllm.model_executor.layers.fused_moe.router.base_router import (
        padding_skip_pays_off,
    )

    assert (
        padding_skip_pays_off(num_tokens, dp_size, top_k=10, num_experts=512)
        is expected
    )


def test_padding_skip_decision_is_uniform_across_dp_ranks(monkeypatch):
    """Eager DP steps keep per-rank sizes; a rank-local gate would dispatch
    top-k ids on small ranks and logits on large ones, hanging the gather."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.router import base_router

    ctx = SimpleNamespace(
        dp_metadata=SimpleNamespace(
            num_tokens_across_dp_cpu=torch.tensor([3, 1351, 5, 8])
        )
    )
    monkeypatch.setattr(base_router, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(base_router, "get_forward_context", lambda: ctx)

    decisions = {
        base_router.padding_skip_pays_off(
            base_router.max_num_tokens_across_dp(n),
            dp_size=4,
            top_k=10,
            num_experts=512,
        )
        for n in (3, 1351, 5, 8)
    }
    assert decisions == {False}
