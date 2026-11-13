# SPDX-License-Identifier: Apache-2.0
"""Tests for the persistent-experts grid clamp (#66524)."""


def _fit(tp, grid, sms):
    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime,
    )

    return experts_runtime._fit_grids_to_device(tp, grid, sms)


def test_grids_clamped_to_sm_count():
    # 148-SM part (B200/GB200): every tuned 152-CTA entry comes down.
    grid, grids = _fit(tp=8, grid=136, sms=148)
    assert grid == 136
    assert max(grids.values()) <= 148
    assert grids[9] == 148 and grids[64] == 148 and grids[65] == 148
    assert grids[3] == 148  # already fits


def test_grids_untouched_on_152_sm_part():
    grid, grids = _fit(tp=8, grid=136, sms=152)
    assert grid == 136
    assert grids[9] == 152 and grids[65] == 152


def test_tp16_default_grid_clamped():
    grid, _ = _fit(tp=16, grid=152, sms=148)
    assert grid == 148
    grid, _ = _fit(tp=16, grid=152, sms=152)
    assert grid == 152


def test_tp16_batch8_tuning_preserved():
    # Current main tunes TP16 batch 8 to 144 CTAs (shared expert runs beside
    # the routed ones); the clamp must keep that entry.
    _, grids = _fit(tp=16, grid=152, sms=152)
    assert grids[8] == 144


def test_constructor_applies_the_clamp():
    """Guard the wiring: removing the clamp call from the constructor must
    fail a test (the helper's own tests do not see that)."""
    import inspect

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime,
    )

    src = inspect.getsource(experts_runtime.PersistentExperts.__init__)
    assert "_fit_grids_to_device(" in src
    assert "self.grid, self.grids = " in src


def test_persistent_experts_grids_fit_real_device():
    """Real construction on the current device: every grid entry and the
    default grid must be co-resident (<= SM count) or startup fails (#66524).
    Rejects mutations that pass a hardcoded count instead of the device SM
    count, which the pure-helper tests cannot see."""
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")

    from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.moe import (
        experts_runtime,
    )

    sms = torch.cuda.get_device_properties(0).multi_processor_count
    experts, rows = 8, 768  # rows==768 selects the TP8 tuned table
    w13 = torch.zeros(experts, rows, 1792, dtype=torch.uint8, device="cuda")
    s13 = torch.zeros(experts, rows, 28, dtype=torch.float8_e4m3fn, device="cuda")
    w2 = torch.zeros(experts, 3584, 192 * 3, dtype=torch.uint8, device="cuda")
    s2 = torch.zeros(experts, 3584, 24, dtype=torch.float8_e4m3fn, device="cuda")

    pe = experts_runtime.PersistentExperts(w13, s13, w2, s2)
    assert pe.grid <= sms
    assert max(pe.grids.values()) <= sms
