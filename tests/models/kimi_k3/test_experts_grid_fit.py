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
