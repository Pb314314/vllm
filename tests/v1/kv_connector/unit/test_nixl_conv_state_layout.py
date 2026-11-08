# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import pytest

from vllm.config import KVTransferConfig, VllmConfig
from vllm.platforms import current_platform

LAYOUT_ENV = "VLLM_SSM_CONV_STATE_LAYOUT"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA platform config")
@pytest.mark.parametrize(
    ("connector", "preset", "expected"),
    [
        ("NixlConnector", None, "DS"),
        ("NixlConnector", "SD", "SD"),
        ("OffloadingConnector", None, None),
    ],
)
def test_platform_selects_ds_conv_state_layout_for_nixl(
    monkeypatch: pytest.MonkeyPatch,
    connector: str,
    preset: str | None,
    expected: str | None,
):
    """NIXL transfers Mamba conv state only in the DS layout; without it the
    prefill instance asserts after loading. The platform selects DS for NIXL
    unless the layout was set explicitly."""
    # setenv first so that teardown restores the variable the platform sets.
    monkeypatch.setenv(LAYOUT_ENV, preset or "")
    if preset is None:
        monkeypatch.delenv(LAYOUT_ENV)
    VllmConfig(
        kv_transfer_config=KVTransferConfig(kv_connector=connector, kv_role="kv_both")
    )
    assert os.environ.get(LAYOUT_ENV) == expected
