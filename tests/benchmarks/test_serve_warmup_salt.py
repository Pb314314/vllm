# SPDX-License-Identifier: Apache-2.0
"""Warmup requests must not prime the shared prefix cache (#66698)."""

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput
from vllm.benchmarks.serve import _salted_warmup_input


def _test_input(extra_body=None):
    return RequestFuncInput(
        prompt="hello world",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=2,
        output_len=4,
        model="m",
        extra_body=extra_body,
    )


def test_warmup_input_gets_unique_cache_salt():
    a = _salted_warmup_input(_test_input(), 0)
    b = _salted_warmup_input(_test_input(), 1)
    assert a.extra_body["cache_salt"] != b.extra_body["cache_salt"]
    # the measured run's input is untouched
    assert _test_input().extra_body is None


def test_warmup_salt_preserves_existing_extra_body():
    out = _salted_warmup_input(_test_input({"stream": True}), 0)
    assert out.extra_body["stream"] is True
    assert out.extra_body["cache_salt"].startswith("warmup-0-")
