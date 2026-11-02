# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay metadata and attention boundary for experimental encoder graphs."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import torch

from vllm.compilation.breakable_cudagraph import record_eager_call
from vllm.utils.torch_utils import direct_register_custom_op


@dataclass(frozen=True)
class EncoderAttentionMetadata:
    """One packed self-attention layout; lengths exclude unused capture slots."""

    sequence_lengths: tuple[int, ...]
    cu_seqlens: torch.Tensor
    max_seqlen: torch.Tensor
    sequence_lengths_gpu: torch.Tensor | None = None

    @property
    def num_tokens(self) -> int:
        return sum(self.sequence_lengths)


_encoder_metadata: ContextVar[EncoderAttentionMetadata | None] = ContextVar(
    "encoder_attention_metadata", default=None
)


def get_encoder_attention_metadata() -> EncoderAttentionMetadata | None:
    return _encoder_metadata.get()


@contextmanager
def encoder_attention_context(
    metadata: EncoderAttentionMetadata | None,
) -> Iterator[None]:
    token = _encoder_metadata.set(metadata)
    try:
        yield
    finally:
        _encoder_metadata.reset(token)


def _encoder_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    backend: str,
    fa_version: int | None,
    workspace: torch.Tensor | None = None,
) -> None:
    from vllm.v1.attention.ops.vit_attn_wrappers import (
        torch_sdpa_wrapper,
        vit_flash_attn_wrapper,
        vit_flashinfer_wrapper,
    )

    metadata = get_encoder_attention_metadata()
    if metadata is None:
        raise RuntimeError("Encoder attention replay requires current batch metadata")
    n = metadata.num_tokens
    if query.shape[0] != 1 or not 0 < n <= query.shape[1]:
        raise ValueError("Expected a nonempty packed encoder batch within capture size")
    q, k, v = query[:, :n], key[:, :n], value[:, :n]
    if backend == "FLASH_ATTN":
        result = vit_flash_attn_wrapper(
            q,
            k,
            v,
            batch_size=1,
            is_rocm_aiter=False,
            fa_version=fa_version,
            scale=scale,
            cu_seqlens=metadata.cu_seqlens,
            max_seqlen=metadata.max_seqlen,
        )
    elif backend == "TORCH_SDPA":
        result = torch_sdpa_wrapper(
            q,
            k,
            v,
            scale=scale,
            enable_gqa=q.shape[2] > k.shape[2],
            sequence_lengths=list(metadata.sequence_lengths),
        )
    elif backend == "FLASHINFER":
        assert workspace is not None and metadata.sequence_lengths_gpu is not None
        result = vit_flashinfer_wrapper(
            q,
            k,
            v,
            scale=scale,
            workspace_buffer=workspace,
            cu_seqlens=metadata.cu_seqlens,
            max_seqlen=metadata.max_seqlen,
            sequence_lengths=metadata.sequence_lengths_gpu,
        )
    else:
        raise NotImplementedError(f"Encoder breakable attention: {backend}")
    output[:, :n].copy_(result)
    output[:, n:].zero_()


def encoder_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    backend: str,
    fa_version: int | None,
    workspace: torch.Tensor | None = None,
) -> None:
    record_eager_call(
        _encoder_attention_with_output,
        query,
        key,
        value,
        output,
        scale,
        backend,
        fa_version,
        workspace,
    )


def encoder_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    backend: str,
    fa_version: int | None,
    workspace: torch.Tensor | None = None,
) -> None:
    pass


direct_register_custom_op(
    op_name="encoder_attention_with_output",
    op_func=encoder_attention_with_output,
    mutates_args=["output", "workspace"],
    fake_impl=encoder_attention_with_output_fake,
)
