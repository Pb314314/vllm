# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    TargetDependentARSpeculator,
)

logger = init_logger(__name__)

_FEED_FORWARD_NAMES = ("mlp", "ffn")


class _ZeroFeedForward(nn.Module):
    """Stands in for a decoder block's feed-forward (MLP or MoE) whose output
    is not needed: returns zeros shaped like its input."""

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.zeros_like(hidden_states)


def drop_mtp_feed_forwards(model: nn.Module, target_model: nn.Module) -> int | None:
    """Replace the feed-forward of every MTP layer's decoder block with zeros
    and free its weights, except any shared with the target. Returns the bytes
    freed, or None (nothing changed) unless every MTP layer opts in through
    `context_kv_only_block()`.

    On a prefill instance the MTP layer only has to write its attention K/V.
    In the block an MTP layer returns there, the feed-forward runs after the
    block's only attention and only adds to the residual, so it never affects
    a K/V. Every rank replaces it, so the collectives inside it (expert
    dispatch, all-reduce) are skipped on all ranks alike.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    layers = list(layers.children()) if isinstance(layers, nn.Module) else []
    if not layers or not all(hasattr(m, "context_kv_only_block") for m in layers):
        return None
    shared = {id(p) for p in target_model.parameters()}
    freed_bytes = 0
    for layer in layers:
        block = layer.context_kv_only_block()
        names = [name for name in _FEED_FORWARD_NAMES if hasattr(block, name)]
        if len(names) != 1:
            raise ValueError(
                f"Expected one feed-forward ({' or '.join(_FEED_FORWARD_NAMES)}) "
                f"in MTP decoder block {type(block).__name__}, found {names}."
            )
        feed_forward = getattr(block, names[0])
        for param in feed_forward.parameters():
            if id(param) not in shared:
                freed_bytes += param.nbytes
                param.data = param.data.new_empty(0)
        setattr(block, names[0], _ZeroFeedForward())
    return freed_bytes


class MTPSpeculator(TargetDependentARSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_config
            if spec_config is not None
            else None
        )
        # Detect index_share_for_mtp_iteration. When True, the proposer
        # toggles skip_topk so step 0 computes MTP's own indices and
        # steps 1+ reuse them.
        self.share_mtp_topk_indices = (
            self.vllm_config.parallel_config.prefill_context_parallel_size == 1
            and getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        return draft_model

    def free_unused_weights(self, target_model: nn.Module) -> bool:
        """With prefill_context_kv_only, replace the feed-forward of every MTP
        layer with zeros and free its weights; see `drop_mtp_feed_forwards`."""
        if not self.context_kv_only:
            return False
        freed_bytes = drop_mtp_feed_forwards(self.model, target_model)
        if freed_bytes is None:
            return False
        logger.info(
            "MTP drafter only stores its context K/V (prefill_context_kv_only); "
            "freed %.2f GiB of its feed-forward weights.",
            freed_bytes / (1 << 30),
        )
        return True

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Step 0 computes its own top-k. Unconditional, so a step that died
        # midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 (prefill) wrote topk indices for every query token in the
        # multi-token batch. Compact them down to each request's last token so
        # steps 1+ can reuse them from the shared buffer.
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        # Switch to reuse mode so draft steps 1+ skip the indexer op and read
        # the indices that step 0 wrote into the shared buffer.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
