# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""With `prefill_context_kv_only`, a DFlash-based drafter frees the weights
outside its draft model's `context_kv_modules()`. The context K/V it stores
must not change."""

import pytest
import torch
from torch import nn

from vllm.config import KVTransferConfig, set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode import init_speculator

NUM_CONTEXT_TOKENS = 37


def _load_speculator(target: str, draft: str, method: str, k: int):
    vllm_config = EngineArgs(
        model=target,
        speculative_config={
            "method": method,
            "model": draft,
            "num_speculative_tokens": k,
            "prefill_context_kv_only": True,
        },
        kv_transfer_config=KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={"shared_storage_path": "local_storage"},
        ),
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=1024,
        max_num_seqs=8,
        enforce_eager=True,
    ).create_engine_config()
    model_config = vllm_config.model_config
    vocab_size = model_config.get_vocab_size()
    hidden_size = model_config.get_hidden_size()
    device = torch.device("cuda")
    with set_current_vllm_config(vllm_config):
        speculator = init_speculator(vllm_config, device, req_states=None)
        # Drafts share only the embedding and the LM head with the target.
        target_model = nn.Module()
        target_model.model = nn.Module()
        with device:
            target_model.model.embed_tokens = VocabParallelEmbedding(
                vocab_size, hidden_size, params_dtype=model_config.dtype
            )
            target_model.lm_head = ParallelLMHead(
                vocab_size, hidden_size, params_dtype=model_config.dtype
            )
        speculator.load_model(target_model)
    return speculator, target_model


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.parametrize(
    ("target", "draft", "method", "k"),
    [
        # DFlashQwen3Model's list, which DSpark, DFlash2 and XPress return too.
        ("Qwen/Qwen3-8B", "z-lab/Qwen3-8B-DFlash-b16", "dflash", 16),
        # A draft model with its own context K/V path and list.
        ("google/gemma-4-12B-it", "deepseek-ai/dspark_gemma4_12b_block7", "dspark", 7),
    ],
)
def test_freeing_keeps_context_kv(dist_init, monkeypatch, target, draft, method, k):
    speculator, target_model = _load_speculator(target, draft, method, k)
    model = speculator.model
    attn_layers = model.model._attn_layers

    def context_kv() -> list[torch.Tensor]:
        written: list[torch.Tensor] = []
        for attn in attn_layers:
            monkeypatch.setattr(
                attn.impl,
                "do_kv_cache_update",
                lambda layer, key, value, kv_cache, slots: written.extend(
                    (key.clone(), value.clone())
                ),
            )
        generator = torch.Generator("cuda").manual_seed(0)
        target_states = torch.randn(
            NUM_CONTEXT_TOKENS,
            model.model.fc.input_size,
            generator=generator,
            device="cuda",
            dtype=speculator.dtype,
        )
        positions = torch.arange(NUM_CONTEXT_TOKENS, device="cuda")
        model.precompute_and_store_context_kv(
            model.combine_hidden_states(target_states), positions, positions
        )
        return written

    with torch.inference_mode():
        before = context_kv()
        speculator.free_unused_weights(target_model)
        assert any(param.numel() == 0 for param in model.parameters())
        after = context_kv()

    assert len(before) == len(after) == 2 * len(attn_layers)
    assert all(torch.equal(b, a) for b, a in zip(before, after))


class _ToyMTPLayer(nn.Module):
    def __init__(self, feed_forward_name: str, opt_in: bool):
        super().__init__()
        self.mtp_block = nn.Module()
        self.mtp_block.self_attn = nn.Linear(8, 8)
        setattr(self.mtp_block, feed_forward_name, nn.Linear(8, 8))
        if opt_in:
            self.context_kv_only_block = lambda: self.mtp_block


def _toy_mtp(feed_forward_name: str = "mlp", opt_in: bool = True) -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleDict({"61": _ToyMTPLayer(feed_forward_name, opt_in)})
    return model


@pytest.mark.parametrize("feed_forward_name", ["mlp", "ffn"])
def test_drop_mtp_feed_forwards(feed_forward_name):
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import drop_mtp_feed_forwards

    model = _toy_mtp(feed_forward_name)
    block = model.model.layers["61"].mtp_block
    feed_forward = getattr(block, feed_forward_name)
    # A weight shared with the target (e.g. a tied projection) must stay.
    target = nn.Module()
    target.shared = feed_forward.bias
    attn_weight = block.self_attn.weight

    freed = drop_mtp_feed_forwards(model, target)

    assert freed == 8 * 8 * 4  # the weight; the shared bias stays
    assert feed_forward.weight.numel() == 0
    assert feed_forward.bias.numel() == 8
    assert block.self_attn.weight is attn_weight and attn_weight.numel() == 64
    hidden_states = torch.randn(3, 8)
    out = getattr(block, feed_forward_name)(hidden_states, "extra", key=None)
    assert torch.equal(out, torch.zeros(3, 8))


def test_drop_mtp_feed_forwards_needs_opt_in():
    from vllm.v1.worker.gpu.spec_decode.mtp.speculator import drop_mtp_feed_forwards

    # MTP layers that don't declare context_kv_only_block() keep every weight,
    # including those of the Transformers backend (an nn.ModuleList of layers).
    undeclared = _toy_mtp(opt_in=False)
    module_list = nn.Module()
    module_list.model = nn.Module()
    module_list.model.layers = nn.ModuleList([_ToyMTPLayer("mlp", opt_in=False)])
    for model in (undeclared, module_list):
        params = [p.numel() for p in model.parameters()]
        assert drop_mtp_feed_forwards(model, nn.Module()) is None
        assert [p.numel() for p in model.parameters()] == params

    both = _toy_mtp("mlp")
    both.model.layers["61"].mtp_block.ffn = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="Expected one feed-forward"):
        drop_mtp_feed_forwards(both, nn.Module())
