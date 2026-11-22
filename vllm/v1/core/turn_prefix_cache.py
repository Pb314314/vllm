# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId, KVCacheBlock
from vllm.v1.metrics.stats import TurnPrefixCacheStats
from vllm.v1.request import Request


@dataclass(slots=True)
class _Checkpoint:
    blocks: tuple[KVCacheBlock, ...]
    keys: tuple[BlockHashWithGroupId, ...]
    metadata_bytes: int


class TurnPrefixCache:
    """Bounded, non-owning shortcuts into the existing full-attention APC.

    Accessed only on the scheduler thread. Keeping a Python block reference
    does not acquire a GPU cache reference or change the block's eviction order.
    """

    def __init__(
        self, block_size: int, max_entries: int = 1024, max_bytes: int = 64 << 20
    ) -> None:
        self.block_size = block_size
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: OrderedDict[tuple[int, BlockHash], _Checkpoint] = OrderedDict()
        self.metadata_bytes = 0
        self.stats = TurnPrefixCacheStats()

    @staticmethod
    def supports(request: Request) -> bool:
        return (
            request.prefix_match_positions is not None
            and request.sampling_params is not None
            and not request.resumable
            and not request.mm_features
            and request.prompt_embeds is None
            and request.lora_request is None
            and not (request.sampling_params.extra_args or {}).get("session_cache")
        )

    def lookup(self, request: Request, max_length: int) -> list[KVCacheBlock] | None:
        if not self.supports(request) or not request.prefix_match_positions:
            return None
        self.stats.queries += 1
        # The sampled final token may not have KV, and retokenization can
        # change a turn's tail. Probe at most two neighboring full blocks.
        position = request.prefix_match_positions[-1]
        end = min(position, max_length) // self.block_size
        for count in (end, end - 1):
            if count <= 0 or count > len(request.block_hashes):
                continue
            key = (count, request.block_hashes[count - 1])
            entry = self.entries.get(key)
            if entry is None:
                continue
            # Keep the original hash objects alive: identity changes whenever
            # a block is evicted/re-keyed, including reuse for identical tokens.
            if any(b.block_hash is not k for b, k in zip(entry.blocks, entry.keys)):
                self.metadata_bytes -= self.entries.pop(key).metadata_bytes
                self.stats.invalidations += 1
                continue
            self.entries.move_to_end(key)
            self.stats.hits += 1
            self.stats.skipped_block_lookups += count
            return list(entry.blocks)
        return None

    def register(
        self, request: Request, blocks: Sequence[KVCacheBlock], num_committed: int
    ) -> None:
        if not self.supports(request):
            return
        count = min(num_committed // self.block_size, len(request.block_hashes))
        if count <= 0 or count > len(blocks):
            return
        key = (count, request.block_hashes[count - 1])
        if key in self.entries:
            old = self.entries.pop(key)
            self.metadata_bytes -= old.metadata_bytes
        selected = tuple(blocks[:count])
        keys = tuple(b.block_hash for b in selected)
        if any(k is None for k in keys):
            return
        # Hashes have a fixed width within this engine. Count retained objects
        # conservatively, even when checkpoints share them.
        size = (
            sys.getsizeof(selected)
            + sys.getsizeof(keys)
            + len(keys) * sys.getsizeof(keys[0])
            + sys.getsizeof(key)
            + sys.getsizeof(key[1])
            + 256
        )
        if size > self.max_bytes:
            return
        while self.entries and (
            len(self.entries) >= self.max_entries
            or self.metadata_bytes + size > self.max_bytes
        ):
            _, old = self.entries.popitem(last=False)
            self.metadata_bytes -= old.metadata_bytes
        if self.max_entries <= 0:
            return
        # None was excluded above; keep the original bytes objects, not copies.
        self.entries[key] = _Checkpoint(
            selected, cast(tuple[BlockHashWithGroupId, ...], keys), size
        )
        self.metadata_bytes += size

    def clear(self) -> None:
        self.entries.clear()
        self.metadata_bytes = 0

    def take_stats(self) -> TurnPrefixCacheStats:
        stats = self.stats
        stats.entries = len(self.entries)
        stats.metadata_bytes = self.metadata_bytes
        self.stats = TurnPrefixCacheStats()
        return stats
