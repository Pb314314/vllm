# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import struct
import sys
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from vllm.utils.gc_utils import gc_untrack
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId, KVCacheBlock
from vllm.v1.metrics.stats import TurnPrefixCacheStats
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.turn_prefix_hash_cache import TurnPrefixHashCache

CheckpointKey = tuple[int, BlockHash, int]


@dataclass(slots=True)
class _Checkpoint:
    blocks: tuple[KVCacheBlock, ...]
    keys: tuple[BlockHashWithGroupId, ...]
    metadata_bytes: int
    generation: int
    groups: tuple[tuple[KVCacheBlock, ...], ...] | None = None


@dataclass(slots=True)
class _PendingCheckpoint:
    blocks: tuple[KVCacheBlock, ...]
    keys: tuple[BlockHashWithGroupId, ...]
    generation: int
    metadata_bytes: int
    canonical_prefix_length: int = 0


@dataclass(slots=True)
class TurnPrefixLookup:
    blocks: dict[int, tuple[KVCacheBlock, ...]]
    skipped_block_lookups: int = 0


class TurnPrefixCache:
    """Bounded, non-owning shortcuts into the existing full-attention APC.

    Accessed only on the scheduler thread. Keeping a Python block reference
    does not acquire a GPU cache reference or change the block's eviction order.
    """

    def __init__(
        self,
        block_size: int,
        max_entries: int = 1024,
        max_bytes: int = 64 << 20,
        *,
        block_pool: "BlockPool | None" = None,
        defer_publication: bool = False,
    ) -> None:
        self.block_size = block_size
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries: OrderedDict[CheckpointKey, _Checkpoint] = OrderedDict()
        self.metadata_bytes = 0
        self.stats = TurnPrefixCacheStats()
        self.block_pool = block_pool
        self.defer_publication = defer_publication and block_pool is not None
        self.pending: OrderedDict[CheckpointKey, _PendingCheckpoint] = OrderedDict()
        self.pending_bytes = 0
        self.max_pending_bytes = min(max_bytes, 8 << 20)
        self.pending_dropped = 0
        self.pending_published = 0
        self.hash_cache: TurnPrefixHashCache | None = None

    def _valid(self, entry: _Checkpoint) -> bool:
        generation = self.block_pool.cache_generation if self.block_pool else -1
        if self.block_pool is not None and entry.generation == generation:
            return True
        # Freeing a request makes its tail the first candidate for eviction.
        if entry.blocks[-1].block_hash is not entry.keys[-1]:
            return False
        if any(b.block_hash is not k for b, k in zip(entry.blocks, entry.keys)):
            return False
        entry.generation = generation
        return True

    @staticmethod
    def supports(request: Request) -> bool:
        return (
            request.prefix_match_positions is not None
            and request.sampling_params is not None
            and not request.resumable
            and not request.mm_features
            and request.prompt_embeds is None
            and request.lora_request is None
        )

    def lookup(
        self,
        request: Request,
        max_length: int,
        *,
        group_id: int = 0,
        block_size: int | None = None,
    ) -> tuple[KVCacheBlock, ...] | None:
        if not self.supports(request) or not request.prefix_match_positions:
            return None
        self.stats.queries += 1
        # The sampled final token may not have KV, and retokenization can
        # change a turn's tail. Probe at most two neighboring full blocks.
        block_size = block_size or self.block_size
        positions = request.prefix_match_positions[-2:]
        candidates = dict.fromkeys(
            count
            for position in reversed(positions)
            for count in (
                min(position, max_length) // block_size,
                min(position, max_length) // block_size - 1,
            )
        )
        for count in candidates:
            hash_count = count * block_size // self.block_size
            if count <= 0 or hash_count > len(request.block_hashes):
                continue
            key = (hash_count, request.block_hashes[hash_count - 1], group_id)
            entry = self.entries.get(key)
            if entry is None:
                continue
            # Keep the original hash objects alive: identity changes whenever
            # a block is evicted/re-keyed, including reuse for identical tokens.
            if not self._valid(entry):
                self.metadata_bytes -= self.entries.pop(key).metadata_bytes
                self.stats.invalidations += 1
                continue
            self.entries.move_to_end(key)
            self.stats.hits += 1
            return entry.blocks
        return None

    def register(
        self,
        request: Request,
        blocks: Sequence[KVCacheBlock],
        num_committed: int,
        *,
        group_id: int = 0,
        block_size: int | None = None,
    ) -> None:
        if not self.supports(request):
            return
        block_size = block_size or self.block_size
        count = (
            min(num_committed, len(request.block_hashes) * self.block_size)
            // block_size
        )
        if count <= 0 or count > len(blocks):
            return
        hash_count = count * block_size // self.block_size
        key = (hash_count, request.block_hashes[hash_count - 1], group_id)
        existing = self.entries.get(key)
        if existing is not None and self._valid(existing):
            self.entries.move_to_end(key)
            return
        if self.max_entries <= 0 or count * 128 + 512 > self.max_bytes:
            return
        generation = self.block_pool.cache_generation if self.block_pool else -1
        if self.defer_publication:
            pending = self.pending.get(key)
            if pending is not None:
                if pending.generation == generation:
                    return
                self.pending_bytes -= self.pending.pop(key).metadata_bytes
            size = count * 128 + 512
            if (
                len(self.pending) >= min(32, self.max_entries)
                or self.pending_bytes + size > self.max_pending_bytes
            ):
                self.pending_dropped += 1
                return
        # Reuse a validated canonical prefix; inspect only the new blocks.
        prefix = self._find_prefix(request, count, group_id, block_size)
        prefix_length = len(prefix.blocks) if prefix is not None else 0
        selected = tuple(blocks[prefix_length:count])
        keys = tuple(b.block_hash for b in selected)
        if any(k is None for k in keys):
            return
        if prefix is not None:
            selected = prefix.blocks + selected
            keys = prefix.keys + keys
        valid_keys = cast(tuple[BlockHashWithGroupId, ...], keys)
        if self.defer_publication:
            self.pending[key] = _PendingCheckpoint(
                gc_untrack(selected),
                gc_untrack(valid_keys),
                generation,
                size,
                prefix_length,
            )
            self.pending_bytes += size
            return
        self._publish(key, selected, valid_keys, prefix_length)

    def _find_prefix(
        self, request: Request, count: int, group_id: int, block_size: int
    ) -> _Checkpoint | None:
        positions = (request.num_prompt_tokens,) + tuple(
            reversed((request.prefix_match_positions or ())[-2:])
        )
        for position in positions:
            for length in (position // block_size, position // block_size - 1):
                if not 0 < length < count:
                    continue
                hash_count = length * block_size // self.block_size
                key = (hash_count, request.block_hashes[hash_count - 1], group_id)
                entry = self.entries.get(key)
                if entry is not None and self._valid(entry):
                    return entry
        return None

    def _publish(
        self,
        key: CheckpointKey,
        selected: tuple[KVCacheBlock, ...],
        keys: tuple[BlockHashWithGroupId, ...],
        canonical_prefix_length: int = 0,
    ) -> None:
        if self.block_pool is not None:
            block_map = self.block_pool.cached_block_hash_to_block
            if block_map.has_duplicates:
                suffix = tuple(
                    (block_map.get_one_block(block.block_hash) or block)
                    if block.block_hash is not None
                    else block
                    for block in selected[canonical_prefix_length:]
                )
                selected = selected[:canonical_prefix_length] + suffix
                keys = keys[:canonical_prefix_length] + cast(
                    tuple[BlockHashWithGroupId, ...],
                    tuple(b.block_hash for b in suffix),
                )
        self._store(key, selected, keys)

    def _store(
        self,
        key: CheckpointKey,
        selected: tuple[KVCacheBlock, ...],
        keys: tuple[BlockHashWithGroupId, ...],
        groups: tuple[tuple[KVCacheBlock, ...], ...] | None = None,
    ) -> None:
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
        if groups is not None:
            size += sys.getsizeof(groups) + sum(sys.getsizeof(g) for g in groups)
        if size > self.max_bytes:
            return
        if key in self.entries:
            self.metadata_bytes -= self.entries.pop(key).metadata_bytes
        while self.entries and (
            len(self.entries) >= self.max_entries
            or self.metadata_bytes + size > self.max_bytes
        ):
            _, old = self.entries.popitem(last=False)
            self.metadata_bytes -= old.metadata_bytes
        if self.max_entries <= 0:
            return
        # Blocks only weakly reference their pool and cannot refer to these
        # immutable tuples. Reference counting still releases all metadata.
        gc_untrack(selected)
        gc_untrack(keys)
        if groups is not None:
            for group in groups:
                gc_untrack(group)
            gc_untrack(groups)
        # None was excluded above; keep the original bytes objects, not copies.
        self.entries[key] = _Checkpoint(
            selected,
            cast(tuple[BlockHashWithGroupId, ...], keys),
            size,
            self.block_pool.cache_generation if self.block_pool else -1,
            groups,
        )
        self.metadata_bytes += size

    def snapshot_key(self, request: Request, position: int) -> CheckpointKey | None:
        """Identify an exact prefix, including the unaligned hash tail."""
        count, tail_size = divmod(position, self.block_size)
        if count <= 0 or count > len(request.block_hashes):
            return None
        tail = request.all_token_ids[position - tail_size : position]
        digest = hashlib.sha256(
            b"vllm-turn-snapshot\0"
            + position.to_bytes(8, "big")
            + request.block_hashes[count - 1]
            + struct.pack(f"!{tail_size}I", *tail)
        ).digest()
        return position, BlockHash(digest), -1

    def register_snapshot(
        self,
        key: CheckpointKey,
        groups: tuple[tuple[KVCacheBlock, ...], ...],
    ) -> None:
        """Index a completed joint state without pinning its GPU blocks."""
        blocks = tuple(b for group in groups for b in group if not b.is_null)
        if not blocks or any(b.block_hash is None for b in blocks):
            return
        keys = cast(
            tuple[BlockHashWithGroupId, ...], tuple(b.block_hash for b in blocks)
        )
        self._store(key, blocks, keys, groups)

    def lookup_snapshot(
        self, request: Request, max_length: int
    ) -> tuple[tuple[tuple[KVCacheBlock, ...], ...], int] | None:
        if not self.supports(request) or not request.prefix_match_positions:
            return None
        # The final sampled token may not have been forwarded. Actual token
        # identity is verified even when the renderer's boundary is approximate.
        positions = sorted(
            {
                p - delta
                for p in request.prefix_match_positions[-2:]
                for delta in (0, 1)
                if 0 < p - delta <= max_length
            },
            reverse=True,
        )
        for position in positions:
            key = self.snapshot_key(request, position)
            if key is None:
                continue
            entry = self.entries.get(key)
            if entry is None or entry.groups is None:
                continue
            if not self._valid(entry):
                self.metadata_bytes -= self.entries.pop(key).metadata_bytes
                self.stats.invalidations += 1
                continue
            self.entries.move_to_end(key)
            return entry.groups, position
        return None

    def publish_pending(
        self, limit: int = 1, *, should_stop: Callable[[], bool] | None = None
    ) -> int:
        """Publish bounded optional work without retaining GPU ownership."""
        processed = 0
        while self.pending and processed < limit:
            if should_stop is not None and should_stop():
                break
            key, pending = self.pending.popitem(last=False)
            self.pending_bytes -= pending.metadata_bytes
            processed += 1
            assert self.block_pool is not None
            if pending.generation != self.block_pool.cache_generation and any(
                block.block_hash is not key
                for block, key in zip(pending.blocks, pending.keys)
            ):
                self.pending_dropped += 1
                continue
            self._publish(
                key,
                pending.blocks,
                pending.keys,
                pending.canonical_prefix_length,
            )
            if key in self.entries:
                self.pending_published += 1
        return processed

    def clear(self) -> None:
        self.entries.clear()
        self.metadata_bytes = 0
        self.pending.clear()
        self.pending_bytes = 0

    def take_stats(self) -> TurnPrefixCacheStats:
        stats = self.stats
        stats.entries = len(self.entries)
        stats.metadata_bytes = self.metadata_bytes
        if self.hash_cache is not None:
            stats.hash_metadata_bytes = self.hash_cache.metadata_bytes
        self.stats = TurnPrefixCacheStats()
        return stats
