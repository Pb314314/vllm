# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.utils.gc_utils import untracked_list
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock, trim_null_head
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    HiddenStateCacheSpec,
    HiSparseHotSpec,
    HiSparseResidentSpec,
    KVCacheGroupRole,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    RSWASpec,
    SinkFullAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

if TYPE_CHECKING:
    from vllm.v1.hisparse.coordinator import HiSparseCoordinator


class SingleTypeKVCacheManager:
    """An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    @property
    def has_positionally_stable_blocks(self) -> bool:
        """Whether positional offload scans can follow this block table.

        True means already-scanned indices will not be reused for a different
        token range. Managers may still null blocks that left their retention
        window, as long as the cursor's positional history remains valid.
        """
        return True

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        needs_kv_cache_zeroing: bool = False,
        max_admission_blocks_per_request: int | None = None,
    ) -> None:
        """Initializes the SingleTypeKVCacheManager.

        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
            scheduler_block_size: The scheduling granularity (LCM of all group
                block sizes); a multiple of this manager's `block_size`.
            dcp_world_size: Decode context parallel world size.
            pcp_world_size: Prefill context parallel world size.
            needs_kv_cache_zeroing: Whether worker-side KV cache zeroing needs
                newly allocated block IDs from this manager.
            max_admission_blocks_per_request: Recycling-aware per-request
                block cap used by `get_num_blocks_to_allocate`. Only set for
                spec types that recycle blocks across chunks (SWA,
                chunked-local); `None` (the default) means no cap, which is
                correct for full-attention-style specs that hold every
                block until the request finishes.

        """
        self.scheduler_block_size = scheduler_block_size
        # The block size for this manager; used for actual block allocation.
        self.dcp_world_size = dcp_world_size if kv_cache_spec.dcp_sharded else 1
        self.pcp_world_size = pcp_world_size if kv_cache_spec.dcp_sharded else 1
        self.block_size = kv_cache_spec.block_size * self.dcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.max_admission_blocks_per_request = max_admission_blocks_per_request
        # Record newly allocated block ids only when worker-side zeroing will
        # consume them and this manager holds a spec type that gets zeroed.
        self._record_new_block_ids = (
            needs_kv_cache_zeroing
            and isinstance(kv_cache_spec, AttentionSpec)
            and not isinstance(kv_cache_spec, CircularBufferSpec)
        )
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished. Blocks hold no strong reference back to these lists or
        # their owner, so the lists cannot be part of a cycle and are hidden from
        # the cyclic GC, which would otherwise visit every block of every request.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(
            untracked_list
        )

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

        # Partial-hit copy-on-write bookkeeping for resumes that end inside a
        # shared block.
        self._partial_hit_reqs: dict[str, tuple[int, KVCacheBlock]] = {}
        self._pending_cow_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def _has_partial_local_hit(
        self,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
    ) -> bool:
        # The local prefix-cache hit ends inside one of this manager's
        # blocks: the shared tail block needs CoW.
        return (
            len(new_computed_blocks) > 0
            and num_local_computed_tokens % self.block_size != 0
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, clamp by `num_required_blocks` by
                `max_admission_blocks_per_request`for recycling-aware specs
                (SWA, chunked-local).

        Returns:
            The number of blocks to allocate.

        """
        num_required_blocks = cdiv(num_tokens, self.block_size)
        if apply_admission_cap and self.max_admission_blocks_per_request is not None:
            # Recycling-aware specs (SWA, chunked-local) cap the per-request
            # reservation here so admission matches the startup pool sizer
            # (`SlidingWindowSpec.max_admission_blocks_per_request` / its
            # chunked-local counterpart). `remove_skipped_blocks` runs from
            # `allocate_slots` before each chunk's `get_num_blocks_to_allocate`,
            # so per-request peak real-held blocks <= this cap, which keeps
            # `sum(reservations) <= pool` <=> `sum(peak_real_held) <= pool`.
            # Drift between the two would re-introduce the deadlock from
            # issue #39734 or, worse, mid-prefill OOM.
            num_required_blocks = min(
                num_required_blocks, self.max_admission_blocks_per_request
            )
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            return max(num_required_blocks - num_req_blocks, 0)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )

        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Reserve the extra block that allocate_new_blocks pulls for the
            # partial-hit CoW redirect.
            num_new_blocks += 1
        return num_new_blocks + num_evictable_blocks

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """Add the locally cached (prefix-hit) blocks to the request:
        1. Touch the computed blocks (paired with adding them to `req_blocks`)
           so their ref_cnt exactly tracks the referencing requests.
        1.5. (Optional) For sliding window, skipped blocks are padded with nulls.
        2. Add the remaining computed blocks.

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.

        """
        # The coordinator only calls this for first-time allocations (running
        # requests are short-circuited there), so the request has no blocks yet.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]

        # Take a reference on reused blocks so their owner cannot free them.
        self.block_pool.touch(new_computed_blocks)

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        # Marks the request as admitted (see `get_num_blocks_to_allocate`).
        self.num_cached_block[request_id] = len(req_blocks)
        if self._has_partial_local_hit(new_computed_blocks, num_local_computed_tokens):
            # Record the partial tail for the CoW redirect in allocate_new_blocks.
            block_idx = num_local_computed_tokens // self.block_size
            self._partial_hit_reqs[request_id] = (block_idx, new_computed_blocks[-1])

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        """Allocate new blocks for external (KV-connector) computed tokens.

        Must run only after every group's local blocks have been touched via
        `add_local_computed_blocks`, so this group's `get_new_blocks` cannot
        evict another group's cache-hit blocks (issue #33775).

        Args:
            request_id: The request ID.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
            record_for_zeroing: Whether the new blocks need zeroing. False when
                the load writes them after this step, which zeroing would race.

        """
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        if num_skipped_tokens > 0:
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens <= 0:
            return

        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = max(
            0, cdiv(num_total_computed_tokens, self.block_size) - len(req_blocks)
        )
        allocated_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids and record_for_zeroing:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.

        Returns:
            The new allocated blocks.

        """
        cow_blocks: list[KVCacheBlock] = []
        if request_id in self._partial_hit_reqs:
            # Partial hit: redirect the shared tail to a private CoW block.
            # Replacing in place keeps the length-based allocation below
            # correct; the extra block was reserved by
            # get_num_blocks_to_allocate.
            block_idx, source_block = self._partial_hit_reqs.pop(request_id)
            cow_block = self.block_pool.get_new_blocks(1)[0]
            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            cow_blocks.append(cow_block)

        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return cow_blocks
        else:
            new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return cow_blocks + new_blocks

    @property
    def records_new_block_ids(self) -> bool:
        """Whether this manager's new blocks are zeroed by the worker."""
        return self._record_new_block_ids

    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def take_pending_cow_copies(
        self,
    ) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain pending CoW source and destination block pairs."""
        pending_copies = self._pending_cow_copies
        self._pending_cow_copies = []
        return pending_copies

    def _apply_cow(
        self,
        request_id: str,
        block_idx: int,
        source_block: KVCacheBlock,
        cow_block: KVCacheBlock,
    ) -> None:
        """Redirect a partial prefix-cache hit to a private CoW block.

        Both copy endpoints stay retained until the copy has run on the worker,
        so a same-step free cannot recycle them: `source_block` keeps its
        hit-ref, `cow_block` takes an extra ref beyond the one handed to the
        request.
        """
        req_blocks = self.req_to_blocks[request_id]
        assert block_idx < len(req_blocks)
        assert req_blocks[block_idx] is source_block
        assert not source_block.is_null and source_block.ref_cnt > 0
        req_blocks[block_idx] = cow_block
        self._pending_cow_copies.append((source_block, cow_block))
        cow_block.ref_cnt += 1

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping and return its blocks without yet
        returning them to the block pool. The caller is responsible for
        eventually passing the returned blocks to `block_pool.free_blocks`,
        freeing them in reverse order (so that tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order, without the leading
            null blocks (e.g. a sliding-window table's skipped prefix).

        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self.num_cached_block.pop(request_id, None)
        self._partial_hit_reqs.pop(request_id, None)
        return trim_null_head(req_blocks)

    def free(self, request_id: str) -> None:
        """Free the blocks for the request.

        Args:
            request_id: The request ID.

        """
        # Free blocks in reverse order so that the tail blocks are freed first.
        self.block_pool.free_blocks(reversed(self.pop_blocks_for_free(request_id)))

    def _remove_blocks_in_range(
        self,
        request_id: str,
        first_block: int,
        last_block: int,
    ) -> None:
        """Free blocks in `[first_block, last_block)` and replace with null_block.

        Iterates backward so newly-evictable tail blocks are reached even after
        earlier blocks in the range were nulled in a prior call.
        """
        if request_id not in self.req_to_blocks:
            return
        if first_block >= last_block:
            return
        blocks = self.req_to_blocks[request_id]
        last_block = min(last_block, len(blocks))

        freed: list[KVCacheBlock] = []
        for i in range(last_block - 1, first_block - 1, -1):
            if blocks[i] == self._null_block:
                break
            freed.append(blocks[i])
            blocks[i] = self._null_block
        if freed:
            self.block_pool.free_blocks(freed)

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove and free blocks no longer needed for attention computation.

        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length for attention types (e.g.
                R-SWA) that evict a middle gap rather than a head prefix. Ignored
                by the default implementation.

        """
        del num_prompt_tokens
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(processed_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        self._remove_blocks_in_range(request_id, 0, num_skipped_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.

        """
        # The default behavior is to not skip any tokens.
        return 0

    def new_step_starts(self) -> None:
        return None


class FullAttentionManager(SingleTypeKVCacheManager):
    pass


class RSWAManager(FullAttentionManager):
    """KV cache manager for Reference Sliding Window Attention (R-SWA).

    When `num_prompt_tokens` is supplied to `remove_skipped_blocks`, frees
    gap blocks between the prefill tail and the current decode window.  This
    bounds per-request KV memory at O(prefix_len + rswa_window) instead of
    growing linearly with decode length.
    """

    def __init__(self, kv_cache_spec: RSWASpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.rswa_window: int = kv_cache_spec.rswa_window

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Free gap blocks that are no longer needed for attention.

        Gap = blocks entirely within
            [ceil(prefix_len / block_size) * block_size,
             max(prefix_len, processed_computed_tokens - rswa_window))

        Freed blocks are replaced with null_block in req_to_blocks so the
        block_table passed to FA4 is valid (null_block KV is all-zero;
        rswa_mask_mod marks gap positions as non-visible so FA4 skips them).
        """
        if num_prompt_tokens is None:
            super().remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )
            return

        bs = self.block_size
        # First block fully after the prefill boundary.
        first_gap_block = cdiv(num_prompt_tokens, bs)
        # Decode window start position; blocks before this are evictable.
        window_start = max(
            num_prompt_tokens, processed_computed_tokens - self.rswa_window
        )
        last_gap_block = window_start // bs  # exclusive upper bound
        self._remove_blocks_in_range(request_id, first_gap_block, last_gap_block)


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window
        # Extra trailing tokens to retain below the window (never attended) so a
        # multi-module MTP store-side lag can still reconstruct the window from
        # cached blocks.
        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        The trailing edge of the window is extended by `extra_retained_tokens`
        so that those extra trailing tokens' blocks are retained (but not
        attended). This is needed for multi-module spec decoding which can
        re-prefill the last num_spec_prefill_tokens - 1 tokens from the end
        of the sequence, and thus needs to delay freeing/caching of blocks.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.

        """
        return max(
            0,
            num_computed_tokens - self.sliding_window + 1 - self.extra_retained_tokens,
        )


class CircularBufferManager(FullAttentionManager):
    """Claims the ring's single block per request; prefix caching disabled."""

    def _claim_ring_block(
        self, request_id: str, record_for_zeroing: bool = True
    ) -> list[KVCacheBlock]:
        req_blocks = self.req_to_blocks[request_id]
        if req_blocks:
            return []
        new_blocks = self.block_pool.get_new_blocks(1)
        req_blocks.extend(new_blocks)
        if self._record_new_block_ids and record_for_zeroing:
            self.new_block_ids.extend(block.block_id for block in new_blocks)
        return new_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        return 0 if self.req_to_blocks.get(request_id) else 1

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        return self._claim_ring_block(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        self._claim_ring_block(request_id, record_for_zeroing)

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        return

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        return

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.

        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens


class MambaManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        # Allocate extra `num_speculative_blocks` blocks for
        # speculative decoding (MTP/EAGLE) with linear attention.
        num_tokens += self.kv_cache_spec.block_size * self.num_speculative_blocks
        return super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
            apply_admission_cap=apply_admission_cap,
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        num_tokens += self.block_size * self.num_speculative_blocks
        return super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model
        )


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        # Cross-attention does not use prefix caching / external KV loads.
        return


class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        kv_cache_group_id: int,
        scheduler_block_size: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec=kv_cache_spec,
            block_pool=block_pool,
            kv_cache_group_id=kv_cache_group_id,
            scheduler_block_size=scheduler_block_size,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0


class HiSparseSourceManager(FullAttentionManager):
    """Host-tier manager with a private pool.

    Every page gets a host block: allocation fails when the host pool cannot
    back it, so each GPU-resident page can be written back and later reclaimed.
    """

    coordinator: "HiSparseCoordinator | None" = None

    @property
    def records_new_block_ids(self) -> bool:
        return False

    def take_new_block_ids(self) -> list[int]:
        self.new_block_ids = []
        return []

    def bind_host_pool(self, num_blocks: int) -> None:
        """Replace the device pool this group was built with by a host one."""
        device_pool = self.block_pool
        self.block_pool = BlockPool(
            num_gpu_blocks=num_blocks,
            metrics_collector=(
                KVCacheMetricsCollector(device_pool.metrics_collector.sample_rate)
                if device_pool.metrics_collector is not None
                else None
            ),
            medium=MEDIUM_CPU,
        )
        self._null_block = self.block_pool.null_block

    def take_pending_cow_copies(self) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Host copies never reach the worker's generic block-copy path."""
        return []

    def take_host_cow_copies(self) -> list[tuple[KVCacheBlock, KVCacheBlock]]:
        """Drain host copies for the coordinator to hand to the connector."""
        copies = self._pending_cow_copies
        self._pending_cow_copies = []
        return copies

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert self.coordinator is not None
        available = self.block_pool.get_num_free_blocks()
        if apply_admission_cap and total_computed_tokens > num_local_computed_tokens:
            num_tokens += self.coordinator.import_step_tokens
            available -= self.coordinator.reserved_import_pages
            self.coordinator.size_import(request_id, cdiv(num_tokens, self.block_size))
        required = super().get_num_blocks_to_allocate(
            request_id,
            num_tokens,
            new_computed_blocks,
            total_computed_tokens,
            num_local_computed_tokens,
            num_tokens_main_model,
        )
        # Host blocks come from a private pool, so they never count against
        # the GPU pool. Use the same admission sentinel as Mamba instead.
        if required > available:
            assert self.coordinator.gpu_pool is not None
            return self.coordinator.gpu_pool.num_gpu_blocks + 1
        return 0

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        if num_external_computed_tokens <= 0:
            return
        # The connector writes these pages; only a successful receive makes
        # them readable, not advancing the request's computed-token count.
        assert self.coordinator is not None
        self.coordinator.record_pending_host_import(
            request_id, num_local_computed_tokens + num_external_computed_tokens
        )
        super().allocate_external_computed_blocks(
            request_id,
            num_local_computed_tokens,
            num_external_computed_tokens,
            record_for_zeroing,
        )
        self.coordinator.update_import_reservation(
            request_id, len(self.req_to_blocks[request_id])
        )

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        new_blocks = super().allocate_new_blocks(
            request_id, num_tokens, num_tokens_main_model
        )
        if new_blocks:
            assert self.coordinator is not None
            self.coordinator.update_import_reservation(
                request_id, len(self.req_to_blocks[request_id])
            )
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        assert self.coordinator is not None
        self.coordinator.free(request_id)
        return super().pop_blocks_for_free(request_id)


class _HiSparseAuxiliaryManager(SingleTypeKVCacheManager):
    """Base for ephemeral groups backed by a HiSparse host source."""

    coordinator: "HiSparseCoordinator | None" = None


class HiSparseHotManager(_HiSparseAuxiliaryManager):
    """Allocate a hot region only after a request acquires CPU-only history."""

    def __init__(self, kv_cache_spec: HiSparseHotSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.blocks_per_request = kv_cache_spec.blocks_per_request
        self.hot_required: set[str] = set()

    def require_hot(self, request_id: str) -> None:
        self.hot_required.add(request_id)

    def has_hot(self, request_id: str) -> bool:
        return len(self.req_to_blocks.get(request_id, ())) == self.blocks_per_request

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        assert not new_computed_blocks
        # A hot region is needed to read host-backed history: an external
        # import, a new request resuming a host prefix, or one already asked
        # to transition. Running requests keep their earlier answer.
        host_import = total_computed_tokens > num_local_computed_tokens
        resumes_host_prefix = (
            num_local_computed_tokens > 0 and request_id not in self.num_cached_block
        )
        if host_import or resumes_host_prefix or request_id in self.hot_required:
            return self.get_num_required_blocks(request_id)
        return 0

    def get_num_required_blocks(self, request_id: str) -> int:
        return max(
            self.blocks_per_request - len(self.req_to_blocks.get(request_id, ())),
            0,
        )

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        self.num_cached_block[request_id] = 0
        assert self.coordinator is not None
        if num_local_computed_tokens > 0 or not self.coordinator.resident_managers:
            self.require_hot(request_id)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        self.require_hot(request_id)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        # Cold admissions can bypass add_local_computed_blocks.
        self.num_cached_block[request_id] = 0
        if request_id not in self.hot_required:
            return []
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = self.blocks_per_request - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        self.hot_required.discard(request_id)
        return super().pop_blocks_for_free(request_id)


class HiSparseResidentManager(_HiSparseAuxiliaryManager):
    """Track GPU-resident pages for otherwise host-backed KV."""

    max_admission_blocks_per_request: int

    def __init__(self, kv_cache_spec: HiSparseResidentSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.residency_changes: dict[str, dict[int, KVCacheBlock]] = {}

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        del num_tokens_main_model
        assert not new_computed_blocks
        if total_computed_tokens > num_local_computed_tokens:
            # Keep the last imported page writable, including the last-token
            # replay on a full prompt hit. Only earlier pages stay host-only.
            imported_pages = cdiv(total_computed_tokens, self.block_size)
            existing = len(self.req_to_blocks.get(request_id, ()))
            needs_tail = self.get_resident_page(request_id, imported_pages - 1) is None
            return int(needs_tail) + max(
                cdiv(num_tokens, self.block_size) - max(imported_pages, existing), 0
            )
        existing = len(self.req_to_blocks.get(request_id, ()))
        host_pages = cdiv(num_local_computed_tokens, self.block_size)
        required = cdiv(num_tokens, self.block_size)
        if apply_admission_cap:
            required = min(required, self.max_admission_blocks_per_request)
        return max(required - max(existing, host_pages), 0)

    def allocate_external_computed_blocks(
        self,
        request_id: str,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        record_for_zeroing: bool = True,
    ) -> None:
        """Reserve a writable final page; earlier imported pages stay on host."""
        assert num_external_computed_tokens > 0
        num_tokens = num_local_computed_tokens + num_external_computed_tokens
        blocks = self.req_to_blocks[request_id]
        tail_page = (num_tokens - 1) // self.block_size
        if len(blocks) <= tail_page:
            blocks.extend([self._null_block] * (tail_page + 1 - len(blocks)))
        if self.get_resident_page(request_id, tail_page) is None:
            blocks[tail_page] = self.block_pool.get_new_blocks(1)[0]

    def add_local_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert not new_computed_blocks
        req_blocks = self.req_to_blocks[request_id]
        assert not req_blocks
        num_host_pages = cdiv(num_local_computed_tokens, self.block_size)
        req_blocks.extend([self._null_block] * num_host_pages)
        self.num_cached_block[request_id] = 0
        assert self.coordinator is not None
        self.coordinator.commit_computed_blocks(request_id, num_host_pages)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        del num_tokens_main_model
        req_blocks = self.req_to_blocks[request_id]
        num_new_blocks = cdiv(num_tokens, self.block_size) - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
        req_blocks.extend(new_blocks)
        return new_blocks

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        assert self.coordinator is not None
        self.coordinator.free(request_id)
        blocks = self.get_residency_row(request_id)
        self.residency_changes.pop(request_id, None)
        super().pop_blocks_for_free(request_id)
        return trim_null_head(blocks)

    def adopt_resident_page(
        self, request_id: str, block_idx: int, block: KVCacheBlock
    ) -> bool:
        """Point a null prefix page at a pinned GPU copy of its contents."""
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or block_idx >= len(blocks):
            return False
        if self.get_resident_page(request_id, block_idx) is not None:
            return False
        self.residency_changes.setdefault(request_id, {})[block_idx] = block
        return True

    def drop_resident_page(
        self, request_id: str, block_idx: int
    ) -> KVCacheBlock | None:
        """Stop reading a page from the GPU; return the block it was read from."""
        block = self.get_resident_page(request_id, block_idx)
        if block is not None:
            changes = self.residency_changes.setdefault(request_id, {})
            changes[block_idx] = self._null_block
        return block

    def get_resident_page(self, request_id: str, block_idx: int) -> KVCacheBlock | None:
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None or block_idx >= len(blocks):
            return None
        block = blocks[block_idx]
        changes = self.residency_changes.get(request_id)
        if changes:
            block = changes.get(block_idx, block)
        return None if block.is_null else block

    def get_residency_row(self, request_id: str, start: int = 0) -> list[KVCacheBlock]:
        """The request's row from `start` with residency changes applied. Unlike
        `get_resident_page`, host-only pages stay as the null block."""
        blocks = self.req_to_blocks.get(request_id, [])[start:]
        changes = self.residency_changes.get(request_id)
        if not changes:
            return blocks
        return [
            changes.get(block_idx, block)
            for block_idx, block in enumerate(blocks, start)
        ]


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec,
    max_in_flight_tokens: int,
    max_model_len: int,
    role: str | None = None,
    **kwargs,
) -> SingleTypeKVCacheManager:
    """Get the appropriate manager for a given KVCacheSpec.

    Uses the KVCacheSpecRegistry to look up the manager class, supporting
    both built-in and custom specs registered via @register_kv_cache_spec
    and KVCacheSpecRegistry.register.

    Args:
        kv_cache_spec: The KVCacheSpec instance
        max_in_flight_tokens: The max tokens scheduled but not yet settled
            (one batch per concurrent step); see `VllmConfig.max_in_flight_tokens`
        max_model_len: The maximum context length the model could serve
        role: Optional KV cache group role used to select a role-specific
            manager
    Returns:
        An instance of the appropriate SingleTypeKVCacheManager subclass

    """
    manager_class = KVCacheSpecRegistry.get_manager_class(kv_cache_spec, role)
    assert manager_class is not None, (
        f"No manager registered for KVCacheSpec {type(kv_cache_spec)}"
    )
    # SlidingWindow / ChunkedLocalAttention managers recycle blocks;
    # the runtime admission cap must match the recycling-aware bound the
    # startup pool sizer uses (single source of truth: the spec method).
    # R-SWA also recycles gap blocks but peak physical KV still fits the
    # full-attention bound (prefix + window <= max_model_len), so it inherits
    # FullAttentionSpec sizing without a separate admission cap.
    if isinstance(
        kv_cache_spec,
        (SlidingWindowSpec, ChunkedLocalAttentionSpec, HiSparseResidentSpec),
    ):
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager


def register_all_kvcache_specs(vllm_config):
    """Built-in spec registration"""
    KVCacheSpecRegistry.register_role_manager(
        KVCacheGroupRole.HISPARSE_SOURCE, HiSparseSourceManager
    )
    KVCacheSpecRegistry.register(
        FullAttentionSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        HiSparseHotSpec,
        HiSparseHotManager,
        uniform_type_base_spec=HiSparseHotSpec,
    )
    KVCacheSpecRegistry.register(
        HiSparseResidentSpec,
        HiSparseResidentManager,
        uniform_type_base_spec=HiSparseResidentSpec,
    )

    KVCacheSpecRegistry.register(
        SlidingWindowSpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowSpec,
    )
    KVCacheSpecRegistry.register(
        CircularBufferSpec,
        CircularBufferManager,
        uniform_type_base_spec=CircularBufferSpec,
    )
    KVCacheSpecRegistry.register(
        SlidingWindowMLASpec,
        SlidingWindowManager,
        uniform_type_base_spec=SlidingWindowMLASpec,
    )

    KVCacheSpecRegistry.register(
        MambaSpec, MambaManager, uniform_type_base_spec=MambaSpec
    )
    KVCacheSpecRegistry.register(
        ChunkedLocalAttentionSpec,
        ChunkedLocalAttentionManager,
        uniform_type_base_spec=ChunkedLocalAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        CrossAttentionSpec,
        CrossAttentionManager,
        uniform_type_base_spec=CrossAttentionSpec,
    )

    # FullAttentionSpec subclasses — grouped with FullAttentionSpec
    KVCacheSpecRegistry.register(
        MLAAttentionSpec, FullAttentionManager, uniform_type_base_spec=FullAttentionSpec
    )
    KVCacheSpecRegistry.register(
        RSWASpec, RSWAManager, uniform_type_base_spec=FullAttentionSpec
    )
    # NOTE(Mengqing): HiddenStateCacheSpec won't take part in
    # grouping, thus the uniform_type_base_spec is just a
    # placeholder.
    KVCacheSpecRegistry.register(
        HiddenStateCacheSpec,
        FullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )
    KVCacheSpecRegistry.register(
        SinkFullAttentionSpec,
        SinkFullAttentionManager,
        uniform_type_base_spec=FullAttentionSpec,
    )

    from vllm.platforms import current_platform

    current_platform.register_custom_kv_cache_specs(vllm_config)
