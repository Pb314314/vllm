# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Sequence

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    CrossAttentionManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import KVCacheConfig


class KVCacheCoordinator:
    """Coordinate the KV cache of different KV cache groups. Supports
    arbitrary numbers of KV cache groups (including 0 groups)."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        max_in_flight_tokens: int,
        dcp_world_size: int,
        pcp_world_size: int,
        scheduler_block_size: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ):
        self.kv_cache_config = kv_cache_config
        # The scheduling granularity (LCM of all group block sizes), must be a
        # multiple of the block size of each group.
        assert all(
            scheduler_block_size % g.kv_cache_spec.block_size == 0
            for g in kv_cache_config.kv_cache_groups
        )
        self.scheduler_block_size = scheduler_block_size

        self.block_pool = BlockPool(
            num_gpu_blocks=kv_cache_config.num_blocks,
            metrics_collector=metrics_collector,
        )

        self.single_type_managers = tuple(
            get_manager_for_kv_cache_spec(
                kv_cache_spec=kv_cache_group.kv_cache_spec,
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
                block_pool=self.block_pool,
                role=kv_cache_group.role,
                kv_cache_group_id=i,
                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
                scheduler_block_size=self.scheduler_block_size,
                needs_kv_cache_zeroing=self.kv_cache_config.needs_kv_cache_zeroing,
            )
            for i, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups)
        )

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_encoder_tokens: int,
        total_computed_tokens: int,
        num_local_computed_tokens: int,
        num_tokens_main_model: int,
        apply_admission_cap: bool = False,
    ) -> int:
        """Get the number of device blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.
            total_computed_tokens: Include both local and external tokens.
            num_local_computed_tokens: The number of local prefix-cache computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            apply_admission_cap: If True, apply the recycling-aware
                per-request admission cap (SWA / chunked-local). Set only by
                the full-sequence admission gate; per-step allocation must
                leave it False so the predictor matches `allocate_new_blocks`.

        Returns:
            The number of blocks to allocate.

        """
        num_blocks_to_allocate = 0
        for i, manager in enumerate(self.single_type_managers):
            if isinstance(manager, CrossAttentionManager):
                # For cross-attention, we issue a single static allocation
                # of blocks based on the number of encoder input tokens.
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_encoder_tokens,
                    [],
                    0,
                    0,
                    num_encoder_tokens,
                    apply_admission_cap=apply_admission_cap,
                )
            else:
                num_blocks_to_allocate += manager.get_num_blocks_to_allocate(
                    request_id,
                    num_tokens,
                    new_computed_blocks[i],
                    total_computed_tokens,
                    num_local_computed_tokens,
                    num_tokens_main_model,
                    apply_admission_cap=apply_admission_cap,
                )
        return num_blocks_to_allocate

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[Sequence[KVCacheBlock], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
        skip_zeroing_group_ids: tuple[int, ...] = (),
    ) -> None:
        """Add the new computed blocks to the request. Optionally allocate new
            blocks for external computed tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
            skip_zeroing_group_ids: Groups whose external-token blocks are
                written by an async load and must not be zeroed.

        """
        # A running request is already tracked in num_cached_block and won't
        # have new prefix-cache hits, so this is a no-op for it.
        if any(
            request_id in manager.num_cached_block
            for manager in self.single_type_managers
        ):
            assert all(len(blocks) == 0 for blocks in new_computed_blocks)
            return

        # Two-phase allocation (issue #33775): first touch every group's local
        # cache-hit blocks, then allocate external blocks for every group. This
        # ensures an earlier group's external `get_new_blocks` cannot evict a
        # later group's not-yet-touched cache-hit blocks.
        for i, manager in enumerate(self.single_type_managers):
            manager.add_local_computed_blocks(
                request_id,
                new_computed_blocks[i],
                num_local_computed_tokens,
                num_external_computed_tokens,
            )
        if num_external_computed_tokens > 0:
            for i, manager in enumerate(self.single_type_managers):
                manager.allocate_external_computed_blocks(
                    request_id,
                    num_local_computed_tokens,
                    num_external_computed_tokens,
                    record_for_zeroing=i not in skip_zeroing_group_ids,
                )

    def allocate_new_blocks(
        self,
        request_id: str,
        num_tokens: int,
        num_tokens_main_model: int,
        num_encoder_tokens: int = 0,
    ) -> tuple[list[KVCacheBlock], ...]:
        """Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
            num_encoder_tokens: The number of encoder tokens for allocating
                blocks for cross-attention.

        Returns:
            The new allocated blocks.

        """
        return tuple(
            manager.allocate_new_blocks(
                request_id,
                num_encoder_tokens
                if isinstance(manager, CrossAttentionManager)
                else num_tokens,
                num_tokens_main_model,
            )
            for manager in self.single_type_managers
        )

    def reset_prefix_cache(self) -> bool:
        """Reset each manager's pool once."""
        pools = dict.fromkeys(
            manager.block_pool for manager in self.single_type_managers
        )
        return all([pool.reset_prefix_cache() for pool in pools])

    def free(self, request_id: str) -> None:
        """Free the blocks for the request.

        Args:
            request_id: The request ID.

        """
        for manager in self.single_type_managers:
            manager.free(request_id)

    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        """Pop the request's bookkeeping from all single-type managers and
        return its blocks without returning them to the block pool. The
        caller must eventually pass the returned blocks to
        `block_pool.free_blocks`, freeing them in reverse order (so that
        tail blocks are evicted first).

        Args:
            request_id: The request ID.

        Returns:
            The request's blocks in allocation order.

        """
        blocks: list[KVCacheBlock] = []
        for manager in self.single_type_managers:
            blocks.extend(manager.pop_blocks_for_free(request_id))
        return blocks

    def remove_skipped_blocks(
        self,
        request_id: str,
        processed_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            processed_computed_tokens: Computed-token prefix length covering
                fully processed and committed tokens only (safe to free).
            num_prompt_tokens: Optional prompt length. R-SWA managers use this to
                free gap blocks between the prefill tail and decode window; other
                manager types ignore it.

        """
        for manager in self.single_type_managers:
            manager.remove_skipped_blocks(
                request_id, processed_computed_tokens, num_prompt_tokens
            )

    def get_blocks(self, request_id: str) -> tuple[list[KVCacheBlock], ...]:
        """Get the blocks for the request."""
        return tuple(
            manager.req_to_blocks.get(request_id) or []
            for manager in self.single_type_managers
        )

    def new_step_starts(self) -> None:
        """Notify each manager that a new step is starting."""
        for manager in self.single_type_managers:
            manager.new_step_starts()


def get_kv_cache_coordinator(
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    max_in_flight_tokens: int,
    dcp_world_size: int,
    pcp_world_size: int,
    scheduler_block_size: int,
    metrics_collector: KVCacheMetricsCollector | None = None,
) -> KVCacheCoordinator:
    return KVCacheCoordinator(
        kv_cache_config,
        max_model_len,
        max_in_flight_tokens,
        dcp_world_size=dcp_world_size,
        pcp_world_size=pcp_world_size,
        scheduler_block_size=scheduler_block_size,
        metrics_collector=metrics_collector,
    )
