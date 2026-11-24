# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref
from collections.abc import Callable, Iterable, Sequence

from vllm.distributed.kv_events import MEDIUM_GPU
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

logger = init_logger(__name__)


class BlockPool:
    """BlockPool that manages KVCacheBlocks.
    It provides methods to allocate and free kv cache blocks. The
    free_block_queue stores the free blocks in allocation order.

    Args:
        num_gpu_blocks: The number of blocks in the pool.
        metrics_collector: Optional metrics collector for tracking block residency.
        medium: Storage medium of the pool.

    """

    def __init__(
        self,
        num_gpu_blocks: int,
        metrics_collector: KVCacheMetricsCollector | None = None,
        medium: str = MEDIUM_GPU,
    ):
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        self.num_gpu_blocks = num_gpu_blocks
        self.medium = medium
        self._weak_self = weakref.ref(self)
        # All kv-cache blocks.
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx, pool=self._weak_self) for idx in range(num_gpu_blocks)
        ]
        # Free block queue that constructs and manipulates a doubly linked
        # list of free blocks.
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)

        # To represent a placeholder block with block_id=0.
        # The ref_cnt of null_block is not maintained, needs special care to
        # avoid freeing it.
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        self.metrics_collector = metrics_collector
        # Callbacks for blocks released with ``unpin_blocks`` whose contents
        # are still being read until the pool reuses them.
        self._reuse_watchers: dict[int, Callable[[KVCacheBlock], None]] = {}

    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """Get new blocks from the free block pool.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of new block.

        """
        if num_blocks > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")

        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)

        if self._reuse_watchers:
            self._notify_reuse(ret)

        for block in ret:
            assert block.ref_cnt == 0
            block.ref_cnt = 1
            if self.metrics_collector:
                self.metrics_collector.on_block_allocated(block)
        return ret

    def _notify_reuse(self, blocks: Iterable[KVCacheBlock]) -> None:
        for block in blocks:
            watcher = self._reuse_watchers.pop(block.block_id, None)
            if watcher is not None:
                watcher(block)

    def unpin_blocks(
        self,
        blocks: Iterable[KVCacheBlock],
        on_reuse: Callable[[KVCacheBlock], None],
    ) -> None:
        """Release references to blocks that stay readable until reused.

        The blocks become last-resort eviction candidates regardless of prefix
        caching: they join the tail of the free queue and count as free.
        `on_reuse` fires when `get_new_blocks` hands a block out (or the
        cache is reset), so the caller can stop reading it.
        """
        released: list[KVCacheBlock] = []
        for block in blocks:
            assert block.ref_cnt > 0 and not block.is_null
            block.ref_cnt -= 1
            self._reuse_watchers[block.block_id] = on_reuse
            if block.ref_cnt == 0:
                released.append(block)
        self.free_block_queue.append_n(released)

    def touch(self, blocks: Sequence[KVCacheBlock]) -> None:
        """Take a reference on each block, removing it from the free queue if
        it was released with `unpin_blocks`.

        Args:
            blocks: A list of blocks to touch.

        """
        for block in blocks:
            if block.is_null:
                continue
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
            if self.metrics_collector:
                self.metrics_collector.on_block_accessed(block)

    def is_block_writable(self, block: KVCacheBlock) -> bool:
        """Return whether a block can be mutated by its sole owner."""
        return not block.is_null and block.ref_cnt == 1

    def free_blocks(
        self, ordered_blocks: Iterable[KVCacheBlock], evict_first: bool = False
    ) -> None:
        """Free a list of blocks, the first one reused last.

        Args:
            ordered_blocks: A list of blocks to free.
            evict_first: Unused; kept for callers that free blocks another
                tier already holds.

        """
        freed = []
        other_pools: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in ordered_blocks:
            if block.is_null:  # never pooled; its ref_cnt is not maintained
                continue
            if block.pool is not None and block.pool is not self._weak_self:
                pool = block.pool()
                if pool is not None:
                    other_pools.setdefault(pool, []).append(block)
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                freed.append(block)

        # LIFO reuse for better GPU locality.
        self.free_block_queue.prepend_n(freed)
        for pool, blocks in other_pools.items():
            pool.free_blocks(blocks, evict_first=evict_first)

    def reset_prefix_cache(self) -> bool:
        """Release blocks still readable through `unpin_blocks`.

        Returns:
            bool: True if no block is in use, False otherwise.

        """
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
        if num_used_blocks != 1:  # The null block is always marked as used
            logger.warning(
                "Failed to reset prefix cache because some "
                "blocks (%d) are not freed yet",
                num_used_blocks - 1,
            )
            return False

        if self._reuse_watchers:
            self._notify_reuse(
                [self.blocks[block_id] for block_id in list(self._reuse_watchers)]
            )

        if self.metrics_collector:
            self.metrics_collector.reset()

        logger.info("Successfully reset prefix cache")
        return True

    def get_num_free_blocks(self) -> int:
        """Get the number of free blocks in the pool.

        Returns:
            The number of free blocks.

        """
        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).

        """
        # Subtract 1 to account for null block.
        total_gpu_blocks = self.num_gpu_blocks - 1
        if not total_gpu_blocks:
            return 0
        return 1.0 - (self.get_num_free_blocks() / total_gpu_blocks)
