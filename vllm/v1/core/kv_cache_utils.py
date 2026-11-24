# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-Cache Utilities."""

import copy
import math
import os
import weakref
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    NewType,
    TypeAlias,
    TypeVar,
    overload,
)

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.hashing import xxhash, xxhash_cbor
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool


# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# For cryptographic hash algorithms it is derived deterministically from a fixed
# default seed, so independent vLLM processes compute identical block hashes for
# identical content and can share a prefix cache (e.g. KV cache reuse across
# nodes) without extra configuration. This does not weaken collision resistance,
# which for SHA-256 does not depend on keeping the seed secret; ``cache_salt``
# remains the mechanism for intentional cache isolation.
#
# Non-cryptographic algorithms keep a per-process random seed, because a
# predictable seed would let an attacker precompute colliding blocks offline
# (see #12621). Setting PYTHONHASHSEED overrides the seed in both cases.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash

# Fixed seed used when the PYTHONHASHSEED environment variable is not set and
# the hash algorithm is cryptographic.
DEFAULT_NONE_HASH_SEED = "vllm-none-hash"

# Algorithms that are not collision resistant, so the seed must stay secret.
_NON_CRYPTO_HASH_FUNCTIONS = frozenset({xxhash, xxhash_cbor})

# The seed NONE_HASH was derived from, set by init_none_hash.
_NONE_HASH_SEED: str | None = None


def resolve_none_hash_seed(hash_fn: Callable[[Any], bytes]) -> str:
    """Resolve the seed to derive NONE_HASH from.

    PYTHONHASHSEED wins if set. Otherwise cryptographic algorithms get the
    fixed default (shareable across processes) and non-cryptographic ones get
    fresh random bytes, keeping the seed unpredictable where collision
    resistance depends on it.
    """
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is not None:
        return hash_seed
    if hash_fn in _NON_CRYPTO_HASH_FUNCTIONS:
        return os.urandom(32).hex()
    return DEFAULT_NONE_HASH_SEED


def get_none_hash_seed() -> str:
    """Return the seed NONE_HASH was derived from.

    Components that must agree on NONE_HASH across processes (the P2P tier
    advertises this during its connect handshake) read the resolved seed here
    instead of re-deriving it, so they observe the random seed too. Falls back
    to the deterministic seed before `init_none_hash` has run.
    """
    if _NONE_HASH_SEED is None:
        return DEFAULT_NONE_HASH_SEED
    return _NONE_HASH_SEED


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH, _NONE_HASH_SEED

    _NONE_HASH_SEED = resolve_none_hash_seed(hash_fn)
    if hash_fn in _NON_CRYPTO_HASH_FUNCTIONS and os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "Using a random per-process NONE_HASH seed because %s is not "
            "collision resistant. Block hashes are therefore not reproducible "
            "across processes; set PYTHONHASHSEED to a shared value to reuse "
            "the prefix cache across instances, or use sha256.",
            hash_fn.__name__,
        )
    NONE_HASH = BlockHash(hash_fn(_NONE_HASH_SEED))


@dataclass(slots=True)
class KVCacheBlock:
    """KV-cache block metadata."""

    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # Deferred frees may contain blocks from multiple pools. Weak to avoid ref cycles.
    pool: "weakref.ref[BlockPool] | None" = field(
        default=None, repr=False, compare=False, kw_only=True
    )
    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: "KVCacheBlock | None" = None
    next_free_block: "KVCacheBlock | None" = None

    # Whether the block is a null block that should never be cached.
    is_null: bool = False

    def __repr__(self) -> str:
        # Use block_id instead of KVCacheBlock object to avoid calling __repr__
        # on KVCacheBlock object recursively.
        prev_block_id = self.prev_free_block.block_id if self.prev_free_block else None
        next_block_id = self.next_free_block.block_id if self.next_free_block else None
        return (
            f"KVCacheBlock(block_id={self.block_id}, "
            f"ref_cnt={self.ref_cnt}, "
            f"prev_free_block={prev_block_id}, "
            f"next_free_block={next_block_id})"
        )


def null_head_len(blocks: Sequence[KVCacheBlock]) -> int:
    """Number of leading null blocks, e.g. a sliding-window table's skipped prefix."""
    if not blocks or not blocks[0].is_null:
        return 0
    lo, hi = 1, len(blocks)
    while lo < hi:
        mid = (lo + hi) // 2
        if blocks[mid].is_null:
            lo = mid + 1
        else:
            hi = mid
    # All null entries are one shared object, so count() checks the run at C speed.
    if blocks[:lo].count(blocks[0]) == lo:
        return lo
    return next(i for i, b in enumerate(blocks) if not b.is_null)


_BlockSeq = TypeVar("_BlockSeq", list[KVCacheBlock], Sequence[KVCacheBlock])


def trim_null_head(blocks: _BlockSeq) -> _BlockSeq:
    """`blocks` without its leading null blocks; `blocks` itself if none."""
    nhl = null_head_len(blocks)
    return blocks[nhl:] if nhl else blocks


class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int


class FreeKVCacheBlockQueue:
    """This class organizes a list of KVCacheBlock objects to a doubly linked
    list of free blocks. We implement this class instead of using Python
    builtin deque to support removing a block in the middle of the queue
    in O(1) time. To close the performance gap to the builtin deque which is
    implemented in C++, this class does not allocate any Python objects when
    manipulating the linked list. Instead, this class manipulates the
    prev_free_block and next_free_block attributes of the given blocks.

    The queue is ordered by block ID in the beginning. When a block is allocated
    and then freed, it will be appended back with the eviction order:
    1. The least recent used block is at the front (LRU).
    2. If two blocks have the same last accessed time (allocated by the
       same sequence), the one with more hash tokens (the tail of a block
       chain) is at the front.
    Note that we maintain this order by reversing the block order when free
    blocks of a request. This operation is outside of this class.

    Args:
        blocks: A list of KVCacheBlock objects.

    """

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.num_free_blocks = len(blocks)

        # Initialize doubly links of consecutive blocks
        for i in range(self.num_free_blocks):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < self.num_free_blocks - 1:
                blocks[i].next_free_block = blocks[i + 1]

        # Create a fake head and a tail block for the doubly linked list to
        # reduce branching in the code
        #
        # The implementation guaranteed that the fake head and tail
        # are NEVER got popped, so we could safely assume each real blocks
        # in the queue has prev and next blocks.
        self.fake_free_list_head = KVCacheBlock(block_id=-1)
        self.fake_free_list_tail = KVCacheBlock(block_id=-1)
        if self.num_free_blocks > 0:
            # Connect fake_head and fake_tail to the first and last block
            # respectively.
            self.fake_free_list_head.next_free_block = blocks[0]
            blocks[0].prev_free_block = self.fake_free_list_head
            self.fake_free_list_tail.prev_free_block = blocks[-1]
            blocks[-1].next_free_block = self.fake_free_list_tail
        else:
            # For empty list, simply connect the fake head and tail.
            self.fake_free_list_head.next_free_block = self.fake_free_list_tail
            self.fake_free_list_tail.prev_free_block = self.fake_free_list_head

    def popleft(self) -> KVCacheBlock:
        """Pop the first free block and reduce num_free_blocks by 1.

        Returns:
            The first free block.

        """
        head = self.fake_free_list_head
        first_block = head.next_free_block
        if first_block is self.fake_free_list_tail or first_block is None:
            assert self.num_free_blocks == 0, (
                f"num_free_blocks ({self.num_free_blocks}) is out of sync "
                "with the free list."
            )
            raise ValueError("No free blocks available")

        next_block = first_block.next_free_block
        if next_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(
                "Invalid block found in popleft() "
                "which doesn't have a valid next_free_block"
            )

        # Connect fake_head and the next block of first_block (i.e. second block
        # or fake tail).
        head.next_free_block = next_block
        next_block.prev_free_block = head

        # Remove the block from the linked list.
        first_block.prev_free_block = first_block.next_free_block = None

        self.num_free_blocks -= 1
        return first_block

    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """Pop the first n free blocks and reduce num_free_blocks by n.

        Args:
            n: The number of blocks to pop.

        Returns:
            A list of n free blocks.

        """
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n

        curr_block = self.fake_free_list_head.next_free_block
        if n == 1:
            assert curr_block is not None
            next_block = curr_block.next_free_block
            curr_block.prev_free_block = None
            curr_block.next_free_block = None
            if next_block is not None:
                self.fake_free_list_head.next_free_block = next_block
                next_block.prev_free_block = self.fake_free_list_head
            return [curr_block]

        # Pop n blocks from the head of the list
        ret = []
        for _ in range(n):
            assert curr_block is not None
            ret.append(curr_block)
            last_block = curr_block
            curr_block = curr_block.next_free_block
            # Reset prev_free_block and next_free_block of all popped blocks
            last_block.prev_free_block = None
            last_block.next_free_block = None

        if curr_block is not None:
            # The queue is not empty, connect the fake head to
            # the new first block.
            self.fake_free_list_head.next_free_block = curr_block
            curr_block.prev_free_block = self.fake_free_list_head
        return ret

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a block in the free list and reduce num_free_blocks by 1.

        Args:
            block: The block to remove.

        """
        prev_block = block.prev_free_block
        next_block = block.next_free_block
        if prev_block is None or next_block is None:
            # This should not happen if the block is from the free list.
            # It indicates a bug in the caller's logic.
            raise RuntimeError(f"remove() called on an invalid block: {block}")

        # Link the previous block to the next block.
        prev_block.next_free_block = next_block
        # Link the next block to the previous block.
        next_block.prev_free_block = prev_block

        # Remove the block from the linked list.
        block.prev_free_block = block.next_free_block = None
        self.num_free_blocks -= 1

    def append(self, block: KVCacheBlock) -> None:
        """Put a block back into the free list and increase
        num_free_blocks by 1.

        Args:
            block: The block to append.

        """
        tail = self.fake_free_list_tail
        last_block = tail.prev_free_block
        if last_block is None:
            raise RuntimeError(
                "prev_free_block of fake_free_list_tail should always exist"
            )
        # Connect the new block after the last block.
        last_block.next_free_block = block
        block.prev_free_block = last_block

        # Connect the fake tail after the new block.
        block.next_free_block = tail
        tail.prev_free_block = block

        self.num_free_blocks += 1

    def prepend_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks at the front of the free list."""
        if len(blocks) == 0:
            return

        first_block = self.fake_free_list_head.next_free_block
        assert first_block is not None, (
            "next_free_block of fake_free_list_head should always exist"
        )

        prev_block = self.fake_free_list_head
        for block in blocks:
            block.prev_free_block = prev_block
            prev_block.next_free_block = block
            prev_block = block

        prev_block.next_free_block = first_block
        first_block.prev_free_block = prev_block

        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[KVCacheBlock]) -> None:
        """Put a list of blocks back into the free list

        Args:
            blocks: The blocks to append.

        """
        if not blocks:
            return

        tail = self.fake_free_list_tail
        last_block = tail.prev_free_block
        assert last_block is not None, (
            "prev_free_block of fake_free_list_tail should always exist"
        )
        # Add inter-connections between consecutive blocks
        for block in blocks:
            block.prev_free_block = last_block
            last_block.next_free_block = block
            last_block = block

        # Connect the last block of <blocks> to the fake tail
        last_block.next_free_block = tail
        tail.prev_free_block = last_block

        self.num_free_blocks += len(blocks)

    def get_all_free_blocks(self) -> list[KVCacheBlock]:
        """Get all free blocks in the free list. Mainly used for testing.

        Returns:
            A list of free blocks.

        """
        ret = []
        if self.fake_free_list_head.next_free_block is None:
            raise RuntimeError(
                "next_free_block of fake_free_list_head should always exist"
            )
        # Start from the first block
        curr_block: KVCacheBlock = self.fake_free_list_head.next_free_block
        # As long as next_free_block is available, we haven't reached to
        # the fake tail yet.
        while curr_block.next_free_block is not None:
            ret.append(curr_block)
            curr_block = curr_block.next_free_block
        return ret

    def iter_blocks_after(
        self,
        cursor: KVCacheBlock | None,
    ) -> Iterator[KVCacheBlock]:
        """Iterate free blocks in eviction order after the cursor."""
        if cursor is None:
            curr_block = self.fake_free_list_head.next_free_block
        else:
            curr_block = cursor.next_free_block

        while curr_block is not None and curr_block is not self.fake_free_list_tail:
            yield curr_block
            curr_block = curr_block.next_free_block


def to_event_extra_keys(
    extra_keys: Iterable[tuple[Any, ...] | None] | None,
) -> list[tuple[Any, ...] | None] | None:
    """Convert block-hash extra keys to the untagged per-block list published
    in KV events.

    External KV event consumers parse the pre-tagging shapes: bare LoRA names
    and cache salts, `(mm_identifier, offset)` pairs and bare prompt-embeds
    digests. Events keep that format until consumers handle the tagged keys.

    Args:
        extra_keys: One entry per block, each as returned by
            `generate_block_hash_extra_keys`, or None.

    Returns:
        One untagged entry per block, or None if there are no entries.

    """
    if not extra_keys:
        return None
    event_keys = [
        None
        if keys is None
        else tuple(key[1:] if key[0] == "mm" else key[1] for key in keys)
        for keys in extra_keys
    ]
    return event_keys or None


def resolve_dcp_kv_block_size(spec: KVCacheSpec, dcp_world_size: int) -> int:
    """Return the token span of a cache block under DCP."""
    return spec.block_size * (dcp_world_size if spec.dcp_sharded else 1)


def resolve_dcp_kv_cache_spec(spec: KVCacheSpec, dcp_world_size: int) -> KVCacheSpec:
    """Return a KV cache spec with block sizes adjusted for DCP."""
    block_size = resolve_dcp_kv_block_size(spec, dcp_world_size)
    if block_size == spec.block_size:
        return spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return replace(
            spec,
            block_size=block_size,
            kv_cache_specs={
                name: resolve_dcp_kv_cache_spec(layer_spec, dcp_world_size)
                for name, layer_spec in spec.kv_cache_specs.items()
            },
        )
    return replace(spec, block_size=block_size)


def resolve_kv_cache_block_sizes(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> tuple[int, int]:
    """Resolve the scheduler block size, the token-alignment invariant used by
    the scheduler. Single group: `cache_config.block_size * dcp`. Multiple
    groups: LCM of every group's effective block size. Attention groups are
    scaled by DCP; Mamba groups keep their full per-rank state and are not
    scaled. Returned twice for callers that still unpack a hash block size.
    """
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        if groups and not groups[0].kv_cache_spec.dcp_sharded:
            dcp = 1
        bs = cache_config.block_size * dcp
        return bs, bs

    group_block_sizes = [
        resolve_dcp_kv_block_size(g.kv_cache_spec, dcp) for g in groups
    ]
    scheduler_block_size = math.lcm(*group_block_sizes)

    logger.info_once("kv cache group sizes %s", tuple(group_block_sizes))
    logger.info_once("kv lcm block sizes %s", scheduler_block_size)
    return scheduler_block_size, scheduler_block_size


def get_max_concurrency_for_kv_cache_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> float:
    """Get the maximum concurrency for the given KV cache configuration.

    A request at max_model_len consumes whole blocks from each group's block
    table — cdiv(per-request bytes, page bytes) of the group's spec — and all
    device groups draw those block ids from one shared pool, so the per-request
    total is the sum over groups. The memory/page ratio is identical whether
    a group carries an aggregated UniformTypeKVCacheSpecs (worker config) or
    a representative per-layer spec (scheduler config), so both capacity
    call sites agree.

    Host groups use a separate pool; the smaller concurrency limit applies.
    """
    num_blocks_per_request = 0
    host_blocks_per_request = 0
    for group in kv_cache_config.kv_cache_groups:
        required = cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(vllm_config),
            group.kv_cache_spec.page_size_bytes,
        )
        if group.host_resident:
            host_blocks_per_request += required
        else:
            num_blocks_per_request += required
    limits = [kv_cache_config.num_blocks / num_blocks_per_request]
    if host_blocks_per_request:
        assert kv_cache_config.hisparse_host_num_blocks is not None
        limits.append(
            kv_cache_config.hisparse_host_num_blocks / host_blocks_per_request
        )
    return min(limits)


def generate_scheduler_kv_cache_config(
    kv_cache_configs: list[KVCacheConfig],
) -> KVCacheConfig:
    """Generate the KV cache configuration for the scheduler."""
    assert all(
        [cfg.num_blocks == kv_cache_configs[0].num_blocks for cfg in kv_cache_configs]
    )
    assert all(
        cfg.hisparse_host_num_blocks == kv_cache_configs[0].hisparse_host_num_blocks
        for cfg in kv_cache_configs
    )
    # Worker configs share scheduler geometry and the resolved zeroing
    # decision, so use an arbitrary one to initialize the scheduler.
    cfg = copy.deepcopy(kv_cache_configs[0])
    for group in cfg.kv_cache_groups:
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            # All layers in the UniformTypeKVCacheSpecs have the same type,
            # so use an arbitrary one to initialize the scheduler.
            group.kv_cache_spec = next(
                iter(group.kv_cache_spec.kv_cache_specs.values())
            )
    return cfg


def get_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> tuple[int, float]:
    """Get the group-aware KV cache token capacity and max concurrency."""
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        vllm_config, kv_cache_config
    )
    return int(max_concurrency * max_model_len), max_concurrency


def _kv_cache_device_name(vllm_config: VllmConfig) -> str:
    device_type = vllm_config.device_config.device_type
    return "GPU" if device_type == "cuda" else device_type.upper()


def update_kv_cache_capacity(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> None:
    """Store and log the resolved KV cache capacity."""
    num_tokens, max_concurrency = get_kv_cache_capacity(vllm_config, kv_cache_config)
    vllm_config.cache_config.kv_cache_size_tokens = num_tokens
    vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency
    max_model_len = vllm_config.model_config.max_model_len
    pools = _kv_cache_device_name(vllm_config)
    if vllm_config.attention_config.hisparse_config is not None:
        pools += " + host"
    logger.info_once(
        "%s KV cache size: %s tokens, "
        "Maximum concurrency for %s tokens per request: %.2fx",
        pools,
        f"{num_tokens:,}",
        f"{max_model_len:,}",
        max_concurrency,
    )


class BlockHashListWithBlockSize:
    """Convert block-hash granularity from `hash_block_size` to `target_block_size`.
    Used when KV cache groups have different block sizes: `hash_block_size`
    is the size used to compute the original `block_hashes`; `target_block_size`
    is the group's actual block size.

    Currently, only scaling up by an integer factor is supported (i.e.,
    `target_block_size` is a multiple of `hash_block_size`). Conversion is
    performed lazily on access for efficiency. Each `hash_block_size` hash is
    already chained over its entire prefix, so the hash at the last
    `hash_block_size` boundary of a `target_block_size` block uniquely
    fingerprints that block's prefix; we use it directly.

    Example (`hash_block_size` = 16, `target_block_size` = 32):
    the second 16-size hash already covers tokens 0-31, so it is the 32-size
    hash:

    Block hashes with block_size 16:
    | Token Range | 0-15 | 16-31 | 32-47 | 48-63 |
    |-------------|------|-------|-------|-------|
    | Hash        | A    | B     | C     | D     |

    Block hashes with block_size 32:
    | Token Range | 0-31 | 32-63 |
    |-------------|------|-------|
    | Hash        | B    | D     |

    Args:
        block_hashes: Block hashes to convert, computed at `hash_block_size`.
        hash_block_size: Block size at which `block_hashes` were computed.
        target_block_size: Desired block size; must be a multiple of `hash_block_size`.

    """

    def __init__(
        self,
        block_hashes: list[BlockHash],
        hash_block_size: int,
        target_block_size: int,
    ):
        self.block_hashes = block_hashes
        assert target_block_size % hash_block_size == 0
        self.scale_factor = target_block_size // hash_block_size

    def __len__(self) -> int:
        return len(self.block_hashes) // self.scale_factor

    @overload
    def __getitem__(self, idx: int) -> BlockHash: ...

    @overload
    def __getitem__(self, idx: slice) -> list[BlockHash]: ...

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_value_at(idx)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self._get_value_at(i) for i in range(start, stop, step)]

        raise TypeError(f"Invalid index type: {type(idx)!r}")

    def __iter__(self) -> Iterator[BlockHash]:
        for i in range(len(self)):
            yield self._get_value_at(i)

    def _get_value_at(self, idx: int) -> BlockHash:
        # The last hash_block_size hash within the target block already chains
        # over the whole prefix, so it is the target block's hash.
        return self.block_hashes[(idx + 1) * self.scale_factor - 1]


BlockHashList = list[BlockHash] | BlockHashListWithBlockSize


def resolve_block_hashes(
    block_hashes: BlockHashList,
    hash_block_size: int,
    block_size: int,
    *,
    supports_fine_grained_hash_lookup: bool = False,
    alignment_tokens: int | None = None,
) -> BlockHashList:
    """Resolve the block-hash view at `block_size`.

    When `block_size` equals `hash_block_size`, reuse the precomputed block
    hashes directly; otherwise view them at `block_size` granularity.
    Fine-grained lookup keeps the original hashes for partial cache hits.
    """
    if block_size == hash_block_size:
        return block_hashes
    if isinstance(block_hashes, BlockHashListWithBlockSize):
        # Already a block-size view
        assert block_hashes.scale_factor == block_size // hash_block_size
        return block_hashes
    # Fine-grained partial hits keep the raw hashes. The caller passes
    # alignment_tokens = hash_block_size to enable them, else >= block_size.
    if (
        supports_fine_grained_hash_lookup
        and alignment_tokens is not None
        and alignment_tokens < block_size
        and block_size % alignment_tokens == 0
    ):
        return block_hashes
    assert block_size % hash_block_size == 0
    return BlockHashListWithBlockSize(block_hashes, hash_block_size, block_size)
