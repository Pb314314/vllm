# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical side of session KV reuse: a tree of immutable committed states.

A node holds the blocks for path positions `[start, end)` of one computed
sequence; its parent holds the positions before `start`. Concatenating the
blocks along a root-to-node path gives a block table. The store never compares
tokens or hashes blocks: the resolver tells it which node a request matches,
and the store only checks that the node still holds that position.

Invariants:
  1. Only the store publishes nodes, from the blocks of the request that
     computed them.
  2. A published node never changes except by truncating its tail. Only the
     publishing request may still write its tail block, and only past `end`.
     A request resuming inside a block gets a private copy of it.
  3. A block returns to the pool only when its `ref_cnt` reaches 0; each node
     holds one reference on each of its blocks.
"""

import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.session.protocol import (
    NodePublished,
    NodeTruncated,
    RequestFinished,
    SessionBinding,
    StoreEvent,
    StoreReset,
)
from vllm.v1.session.resolver import pack_tokens

_COMMITTED = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
# Aborted requests (including stop strings, which the frontend aborts) still
# computed valid KV; they publish it without advancing their session.
_PUBLISHED = (*_COMMITTED, RequestStatus.FINISHED_ABORTED)


def validate_session_cache_config(vllm_config: VllmConfig) -> None:
    unsupported = {
        "speculative decoding": vllm_config.speculative_config is not None,
        "KV connectors": vllm_config.kv_transfer_config is not None,
        "EC connectors": vllm_config.ec_transfer_config is not None,
        "LoRA": vllm_config.lora_config is not None,
        "encoder-decoder models": vllm_config.model_config.is_encoder_decoder,
        "pipeline parallelism": (
            vllm_config.parallel_config.pipeline_parallel_size > 1
        ),
    }
    found = [name for name, present in unsupported.items() if present]
    if found:
        raise ValueError(f"Session KV cache does not support {', '.join(found)} yet")


def validate_session_request(request: Request) -> None:
    if request.prompt_embeds is not None or request.mm_features:
        raise ValueError("Session KV cache only supports text token prompts")


class _Node:
    __slots__ = (
        "node_id",
        "parent",
        "start",
        "end",
        "blocks",
        "num_children",
        "commit_points",
    )

    def __init__(
        self,
        node_id: int,
        parent: "_Node | None",
        start: int,
        end: int,
        blocks: tuple[list[KVCacheBlock], ...],
        commit_points: tuple[int, ...] | None,
    ) -> None:
        self.node_id = node_id
        self.parent = parent
        self.start = start
        self.end = end
        # Per KV cache group: blocks for block indices
        # `[start // block_size, cdiv(end, block_size))`.
        self.blocks = blocks
        self.num_children = 0
        self.commit_points = commit_points

    def resume_length(self, length: int, alignment: int) -> int:
        """Longest position <= `length` a request can resume this node at."""
        if self.commit_points is None:
            length = min(length, self.end)
            length -= length % alignment
            return length if length > self.start else 0
        return max(
            (c for c in self.commit_points if self.start < c <= length), default=0
        )


class SessionKVCacheManager(KVCacheManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        managers = self.coordinator.single_type_managers
        supported = (FullAttentionManager, MambaManager)
        if any(type(m) not in supported for m in managers):
            raise ValueError(
                "Session KV cache supports full attention and Mamba layers only; "
                f"got {sorted({type(m).__name__ for m in managers})}"
            )
        if any(isinstance(m, MambaManager) and m.num_speculative_blocks for m in managers):
            raise ValueError("Session KV cache does not support speculative decoding")
        # Mamba state is valid only where a request stopped, so hybrid models
        # resume at commit points; attention-only models at any position.
        self.commit_points_only = any(isinstance(m, MambaManager) for m in managers)
        self.block_sizes = [m.block_size for m in managers]
        # Attention-only models resume at block boundaries and recompute the
        # rest of the block, which is cheaper than copying it. Hybrid models
        # must resume exactly at a commit point, copying the boundary blocks.
        self.resume_alignment = math.lcm(*self.block_sizes)

        self.epoch = 0
        self.seq = 0
        self._publish_seq = 0
        self._next_node_id = 1
        # Least recently used first.
        self.nodes: OrderedDict[int, _Node] = OrderedDict()
        self._listeners: list[Callable[[StoreEvent], None]] = []
        self._rebind: Callable[[Request], SessionBinding] | None = None
        # Attaches handed to the scheduler but not yet allocated: protected
        # from eviction until the request holds its own references.
        self._pending: dict[str, tuple[list[_Node], int]] = {}
        # Deepest position on each running request's path held by a node.
        self._tip: dict[str, tuple[_Node | None, int]] = {}
        # Prompt node published by each running request; replaced by the
        # request's final node when nothing else hangs off it.
        self._prompt_node: dict[str, _Node] = {}

        self.num_attached_requests = 0
        self.num_attached_tokens = 0
        self.num_stale_bindings = 0
        self.num_evicted_blocks = 0

    def connect(
        self,
        listener: Callable[[StoreEvent], None],
        rebind: Callable[[Request], SessionBinding] | None = None,
    ) -> None:
        self._listeners.append(listener)
        self._rebind = rebind

    def _emit(self, make_event: Callable[[int], StoreEvent]) -> None:
        self.seq += 1
        event = make_event(self.seq)
        for listener in self._listeners:
            listener(event)

    # ------------------------------------------------------------------ attach
    def prefix_cache_lookup_enabled(self, request: Request) -> bool:
        return request.prefix_cache_hit_limit != 0

    def get_computed_blocks(
        self, request: Request, max_cache_hit_length: int | None = None
    ) -> tuple[KVCacheBlocks, int, int]:
        self._pending.pop(request.request_id, None)
        binding = request.session_binding
        if not self.prefix_cache_lookup_enabled(request) or binding is None:
            return self.empty_kv_cache_blocks, 0, 0
        if binding.seq < self._publish_seq and self._rebind is not None:
            # Only a new node can offer a longer match.
            binding = request.session_binding = self._rebind(request)
        if binding.epoch != self.epoch:
            self.num_stale_bindings += 1
            return self.empty_kv_cache_blocks, 0, 0

        limit = request.get_max_cache_hit_length(max_cache_hit_length)
        for candidate in binding.candidates:
            found = self._resume_point(
                self.nodes.get(candidate.node_id), min(candidate.length, limit)
            )
            if found is not None:
                break
        else:
            if binding.candidates:
                self.num_stale_bindings += 1
            return self.empty_kv_cache_blocks, 0, 0

        node, length = found
        path = self._path(node)
        self._pending[request.request_id] = (path, length)
        blocks = tuple(
            self._assemble(path, length, group) for group in range(len(self.block_sizes))
        )
        return self.create_kv_cache_blocks(blocks), length, 0

    def _resume_point(self, node: _Node | None, length: int) -> tuple[_Node, int] | None:
        # A truncated or partially matching node falls back to its ancestors.
        while node is not None and length > 0:
            resume = node.resume_length(length, self.resume_alignment)
            if resume:
                return node, resume
            length = min(length, node.start)
            node = node.parent
        return None

    @staticmethod
    def _path(node: _Node) -> list[_Node]:
        path = []
        current: _Node | None = node
        while current is not None:
            path.append(current)
            current = current.parent
        return path

    def _assemble(self, path: list[_Node], length: int, group: int) -> list[KVCacheBlock]:
        """Block table for positions `[0, length)`; deeper nodes win where a
        child's first block re-covers its parent's last one."""
        block_size = self.block_sizes[group]
        upper = cdiv(length, block_size)
        segments = []
        for node in path:
            lower = node.start // block_size
            if lower < upper:
                segments.append(node.blocks[group][: upper - lower])
                upper = lower
        assert upper == 0, "path does not cover the resumed prefix"
        return [block for segment in reversed(segments) for block in segment]

    # ------------------------------------------------------------- allocation
    def allocate_slots(
        self, request: Request, num_new_tokens: int, *args: Any, **kwargs: Any
    ) -> KVCacheBlocks | None:
        new_blocks = super().allocate_slots(request, num_new_tokens, *args, **kwargs)
        if new_blocks is None:
            return None
        num_scheduled_end = request.num_computed_tokens + num_new_tokens
        pending = self._pending.pop(request.request_id, None)
        if pending is not None:
            path, length = pending
            num_scheduled_end += length
            self._tip[request.request_id] = (path[0], length)
            for node in path:
                self.nodes.move_to_end(node.node_id)
            self.num_attached_requests += 1
            self.num_attached_tokens += length
        if (
            not self.commit_points_only
            and request.request_id not in self._prompt_node
            and num_scheduled_end >= request.num_prompt_tokens
        ):
            # Like prefix caching, share the prompt from the step that computes
            # it: each attention layer writes every token's KV before any
            # request in the batch reads it. Recurrent state has no such
            # ordering, so hybrid models only publish finished requests.
            node = self._publish(request, request.num_prompt_tokens)
            if node is not None:
                self._prompt_node[request.request_id] = node
        return new_blocks

    def _publish(self, request: Request, end: int) -> _Node | None:
        parent, start = self._tip.get(request.request_id, (None, 0))
        if parent is not None and (
            parent.node_id not in self.nodes or parent.end < start
        ):
            # The request still holds every block of its own path.
            parent, start = None, 0
        if end <= start:
            return parent if end == start else None
        managers = self.coordinator.single_type_managers
        blocks = tuple(
            list(
                m.req_to_blocks[request.request_id][
                    start // m.block_size : cdiv(end, m.block_size)
                ]
            )
            for m in managers
        )
        if any(block.is_null for group in blocks for block in group):
            return None
        self.block_pool.touch([b for group in blocks for b in group])
        commit_points = (end,) if self.commit_points_only else None
        node = _Node(self._next_node_id, parent, start, end, blocks, commit_points)
        self._next_node_id += 1
        self.nodes[node.node_id] = node
        if parent is not None:
            parent.num_children += 1
        self._tip[request.request_id] = (node, end)
        self._publish_seq = self.seq + 1
        tokens = pack_tokens(request.all_token_ids[start:end])
        self._emit(
            lambda seq: NodePublished(
                seq,
                node.node_id,
                parent.node_id if parent is not None else None,
                start,
                end,
                tokens,
                commit_points,
                request.request_id,
            )
        )
        return node

    def free(self, request: Request) -> None:
        request_id = request.request_id
        self._pending.pop(request_id, None)
        if request.is_finished():
            committed = None
            end = request.num_computed_tokens
            prompt_node = self._prompt_node.pop(request_id, None)
            if request.status in _PUBLISHED and 0 < end <= request.num_tokens:
                if (
                    prompt_node is not None
                    and prompt_node.node_id in self.nodes
                    and not prompt_node.num_children
                    and self._tip.get(request_id, (None, 0))[0] is prompt_node
                ):
                    # One node per turn: republish the whole turn, then drop
                    # the prompt-only node it supersedes.
                    parent = prompt_node.parent
                    self._tip[request_id] = (parent, prompt_node.start)
                    committed = self._publish(request, end)
                    self._remove(prompt_node)
                else:
                    committed = self._publish(request, end)
            self._tip.pop(request_id, None)
            node_id = (
                committed.node_id
                if committed is not None and request.status in _COMMITTED
                else None
            )
            self._emit(lambda seq: RequestFinished(seq, request_id, node_id))
        super().free(request)

    # ---------------------------------------------------------------- eviction
    def _ensure_free_blocks(self, num_blocks: int) -> bool:
        """Evict idle states only if that makes room for `num_blocks`."""
        deficit = num_blocks - self.block_pool.get_num_free_blocks()
        if deficit <= 0:
            return True
        if self._num_reclaimable_blocks() < deficit:
            return False
        while self.block_pool.get_num_free_blocks() < num_blocks and self._evict():
            pass
        return self.block_pool.get_num_free_blocks() >= num_blocks

    def _protected(self) -> set[int]:
        protected = {
            node.node_id for path, _ in self._pending.values() for node in path
        }
        protected.update(
            node.node_id for node, _ in self._tip.values() if node is not None
        )
        return protected

    def _num_reclaimable_blocks(self) -> int:
        """Blocks only idle states hold (a lower bound: blocks two idle states
        share are not counted)."""
        protected = self._protected()
        return len(
            {
                block.block_id
                for node in self.nodes.values()
                if node.node_id not in protected
                for group in node.blocks
                for block in group
                if block.ref_cnt == 1
            }
        )

    def _evict(self) -> bool:
        """Free the least recently used leaf that has blocks of its own."""
        protected = self._protected()
        idle_leaves = [
            node
            for node in self.nodes.values()
            if not node.num_children and node.node_id not in protected
        ]
        for node in idle_leaves:
            if self._trim(node):
                return True
        if idle_leaves:
            # Every idle leaf only shares blocks; dropping one exposes its
            # parent as the next leaf.
            self._remove(idle_leaves[0])
            return True
        return False

    def _trim(self, node: _Node) -> int:
        if self.commit_points_only or len(self.block_sizes) > 1:
            if not any(b.ref_cnt == 1 for group in node.blocks for b in group):
                return 0
            freed = sum(b.ref_cnt == 1 for group in node.blocks for b in group)
            self._remove(node)
            return freed
        blocks = node.blocks[0]
        keep = len(blocks)
        while keep and blocks[keep - 1].ref_cnt == 1:
            keep -= 1
        freed = len(blocks) - keep
        if not freed:
            return 0
        block_size = self.block_sizes[0]
        end = min(node.end, (node.start // block_size + keep) * block_size)
        if end <= node.start:
            self._remove(node)
            return freed
        self.block_pool.free_blocks(reversed(blocks[keep:]))
        del blocks[keep:]
        node.end = end
        self.num_evicted_blocks += freed
        self._emit(lambda seq: NodeTruncated(seq, node.node_id, end, None))
        return freed

    def _remove(self, node: _Node) -> None:
        del self.nodes[node.node_id]
        if node.parent is not None:
            node.parent.num_children -= 1
        freed = sum(b.ref_cnt == 1 for group in node.blocks for b in group)
        self.num_evicted_blocks += freed
        for group in node.blocks:
            self.block_pool.free_blocks(reversed(group))
        self._emit(lambda seq: NodeTruncated(seq, node.node_id, node.start, None))

    def reset_prefix_cache(self) -> bool:
        for node in reversed(self.nodes.values()):
            for group in node.blocks:
                self.block_pool.free_blocks(reversed(group))
        self.nodes.clear()
        self._pending.clear()
        self._tip.clear()
        self._prompt_node.clear()
        self.epoch += 1
        self._emit(lambda seq: StoreReset(seq, self.epoch))
        return super().reset_prefix_cache()

    def session_stats(self) -> dict[str, int]:
        held = {
            b.block_id for n in self.nodes.values() for g in n.blocks for b in g
        }
        return {
            "nodes": len(self.nodes),
            "node_blocks": len(held),
            "attached_requests": self.num_attached_requests,
            "attached_tokens": self.num_attached_tokens,
            "stale_bindings": self.num_stale_bindings,
            "evicted_blocks": self.num_evicted_blocks,
        }
