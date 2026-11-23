# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from typing_extensions import Buffer

from vllm.v1.token_buffer import TokenBuffer

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request


@dataclass(frozen=True, slots=True)
class _HashPrefix:
    hashes: tuple["BlockHash", ...]
    metadata_bytes: int


class TurnPrefixHashCache:
    """Memoize content hashes, independently of GPU cache residency.

    Input processing reads this index while the scheduler publishes completed
    turns. Fingerprinting runs outside the metadata lock. No token buffers or
    GPU blocks are retained.
    """

    def __init__(
        self,
        block_size: int,
        lookahead: int = 0,
        max_entries: int = 1024,
        max_bytes: int = 64 << 20,
    ) -> None:
        self.block_size = block_size
        self.lookahead = lookahead
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: OrderedDict[tuple[str | None, int, bytes], _HashPrefix] = (
            OrderedDict()
        )
        self._lengths: dict[tuple[str | None, int], int] = {}
        self._metadata_bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def supports(request: "Request | EngineCoreRequest") -> bool:
        return (
            request.prefix_match_positions is not None
            and request.sampling_params is not None
            and not request.resumable
            and not request.mm_features
            and request.prompt_embeds is None
            and request.lora_request is None
            and request.prompt_token_ids is not None
        )

    @staticmethod
    def _view(tokens) -> memoryview:
        try:
            view = memoryview(tokens)
            if view.ndim == 1 and view.format in ("i", "I") and view.contiguous:
                return view.cast("B")
        except TypeError:
            pass
        return memoryview(cast(Buffer, TokenBuffer(tokens))).cast("B")

    def lookup(
        self, request: "Request | EngineCoreRequest"
    ) -> list["BlockHash"] | None:
        if not self.supports(request) or not request.prefix_match_positions:
            return None
        tokens = request.prompt_token_ids
        assert tokens is not None
        max_count = max(0, len(tokens) - self.lookahead) // self.block_size
        counts = sorted(
            {
                min(position // self.block_size, max_count) - delta
                for position in request.prefix_match_positions[-2:]
                for delta in (0, 1)
            },
            reverse=True,
        )
        with self._lock:
            counts = [
                count
                for count in counts
                if count > 0 and (request.cache_salt, count) in self._lengths
            ]
        if not counts:
            return None
        data = self._view(tokens)
        for count in counts:
            end = count * self.block_size + self.lookahead
            digest = hashlib.sha256(data[: end * 4]).digest()
            key = (request.cache_salt, count, digest)
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
            if entry is not None:
                return list(entry.hashes)
        return None

    def register(self, request: "Request", position: int) -> None:
        if not self.supports(request) or self.max_entries <= 0:
            return
        count = min(position // self.block_size, len(request.block_hashes))
        end = count * self.block_size + self.lookahead
        if count <= 0 or end > request.num_tokens:
            return
        estimate = (
            count * (8 + sys.getsizeof(request.block_hashes[0]))
            + sys.getsizeof(request.cache_salt)
            + 512
        )
        if estimate > self.max_bytes:
            return
        data = self._view(request._all_token_ids)
        digest = hashlib.sha256(data[: end * 4]).digest()
        key = (request.cache_salt, count, digest)
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
        hashes = tuple(request.block_hashes[:count])
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return
            while self._entries and (
                len(self._entries) >= self.max_entries
                or self._metadata_bytes + estimate > self.max_bytes
            ):
                old_key, old = self._entries.popitem(last=False)
                self._metadata_bytes -= old.metadata_bytes
                length_key = old_key[:2]
                self._lengths[length_key] -= 1
                if not self._lengths[length_key]:
                    del self._lengths[length_key]
            self._entries[key] = _HashPrefix(hashes, estimate)
            self._metadata_bytes += estimate
            length_key = key[:2]
            self._lengths[length_key] = self._lengths.get(length_key, 0) + 1

    @property
    def metadata_bytes(self) -> int:
        with self._lock:
            return self._metadata_bytes

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._lengths.clear()
            self._metadata_bytes = 0
