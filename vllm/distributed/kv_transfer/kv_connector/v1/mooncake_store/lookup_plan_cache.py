# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(slots=True)
class LookupPlan:
    keys: tuple[str, ...]
    meta: tuple[tuple[int, bytes, int], ...]
    metadata_bytes: int


class LookupPlanCache:
    """Bounded immutable key construction results, never cached remote hits.

    Each worker owns one instance and its fixed Store namespaces. Every lookup
    still checks all keys with the Store, including keys returned here.
    """

    def __init__(self, max_bytes: int, max_entries: int = 1024) -> None:
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self.entries: OrderedDict[tuple[int, int, bytes], LookupPlan] = OrderedDict()
        self.metadata_bytes = 0
        self.hits = 0
        self.reused_keys = 0

    def get(
        self,
        group_id: int,
        unit_size: int,
        hash_block_size: int,
        block_hashes: Sequence[BlockHash],
        positions: Sequence[int],
        max_units: int,
    ) -> LookupPlan | None:
        candidates = dict.fromkeys(
            count
            for position in reversed(positions[-2:])
            for count in (position // unit_size - 1, position // unit_size - 2)
        )
        for count in candidates:
            if count <= 0 or count > max_units:
                continue
            hash_count = count * unit_size // hash_block_size
            key = (group_id, count, bytes(block_hashes[hash_count - 1]))
            plan = self.entries.get(key)
            if plan is not None:
                self.entries.move_to_end(key)
                self.hits += 1
                self.reused_keys += len(plan.keys)
                return plan
        return None

    def put(
        self,
        group_id: int,
        keys: list[str],
        meta: list[tuple[int, bytes, int]],
        num_namespaces: int,
    ) -> None:
        # Leave the final unit out so a changed generation header can still
        # reuse key construction. The caller continues probing every unit.
        count = len(meta) - 1
        if count <= 0 or self.max_entries <= 0:
            return
        key = (group_id, count, meta[count - 1][1])
        key_count = count * num_namespaces
        per_unit = (
            sys.getsizeof(meta[0])
            + sys.getsizeof(meta[0][1])
            + sys.getsizeof(meta[0][2])
        )
        per_key = max(sys.getsizeof(k) for k in keys[:num_namespaces])
        size = 512 + count * (per_unit + 8) + key_count * (per_key + 8)
        if size > self.max_bytes:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.metadata_bytes -= previous.metadata_bytes
        while self.entries and (
            len(self.entries) >= self.max_entries
            or self.metadata_bytes + size > self.max_bytes
        ):
            _, old = self.entries.popitem(last=False)
            self.metadata_bytes -= old.metadata_bytes
        self.entries[key] = LookupPlan(
            tuple(keys[:key_count]), tuple(meta[:count]), size
        )
        self.metadata_bytes += size

    def clear(self) -> None:
        self.entries.clear()
        self.metadata_bytes = 0
