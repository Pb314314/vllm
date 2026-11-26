# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content side of session KV reuse.

The resolver mirrors the token content of every state the KV store has
published in a radix tree, assigns each request to a session, and hands the
store candidate `(state, length)` references whose content it has compared
token by token. Each state owns every vertex on its path, so states come and go
independently. The resolver never sees physical blocks. If it lags behind the
store it can only be more optimistic (the store rejects dead references), never
wrong.
"""

import enum
import time
from array import array
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

from vllm.v1.request import Request
from vllm.v1.session.protocol import (
    Candidate,
    NodeExtended,
    NodeId,
    NodePublished,
    NodeTruncated,
    RequestFinished,
    SessionBinding,
    StoreEvent,
    StoreReset,
)

MAX_CANDIDATES = 4
_ITEMSIZE = 4


def pack_tokens(tokens: Sequence[int]) -> bytes:
    return array("i", tokens).tobytes()


def _common_prefix(a: bytes, b: bytes) -> int:
    """Number of leading tokens two packed sequences share."""
    n = min(len(a), len(b)) // _ITEMSIZE
    if a[: n * _ITEMSIZE] == b[: n * _ITEMSIZE]:
        return n
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[: mid * _ITEMSIZE] == b[: mid * _ITEMSIZE]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _token_at(tokens: bytes, pos: int) -> int:
    return int.from_bytes(
        tokens[pos * _ITEMSIZE : (pos + 1) * _ITEMSIZE], "little", signed=True
    )


class _Vertex:
    """A radix edge and the vertex it leads to. Covers path positions
    `[depth - len(label), depth)`."""

    __slots__ = ("label", "depth", "parent", "children", "owners", "ends")

    def __init__(self, label: bytes, depth: int, parent: "_Vertex | None") -> None:
        self.label = label
        self.depth = depth
        self.parent = parent
        self.children: dict[int, _Vertex] = {}
        # States whose path reaches into this span, oldest first.
        self.owners: dict[NodeId, None] = {}
        # Those of them that end inside it.
        self.ends: dict[NodeId, None] = {}

    @property
    def start(self) -> int:
        return self.depth - len(self.label) // _ITEMSIZE


@dataclass
class _Node:
    end: int
    commit_points: tuple[int, ...] | None
    namespace: Hashable
    # The vertex the state ends in.
    terminal: _Vertex | None = None


class SessionState(enum.Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass
class Session:
    session_id: int
    namespace: Hashable
    cursor: NodeId | None = None
    state: SessionState = SessionState.IDLE
    writer: str | None = None
    client_id: str | None = None


class SessionResolver:
    """Single-threaded; colocated with the engine core in this version."""

    def __init__(self) -> None:
        self.epoch = 0
        self.seq = 0
        self.roots: dict[Hashable, _Vertex] = {}
        self.nodes: dict[NodeId, _Node] = {}
        self.sessions: dict[int, Session] = {}
        self.cursor_owner: dict[NodeId, int] = {}
        self.aliases: dict[tuple[Hashable, str], int] = {}
        # States published for each client session id, so a caller can
        # release a session it knows has ended.
        self.client_nodes: dict[str, set[NodeId]] = {}
        self.node_client: dict[NodeId, str] = {}
        self.request_session: dict[str, int] = {}
        self._packed: dict[str, bytes] = {}
        self._next_session_id = 1
        # Whether states resume at any position (attention-only models) rather
        # than only at their commit points.
        self.resume_anywhere = False
        self.num_resolves = 0
        self.num_continued = 0
        self.resolve_seconds = 0.0

    @staticmethod
    def namespace(request: Request) -> Hashable:
        lora = request.lora_request.lora_name if request.lora_request else None
        return (request.cache_salt, lora)

    # ------------------------------------------------------------------ resolve
    def resolve(self, request: Request) -> SessionBinding:
        start_time = time.perf_counter()
        namespace = self.namespace(request)
        tokens = self._packed_tokens(request)
        limit = request.get_max_cache_hit_length()
        path, matched = self._match(namespace, tokens, limit)
        candidates = self._candidates(path, matched)

        session_id = self.request_session.get(request.request_id)
        if session_id is None:
            session_id = self._assign_session(request, namespace, candidates)
        self.resolve_seconds += time.perf_counter() - start_time
        self.num_resolves += 1
        return SessionBinding(self.epoch, session_id, candidates, self.seq, matched)

    def _packed_tokens(self, request: Request) -> bytes:
        """Pack a request's tokens once; re-resolves only pack new tokens."""
        tokens = request.all_token_ids
        packed = self._packed.get(request.request_id, b"")
        num_packed = len(packed) // _ITEMSIZE
        if num_packed > len(tokens):
            packed, num_packed = b"", 0
        if num_packed < len(tokens):
            packed += pack_tokens(tokens[num_packed:])
            self._packed[request.request_id] = packed
        return packed

    def _match(
        self, namespace: Hashable, tokens: bytes, limit: int
    ) -> tuple[list[_Vertex], int]:
        """Vertices along the longest match and the matched length."""
        vertex = self.roots.get(namespace)
        path: list[_Vertex] = []
        matched = 0
        while vertex is not None and matched < limit:
            child = vertex.children.get(_token_at(tokens, matched))
            if child is None:
                break
            n = _common_prefix(child.label, tokens[matched * _ITEMSIZE :])
            path.append(child)
            matched = min(matched + n, limit)
            if n < len(child.label) // _ITEMSIZE:
                break
            vertex = child
        return path, matched

    def _candidates(self, path: list[_Vertex], matched: int) -> tuple[Candidate, ...]:
        found: dict[NodeId, int] = {}
        for vertex in reversed(path):
            reach = min(vertex.depth, matched)
            if self.resume_anywhere:
                # Newest first, the states reaching `reach`; only the vertex's
                # few ending states are skipped on the way.
                for node_id in reversed(vertex.owners):
                    if len(found) >= MAX_CANDIDATES:
                        break
                    if node_id not in found and self.nodes[node_id].end >= reach:
                        found[node_id] = reach
            # Commit points of other states lie where those end.
            for node_id in reversed(vertex.ends):
                if len(found) >= MAX_CANDIDATES:
                    break
                if node_id not in found:
                    length = self._resume_length(self.nodes[node_id], reach)
                    if length:
                        found[node_id] = length
            if len(found) >= MAX_CANDIDATES:
                break
        ranked = sorted(found.items(), key=lambda kv: (-kv[1], -kv[0].local))
        return tuple(Candidate(n, length) for n, length in ranked[:MAX_CANDIDATES])

    @staticmethod
    def _resume_length(node: _Node, reach: int) -> int:
        if node.commit_points is None:
            return min(reach, node.end)
        return max((c for c in node.commit_points if c <= reach), default=0)

    def _assign_session(
        self,
        request: Request,
        namespace: Hashable,
        candidates: tuple[Candidate, ...],
    ) -> int:
        client_id = request.session_id
        preferred = (
            self.aliases.get((namespace, client_id)) if client_id is not None else None
        )
        # Continue the session whose whole cursor state this request extends;
        # candidates are longest first.
        chosen: Session | None = None
        for cand in candidates:
            if (
                chosen is not None
                and chosen.cursor is not None
                and cand.length < self.nodes[chosen.cursor].end
            ):
                break
            owner = self.cursor_owner.get(cand.node_id)
            if owner is None or cand.length != self.nodes[cand.node_id].end:
                continue
            session = self.sessions[owner]
            if session.state != SessionState.IDLE:
                continue
            if chosen is None or owner == preferred:
                chosen = session
        if chosen is None:
            chosen = Session(self._next_session_id, namespace, client_id=client_id)
            self._next_session_id += 1
            self.sessions[chosen.session_id] = chosen
        else:
            self.num_continued += 1
        if client_id is not None:
            self.aliases[namespace, client_id] = chosen.session_id
        chosen.state = SessionState.ACTIVE
        chosen.writer = request.request_id
        self.request_session[request.request_id] = chosen.session_id
        return chosen.session_id

    # ------------------------------------------------------------------- events
    def apply(self, event: StoreEvent) -> None:
        assert event.seq == self.seq + 1, (event.seq, self.seq)
        self.seq = event.seq
        if isinstance(event, NodePublished):
            self._insert(event)
        elif isinstance(event, NodeExtended):
            self._extend(event)
        elif isinstance(event, NodeTruncated):
            self._truncate(event.node_id, event.end, event.commit_points)
        elif isinstance(event, RequestFinished):
            self._finish(event.request_id, event.node_id)
        elif isinstance(event, StoreReset):
            self._reset(event.epoch)

    def _insert(self, event: NodePublished) -> None:
        request_session = self.request_session.get(event.request_id)
        namespace = (
            self.sessions[request_session].namespace
            if request_session is not None
            else None
        )
        self.nodes[event.node_id] = _Node(event.end, event.commit_points, namespace)
        self.resume_anywhere = event.commit_points is None
        client_id = (
            self.sessions[request_session].client_id
            if request_session is not None
            else None
        )
        if client_id is not None:
            self.node_client[event.node_id] = client_id
            self.client_nodes.setdefault(client_id, set()).add(event.node_id)
        self._add_path(event.node_id, event.tokens)

    def _extend(self, event: NodeExtended) -> None:
        node = self.nodes[event.node_id]
        assert node.end == event.start
        if event.commit_points is not None:
            node.commit_points = event.commit_points
        if event.end == event.start:
            return
        # Splits during the insert see the old end: a path that leaves an edge
        # where the state used to end gives up the rest of that edge.
        self._add_path(event.node_id, self._node_path(node) + event.tokens)
        node.end = event.end

    @staticmethod
    def _node_path(node: _Node) -> bytes:
        parts: list[bytes] = []
        vertex = node.terminal
        while vertex is not None:
            parts.append(vertex.label)
            vertex = vertex.parent
        return b"".join(reversed(parts))[: node.end * _ITEMSIZE]

    def _add_path(self, node_id: NodeId, path: bytes) -> None:
        """Insert `path` and make `node_id` an owner of every vertex on it."""
        node = self.nodes[node_id]
        end = len(path) // _ITEMSIZE
        vertex = self.roots.setdefault(node.namespace, _Vertex(b"", 0, None))
        depth = 0
        while depth < end:
            first = _token_at(path, depth)
            child = vertex.children.get(first)
            if child is None:
                child = _Vertex(path[depth * _ITEMSIZE :], end, vertex)
                vertex.children[first] = child
            else:
                cut = _common_prefix(child.label, path[depth * _ITEMSIZE :])
                if cut < len(child.label) // _ITEMSIZE and depth + cut < end:
                    child = self._split(child, cut)
            child.owners[node_id] = None
            vertex, depth = child, child.depth
        if node.terminal is not None:
            node.terminal.ends.pop(node_id, None)
        node.terminal = vertex
        vertex.ends[node_id] = None

    def _split(self, vertex: _Vertex, cut: int) -> _Vertex:
        """Split `vertex` after `cut` tokens; return the upper half."""
        assert vertex.parent is not None and cut > 0
        upper = _Vertex(vertex.label[: cut * _ITEMSIZE], vertex.start + cut, None)
        upper.parent = vertex.parent
        upper.owners = dict(vertex.owners)
        vertex.parent.children[_token_at(vertex.label, 0)] = upper
        vertex.label = vertex.label[cut * _ITEMSIZE :]
        vertex.parent = upper
        upper.children[_token_at(vertex.label, 0)] = vertex
        for node_id in list(vertex.ends):
            node = self.nodes[node_id]
            if node.end <= upper.depth:
                del vertex.ends[node_id]
                del vertex.owners[node_id]
                upper.ends[node_id] = None
                node.terminal = upper
        return upper

    def _truncate(
        self, node_id: NodeId, end: int, commit_points: tuple[int, ...] | None
    ) -> None:
        node = self.nodes.get(node_id)
        if node is None:
            return
        vertex = node.terminal
        assert vertex is not None
        vertex.ends.pop(node_id, None)
        while vertex.parent is not None and vertex.start >= end:
            vertex.owners.pop(node_id, None)
            parent = vertex.parent
            self._prune(vertex)
            vertex = parent
        node.end = min(node.end, end)
        node.commit_points = commit_points
        if end > 0:
            node.terminal = vertex
            vertex.ends[node_id] = None
            return
        del self.nodes[node_id]
        client_id = self.node_client.pop(node_id, None)
        if client_id is not None:
            published = self.client_nodes[client_id]
            published.discard(node_id)
            if not published:
                del self.client_nodes[client_id]
        session_id = self.cursor_owner.pop(node_id, None)
        if session_id is not None:
            self._retire_cursor(self.sessions[session_id])

    def _prune(self, vertex: _Vertex) -> None:
        while vertex.parent is not None and not vertex.owners and not vertex.children:
            parent = vertex.parent
            del parent.children[_token_at(vertex.label, 0)]
            vertex = parent

    def _retire_cursor(self, session: Session) -> None:
        session.cursor = None
        if session.state == SessionState.IDLE:
            # Nothing left to continue; a new request starts a new session.
            self._drop_session(session)

    def _finish(self, request_id: str, node_id: NodeId | None) -> None:
        self._packed.pop(request_id, None)
        session_id = self.request_session.pop(request_id, None)
        if session_id is None:
            return
        session = self.sessions[session_id]
        if session.writer != request_id:
            return
        session.writer = None
        session.state = SessionState.IDLE
        if node_id is not None and node_id in self.nodes:
            if session.cursor is not None:
                self.cursor_owner.pop(session.cursor, None)
            previous = self.cursor_owner.get(node_id)
            if previous is not None and previous != session_id:
                self._retire_cursor(self.sessions[previous])
            session.cursor = node_id
            self.cursor_owner[node_id] = session_id
        elif session.cursor is None:
            self._drop_session(session)

    def _drop_session(self, session: Session) -> None:
        del self.sessions[session.session_id]
        if session.client_id is not None:
            key = (session.namespace, session.client_id)
            if self.aliases.get(key) == session.session_id:
                del self.aliases[key]

    def release(self, client_ids: list[str]) -> list[NodeId]:
        """Forget the sessions a caller names as ended and return the states
        they published, for the store to free."""
        released: set[NodeId] = set()
        for client_id in client_ids:
            released |= self.client_nodes.pop(client_id, set())
            for session in list(self.sessions.values()):
                if session.client_id != client_id:
                    continue
                self.aliases.pop((session.namespace, client_id), None)
                session.client_id = None
                if session.state == SessionState.IDLE:
                    if session.cursor is not None:
                        self.cursor_owner.pop(session.cursor, None)
                    self._drop_session(session)
        for node_id in released:
            self.node_client.pop(node_id, None)
        return list(released)

    def _reset(self, epoch: int) -> None:
        self.epoch = epoch
        self.roots.clear()
        self.nodes.clear()
        self.client_nodes.clear()
        self.node_client.clear()
        self.cursor_owner.clear()
        for session in list(self.sessions.values()):
            session.cursor = None
            if session.state == SessionState.IDLE:
                self._drop_session(session)

    def stats(self) -> dict[str, float]:
        return {
            "resolves": self.num_resolves,
            "continued_sessions": self.num_continued,
            "resolve_seconds": self.resolve_seconds,
            "nodes": len(self.nodes),
            "sessions": len(self.sessions),
        }
