# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental session-owned KV cache, without automatic prefix matching."""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from vllm.outputs import RequestError
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.token_buffer import extend_from_slice
from vllm.v1.utils import ConstantList


@dataclass
class SessionState:
    """A logical token history and its optional, exclusively owned resident KV."""

    token_ids: list[int]
    prompt_token_ids: list[int]
    cache_salt: str | None
    version: int = 0
    num_computed_tokens: int = 0
    blocks: tuple[list[KVCacheBlock], ...] = ()
    request_id: str | None = None


class SessionKVCacheManager(KVCacheManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.idle: OrderedDict[str, SessionState] = OrderedDict()
        self.eviction_policy = "whole"
        self.num_session_evictions = 0
        self.num_session_trims = 0
        self.num_session_evicted_blocks = 0

    def evict(self, session_id: str, num_blocks: int | None = None) -> None:
        session = self.idle[session_id]
        if num_blocks is not None:
            assert len(session.blocks) == 1
            blocks = session.blocks[0]
            keep = max(0, len(blocks) - num_blocks)
            self.block_pool.free_blocks(reversed(blocks[keep:]))
            del blocks[keep:]
            session.num_computed_tokens = min(
                session.num_computed_tokens,
                keep * self.coordinator.single_type_managers[0].block_size,
            )
            if keep:
                return
        self.idle.pop(session_id)
        for blocks in session.blocks:
            self.block_pool.free_blocks(reversed(blocks))
        session.blocks = ()
        session.num_computed_tokens = 0

    def allocate_slots(
        self, request: Request, num_new_tokens: int, *args: Any, **kwargs: Any
    ) -> KVCacheBlocks | None:
        while True:
            blocks = super().allocate_slots(request, num_new_tokens, *args, **kwargs)
            if blocks is not None or not self.idle:
                return blocks
            session_id = next(iter(self.idle))
            session = self.idle[session_id]
            before = sum(len(group) for group in session.blocks)
            self.evict(
                session_id,
                self._allocation_shortfall
                if self.eviction_policy == "prefix"
                else None,
            )
            self.num_session_evictions += int(not session.blocks)
            self.num_session_trims += int(bool(session.blocks))
            self.num_session_evicted_blocks += before - sum(
                len(group) for group in session.blocks
            )

    def retain(self, session_id: str, session: SessionState, request: Request) -> None:
        assert not request.num_in_flight_tokens
        session.num_computed_tokens = min(
            request.num_computed_tokens, request.num_tokens - 1
        )
        session.blocks = tuple(
            manager.pop_blocks_for_free(request.request_id)
            for manager in self.coordinator.single_type_managers
        )
        self.idle[session_id] = session

    def resume(self, session_id: str, session: SessionState, request: Request) -> None:
        self.idle.pop(session_id, None)
        if session.blocks:
            for manager, blocks in zip(
                self.coordinator.single_type_managers, session.blocks, strict=True
            ):
                assert request.request_id not in manager.req_to_blocks
                manager.req_to_blocks[request.request_id] = blocks
            request.num_computed_tokens = session.num_computed_tokens
            session.blocks = ()

    def reset_prefix_cache(self) -> bool:
        while self.idle:
            self.evict(next(iter(self.idle)))
        return super().reset_prefix_cache()

    def estimate_cached_tokens(self, request: Request) -> int:
        # No block hashes exist in this mode; do not scan the history for them.
        return 0


class SessionScheduler(Scheduler):
    """Select with `scheduler_cls`, APC and async scheduling disabled.

    `SamplingParams.extra_args["session_cache"]` contains `id` and the
    expected `version` (0 creates a session). Each successful turn increments
    the version. The prompt contains only the new token IDs. Eviction releases
    idle KV according to the selected policy; token history remains for replay.
    """

    kv_cache_manager_cls = SessionKVCacheManager
    kv_cache_manager: SessionKVCacheManager

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        config = self.vllm_config
        parallel = config.parallel_config
        if (
            self.cache_config.enable_prefix_caching
            or self.scheduler_config.async_scheduling
            or config.kv_transfer_config is not None
            or config.ec_transfer_config is not None
            or config.aux_output_config.enabled
            or config.lora_config is not None
            or config.model_config.runner_type != "generate"
            or config.model_config.is_encoder_decoder
            or config.model_config.is_multimodal_model
            or parallel.world_size != 1
            or parallel.data_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or any(
                group.manager_cls is not FullAttentionManager
                or any(type(spec) is not AttentionSpec for spec in group.distinct_specs)
                for group in self.kv_cache_config.kv_cache_groups
            )
            or (
                config.speculative_config is not None
                and config.speculative_config.method != "ngram"
            )
        ):
            raise ValueError(
                "SessionScheduler requires a single-rank, text-only full-attention "
                "model, APC and async scheduling disabled, and no connectors or "
                "LoRA. Only stateless ngram speculative decoding is supported."
            )
        if not isinstance(config.additional_config, dict):
            raise ValueError("SessionScheduler requires a dictionary additional_config")
        options = config.additional_config.get("session_cache", {})
        if not isinstance(options, dict):
            raise ValueError("session_cache options must be a dictionary")
        eviction_policy = options.get("eviction_policy", "whole")
        if eviction_policy not in ("whole", "prefix"):
            raise ValueError("session_cache.eviction_policy must be whole or prefix")
        if (
            eviction_policy == "prefix"
            and len(self.kv_cache_config.kv_cache_groups) != 1
        ):
            raise ValueError("Prefix-preserving eviction requires one KV cache group")
        self.kv_cache_manager.eviction_policy = eviction_policy
        self.max_sessions = options.get("max_sessions", 1024)
        if type(self.max_sessions) is not int or self.max_sessions <= 0:
            raise ValueError("session_cache.max_sessions must be a positive integer")
        self.sessions: dict[str, SessionState] = {}
        self.request_sessions: dict[str, str] = {}
        self.pending_errors: list[tuple[int, EngineCoreOutput]] = []

    def add_request(self, request: Request) -> None:
        params = request.sampling_params
        extra = params.extra_args if params is not None else None
        command = extra.get("session_cache") if extra else None
        if command is None:
            super().add_request(request)
            return
        try:
            self._bind_session(request, command)
        except ValueError as error:
            self.pending_errors.append(
                (
                    request.client_index,
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.ERROR,
                        stop_reason=str(error),
                        error=RequestError("validation_error", str(error)),
                    ),
                )
            )
            return
        super().add_request(request)

    def _bind_session(self, request: Request, command: dict[str, Any]) -> None:
        params = request.sampling_params
        if (
            params is None
            or params.n != 1
            or params.prompt_logprobs is not None
            or params.prompt_logprob_token_ids is not None
            or request.resumable
            or request.prompt_token_ids is None
            or request.prompt_embeds is not None
            or request.mm_features
            or request.lora_request is not None
            or request.block_hashes
        ):
            raise ValueError("Session turns require un-hashed text tokens and n=1")
        if not isinstance(command, dict) or set(command) != {"id", "version"}:
            raise ValueError("session_cache requires exactly id and version")
        session_id, version = command["id"], command["version"]
        if (
            not isinstance(session_id, str)
            or not session_id
            or type(version) is not int
        ):
            raise ValueError("session_cache requires a nonempty id and integer version")
        session = self.sessions.get(session_id)
        if request.num_prompt_tokens >= self.max_model_len:
            raise ValueError("Session input exceeds the model context limit")
        if session is None:
            if version != 0:
                raise ValueError("Unknown session; create a new session with version 0")
            if len(self.sessions) >= self.max_sessions:
                raise ValueError("Session limit reached; close an idle session first")
            session = SessionState(
                request._all_token_ids, request.prompt_token_ids, request.cache_salt
            )
        else:
            if session.request_id is not None:
                raise ValueError("Session already has an active turn")
            if version != session.version or request.cache_salt != session.cache_salt:
                raise ValueError("Session version or cache namespace mismatch")
            if len(session.token_ids) + request.num_prompt_tokens >= self.max_model_len:
                raise ValueError("Session continuation exceeds the model context limit")
            # Preserve both token buffers; only append the previous output and delta.
            extend_from_slice(
                session.prompt_token_ids,
                session.token_ids,
                len(session.prompt_token_ids),
                len(session.token_ids),
            )
            session.prompt_token_ids.extend(request.prompt_token_ids)
            session.token_ids.extend(request.prompt_token_ids)
            request.prompt_token_ids = session.prompt_token_ids
            request._all_token_ids = session.token_ids
            request.all_token_ids = ConstantList(session.token_ids)
            request.num_prompt_tokens = len(session.token_ids)
        session.request_id = request.request_id
        self.sessions[session_id] = session
        self.request_sessions[request.request_id] = session_id

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        if not super()._handle_blocked_waiting_request(request):
            return False
        session_id = self.request_sessions.get(request.request_id)
        if session_id is not None:
            self.kv_cache_manager.resume(session_id, self.sessions[session_id], request)
            if request.prefill_stats is not None and request.num_preemptions == 0:
                request.prefill_stats.set(
                    num_prompt_tokens=request.num_prompt_tokens,
                    num_local_cached_tokens=request.num_computed_tokens,
                    num_external_cached_tokens=0,
                )
        return True

    def _free_request_blocks(self, request: Request) -> None:
        session_id = self.request_sessions.get(request.request_id)
        if session_id is not None and request.is_finished():
            session = self.sessions[session_id]
            del self.request_sessions[request.request_id]
            if request.status in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            ):
                self.kv_cache_manager.retain(session_id, session, request)
                session.request_id = None
                session.version += 1
                return
            # An interrupted turn has no committed continuation contract.
            if session_id in self.kv_cache_manager.idle:
                self.kv_cache_manager.evict(session_id)
            del self.sessions[session_id]
        super()._free_request_blocks(request)

    def close_session(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        if session is None:
            return False
        if session.request_id is not None:
            raise ValueError("Cannot close a session with an active turn")
        if session_id in self.kv_cache_manager.idle:
            self.kv_cache_manager.evict(session_id)
        del self.sessions[session_id]
        return True

    def has_finished_requests(self) -> bool:
        return bool(self.pending_errors) or super().has_finished_requests()

    def update_from_output(
        self, scheduler_output: SchedulerOutput, model_runner_output: ModelRunnerOutput
    ) -> dict[int, EngineCoreOutputs]:
        outputs = super().update_from_output(scheduler_output, model_runner_output)
        for client_index, error in self.pending_errors:
            outputs.setdefault(client_index, EngineCoreOutputs()).outputs.append(error)
        self.pending_errors.clear()
        return outputs
