# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.inputs.engine import EngineInput
from vllm.tokenizers import TokenizerLike

# Qwen2.5-Instruct's shipped template. New templates need round-trip tests;
# recognizing a model name or a delimiter alone is not sufficient.
_QWEN25_TEMPLATE_SHA256 = (
    "cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f"
)
_MINIMAX31_TEMPLATE_SHA256 = (
    "961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431"
)


class ChatTurnBoundaries:
    def __init__(self, tokenizer: TokenizerLike, template: str | None) -> None:
        template = template or getattr(tokenizer, "chat_template", None)
        digest = (
            hashlib.sha256(template.encode()).hexdigest()
            if isinstance(template, str)
            else None
        )
        # Renderer pooling wraps public methods but keeps the native encoder
        # in the MRO. Ignore only vLLM's own transparent HF wrappers.
        template_method = next(
            (
                cls.__dict__["apply_chat_template"]
                for cls in type(tokenizer).__mro__
                if "apply_chat_template" in cls.__dict__
                and cls.__module__ != "vllm.tokenizers.hf"
            ),
            None,
        )
        self.deepseek = (
            getattr(template_method, "__module__", None)
            == "vllm.tokenizers.deepseek_v41"
        )
        self.supported = self.deepseek or digest in (
            _QWEN25_TEMPLATE_SHA256,
            _MINIMAX31_TEMPLATE_SHA256,
        )
        self.delimiters: tuple[str, ...] = ("<|im_start|>", "<|im_end|>")
        start, end = "<|im_start|>assistant\n", "<|im_end|>"
        if digest == _MINIMAX31_TEMPLATE_SHA256:
            start, end = "]~b]ai\n", "[e~["
            self.delimiters = ("]~b]", "[e~[")
        elif self.deepseek:
            start, end = "<｜Assistant｜>", "<｜end▁of▁sentence｜>"
            self.delimiters = (start, end, "<｜User｜>")
        self.end_token = None
        self.assistant_start = []
        if self.supported:
            self.end_token = tokenizer.convert_tokens_to_ids(end)
            self.assistant_start = tokenizer.encode(start, add_special_tokens=False)
        self.generation_suffixes = [self.assistant_start]
        if self.deepseek:
            self.generation_suffixes = [
                tokenizer.encode(start + thinking, add_special_tokens=False)
                for thinking in ("<think>", "</think>")
            ]

    def add_hint(
        self, request: ChatCompletionRequest, engine_input: EngineInput
    ) -> None:
        if (
            not self.supported
            or engine_input["type"] != "token"
            or request.chat_template is not None
            or request.chat_template_kwargs
            or request.tools
            or request.use_beam_search
            or request.truncate_prompt_tokens is not None
            or request.continue_final_message
            or not request.add_generation_prompt
        ):
            return
        messages = request.messages
        if not messages or messages[-1]["role"] != "user":
            return
        if any(
            m["role"] not in ("system", "user", "assistant")
            or not isinstance(m.get("content"), str)
            or any(
                k not in ("role", "content", "reasoning", "reasoning_content")
                for k in m
            )
            or any(delimiter in m["content"] for delimiter in self.delimiters)
            for m in messages
        ):
            return
        tokens = engine_input["prompt_token_ids"]
        generation_length = next(
            (
                len(suffix)
                for suffix in self.generation_suffixes
                if suffix and tokens[-len(suffix) :] == suffix
            ),
            None,
        )
        if generation_length is None:
            return
        # Empty hints mark eligible first turns for checkpoint registration.
        if not any(m["role"] == "assistant" for m in messages):
            engine_input["prefix_match_positions"] = ()
            return
        if len(messages) < 2 or messages[-2]["role"] != "assistant":
            return
        header_length = len(self.assistant_start)
        for pos in range(len(tokens) - generation_length - 1, -1, -1):
            if tokens[pos] != self.assistant_start[0]:
                continue
            if tokens[pos : pos + header_length] != self.assistant_start:
                continue
            start = pos + header_length
            # The previous prompt is also useful when reasoning is rewritten
            # or a trace replays recorded rather than generated assistant text.
            if self.deepseek:
                start += 1
            for end in range(start, len(tokens) - generation_length):
                if tokens[end] == self.end_token:
                    engine_input["prefix_match_positions"] = (start, end)
                    return
            return
