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


class ChatTurnBoundaries:
    def __init__(self, tokenizer: TokenizerLike, template: str | None) -> None:
        template = template or getattr(tokenizer, "chat_template", None)
        self.supported = (
            isinstance(template, str)
            and hashlib.sha256(template.encode()).hexdigest() == _QWEN25_TEMPLATE_SHA256
        )
        self.end_token = None
        self.assistant_start = []
        if self.supported:
            self.end_token = tokenizer.convert_tokens_to_ids("<|im_end|>")
            self.assistant_start = tokenizer.encode(
                "<|im_start|>assistant\n", add_special_tokens=False
            )

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
            or any(k not in ("role", "content") for k in m)
            or "<|im_start|>" in m["content"]
            or "<|im_end|>" in m["content"]
            for m in messages
        ):
            return
        tokens = engine_input["prompt_token_ids"]
        if tokens[-len(self.assistant_start) :] != self.assistant_start:
            return
        # Empty hints mark eligible first turns for checkpoint registration.
        if not any(m["role"] == "assistant" for m in messages):
            engine_input["prefix_match_positions"] = ()
            return
        if len(messages) < 2 or messages[-2]["role"] != "assistant":
            return
        ends = 0
        for pos in range(len(tokens) - 1, -1, -1):
            if tokens[pos] == self.end_token:
                ends += 1
                if ends == 2:
                    # Position before the previous assistant's closing marker.
                    engine_input["prefix_match_positions"] = (pos,)
                    return
