# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.inputs.engine import EngineInput
from vllm.tokenizers import TokenizerLike

# Tokenizer wire formats, not model identities. These produce advisory lookup
# positions only; the engine still verifies the complete prefix and cache state.
_ASSISTANT_MARKERS = (
    ("<|im_start|>", "<|im_start|>assistant\n", "<|im_end|>"),
    ("<|im_assistant|>", "<|im_assistant|>assistant<|im_middle|>", "<|im_end|>"),
    ("]~b]", "]~b]ai\n", "[e~["),
    ("<｜Assistant｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>"),
)


class ChatTurnBoundaries:
    def __init__(self, tokenizer: TokenizerLike, template: str | None) -> None:
        vocab = tokenizer.get_vocab()
        markers = next(
            (m for m in _ASSISTANT_MARKERS if m[0] in vocab and m[2] in vocab),
            None,
        )
        self.supported = markers is not None
        self.deepseek = markers is not None and markers[0] == "<｜Assistant｜>"
        self.delimiters: tuple[str, ...] = ()
        self.end_token = None
        self.assistant_start = []
        self.generation_suffixes: list[list[int]] = []
        if markers is None:
            return
        marker, start, end = markers
        self.delimiters = (marker, end)
        self.end_token = vocab[end]
        self.assistant_start = tokenizer.encode(start, add_special_tokens=False)
        self.generation_suffixes = sorted(
            (
                tokenizer.encode(start + thinking, add_special_tokens=False)
                for thinking in ("", "<think>", "<think>\n", "</think>")
            ),
            key=len,
            reverse=True,
        )

    def add_hint(
        self, request: ChatCompletionRequest, engine_input: EngineInput
    ) -> None:
        if (
            not self.supported
            or engine_input["type"] != "token"
            or request.chat_template is not None
            or request.chat_template_kwargs
            or request.use_beam_search
            or request.truncate_prompt_tokens is not None
            or request.continue_final_message
            or not request.add_generation_prompt
        ):
            return
        messages = request.messages
        if not messages:
            return
        # Delimiters inside content may suggest an unhelpful boundary. Hints
        # never authorize reuse: the engine verifies the entire prefix hash.
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
        if len(messages) < 2 or self.end_token is None:
            return
        header_length = len(self.assistant_start)
        stop = len(tokens) - generation_length
        window = 128
        while stop > 0:
            chunk_start = max(0, stop - window)
            cursor = chunk_start
            header = None
            while cursor < stop:
                try:
                    pos = tokens.index(self.assistant_start[0], cursor, stop)
                except ValueError:
                    break
                if tokens[pos : pos + header_length] == self.assistant_start:
                    header = pos
                cursor = pos + 1
            if header is None:
                stop = chunk_start
                window = min(window * 2, 4096)
                continue
            start = header + header_length
            # The previous prompt is also useful when reasoning is rewritten
            # or a trace replays recorded rather than generated assistant text.
            start += generation_length - header_length
            try:
                end = tokens.index(
                    self.end_token, start, len(tokens) - generation_length
                )
            except ValueError:
                return
            engine_input["prefix_match_positions"] = (start, end)
            return
