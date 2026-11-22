# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

from tests.utils import RemoteOpenAIServer

# any model with a chat template defined in tokenizer_config should work here
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def test_turn_prefix_cache_real_chat_round_trip():
    """Ordinary Chat API requests take the shortcut without changing APC hits."""
    import copy
    import time

    import requests

    from tests.models.utils import check_logprobs_close

    history = [
        {"role": "system", "content": "You are helpful. " * 160},
        {"role": "user", "content": "Write a list of twenty different animals."},
    ]
    replay: list[list[dict[str, str]]] = []
    reference: list[tuple[list[str], str, list[dict[str, float]]]] = []
    reference_hits: list[int] = []
    reference_tokens: list[list[int]] = []
    args = [
        "--enforce-eager",
        "--attention-backend",
        "TRITON_ATTN",
        "--max-model-len",
        "4096",
        "--max-num-batched-tokens",
        "256",
        "--max-num-seqs",
        "8",
        "--kv-cache-memory-bytes",
        str(1 << 30),
        "--enable-prefix-caching",
        "--enable-prompt-tokens-details",
    ]
    for enabled in (False, True):
        with (
            RemoteOpenAIServer(
                MODEL_NAME,
                args,
                env_dict={
                    "VLLM_TURN_PREFIX_CACHE": str(int(enabled)),
                    "VLLM_USE_RUST_FRONTEND": "0",
                    "VLLM_SERVER_DEV_MODE": "1",
                },
            ) as server,
            server.get_client() as client,
        ):
            for turn in range(6):
                if enabled:
                    messages = replay[turn]
                elif turn < 3:
                    messages = copy.deepcopy(history)
                elif turn == 3:
                    messages = copy.deepcopy(replay[1])
                    messages[-1]["content"] = "Name some birds instead."
                elif turn == 4:
                    messages = [
                        history[0],
                        {
                            "role": "user",
                            "content": "Summary: we listed animals. Go on.",
                        },
                    ]
                else:
                    messages = copy.deepcopy(replay[2])
                    messages[2]["content"] = "A revised answer about birds."
                token_response = requests.post(
                    server.url_for("tokenize"),
                    json={
                        "model": MODEL_NAME,
                        "messages": messages,
                        "add_generation_prompt": True,
                    },
                )
                token_response.raise_for_status()
                tokens = token_response.json()["tokens"]
                stream = turn == 1
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0,
                    max_tokens=32,
                    logprobs=True,
                    top_logprobs=5,
                    stream=stream,
                    stream_options={"include_usage": True} if stream else None,
                    extra_body={"ignore_eos": True},
                )
                if stream:
                    text, probs, usage = "", [], None
                    for chunk in response:
                        if chunk.usage is not None:
                            usage = chunk.usage
                        for choice in chunk.choices:
                            text += choice.delta.content or ""
                            if choice.logprobs and choice.logprobs.content:
                                probs.extend(choice.logprobs.content)
                else:
                    text = response.choices[0].message.content
                    probs = response.choices[0].logprobs.content
                    usage = response.usage
                assert text and probs and usage is not None
                output = (
                    [p.token for p in probs],
                    text,
                    [{p.token: p.logprob for p in item.top_logprobs} for item in probs],
                )
                cached = usage.prompt_tokens_details.cached_tokens
                if enabled:
                    assert tokens == reference_tokens[turn]
                    assert cached == reference_hits[turn]
                    check_logprobs_close(
                        outputs_0_lst=[reference[turn]],
                        outputs_1_lst=[output],
                        name_0="APC",
                        name_1="turn lookup",
                        always_check_logprobs=True,
                    )
                else:
                    replay.append(messages)
                    reference.append(output)
                    reference_hits.append(cached)
                    reference_tokens.append(tokens)
                    history += [
                        {"role": "assistant", "content": text},
                        {"role": "user", "content": "Continue with more animals."},
                    ]

            # Text/token callers carry no hints and still get ordinary APC.
            completion = client.completions.create(
                model=MODEL_NAME,
                prompt=tokens,
                max_tokens=1,
                temperature=0,
            )
            assert completion.usage.prompt_tokens_details.cached_tokens > 0

            if enabled:
                deadline = time.monotonic() + 15
                while True:
                    metrics = requests.get(server.url_for("metrics")).text
                    hits = sum(
                        float(line.rsplit(" ", 1)[-1])
                        for line in metrics.splitlines()
                        if line.startswith("vllm:turn_prefix_cache_hits_total{")
                    )
                    if hits >= 3:
                        break
                    assert time.monotonic() < deadline, metrics
                    time.sleep(0.05)
                assert "vllm:turn_prefix_cache_skipped_block_lookups_total" in metrics
                reset = requests.post(server.url_for("reset_prefix_cache"))
                reset.raise_for_status()
                cold = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=replay[-1],
                    max_tokens=1,
                    temperature=0,
                )
                assert cold.usage.prompt_tokens_details.cached_tokens == 0


@pytest.fixture(scope="module")
def default_server_args():
    return [
        # use half precision for speed and memory savings in CI environment
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enforce-eager",
        "--enable-per-request-metrics",
    ]


@pytest.fixture(scope="module")
def server(default_server_args):
    with RemoteOpenAIServer(MODEL_NAME, default_server_args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_json_schema(client: openai.AsyncOpenAI, model_name: str) -> None:
    invalid_json_schema = {
        "$defs": {
            "CarType": {
                "enum": ["sedan", "SUV", "Truck", "Coupe"],
                "title": "CarType",
                "type": "string",
            }
        },
        "properties": {
            "brand": {"title": "Brand", "type": "string"},
            "model": {"title": "Model", "type": "string"},
            "car_type": {"$ref": "#/$defs/CarType"},
            "foo": "bar",
        },
        "required": ["brand", "model", "car_type"],
        "title": "CarDescription",
        "type": "object",
    }
    prompt = (
        "Generate a JSON with the brand, model and car_type of"
        "the most iconic car from the 90's"
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"json": invalid_json_schema}},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_regex(client: openai.AsyncOpenAI, model_name: str):
    prompt = (
        "Generate an email address for Alan Turing, who works in Enigma."
        "End in .com and new line. Example result:"
        "alan.turing@enigma.com\n"
    )

    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"regex": r"[.*"}, "stop": ["\n"]},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_invalid_grammar(client: openai.AsyncOpenAI, model_name: str):
    invalid_simplified_sql_grammar = """
        root ::= select_statementinvalidsyntax

        select_statement ::= "SELECT " column " from " table " where " condition

        column ::= "col_1 " | "col_2 "

        table ::= "table_1 " | "table_2 "

        condition ::= column "= " number

        number ::= "1 " | "2 "
    """

    prompt = (
        "Generate an SQL query to show the 'username' and 'email'"
        "from the 'users' table."
    )
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={
                "structured_outputs": {"grammar": invalid_simplified_sql_grammar}
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name",
    [MODEL_NAME],
)
async def test_empty_grammar(client: openai.AsyncOpenAI, model_name: str) -> None:
    prompt = "Say hello"
    with pytest.raises((openai.BadRequestError, openai.APIError)):
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"structured_outputs": {"grammar": ""}},
        )


# Decode-side token reuse for disaggregated serving. The router forwards the
# prefill stage's prompt token ids in kv_transfer_params so the decode stage
# skips re-tokenizing.

TOKEN_IN_MESSAGES = [{"role": "user", "content": "Hello, how are you today?"}]
DECODE_MESSAGES = [{"role": "user", "content": "unrelated decode-side text"}]


@pytest.mark.asyncio
async def test_kv_transfer_prompt_token_ids_round_trip(client: openai.AsyncOpenAI):
    """Ids forwarded in kv_transfer_params are used verbatim, skipping tokenize.

    The decode request carries different messages, so a response whose
    prompt_token_ids match the forwarded ids proves the ids were used rather
    than the request's own messages. Generated text is not compared across
    requests because vLLM greedy decoding is not bitwise-reproducible.
    """
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    reused_ids = baseline.prompt_token_ids
    assert reused_ids

    decode = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=DECODE_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={
            "kv_transfer_params": {"prompt_token_ids": reused_ids},
            "return_token_ids": True,
        },
    )

    # The engine saw the forwarded ids, not the decode request's own messages.
    assert decode.prompt_token_ids == reused_ids
    # text-out: reuse still yields a detokenized message.
    assert decode.choices[0].message.content


@pytest.mark.asyncio
async def test_kv_transfer_prompt_token_ids_streaming(client: openai.AsyncOpenAI):
    """Decode-side token reuse streams chat-formatted text-out."""
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        extra_body={"return_token_ids": True},
    )
    reused_ids = baseline.prompt_token_ids
    assert reused_ids

    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=DECODE_MESSAGES,
        max_completion_tokens=16,
        temperature=0,
        stream=True,
        extra_body={
            "kv_transfer_params": {"prompt_token_ids": reused_ids},
            "return_token_ids": True,
        },
    )

    content = ""
    delta_token_ids: list[int] = []
    first_chunk = True
    async for chunk in stream:
        if first_chunk:
            # prompt_token_ids arrives once, on the first chunk.
            assert chunk.prompt_token_ids == reused_ids
            first_chunk = False
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
        if tids := getattr(chunk.choices[0], "token_ids", None):
            delta_token_ids.extend(tids)

    # streamed text-out, reconstructed from deltas, with generated token ids.
    assert content
    assert delta_token_ids


@pytest.mark.asyncio
async def test_kv_transfer_prompt_token_ids_return_prompt_text(
    client: openai.AsyncOpenAI,
):
    """`return_prompt_text` returns the rendered prompt, so the ids are
    ignored and `messages` is rendered."""
    baseline = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=1,
        extra_body={"return_token_ids": True, "return_prompt_text": True},
    )
    assert baseline.model_extra["prompt_text"]

    decode = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=TOKEN_IN_MESSAGES,
        max_completion_tokens=1,
        extra_body={
            "kv_transfer_params": {"prompt_token_ids": baseline.prompt_token_ids},
            "return_prompt_text": True,
        },
    )
    assert decode.model_extra["prompt_text"] == baseline.model_extra["prompt_text"]


def _assert_complete_timing_metrics(metrics: object) -> None:
    assert isinstance(metrics, dict)
    timing_fields = {
        "time_to_first_token_ms",
        "generation_time_ms",
        "queue_time_ms",
        "mean_itl_ms",
        "tokens_per_second",
    }
    assert timing_fields <= metrics.keys()
    assert all(metrics[field] >= 0 for field in timing_fields)


@pytest.mark.asyncio
async def test_per_request_metrics_non_streaming(client: openai.AsyncOpenAI):
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Count to four."}],
        max_completion_tokens=4,
        temperature=0,
        extra_body={"min_tokens": 4},
    )

    _assert_complete_timing_metrics(response.model_extra.get("metrics"))


@pytest.mark.asyncio
async def test_per_request_metrics_on_final_usage_chunk(client: openai.AsyncOpenAI):
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello."}],
        max_completion_tokens=1,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"min_tokens": 1},
    )

    chunks = [chunk async for chunk in stream]
    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.usage is None
        assert "metrics" not in (chunk.model_extra or {})

    usage_chunk = chunks[-1]
    assert usage_chunk.usage is not None
    assert usage_chunk.choices == []
    metrics = (usage_chunk.model_extra or {}).get("metrics")
    assert isinstance(metrics, dict)
    assert metrics["time_to_first_token_ms"] >= 0
    assert metrics["generation_time_ms"] >= 0
    assert metrics["queue_time_ms"] >= 0
    assert metrics["tokens_per_second"] >= 0
    assert "mean_itl_ms" not in metrics
