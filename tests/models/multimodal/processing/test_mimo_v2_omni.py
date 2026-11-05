# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiMo-V2 omni video timestamp construction.

Regression tests for https://github.com/vllm-project/vllm/issues/65523: the
processor used to label sampled frames with ``arange(T) / 1 fps``, so every
clip's injected timestamps read 00:00..00:30 regardless of the real frame
times. The timestamps must come from the video loader metadata
(``frames_indices / fps``).

Self-contained: synthetic byte-level BPE tokenizer and in-memory config, no
checkpoint download, runs on CPU anywhere.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.model_executor.models.mimo_v2_omni import (
    MiMoV2OmniMultiModalDataParser,
    MiMoV2OmniMultiModalProcessor,
    _video_timestamps_from_metadata,
)
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, VideoProcessorItems
from vllm.transformers_utils.processors.mimo_v2_omni import MiMoOmniProcessor

pytestmark = pytest.mark.cpu_test

SPECIAL_TOKENS = [
    "<|vision_start|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_end|>",
    "<|video_start|>",
    "<|video_end|>",
]


@pytest.fixture
def tokenizer():
    vocab = {
        token: index
        for index, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    return TokenizersBackend(
        tokenizer_object=backend,
        additional_special_tokens=SPECIAL_TOKENS,
        clean_up_tokenization_spaces=False,
    )


@pytest.fixture
def processor(tokenizer):
    """A MiMoV2OmniMultiModalProcessor with an in-memory info namespace."""
    vocab = tokenizer.get_vocab()
    hf_processor = MiMoOmniProcessor(
        tokenizer,
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        temporal_compression_ratio=1,
        image_token_id=vocab["<|image_pad|>"],
        video_token_id=vocab["<|video_pad|>"],
        vision_start_token_id=vocab["<|vision_start|>"],
        vision_end_token_id=vocab["<|vision_end|>"],
        video_start_token_id=vocab["<|video_start|>"],
        video_end_token_id=vocab["<|video_end|>"],
    )
    proc = object.__new__(MiMoV2OmniMultiModalProcessor)
    proc.info = SimpleNamespace(
        get_tokenizer=lambda: tokenizer,
        get_hf_config=lambda: SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=2)
        ),
        get_hf_processor=lambda **kwargs: hf_processor,
    )
    return proc


def _make_mm_items(videos):
    """Parse (video, metadata) tuples through the real MiMo data parser."""
    parser = MiMoV2OmniMultiModalDataParser(
        2,  # spatial_merge_size
        target_sr=24000.0,
        video_needs_metadata=True,
    )
    return parser.parse_mm_data({"video": videos})


def _video_metadata(frames_indices, fps=25.0):
    return {
        "fps": fps,
        "duration": (frames_indices[-1] + 1) / fps,
        "total_num_frames": frames_indices[-1] + 1,
        "frames_indices": frames_indices,
        "video_backend": "opencv",
        "do_sample_frames": False,
    }


# ---------------------------------------------------------------------------
# _video_timestamps_from_metadata
# ---------------------------------------------------------------------------


def test_video_timestamps_from_metadata_uniform_sampling():
    # A 120 s @ 25 fps clip uniformly sampled to 16 frames, 8 s apart.
    indices = [i * 200 for i in range(16)]
    ts = _video_timestamps_from_metadata(16, _video_metadata(indices))
    assert ts is not None
    torch.testing.assert_close(
        ts, torch.arange(16, dtype=torch.float32) * 8.0, rtol=0, atol=1e-4
    )


def test_video_timestamps_from_metadata_start_offset():
    indices = [500 + i * 200 for i in range(8)]  # first frame at t=20 s
    ts = _video_timestamps_from_metadata(8, _video_metadata(indices))
    assert ts is not None
    assert float(ts[0]) == pytest.approx(20.0)
    assert float(ts[-1]) == pytest.approx(20.0 + 7 * 8.0)


def test_video_timestamps_from_metadata_numpy_indices():
    # User-supplied metadata may carry numpy arrays (offline tuple path);
    # `not frames_indices` must not raise "truth value of an array is
    # ambiguous".
    indices = np.array([i * 200 for i in range(16)])
    metadata = _video_metadata([0] * 16) | {"frames_indices": indices}
    ts = _video_timestamps_from_metadata(16, metadata)
    assert ts is not None
    assert float(ts[1]) == pytest.approx(8.0)


@pytest.mark.parametrize(
    "num_frames,metadata",
    [
        (4, None),  # no metadata
        (4, "not-a-dict"),
        (4, {"fps": 25.0}),  # missing frames_indices
        (4, {"frames_indices": [0, 1, 2, 3]}),  # missing fps
        (4, {"frames_indices": [0, 1, 2, 3], "fps": 0.0}),  # non-positive fps
        (4, {"frames_indices": [0, 1, 2, 3], "fps": -25.0}),
        (4, {"frames_indices": [0, 1, 2], "fps": 25.0}),  # length mismatch
        (4, {"frames_indices": [], "fps": 25.0}),  # empty
        (4, {"frames_indices": [3, 2, 1, 0], "fps": 25.0}),  # not increasing
    ],
)
def test_video_timestamps_from_metadata_rejects_unusable(num_frames, metadata):
    assert _video_timestamps_from_metadata(num_frames, metadata) is None


def test_video_timestamps_from_metadata_single_frame():
    ts = _video_timestamps_from_metadata(1, _video_metadata([100]))
    assert ts is not None
    assert ts.tolist() == [4.0]  # 100 / 25 fps


# ---------------------------------------------------------------------------
# _get_hf_mm_inputs: parsed (video, metadata) tuples drive the timestamps
# ---------------------------------------------------------------------------


def _get_video_timestamps(processor, videos):
    mm_items = _make_mm_items(videos)
    hf_inputs = processor._get_hf_mm_inputs(mm_items, {})
    out = hf_inputs.hf_data["videos"]
    assert len(out) == len(videos)
    frames, timestamps = out[0]
    assert frames.ndim == 4 and frames.shape[1] in (1, 3, 4)  # TCHW
    return timestamps


def test_hf_mm_inputs_timestamps_use_frame_media_time(processor):
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    # 8 s between sampled frames -> 16 s per 2-frame grid, not 1 fps.
    ts = _get_video_timestamps(
        processor, [(frames, _video_metadata([i * 200 for i in range(16)]))]
    )
    torch.testing.assert_close(
        ts, torch.arange(16, dtype=torch.float32) * 8.0, rtol=0, atol=1e-4
    )


def test_hf_mm_inputs_timestamps_fall_back_to_1fps(processor):
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    metadata = _video_metadata([0] * 16)
    metadata.pop("frames_indices")
    ts = _get_video_timestamps(processor, [(frames, metadata)])
    torch.testing.assert_close(
        ts, torch.arange(16, dtype=torch.float32), rtol=0, atol=0
    )


def test_hf_mm_inputs_preserve_prebuilt_mimo_tuples(processor):
    # Already-converted (TCHW, timestamps) tuples pass through untouched.
    frames = torch.zeros(4, 3, 56, 56)
    given_ts = torch.tensor([3.0, 5.0, 9.0, 10.0])
    mm_items = MultiModalDataItems({"video": VideoProcessorItems([(frames, given_ts)])})
    hf_inputs = processor._get_hf_mm_inputs(mm_items, {})
    out_frames, out_ts = hf_inputs.hf_data["videos"][0]
    assert out_frames is frames
    assert out_ts is given_ts


def test_raw_video_without_metadata_raises(processor):
    # With video_needs_metadata=True, raw arrays must carry metadata
    # (same contract as Qwen2.5-VL and other metadata-driven models).
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Video metadata is required"):
        _make_mm_items([frames])


# ---------------------------------------------------------------------------
# Prompt-update reconstruction: labels in the final prompt carry media time
# ---------------------------------------------------------------------------


def test_prompt_timestamp_labels_match_media_time(processor):
    tokenizer = processor.info.get_tokenizer()
    vocab = tokenizer.get_vocab()

    # 8 grids at 16 s per grid starting at t=4 s (a mid-clip segment).
    start, spg, t_grid = 4.0, 16.0, 8
    mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
        {
            "video_grid_thw": torch.tensor([[t_grid, 2, 2]]),
            "second_per_grid_ts": torch.tensor([spg]),
            "video_start_times": torch.tensor([start]),
        },
        {
            "video_grid_thw": MultiModalFieldConfig.batched("video"),
            "second_per_grid_ts": MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            ),
            "video_start_times": MultiModalFieldConfig.batched(
                "video", keep_on_cpu=True
            ),
        },
    )
    mm_items = _make_mm_items(
        [(np.zeros((2 * t_grid, 56, 56, 3), dtype=np.uint8), _video_metadata([0]))]
    )

    updates = processor._get_mm_prompt_updates(mm_items, {}, mm_kwargs)
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    actual_ids, placeholders = processor._apply_prompt_updates(prompt_ids, updates)

    assert len(placeholders["video"]) == 1
    text = tokenizer.decode(actual_ids)

    # One label per temporal grid, each the real media time.
    for j in range(t_grid):
        seconds = int(start + j * spg)
        label = f"{seconds // 60:02d}:{seconds % 60:02d}"
        assert label in text, f"missing timestamp label {label}"

    # The video placeholder still expands to grid tokens.
    video_pad_id = vocab["<|video_pad|>"]
    assert actual_ids.count(video_pad_id) == t_grid  # H*W/merge^2 == 1 per grid
