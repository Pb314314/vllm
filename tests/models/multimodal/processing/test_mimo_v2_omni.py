# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MiMo-V2 omni video timestamp construction.

Regression tests for https://github.com/vllm-project/vllm/issues/65523: the
processor used to label sampled frames with ``arange(T) / 1 fps``, so every
clip's injected timestamps read 00:00..00:30 regardless of the real frame
times. The timestamps must come from the video loader metadata
(``frames_indices / fps``).

The end-to-end tests run the model's own data parser and HF processor against
the public ``XiaomiMiMo/MiMo-V2.5`` config/tokenizer (a ~12 MB download, no
weights); everything else is self-contained (synthetic tokenizer, in-memory
config) and runs on CPU anywhere.
"""

from types import SimpleNamespace

from collections import UserList

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import TokenizersBackend

from vllm.model_executor.models.mimo_v2_omni import (
    MiMoV2OmniMultiModalDataParser,
    MiMoV2OmniMultiModalProcessor,
)
from vllm.multimodal.inputs import (
    MediaWithBytes,
)
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


def _timestamps_from_metadata(*args):
    # Local import so that on stock code (which has no such helper) the
    # end-to-end tests below still collect and fail with a wrong-value
    # assertion instead of a collection error.
    from vllm.model_executor.models.mimo_v2_omni import (
        _video_timestamps_from_metadata,
    )

    return _video_timestamps_from_metadata(*args)


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
# End-to-end: the model's own parser + HF processor must carry real frame
# times into the prompt (this is what breaks if the data parser drops the
# loader metadata, e.g. if `video_needs_metadata=True` is lost).
# ---------------------------------------------------------------------------

# Public config/tokenizer/remote code (~12 MB, no weights needed); the gated
# MiMo-V2.5-Omni registry entry cannot be downloaded in CI.
MIMO_V25 = "XiaomiMiMo/MiMo-V2.5"


@pytest.fixture(scope="module")
def e2e_processor():
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY

    model_config = ModelConfig(
        MIMO_V25,
        tokenizer=MIMO_V25,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _run_video_e2e(processor, video_item):
    mm_items = processor.info.parse_mm_data({"video": [video_item]})
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    processed = processor(prompt, mm_items=mm_items)
    data = processed["mm_kwargs"]["video"][0].get_data()
    text = processor.info.get_tokenizer().decode(processed["prompt_token_ids"])
    return data, text


def test_e2e_video_timestamps_use_frame_media_time(e2e_processor):
    """A 120 s @ 25 fps clip sampled to 16 frames must not read as 15 s."""
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    metadata = _video_metadata([i * 200 for i in range(16)])  # 8 s apart

    data, text = _run_video_e2e(e2e_processor, (frames, metadata))

    # The real sampling interval is 8 s per frame, i.e. 16 s per 2-frame
    # grid - not 2 s as produced by the old hardcoded 1 fps timestamps.
    assert float(data["second_per_grid_ts"]) == pytest.approx(16.0)
    assert float(data["video_start_times"]) == pytest.approx(0.0)
    # Labels in the processed prompt carry the real media time.
    assert "00:16" in text and "01:52" in text
    assert "00:14" not in text  # the old bug's last label


def test_e2e_video_timestamps_media_with_bytes(e2e_processor):
    """The serving path delivers MediaWithBytes((frames, metadata))."""
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    metadata = _video_metadata([i * 200 for i in range(16)])
    item = MediaWithBytes((frames, metadata), b"original-encoded-bytes")

    data, _ = _run_video_e2e(e2e_processor, item)

    assert float(data["second_per_grid_ts"]) == pytest.approx(16.0)


def test_e2e_video_timestamps_fall_back_to_1fps(e2e_processor):
    """Metadata without frames_indices keeps the old 1 fps behavior."""
    frames = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    metadata = _video_metadata([0] * 16)
    metadata.pop("frames_indices")

    data, _ = _run_video_e2e(e2e_processor, (frames, metadata))

    assert float(data["second_per_grid_ts"]) == pytest.approx(2.0)


def test_e2e_raw_video_without_metadata_keeps_1fps_fallback(e2e_processor):
    """Raw frames carry no timing info: they keep the pre-PR 1 fps behavior.

    Regression guard: `multi_modal_data={"video": ndarray}` and a list of PIL
    frames were accepted before this PR and must stay accepted.
    """
    from PIL import Image

    ndarray_video = np.zeros((16, 56, 56, 3), dtype=np.uint8)
    pil_video = [Image.fromarray(frame) for frame in ndarray_video]

    for raw_video in (ndarray_video, pil_video):
        data, _ = _run_video_e2e(e2e_processor, raw_video)
        assert float(data["second_per_grid_ts"]) == pytest.approx(2.0)
        assert float(data["video_start_times"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _video_timestamps_from_metadata
# ---------------------------------------------------------------------------


def test_video_timestamps_from_metadata_uniform_sampling():
    # A 120 s @ 25 fps clip uniformly sampled to 16 frames, 8 s apart.
    indices = [i * 200 for i in range(16)]
    ts = _timestamps_from_metadata(16, _video_metadata(indices))
    assert ts is not None
    torch.testing.assert_close(
        ts, torch.arange(16, dtype=torch.float32) * 8.0, rtol=0, atol=1e-4
    )


def test_video_timestamps_from_metadata_start_offset():
    indices = [500 + i * 200 for i in range(8)]  # first frame at t=20 s
    ts = _timestamps_from_metadata(8, _video_metadata(indices))
    assert ts is not None
    assert float(ts[0]) == pytest.approx(20.0)
    assert float(ts[-1]) == pytest.approx(20.0 + 7 * 8.0)


def test_video_timestamps_from_metadata_numpy_indices():
    # User-supplied metadata may carry numpy arrays (offline tuple path);
    # `not frames_indices` must not raise "truth value of an array is
    # ambiguous".
    indices = np.array([i * 200 for i in range(16)])
    metadata = _video_metadata([0] * 16) | {"frames_indices": indices}
    ts = _timestamps_from_metadata(16, metadata)
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
    assert _timestamps_from_metadata(num_frames, metadata) is None


def test_video_timestamps_from_metadata_single_frame():
    ts = _timestamps_from_metadata(1, _video_metadata([100]))
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


class _FakeVAItems(UserList):
    """Minimal MultiModalDataItems entry carrying raw video_audio content.

    The vLLM data parser registers no "video_audio" modality (chat-template
    video_audio content is expanded inside the in-tree MiMoOmniProcessor
    before parsing), so this exercises the defensive video_audio conversion
    loop in _get_hf_mm_inputs for direct processor callers.
    """

    def __init__(self, va_list):
        super().__init__([None])  # non-empty so the base loop visits it
        self._va_list = va_list

    def get_processor_data(self):
        return {"video_audio": self._va_list}

    def get_passthrough_data(self):
        return {}

    def get_count(self):
        return len(self._va_list)


def _run_video_audio(processor, va_item):
    mm_items = MultiModalDataItems({"video_audio": _FakeVAItems([va_item])})
    hf_inputs = processor._get_hf_mm_inputs(mm_items, {})
    converted = hf_inputs.hf_data["video_audio"]
    assert len(converted) == 1
    frames, timestamps = converted[0].video
    assert frames.shape[0] == timestamps.shape[0]
    return timestamps


def test_video_audio_timestamps_use_frame_media_time(processor):
    frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)
    audio = np.zeros(4, dtype=np.float32)
    # 0.4s-spaced frames: must NOT collapse onto the 1 fps fallback.
    metadata = _video_metadata([10, 35, 60, 85, 110, 135, 160, 185])
    ts = _run_video_audio(processor, ((frames, metadata), audio))
    expect = 0.4 + torch.arange(8, dtype=torch.float32)
    assert torch.allclose(ts, expect, atol=1e-6)


def test_video_audio_raw_frames_fall_back_to_1fps(processor):
    frames = np.zeros((8, 4, 4, 3), dtype=np.uint8)
    audio = np.zeros(4, dtype=np.float32)
    ts = _run_video_audio(processor, (frames, audio))
    assert torch.allclose(ts, torch.arange(8, dtype=torch.float32) / 1.0)
