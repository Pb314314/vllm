"""Probe 2: the real server frontend path — parse_chat_messages with a data URI,
then the processor, printing the timestamp labels the model would see."""
import base64
import os
import re
import sys

sys.path.insert(0, os.environ.get("VLLM_TREE", "/tmp/wt-fix4-main2"))

import numpy as np

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

MODEL = os.environ.get("MODEL", "XiaomiMiMo/MiMo-V2.5")  # public config/tokenizer; env overrides
CLIP = os.environ.get("CLIP", "/tmp/fix4-e2e/probe30.mp4")

mc = ModelConfig(MODEL, tokenizer=MODEL, trust_remote_code=True, dtype="bfloat16",
                 seed=0, limit_mm_per_prompt={"image": 0, "video": 1})

uri = "data:video/mp4;base64," + base64.b64encode(open(CLIP, "rb").read()).decode()
msgs = [{"role": "user", "content": [
    {"type": "video_url", "video_url": {"url": uri}},
    {"type": "text", "text": "Describe."},
]}]

from vllm.renderers.chat_utils.chat_message import parse_chat_messages
from vllm.renderers.hf import resolve_chat_template_content_format

from transformers import AutoTokenizer
_tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
try:
    cf = resolve_chat_template_content_format(
        chat_template=None, tools=None, given_format="auto",
        tokenizer=_tok, model_config=mc,
    )
except Exception as e:
    print("content format resolve failed, using 'string':", e)
    cf = "string"
conversation, mm_data, mm_uuids = parse_chat_messages(
    msgs, mc, content_format=cf,
    media_io_kwargs=None, mm_processor_kwargs=None,
)
v = mm_data["video"][0]
print("mm_data video item type:", type(v).__name__, "| inner:", type(getattr(v, "media", v)).__name__)
inner = getattr(v, "media", v)
if isinstance(inner, tuple):
    fr, mdv = inner
    print("frames shape:", getattr(fr, "shape", None))
    if isinstance(mdv, dict):
        fi = mdv.get("frames_indices")
        print("frames_indices len:", len(fi) if fi is not None else None,
              "fps:", mdv.get("fps"), "backend:", mdv.get("video_backend"),
              "sampled:", mdv.get("do_sample_frames"))

# direct helper probe with the served metadata
try:
    from vllm.model_executor.models.mimo_v2_omni import _video_timestamps_from_metadata
except ImportError:  # baseline arms: helper does not exist (expected)
    _video_timestamps_from_metadata = None
if isinstance(inner, tuple) and isinstance(inner[1], dict):
    import torch as _t
    _fr, _md = inner
    _T = _fr.shape[0]
    _ts = _video_timestamps_from_metadata(_T, _md) if _video_timestamps_from_metadata else None
    print("helper direct: T=", _T, "->", None if _ts is None else [_ts[0].item(), _ts[1].item(), _ts[-1].item()])

proc = MULTIMODAL_REGISTRY.create_processor(mc)
mm_items = proc.info.parse_mm_data(mm_data)
processed = proc("<|vision_start|><|video_pad|><|vision_end|>", mm_items=mm_items)
text = proc.info.get_tokenizer().decode(processed["prompt_token_ids"])
labels = re.findall(r"\d{1,3}:\d{2}", text)
print("labels:", labels[:6], "...", labels[-3:] if len(labels) > 6 else "")

# --- multi-item prompt cells: two videos, and image+video mixed ---
import io

import PIL.Image

mc2 = ModelConfig(MODEL, tokenizer=MODEL, trust_remote_code=True, dtype="bfloat16",
                  seed=0, limit_mm_per_prompt={"image": 2, "video": 2})
_buf = io.BytesIO()
PIL.Image.new("RGB", (224, 224), (128, 64, 32)).save(_buf, format="PNG")
img_uri = "data:image/png;base64," + base64.b64encode(_buf.getvalue()).decode()

multi_cells = {
    "two-videos": [
        {"type": "video_url", "video_url": {"url": uri}},
        {"type": "video_url", "video_url": {"url": uri}},
    ],
    "image+video": [
        {"type": "image_url", "image_url": {"url": img_uri}},
        {"type": "video_url", "video_url": {"url": uri}},
    ],
}
for tag, items in multi_cells.items():
    msgs2 = [{"role": "user", "content": items + [{"type": "text", "text": "Describe."}]}]
    conv2, mm_data3, _uuids = parse_chat_messages(
        msgs2, mc2, content_format=cf,
        media_io_kwargs=None, mm_processor_kwargs=None,
    )
    mm_items3 = proc.info.parse_mm_data(mm_data3)
    nvid = len(mm_data3.get("video", []))
    nimg = len(mm_data3.get("image", []))
    prompt3 = ("<|vision_start|><|image_pad|><|vision_end|>" * nimg
               + "<|vision_start|><|video_pad|><|vision_end|>" * nvid)
    processed3 = proc(prompt3, mm_items=mm_items3)
    text3 = proc.info.get_tokenizer().decode(processed3["prompt_token_ids"])
    blocks = re.split(r"<\|vision_start\|>", text3)
    for bi, blk in enumerate(blocks):
        lab = re.findall(r"\d{1,3}:\d{2}", blk)
        if lab:
            print(f"multi[{tag}] block{bi} labels:", lab[:3], "...", lab[-2:])
    print(f"multi[{tag}] videos={nvid} images={nimg} prompt_chars={len(text3)}")
