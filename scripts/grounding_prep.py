#!/usr/bin/env python3
"""Grounding prep: build clips whose color flashes sit exactly on sampled frames.

Two-pass per clip: (1) encode a solid dummy of the target length, run vllm's
real video loader to get frames_indices/fps; (2) encode the real clip with a
moving gradient and solid-color flashes centered on chosen sampled-frame
times, each ~0.8 label-frame-steps wide so exactly one sampled frame is
inside the flash. Writes manifest.json for the client/scoring.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO

OUT = Path("$TASKDIR/clips2")
W, H = 320, 240

# (name, duration_s, fps, [(color_bgr, color_name, sampled_index)])
PLANS = [
    ("g60", 60, 25, [((0, 0, 255), "red", 9), ((255, 0, 0), "blue", 22)]),
    ("g150", 150, 25, [
        ((0, 0, 255), "red", 6),
        ((0, 255, 0), "green", 12),
        ((255, 0, 0), "blue", 19),
        ((0, 255, 255), "yellow", 26),
    ]),
    ("g600", 600, 10, [
        ((0, 0, 255), "red", 4),
        ((0, 255, 0), "green", 9),
        ((255, 0, 0), "blue", 14),
        ((0, 255, 255), "yellow", 19),
        ((255, 0, 255), "magenta", 24),
        ((255, 255, 255), "white", 28),
    ]),
]


def encode(path, duration_s, fps, events_idx):
    """events_idx: list of (bgr, frame_index); flash covers +-40% of the step."""
    total = int(duration_s * fps)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for i in range(total):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = (np.arange(W)[None, :] + 2 * i) % 256
        frame[:, :, 1] = (np.arange(H)[:, None] + i) % 256
        for bgr, idx, half in events_idx:
            if abs(i - idx) <= half:
                frame[:, :] = bgr
        vw.write(frame)
    vw.release()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"clips": []}
    for name, dur, fps, events in PLANS:
        dummy = OUT / f"{name}_dummy.mp4"
        encode(dummy, dur, fps, [])
        _, meta = VideoMediaIO(ImageMediaIO()).load_file(dummy)
        indices = meta["frames_indices"]
        real_fps = meta["fps"]
        n = len(indices)
        step_frames = indices[1] - indices[0]
        half = max(1, int(0.4 * step_frames))
        events_idx = []
        ev_meta = []
        for bgr, color, k in events:
            fi = indices[k]
            events_idx.append((bgr, fi, half))
            ev_meta.append(
                {
                    "color": color,
                    "frame_index": fi,
                    "time_s": fi / real_fps,
                    "sampled_position": k,
                }
            )
        real = OUT / f"{name}.mp4"
        encode(real, dur, fps, events_idx)
        # verify: reload the real clip; the flashed sampled frames must be
        # exactly the planned ones
        frames, meta2 = VideoMediaIO(ImageMediaIO()).load_file(real)
        got = []
        for bgr, color, k in events:
            fr = frames[k].astype(np.int16)  # sampled frame at position k
            # vllm's loader returns RGB frames; our event colors are BGR
            target = np.array(bgr[::-1], dtype=np.int16)
            # lossy codec: per-channel mean within 32 of the flash color
            got.append(bool(np.all(np.abs(fr.mean(axis=(0, 1)) - target) < 32)))
        assert all(got), f"{name}: flash not on sampled frame: {got}"
        dummy.unlink()
        manifest["clips"].append(
            {
                "file": str(real),
                "duration_s": dur,
                "fps": real_fps,
                "num_sampled": n,
                "frame_step_s": step_frames / real_fps,
                "events": ev_meta,
            }
        )
        print(f"{name}: {len(events)} events verified on sampled frames", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("PREP_DONE")


if __name__ == "__main__":
    main()
