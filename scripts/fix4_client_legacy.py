#!/usr/bin/env python3
"""FIX4 client: correctness (flash-time grounding + collateral identity) and
perf (TTFT/TPOT/throughput at concurrency 1 and 4) against a served MiMo-V2.6.
"""
import argparse
import base64
import concurrent.futures as cf
import json
import os
import statistics
import sys
import time
import urllib.request

CLIPS = {30: "flash_030.mp4", 60: "flash_060.mp4", 120: "flash_120.mp4"}
CLIP_DIR = "$TASKDIR/clips"


def b64_video(name):
    with open(f"{CLIP_DIR}/{name}", "rb") as f:
        return base64.b64encode(f.read()).decode()


def chat(url, model, content, max_tokens=64, seed=0, timeout=600):
    body = {
        "model": model,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "messages": [{"role": "user", "content": content}],
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    text = ""
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if not line.startswith(b"data: "):
                continue
            payload = line[6:].strip()
            if payload == b"[DONE]":
                break
            obj = json.loads(payload)
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0]["delta"].get("content") or ""
                if delta and ttft is None:
                    ttft = time.time() - t0
                text += delta
            if obj.get("usage"):
                usage = obj["usage"]
    e2e = time.time() - t0
    return text, ttft, e2e, usage


def video_content(prompt, clip_name):
    return [
        {"type": "text", "text": prompt},
        {
            "type": "video_url",
            "video_url": {"url": "data:video/mp4;base64," + b64_video(clip_name)},
        },
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    out = {"model": a.model, "grounding": {}, "collateral": {}, "perf": {}}

    # 1. temporal grounding: manifest-driven, events on sampled frames
    if os.environ.get("SKIP_GROUNDING") == "1":
        manifest = {"clips": []}
        print("[grounding skipped]", flush=True)
    else:
        import re as _re
        manifest = json.load(open("$TASKDIR/clips2/manifest.json"))
    # load clips once
    clip_b64 = {}
    def get_b64(path):
        if path not in clip_b64:
            with open(path, "rb") as f:
                clip_b64[path] = base64.b64encode(f.read()).decode()
        return clip_b64[path]

    def ask(clip, ev):
        q = ("Each frame of this video is preceded by a timestamp label like 00:09. "
             f"Exactly one frame is solid {ev['color']}. Reply with only the timestamp "
             f"label printed just before that solid {ev['color']} frame.")
        content = [
            {"type": "text", "text": q},
            {"type": "video_url",
             "video_url": {"url": "data:video/mp4;base64," + get_b64(clip["file"])}},
        ]
        text, ttft, e2e, usage = chat(a.url, a.model, content, max_tokens=2048)
        tail = text.split("</think>")[-1]
        stamps = _re.findall(r"(\d{1,3}):(\d{2})", tail)
        ans = int(m) * 60 + int(ss) if (stamps and (m := stamps[-1][0]) and (ss := stamps[-1][1])) else None
        return {
            "clip": clip["file"].split("/")[-1],
            "color": ev["color"],
            "true_s": ev["time_s"],
            "answer_s": ans,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tail": tail[-300:],
        }

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(ask, clip, ev)
                for clip in manifest["clips"] for ev in clip["events"]]
        for f in futs:
            r = f.result()
            out["grounding"][f"{r['clip']}:{r['color']}"] = r
            print(f"[grounding {r['clip']}:{r['color']}] true={r['true_s']:.1f}s ans={r['answer_s']} ptok={r['prompt_tokens']}", flush=True)

    # 1b. label readout: the model lists the label sequence directly (reveals the
    # injected timeline regardless of its temporal-grounding ability)
    if os.environ.get("SKIP_GROUNDING") != "1":
        q = ("This video's frames carry timestamp labels like 00:09. List the FIRST "
             "three and the LAST label exactly as printed, comma-separated, nothing else.")
        content = [
            {"type": "text", "text": q},
            {"type": "video_url",
             "video_url": {"url": "data:video/mp4;base64," + get_b64("$TASKDIR/clips2/g60.mp4")}},
        ]
        text, ttft, e2e, usage = chat(a.url, a.model, content, max_tokens=256)
        out["label_readout"] = {"text": text.split("</think>")[-1][-300:],
                                "prompt_tokens": usage.get("prompt_tokens")}
        print("[label_readout]", out["label_readout"]["text"][:120].replace("\n", " "), flush=True)

    # 2. collateral: greedy text-only identity
    texts = [
        "What is the capital of France?",
        "List the first five prime numbers.",
        "Explain photosynthesis in one sentence.",
        "Write a haiku about the ocean.",
        "What is 17 * 23?",
    ]
    for t in texts:
        text, _, _, usage = chat(a.url, a.model, [{"type": "text", "text": t}], max_tokens=128)
        out["collateral"].setdefault("text", []).append(text.strip())
    print("[collateral] text done", flush=True)

    # 2b. collateral: greedy image prompts
    from PIL import Image, ImageDraw
    import io
    for color, shape in (("red", "circle"), ("blue", "square")):
        img = Image.new("RGB", (336, 336), (255, 255, 255))
        d = ImageDraw.Draw(img)
        c = {"red": (220, 30, 30), "blue": (30, 30, 220)}[color]
        m = 84
        if shape == "circle":
            d.ellipse([m, m, 252, 252], fill=c)
        else:
            d.rectangle([m, m, 252, 252], fill=c)
        buf = io.BytesIO(); img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        content = [{"type": "text", "text": "What color and shape is in this image?"},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]
        text, _, _, usage = chat(a.url, a.model, content, max_tokens=128)
        out["collateral"].setdefault("image", []).append({"subject": f"{color}-{shape}", "answer": text.strip(), "ptok": usage.get("prompt_tokens")})
    print("[collateral] image done", flush=True)

    # 3. perf: video requests at concurrency 1 and 4
    perf_q = video_content("Describe this video briefly.", "flash_060.mp4")
    for conc in (1, 4):
        n = 8
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=conc) as ex:
            res = list(ex.map(lambda _: chat(a.url, a.model, perf_q, max_tokens=128), range(n)))
        wall = time.time() - t0
        ttfts = [r[1] for r in res if r[1]]
        e2es = [r[2] for r in res]
        comp = [r[3].get("completion_tokens", 0) for r in res]
        ptok = [r[3].get("prompt_tokens") for r in res]
        out["perf"][f"c{conc}"] = {
            "n": n,
            "wall": wall,
            "ttft_p50": statistics.median(ttfts),
            "ttft_max": max(ttfts),
            "e2e_p50": statistics.median(e2es),
            "tpot_ms": 1000 * (statistics.median(e2es) - statistics.median(ttfts)) / max(1, (max(comp) - 1)),
            "throughput_tok_s": sum(comp) / wall,
            "prompt_tokens": ptok[0],
            "prompt_tokens_all_same": len(set(ptok)) == 1,
        }
        print(f"[perf c{conc}] ttft_p50={out['perf'][f'c{conc}']['ttft_p50']:.3f}s", flush=True)

    with open(a.output, "w") as f:
        json.dump(out, f, indent=1)
    print("CLIENT_DONE")


if __name__ == "__main__":
    main()
