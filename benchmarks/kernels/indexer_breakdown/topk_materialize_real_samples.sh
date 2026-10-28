#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
    echo "Usage: $0 OUTPUT_DIRECTORY" >&2
    exit 2
fi
topk_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
"$topk_repo/.venv/bin/python" - \
    "$topk_repo/benchmarks/kernels/indexer_breakdown/results/topk_real_samples_archive.jsonl" \
    "$1" <<'PY'
import base64
import hashlib
import json
import sys
from pathlib import Path

import torch

archive, destination = Path(sys.argv[1]), Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
for line in archive.read_text().splitlines():
    entry = json.loads(line)
    raw = base64.b64decode(entry["logits_b64"])
    assert hashlib.sha256(raw).hexdigest() == entry["raw_sha256"]
    logits = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).clone()
    logits = logits.reshape(entry["shape"])
    metadata = dict(entry["metadata"])
    metadata["original_pt_sha256"] = entry["original_pt_sha256"]
    metadata["raw_logits_sha256"] = entry["raw_sha256"]
    metadata["raw_context_tokens"] = metadata["max_seq_len_host"]
    metadata["capacity"] = metadata["full_shape"][1]
    metadata["original_logits_stride"] = metadata["full_stride"]
    payload = {
        **entry["original_payload_metadata"],
        "logits": logits,
        "seq_lens": torch.tensor(entry["seq_lens"], dtype=torch.int32),
        "metadata": metadata,
    }
    output = destination / Path(entry["name"]).name
    torch.save(payload, output)
    print(output)
PY
