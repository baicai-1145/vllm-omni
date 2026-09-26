"""Merge Turbo 4-step LoRA into the pruned MiniMax-H3 bf16 weights.

Streaming merge: iterate checkpoint shards, add LoRA deltas on the fly
(never materializing all deltas at once), write a new transformer dir.

Run with:  python3 merge_turbo_lora.py
"""
import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = "/workspace/models/MiniMax-H3-Pruned/transformer"
LORA = "/workspace/models/turbo-lora/turbo4/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
OUT = "/workspace/models/MiniMax-H3-Pruned-turbo4/transformer"

# scale = lora_alpha / r. lightx2v turbo: alpha=128, r=128 -> 1.0
SCALE = 1.0

os.makedirs(OUT, exist_ok=True)

# 1) load lora pairs (bf16, ~1.4G on disk -> ~2.8G ram in fp32; keep bf16 math per-pair)
pairs_a = {}
pairs_b = {}
with safe_open(LORA, framework="pt") as fh:
    for k in fh.keys():
        t = fh.get_tensor(k)
        if ".lora_A." in k:
            pairs_a[k.split(".lora_A.")[0]] = t
        elif ".lora_B." in k:
            pairs_b[k.split(".lora_B.")[0]] = t

print(f"lora modules: {len(pairs_a)}")

# 2) stream-merge per shard
idx = json.load(open(f"{BASE}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
shards = sorted(set(idx.values()))
merged = 0
for shard in shards:
    sd = load_file(f"{BASE}/{shard}")
    # collect key -> delta name for this shard only
    for name, a in pairs_a.items():
        # diffusers name -> pruned ckpt name (pruned repo uses diffusers names already)
        wa = sd.get(f"{name}.lora_A")  # not present; look up base weight instead
    # compute per-key merges for weights present in this shard
    for key in list(sd.keys()):
        # base key like transformer_blocks.0.attn.to_q.weight
        if not key.endswith(".weight"):
            continue
        lora_name = key[: -len(".weight")]
        a = pairs_a.get(lora_name)
        b = pairs_b.get(lora_name)
        if a is None or b is None:
            continue
        w = sd[key]
        delta = (b.float() @ a.float()) * SCALE
        if delta.shape != w.shape:
            print(f"  shape skip {key}: delta {tuple(delta.shape)} vs w {tuple(w.shape)}")
            continue
        # fc1: diffusers delta rows are [gate; up] (runtime chunks gate-first),
        # but the raw pruned checkpoint stores value-first [up; gate] (the
        # load-time patch swaps halves). Swap delta halves to match raw layout.
        if key.endswith("ff.net.0.proj.weight") and delta.shape[0] == w.shape[0] and w.shape[0] % 2 == 0:
            h = delta.shape[0] // 2
            delta = torch.cat([delta[h:], delta[:h]], dim=0)
        sd[key] = (w.float() + delta).to(w.dtype)
        merged += 1
    save_file(sd, f"{OUT}/{shard}", metadata={"format": "pt"})
    print(f"merged shard {shard}")

# 3) copy config + index
for f in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
    src = f"{BASE}/{f}"
    if os.path.exists(src):
        with open(src) as fh:
            data = fh.read()
        # rewrite relative shard names (they are unchanged)
        with open(f"{OUT}/{f}", "w") as fh:
            fh.write(data)

print(f"total merged tensors: {merged}")
print(f"output: {OUT}")
