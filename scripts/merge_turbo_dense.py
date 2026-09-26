"""Merge Turbo 4-step LoRA into the official DENSE MiniMax-H3 FL2VA weights.

The pruned repo's attention/MLP weights differ from dense (independently
distilled), so LoRA deltas trained on dense cannot transfer. Merge into the
true dense base instead.

Layout notes (verified):
- dense qkv_proj: [q(7168); k(7168); v(7168)] x 5376 -> add turbo to_q/to_k/to_v
  deltas at the right slices.
- dense mlp.fc1 raw layout is value-first [up; gate] (same as the pruned raw
  checkpoint); diffusers turbo delta rows are gate-first [gate; up] -> swap
  halves before adding.
- everything else (to_out.0, ff.net.2, refiner) adds directly.

Streaming: one shard at a time. Output: /workspace/models/MiniMax-H3-turbo4
"""
import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = "/workspace/models/MiniMax-H3/FL2VA/transformer"
LORA = "/workspace/models/turbo-lora/turbo4/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
OUT = "/workspace/models/MiniMax-H3-turbo4/transformer"

SCALE = 1.0  # alpha=r=128
H = 7168     # attention inner dim

os.makedirs(OUT, exist_ok=True)

# 1) load lora pairs
pairs_a, pairs_b = {}, {}
with safe_open(LORA, framework="pt") as fh:
    for k in fh.keys():
        t = fh.get_tensor(k)
        if ".lora_A." in k:
            pairs_a[k.split(".lora_A.")[0]] = t
        elif ".lora_B." in k:
            pairs_b[k.split(".lora_B.")[0]] = t
print(f"lora modules: {len(pairs_a)}")

# precompute deltas grouped by target base key
# turbo diffusers name -> dense base key
def dense_key(diff_name: str):
    # diffusers turbo name -> dense key. Dense repo already renamed:
    #   token_refiner.refiner_blocks.N -> token_refiner.blocks.N
    #   transformer_blocks.N -> blocks.N
    out = diff_name.replace("transformer_blocks.", "blocks.")
    out = out.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    return out

ref_by_base = defaultdict(list)  # base key -> (diff_name, pos)
for name in pairs_a:
    base = dense_key(name)
    if ".attn.to_q" in name:
        ref_by_base[base.replace("attn.to_q", "attn.qkv_proj") + ".weight"].append((name, 0))
    elif ".attn.to_k" in name:
        ref_by_base[base.replace("attn.to_k", "attn.qkv_proj") + ".weight"].append((name, 1))
    elif ".attn.to_v" in name:
        ref_by_base[base.replace("attn.to_v", "attn.qkv_proj") + ".weight"].append((name, 2))
    elif ".attn.to_out.0" in name:
        ref_by_base[base.replace("attn.to_out.0", "attn.out_proj") + ".weight"].append((name, None))
    elif "ff.net.0.proj" in name:
        ref_by_base[base.replace("ff.net.0.proj", "mlp.fc1") + ".weight"].append((name, "fc1"))
    elif "ff.net.2" in name:
        ref_by_base[base.replace("ff.net.2", "mlp.fc2") + ".weight"].append((name, None))

print(f"target base keys: {len(ref_by_base)}")

# 2) stream-merge per shard
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
shards = sorted(set(idx.values()))
merged = 0
for shard in shards:
    sd = load_file(f"{BASE}/{shard}")
    hit = False
    for key in list(sd.keys()):
        if key not in ref_by_base:
            continue
        w = sd[key].float()
        for name, pos in ref_by_base[key]:
            a, b = pairs_a[name], pairs_b[name]
            delta = (b.float() @ a.float()) * SCALE
            if pos == "fc1":
                h = delta.shape[0] // 2
                delta = torch.cat([delta[h:], delta[:h]], dim=0)
                pos = None
            if pos is None:
                if delta.shape != w.shape:
                    print(f"  shape skip {key}: {tuple(delta.shape)} vs {tuple(w.shape)}", flush=True)
                    del delta
                    continue
                w += delta
            else:
                if delta.shape != (H, w.shape[1]):
                    print(f"  qkv slice skip {key} pos{pos}", flush=True)
                    del delta
                    continue
                w[pos*H:(pos+1)*H] += delta
            del delta
            merged += 1
        sd[key] = w.to(torch.bfloat16)
        hit = True
    save_file(sd, f"{OUT}/{shard}", metadata={"format": "pt"})
    del sd
    if hit:
        print(f"merged {shard}", flush=True)

# 3) copy configs
for f in ("config.json", "model.safetensors.index.json"):
    src = f"{BASE}/{f}"
    if os.path.exists(src):
        data = open(src).read()
        open(f"{OUT}/{f}", "w").write(data)
print(f"total tensors updated: {merged}")
print(f"output: {OUT}")
