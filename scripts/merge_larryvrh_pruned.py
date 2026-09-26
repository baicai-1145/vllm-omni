"""Merge larryvrh turbo LoRA (v4 step600 ema) into the PRUNED MiniMax-H3.

The LoRA is trained against the dense adaln input (silu(t_emb), 2688-d) but
the pruned base consumes 8-d curve coordinates: silu_te = mean + c @ basis.
Fold the adaln delta analytically:
    delta_out(c) = B @ A @ silu_te = B@(A@mean) + (B @ (A @ basis.T)) @ c
  -> weight delta (pruned, in=8):   dW = B @ (A @ basis.T)
  -> bias delta (folded_bias):      db = B @ (A @ mean)
Backbone deltas (qkv/out/fc1/fc2) apply directly; qkv delta rows [q;k;v] map
to diffusers to_q/to_k/to_v; fc1 rows are gate-first while the raw pruned
checkpoint is value-first -> swap halves.

Streaming per shard, peak RAM < 6G (32G container safe).
"""
import json
import os
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

BASE = "/workspace/models/MiniMax-H3-Pruned/transformer"
LORA = "/workspace/models/turbo-lora/minimax_h3_turbo_v4_step600_ema.safetensors"
OUT = "/workspace/models/MiniMax-H3-Pruned-turboL/transformer"
SCALE = 1.0
H = 7168  # attention inner dim

os.makedirs(OUT, exist_ok=True)

# basis/mean for adaln fold
with safe_open(f"{BASE}/adaln_affine.safetensors", framework="pt") as fh:
    BASIS = fh.get_tensor("adaln_basis").float()   # [8, 2688]
    MEAN = fh.get_tensor("adaln_mean").float()     # [2688,]

# 1) index lora tensors by module name (stream: keep only A/B refs, bf16)
lora_path = LORA
mod_names = set()
with safe_open(lora_path, framework="pt") as fh:
    for k in fh.keys():
        if ".lora_A." in k:
            mod_names.add(k.split(".lora_A.")[0])
print(f"lora modules: {len(mod_names)}")

# 2) plan: base weight key -> handler spec
# diffusers base key patterns in the pruned ckpt:
#   transformer_blocks.N.attn.to_q|to_k|to_v.weight   <- qkv_proj delta slices
#   transformer_blocks.N.attn.to_out.0.weight          <- out_proj delta
#   transformer_blocks.N.ff.net.0.proj.weight          <- fc1 delta (swap halves)
#   transformer_blocks.N.ff.net.2.weight               <- fc2 delta
#   transformer_blocks.N.adaln_proj.linear.weight      <- adaln dW (folded)
#   (folded_bias lives where? check index)
idx_all = json.load(open(f"{BASE}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
fb_key = [k for k in idx_all if "folded" in k or ("adaln" in k and "bias" in k)]
print("adaln bias keys in ckpt:", fb_key[:4])

plan = {}  # ckpt key -> (diff_name, kind)
for name in mod_names:
    # name like blocks.0.attn.qkv_proj
    parts = name.split(".")
    n = parts[1]
    kind = None
    if "attn.qkv_proj" in name: kind = "qkv"
    elif "attn.out_proj" in name: kind = "out"
    elif "mlp.fc1" in name: kind = "fc1"
    elif "mlp.fc2" in name: kind = "fc2"
    elif "adaln_proj.linear" in name: kind = "adaln"
    if kind is None:
        continue
    if kind == "qkv":
        for sub, sl in (("to_q", 0), ("to_k", 1), ("to_v", 2)):
            key = f"transformer_blocks.{n}.attn.{sub}.weight"
            plan.setdefault(key, []).append((name, kind, sl))
    elif kind == "out":
        plan.setdefault(f"transformer_blocks.{n}.attn.to_out.0.weight", []).append((name, kind, None))
    elif kind == "fc1":
        plan.setdefault(f"transformer_blocks.{n}.ff.net.0.proj.weight", []).append((name, kind, None))
    elif kind == "fc2":
        plan.setdefault(f"transformer_blocks.{n}.ff.net.2.weight", []).append((name, kind, None))
    elif kind == "adaln":
        plan.setdefault(f"transformer_blocks.{n}.adaln_proj.linear.weight", []).append((name, kind, None))
        plan.setdefault(f"transformer_blocks.{n}.adaln_proj.folded_bias", []).append((name, kind + ":bias", None))

print(f"planned base keys: {len(plan)}")
missing = [k for k in plan if k not in idx_all]
if missing:
    print("MISSING in ckpt:", missing[:4])

def get_delta(fh, name):
    a = fh.get_tensor(f"{name}.lora_A.weight").float()
    b = fh.get_tensor(f"{name}.lora_B.weight").float()
    return (b @ a) * SCALE

# 3) stream merge per shard
shards = sorted(set(idx_all.values()))
merged = 0
for shard in shards:
    sd = load_file(f"{BASE}/{shard}")
    hit = False
    for key in list(sd.keys()):
        if key not in plan:
            continue
        w = sd[key].float()
        ok = True
        for name, kind, sl in plan[key]:
            with safe_open(lora_path, framework="pt") as fh:
                if kind.startswith("adaln"):
                    a = fh.get_tensor(f"{name}.lora_A.weight").float()
                    b = fh.get_tensor(f"{name}.lora_B.weight").float()
                    if kind == "adaln":
                        delta = (b @ (a @ BASIS.T)) * SCALE      # [out, 8]
                    else:
                        delta = (b @ (a @ MEAN)) * SCALE         # [out]
                    del a, b
                else:
                    delta = get_delta(fh, name)
            if kind == "qkv":
                if delta.shape[0] != 3 * H:
                    print(f"  qkv delta unexpected {tuple(delta.shape)}"); ok = False; break
                seg = delta[sl*H:(sl+1)*H]
                if seg.shape != w.shape:
                    print(f"  seg skip {key}: {tuple(seg.shape)} vs {tuple(w.shape)}"); ok = False; break
                w += seg
            elif kind == "fc1":
                h = delta.shape[0] // 2
                delta = torch.cat([delta[h:], delta[:h]], dim=0)  # gate-first -> value-first
                if delta.shape != w.shape:
                    print(f"  fc1 skip {key}"); ok = False; break
                w += delta
            elif kind == "adaln":
                if delta.shape != w.shape:
                    print(f"  adaln skip {key}: {tuple(delta.shape)} vs {tuple(w.shape)}"); ok = False; break
                w += delta
            elif kind == "adaln:bias":
                # db = delta_plain_offset = B @ (A @ mean), as a vector
                if delta.shape != w.shape:
                    print(f"  adaln-bias skip {key}: {tuple(delta.shape)} vs {tuple(w.shape)}"); ok = False; break
                w += delta
            elif kind == "adaln:bias":
                pass  # delta already the bias vector (computed below)
            else:
                if delta.shape != w.shape:
                    print(f"  skip {key}"); ok = False; break
                w += delta
            del delta
            merged += 1
        if ok:
            sd[key] = w.to(torch.bfloat16)
            hit = True
    save_file(sd, f"{OUT}/{shard}", metadata={"format": "pt"})
    del sd
    if hit:
        print(f"merged {shard}", flush=True)

# 4) adaln bias fold (per-block bias vector) + copy aux files
# find where the per-block folded bias lives
print("\nlooking for per-block adaln bias in ckpt keys...")
cand = [k for k in idx_all if "adaln" in k.lower()]
print(cand[:8])

for f in ("config.json", "diffusion_pytorch_model.safetensors.index.json", "adaln_affine.safetensors"):
    src = f"{BASE}/{f}"
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, f"{OUT}/{f}")
print(f"total tensors updated: {merged}")
print(f"output: {OUT}")
