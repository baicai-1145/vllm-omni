"""Convert Turbo/FlashGen LoRA for the pruned low-rank MiniMax-H3.

Official LoRAs train `adaln_proj.linear` with 2688-wide inputs (dense
time embedding). The pruned model consumes 8-wide interpolated
coordinates, so a dense LoRA delta `B @ A` must be projected onto the
pruned basis:  A' = A @ basis.T  (shape [rank, 8]), B unchanged.

QKV / fc1 / fc2 / out_proj LoRAs are shape-compatible as-is.

Usage:
    python3 convert_lora_adaln.py <in.safetensors> <out.safetensors>
"""
import sys
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file

PRUNED = "/workspace/models/MiniMax-H3-Pruned/transformer"


def load_basis():
    idx = json.load(open(f"{PRUNED}/diffusion_pytorch_model.safetensors.index.json"))["weight_map"]
    with safe_open(f"{PRUNED}/{idx['adaln_basis']}", framework="pt") as fh:
        return fh.get_tensor("adaln_basis").float()  # [8, 2688]


def main(src, dst):
    basis = load_basis()
    out = {}
    converted = 0
    with safe_open(src, framework="pt") as fh:
        keys = list(fh.keys())
        for k in keys:
            t = fh.get_tensor(k)
            if ".adaln_proj.linear.lora_A." in k:
                # [rank, 2688] -> [rank, 8]:  A' = A @ basis.T
                t = (t.float() @ basis.T).to(t.dtype).contiguous()
                converted += 1
            elif ".adaln_proj.linear.lora_B." in k:
                pass  # [out, rank] unchanged
            out[k] = t
    save_file(out, dst)
    print(f"converted {converted} adaln lora_A tensors -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
