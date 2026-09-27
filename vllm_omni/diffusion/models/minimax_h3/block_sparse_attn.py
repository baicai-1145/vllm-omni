# SPDX-License-Identifier: Apache-2.0
# Block-sparse attention for MiniMax-H3 on Ascend 950PR.
#
# Why: FlashAttention is 68% of the H3 denoise step and already runs at 99% MAC
# utilisation, so the only lever is computing fewer S^2 blocks. CANN 9.1.0 ships
# `npu_block_sparse_attention`, which skips masked blocks entirely (latency is
# strictly linear in blocks kept: 147ms dense -> 71ms at 50% for S=37760,
# H=56, D=128) but requires a caller-built [1, H, nb, nb] int8 block mask.
#
# The mask is produced without any dense pass: block importance is estimated as
# <mean_pool(q_block), mean_pool(k_block)>, a [nb, D] x [D, nb] GEMM per head
# (~0.03% of the S^2 attention FLOPs), then top-k per query block row. On
# captured real activations this estimator recovers 90-97% of the oracle's
# captured attention mass.
#
# 950PR constraint: `inner_precise=4` is mandatory (0/1 raise 561103).
#
# Env: H3_BLOCK_SPARSE_RATIO=<float 0..1> (0 disables; default disabled).
"""Pooled-QK block mask + npu_block_sparse_attention for the H3 packed layout."""

import os

import torch
import torch_npu

BLK = 128


def ratio() -> float:
    """Configured keep-ratio; <=0 disables the block-sparse path.

    ``H3_BLOCK_SPARSE_RATIO`` accepts either a single float (uniform across
    layers) or ``"lo:hi"``, which keeps ``lo`` for the first 60% of the DiT
    layers and ``hi`` for the rest. The split exists because attention mass is
    far more concentrated in shallow layers (block map capture at 30% of blocks:
    L5 0.82, L20 0.77, L40 0.61), so a uniform budget either wastes blocks early
    or destroys the deep layers.
    """
    try:
        raw = (os.environ.get("H3_BLOCK_SPARSE_RATIO", "0") or "0").strip()
    except Exception:  # noqa: BLE001
        return 0.0
    if ":" in raw:
        try:
            return float(raw.split(":", 1)[0])
        except ValueError:
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def ratio_for_layer(layer_idx: int, num_layers: int = 50) -> float:
    """Resolve the keep-ratio for one layer (see :func:`ratio` for the syntax)."""
    raw = (os.environ.get("H3_BLOCK_SPARSE_RATIO", "0") or "0").strip()
    if ":" not in raw:
        return ratio()
    try:
        lo_s, hi_s = raw.split(":", 1)
        lo, hi = float(lo_s), float(hi_s)
    except ValueError:
        return 0.0
    cut = int(round(num_layers * 0.6))
    return lo if layer_idx < cut else hi


def build_mask(q: torch.Tensor, k: torch.Tensor, keep: float) -> torch.Tensor:
    """q,k: [S, H, D] packed. Returns int8 block mask [1, H, nb, nb]."""
    s, h, d = q.shape
    nb = (s + BLK - 1) // BLK
    if nb < 2:
        return torch.ones(1, h, nb, nb, dtype=torch.int8, device=q.device)
    # [S,H,D] -> [H, nb, D] by averaging each BLK-token block
    qb = q[: nb * BLK].reshape(nb, BLK, h, d).permute(2, 0, 1, 3).float().mean(2)
    kb = k[: nb * BLK].reshape(nb, BLK, k.shape[1], d).permute(2, 0, 1, 3).float().mean(2)
    est = torch.bmm(qb, kb.transpose(1, 2)) * (d ** -0.5)      # [H, nb, nb]
    nsel = max(1, min(nb, int(round(keep * nb))))
    if nsel >= nb:
        return torch.ones(1, h, nb, nb, dtype=torch.int8, device=q.device)
    idx = est.topk(nsel, dim=-1).indices
    mask = torch.zeros(h, nb, nb, dtype=torch.int8, device=q.device)
    mask.scatter_(-1, idx, 1)
    return mask.unsqueeze(0)


def block_sparse_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           keep: float) -> torch.Tensor:
    """Packed [S, H, D] attention via npu_block_sparse_attention. Returns [S, H, D]."""
    s, h, d = q.shape
    nb = (s + BLK - 1) // BLK
    mask = build_mask(q, k, keep)
    # BNSD views of the packed tensors
    q4 = q.permute(1, 0, 2).unsqueeze(0).contiguous()
    k4 = k.permute(1, 0, 2).unsqueeze(0).contiguous()
    v4 = v.permute(1, 0, 2).unsqueeze(0).contiguous()
    out, _ = torch_npu.npu_block_sparse_attention(
        q4, k4, v4, mask, [BLK, BLK],
        q_input_layout="BNSD", kv_input_layout="BNSD",
        num_key_value_heads=h, scale_value=d ** -0.5, inner_precise=4,
    )
    return out.squeeze(0).permute(1, 0, 2).contiguous()
