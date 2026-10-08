import os
import torch
import torch.nn.functional as F

_CAPTURED = set()
_STEP_COUNTS = {}
_QKLOG = set()

def _fake_quant_per_token(t, bits):
    """Symmetric per-token uniform quantisation, in place.

    Matches what a real int8 attention kernel would do to Q and K: one scale
    per token (row), computed dynamically. This is the same granularity as
    torch_npu.npu_dynamic_quant's pertoken_scale, which is the path the int8
    roof (742 TOP/s) was measured through.
    """
    qmax = float(2 ** (bits - 1) - 1)
    s = t.abs().amax(dim=-1, keepdim=True).float().div(qmax).clamp(min=1e-12)
    t.copy_((t.float() / s).round_().clamp_(-qmax, qmax).mul_(s).to(t.dtype))
    return t


def _apply_qk_int8(layer_idx, q, k, v):
    """Emulate int8 attention inputs on the real activations. Quality only.

    This deliberately does NOT make anything faster: the arithmetic is still the
    bf16 FlashAttention. The question it answers is whether H3 can tolerate the
    int8 Q/K error at all -- if it cannot, writing an int8 AscendC kernel is
    pointless and the whole direction can be closed.
    """
    bits = int(os.environ.get("H3_ATTN_QK_INT8", "0"))
    if bits <= 0:
        return
    lo, hi = 0, 10**9
    rng = os.environ.get("H3_ATTN_QK_LAYERS", "").strip()
    if rng:
        a, _, b = rng.partition("-")
        lo, hi = int(a), int(b or a)
    if not (lo <= layer_idx <= hi):
        return
    for name, t in (("q", q), ("k", k)):
        before = t.float().norm()
        _fake_quant_per_token(t, bits)
        if (name, layer_idx) not in _QKLOG and os.environ.get("H3_ATTN_QK_LOG") == "1":
            after = t.float().norm()
            _QKLOG.add((name, layer_idx))
            print(f"[QKINT8] L{layer_idx:02d} {name} bits={bits} "
                  f"rel_err={((before-after)/before).item()*100:.3f}% "
                  f"absmax={t.abs().max().item():.3f}", flush=True)
    if os.environ.get("H3_ATTN_V_INT8", "0") == "1" and v is not None:
        _fake_quant_per_token(v, bits)


def maybe_capture_attention(layer_idx, q, k, v, block_size=128):
    try:
        _apply_qk_int8(layer_idx, q, k, v)
    except Exception as _e:  # never let the experiment break generation
        print(f"[QKINT8] disabled after error: {type(_e).__name__}: {_e}", flush=True)
    if os.environ.get("H3_CAPTURE_ATTN") != "1":
        return
    if layer_idx is None:
        return
    
    target_layers_str = os.environ.get("H3_CAPTURE_ATTN_LAYERS", "5,20,40")
    target_layers = {int(x.strip()) for x in target_layers_str.split(",") if x.strip()}
    if layer_idx not in target_layers:
        return
    
    step = _STEP_COUNTS.get(layer_idx, 0)
    _STEP_COUNTS[layer_idx] = step + 1
    
    target_step = int(os.environ.get("H3_CAPTURE_ATTN_STEP", "0"))
    if step != target_step:
        return
    
    key = (layer_idx, step)
    if key in _CAPTURED:
        return
    _CAPTURED.add(key)
    
    out_dir = os.environ.get("H3_CAPTURE_OUT_DIR", "/workspace/.tmp/h3-mission/worker-sparsefa/captured_maps")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[SPARSE_HOOK] Capturing attention for layer {layer_idx}, step {step}...", flush=True)
    
    # Compute exact block attention map
    # q, k: [S, N, D]
    S, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    q_h = q.transpose(0, 1).contiguous() # [N, S, D]
    k_h = k.transpose(0, 1).contiguous()
    
    n_blocks = (S + block_size - 1) // block_size
    block_attn = torch.zeros(N, n_blocks, n_blocks, dtype=torch.float32)
    
    head_chunk = 8
    q_block_chunk = 16
    for h_start in range(0, N, head_chunk):
        h_end = min(h_start + head_chunk, N)
        qh = q_h[h_start:h_end]
        kh = k_h[h_start:h_end]
        for qb_start in range(0, n_blocks, q_block_chunk):
            qb_end = min(qb_start + q_block_chunk, n_blocks)
            q_start = qb_start * block_size
            q_end = min(qb_end * block_size, S)
            cur_qb_count = qb_end - qb_start
            
            qc = qh[:, q_start:q_end]
            logits = torch.matmul(qc, kh.transpose(-2, -1)) * scale
            q_len = qc.shape[1]
            if q_len < cur_qb_count * block_size:
                pad_len = cur_qb_count * block_size - q_len
                logits = F.pad(logits, (0, 0, 0, pad_len), value=-1e4)
            probs = F.softmax(logits.float(), dim=-1)
            if S < n_blocks * block_size:
                probs = F.pad(probs, (0, n_blocks * block_size - S), value=0.0)
            probs_5d = probs.view(h_end - h_start, cur_qb_count, block_size, n_blocks, block_size)
            b_mass = probs_5d.sum(dim=-1).mean(dim=2)
            block_attn[h_start:h_end, qb_start:qb_end, :] = b_mass.cpu()
            
    map_path = os.path.join(out_dir, f"block_attn_layer_{layer_idx}_step_{step}.pt")
    torch.save(block_attn, map_path)
    print(f"[SPARSE_HOOK] Saved exact block attention to {map_path}", flush=True)
    
    if os.environ.get("H3_SAVE_QKV") == "1":
        qkv_path = os.path.join(out_dir, f"qkv_layer_{layer_idx}_step_{step}.pt")
        torch.save({"q": q.cpu(), "k": k.cpu(), "v": v.cpu()}, qkv_path)
        print(f"[SPARSE_HOOK] Saved QKV tensors to {qkv_path}", flush=True)
