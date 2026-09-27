import os
import torch
import torch.nn.functional as F

_CAPTURED = set()
_STEP_COUNTS = {}

def maybe_capture_attention(layer_idx, q, k, v, block_size=128):
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
