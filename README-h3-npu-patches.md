# vllm-omni H3 Ascend 950PR Patches

Baseline: vllm-project/vllm-omni 0.28.0 (main snapshot, shipped wheel).

## What changed (commit 2)

| file | purpose |
|---|---|
| `vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py` | AdaLN-pruned checkpoint loading (H3_PRUNED_ADALN=1): time_embedder coord table [1025,8] w/ fp32 lerp, folded adaln bias buffers, pruned key normalization (transformer_blocks.->blocks. etc), QKV plain-cat fusion (per-head grouped layout == plain cat verified), fc1 value-first->gate-first half swap, turbo/native LoRA binding support |
| `vllm_omni/diffusion/attention/ops/minimax_h3_modulation.py` | import os (env access) |
| `vllm_omni/entrypoints/stage_utils.py` | deferred platform import (breaks import cycle) |
| `vllm_omni/model_executor/layers/rotary_embedding/mrope.py` | deferred NPU platform resolution |
| `vllm_omni/platforms/__init__.py` | pre-bind transitional platform object (re-entrant import guard) |
| `scripts/` | run_config.sh (all serving configs: bf16/int8/mxfp8/mxfp4/pruned-*/turbo-merges), gen_sample.sh (generation client), offline LoRA merge scripts (lightx2v turbo 4-step into pruned/dense base, larryvrh v4 with analytic adaln fold dW=B@(A@basis.T), db=B@(A@mean)) |
| `docs/REPORT.md` | full bring-up log: quantization comparison, turbo distillation speedups (4-step 43.4s / 8-step 87.6s e2e at 1344x768x5s), MFU analysis (~55-64% mixed), mosaic artifact investigation |

## Known open issue
Mosaic/checkerboard artifact on all pruned-model outputs (16-20px periodic, token-level decorrelation). Attention backend / RoPE kernel / weight loading / patchify all verified correct; root cause under active numerical bisection vs diffusers 0.40 reference.
