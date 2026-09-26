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

## RESOLVED: mosaic artifact
Root cause: `MiniMaxH3Rope.inv_freq` was registered via `torch.empty` (uninitialized). Dense checkpoints ship `rope.inv_freq` in safetensors so the buffer got overwritten; pruned checkpoints (diffusers convention, non-persistent buffer) do not, leaving malloc garbage (~1e31) as rotary frequencies -> pseudo-random angle rotations -> total spatial decorrelation -> 16-20px checkerboard. Fixed by computing `inv_freq = theta^-(arange(0,32,2)/32)` at init. See docs/FIX_REPORT.md; verification sample: samples/fix_check.mp4 (clean golden-retriever frame).

## New agent onboarding
Start with `AGENTS.md` (hard constraints, baseline numbers, serving commands, TODOs) and `QUANT-NOTES.md` (W4A4 MXFP4 dualscale internals, mul_scale calibration plan, survey of community H3 W4A4 checkpoints). Full history in `docs/REPORT.md`.
