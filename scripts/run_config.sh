#!/bin/bash
# 用法: ./run_config.sh <bf16|mxfp8|int8|mxfp4|flashgen|bf16pruned> [port]
# 单卡 950PR 官方配方适配 (源: MindIE-SD examples/minimax-h3/infer.md, vllm-omni 0.28.0)
# 注意: BASE 数组不含模型路径 —— 模型在各 case 分支显式指定, exec 时拼入
set -u
CFG=${1:?config required}
PORT=${2:-9098}
M=/workspace/.tmp/h3-mission
MODEL_DEFAULT=/workspace/models/MiniMax-H3/FL2VA
FLASHGEN_LORA=$M/flashgen-lora/minimax_h3_t2va_flashgen_4step_v1.0_768p_bf16.safetensors

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export VLLM_LOGGING_LEVEL=INFO

BASE=(vllm serve --omni --host 0.0.0.0 --port "$PORT" --trust-remote-code
      --task-type fl2va --num-gpus 1 --usp 1 --ring 1 --text-encoder-tp-size 1
      --vae-parallel-mode tile --vae-use-tiling --vae-patch-parallel-size 1
      --enable-diffusion-pipeline-profiler
      --init-timeout 3600 --stage-init-timeout 1800)

case "$CFG" in
  bf16)  # 官方推荐无损: DLO + FLASH_ATTN (需 host RAM ~135G, 32G 容器勿用!)
    MODEL=$MODEL_DEFAULT
    EXTRA=(--enable-distributed-layerwise-offload
           --diffusion-attention-backend FLASH_ATTN) ;;
  mxfp8) # 官方推荐有损: 无DLO + mxfp8 + EQBSA(mix, 8/12)
    MODEL=$MODEL_DEFAULT
    EXTRA=(--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"precision":"mix","start_step":8,"end_step":12}}}'
           --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}') ;;
  int8)  # 有损变体: int8 + RainFusion (bf16 精度, start 12)
    MODEL=$MODEL_DEFAULT
    EXTRA=(--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"start_step":12}}}'
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}') ;;
  mxfp4) # 在线 mxfp4: 必须去掉 DLO
    MODEL=$MODEL_DEFAULT
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"mxfp4"}}') ;;
  flashgen) # 官方 FlashGen 4步: 无损+DLO+LoRA (需 host RAM ~135G!)
    MODEL=$MODEL_DEFAULT
    EXTRA=(--enable-distributed-layerwise-offload
           --diffusion-attention-backend FLASH_ATTN
           --lora-backend peft --lora-path "$FLASHGEN_LORA") ;;
  pruned-bf16|bf16pruned) # 剪枝低秩无损参考: DiT 40G + TE 49G + VAE 10G ≈ 100G
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN) ;;
  pruned-int8) # 剪枝模型 + int8 量化 (attn/mlp), AdaLN 保持 bf16
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}') ;;
  pruned-mxfp8) # 剪枝模型 + mxfp8
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}') ;;
  pruned-mxfp4) # 剪枝模型 + mxfp4
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"mxfp4"}}') ;;
  pruned-int8-rf) # 剪枝+int8+RainFusion(官方同款: sparsity 0.8, start 12)
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"start_step":12}}}'
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}') ;;
  pruned-int8-rf-agg) # 更激进: sparsity 0.9, start 6, end 4
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.9,"start_step":6,"end_step":4}}}'
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}') ;;
  pruned-int8-tea) # 剪枝+int8+FLASH+TeaCache (H3 官方标定 0.17)
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}'
           --cache-backend tea_cache) ;;
  pruned-int8-tea-rf) # 叠加: TeaCache + RainFusion 0.9 + int8
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.9,"start_step":6,"end_step":4}}}'
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}'
           --cache-backend tea_cache) ;;
  pruned-turbo8) # 剪枝+Turbo 8步 LoRA (diffusers 布局, 无 adaln lora, 直接兼容)
    MODEL=/workspace/models/MiniMax-H3-Pruned
    export H3_PRUNED_ADALN=1
    TURBO_LORA=/workspace/models/turbo-lora/turbo4
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --lora-backend peft --lora-path "$TURBO_LORA") ;;
  pruned-turbo4) # 剪枝模型 + lightx2v 4-step 合并 + int8 在线量化 (turbo4 与 W8A8 叠加)
    MODEL=/workspace/models/MiniMax-H3-Pruned-turbo4
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}') ;;
  dense-turbo4) # 官方 dense + Turbo 4-step LoRA 离线合并 (DLO 必开, 否则 HBM OOM)
    MODEL=/workspace/models/MiniMax-H3-turbo4
    EXTRA=(--enable-distributed-layerwise-offload
           --diffusion-attention-backend FLASH_ATTN) ;;
  pruned-turboL) # 剪枝模型 + larryvrh 8step 合并 + int8 在线量化
    MODEL=/workspace/models/MiniMax-H3-Pruned-turboL
    export H3_PRUNED_ADALN=1
    EXTRA=(--diffusion-attention-backend FLASH_ATTN
           --diffusion-quantization-config '{"transformer":{"method":"int8"}}') ;;
  *) echo "unknown config: $CFG"; exit 1 ;;
esac

exec "${BASE[@]}" "$MODEL" "${EXTRA[@]}"
