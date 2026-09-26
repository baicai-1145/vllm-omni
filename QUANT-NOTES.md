# 量化笔记 — W4A4 MXFP4 专题 (H3 @ 950PR)

> 配套 AGENTS.md。历史与速度数据见 REPORT.md。

## 本机可用量化栈 (vllm_omni/quantization/)

| method | 精度 | 状态 | 实测 |
|---|---|---|---|
| `int8` | W8A8 在线 | ✅ 可用 | 近无损 (MAD 4.27), ~1.18x |
| `mxfp4` | W4A4 单尺度在线 | ❌ 坍缩 (std 6.1) | 勿用 |
| `mxfp4_dualscale` | W4A4 双尺度在线/离线 | ✅ 可用 | 结构正常, MAD 40.2 色偏 |
| `svdquant` | NVFP4 W4A4 + rank32 低秩 | 未测 (Blackwell 向) | — |
| `mxfp8` | W8A8 | 可用 | — |

### mxfp4_dualscale 内部结构 (mxfp4_config.py)
- 权重: FP4 (float4_e2m1fn_x2, N,K 布局) + 两级 scale: fine per-32 (e8m0) + coarse per-512 (fp32)
- 激活: `npu_dynamic_dual_level_mx_quant(x, smooth_scale=mul_scale)` → `npu_dual_level_quant_matmul`
- 权重存 NZ 硬件格式 (`npu_format_cast(..., 29)`)
- **在线模式 mul_scale = 全1** (无平滑); 离线模式从 checkpoint 读 `mul_scale (K,) fp32`
- `num_bf16_fallback_layers`: 前 N 个块整体 BF16 (默认 5)
- `ignored_layers` (离线): config.json 标记哪些层保持 BF16

## 头号优化: mul_scale 校准 (预计消掉大部分 W4A4 色偏)

原理 (SmoothQuant/SVDQuant 共同思想): 激活某通道离群 → 用 per-channel scale 平滑迁移到权重侧, 让两者都落进 FP4 动态范围。

步骤草案:
1. 校准集: 8 prompts × 20 步 (参考 rootonchair/MiniMax-H3-nunchaku-lite-int4 "calibrated 8x20" 版本; data-free 版 = 只有 weight-span smoothing, 无激活校准)
2. hook 每个 quantized Linear 的 forward 输入 x, 记录 per-channel (K 维) 统计: absmax / 均值 (SmoothQuant 用 s_j = max|X_j|^α / max|W_j|^(1-α), α≈0.5)
3. mul_scale = 平滑因子 (K,) fp32; 离线: 写入 checkpoint 每层 `mul_scale` tensor; 或在线: 替换 process_weights_after_loading 里的全1向量 (可在 mxfp4_config.py 打补丁从 json 读)
4. 验证: 同 seed 生成, MAD vs bf16 从 40 目标 <15; 看图确认色偏消失

注意: x 的 K 维是 per-head grouped 还是 merged 取决于层 (qkv 21504 = 3×7168; fc1 28672 = 2×14336)——per-channel 统计要按运行时实际布局。

## fallback / ignored 权衡
- `num_bf16_fallback_layers`: 5→8 约 +3% 时间; 直接保护最敏感的前几块 (早期块决定全局结构)
- `ignored_layers` (需离线 checkpoint): 把 `blocks.*.attn.(qkv_proj|out_proj)` 留 BF16, 只量化 MLP — attention 对离群最敏感且只占 FLOPs 一部分

## 别人的 H3 W4A4 (调研结论, 2026-09)
- SVDQuant/nunchaku 系: smooth 迁移离群 → rank-32 bf16 低秩支路吸收 → 残差 GPTQ int4 → 激活 per-token int4。H3 已有: felipesztutman/MiniMax-H3-W4A4, ModelsLab/MiniMax-H3-svdquant-int4_r32 (GPTQ), rootonchair/MiniMax-H3-nunchaku-lite-int4 (data-free 与 calibrated-8x20 两版)
- ConvRot 系 (ComfyUI): 旋转重排 + INT8/INT4, pruned 单文件 11G 级
- NVFP4: Blackwell 原生; vllm-omni 的 svdquant config 对应此路线
- 本机对应物: MindIE-SD W4A4MXFP4(Dual)QuantLinear — 即 mxfp4_dualscale 的上游
- 共同点: **没有朴素 RTN**; 全部有离群吸收结构 + 校准

## 测量口径 (对比实验必须统一)
- 固定: seed 1101, 1344x768, 5.0s, 8 步 (或 4 步声明)
- 质量三件套: MAD vs bf16 基线帧 (中帧), std/entropy 参考值, **必须看图** (统计量漏检过棋盘)
- 速度: gen_sample.sh 的 wall 时间; 每步耗时从 server log `denoise_step_latency_ms`
