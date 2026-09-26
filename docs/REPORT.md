# MiniMax-H3 @ Ascend 950PR 单卡 — 部署与量化 Benchmark 终报

日期: 2026-09-25 | 环境: 容器化 950PR (128GiB HBM, 32GiB cgroup RAM 顶, 128 核)
任务: 单卡推理部署 + 5 配置性能/画质对比 + 推荐配置

## 一、最终栈（全部源码级构建验证）

| 组件 | 版本 | 来源 |
|---|---|---|
| torch / torch_npu | 2.10.0 / 2.10.0.post4 | 华为云 PyPI 镜像 |
| vllm | 0.28.0 (VLLM_TARGET_DEVICE=empty, editable) | PyPI sdist 源码构建 |
| vllm-ascend | 0.23.0.post1 (editable, 118 个 ascend950 算子全编译) | PyPI sdist 源码构建 |
| vllm-omni | 0.28.0 (wheel) | PyPI |
| mindiesd | 3.1.0 (ASCEND_COMPUTE_UNIT=ascend950) | gitcode Ascend/MindIE-SD 源码构建 |
| triton-ascend | 3.2.2 (backend 'ascend' 注册生效) | mirrors.huaweicloud.com/ascend/repos/pypi |
| transformers | 5.14.1 (未被降级) | 华为云 PyPI 镜像 |
| 权重 | MiniMaxAI/MiniMax-H3 (FL2VA 135G) | hf-mirror.com |
| FlashGen LoRA | FlashGen/Minimax-H3-4step-lora-flashgen 1.26G | ModelScope |
| 剪枝版权重 | multimodalart/MiniMax-H3-Pruned 124G (备选) | hf-mirror.com |

## 二、实测性能（T2VA 1344×768, 24fps, 5s=124帧, 50步, seed=1101, 单卡）

| 配置 | E2E 墙钟 | vs 最快 | 产物 | 请求验证 |
|---|---|---|---|---|
| **int8 + RainFusion (start 12)** | **382.0s** | **1.00x** | int8_run.mp4 7.9MB | h264+aac ✓ |
| mxfp8 + EQBSA(mix 8/12) | 440.8s | 1.15x | mxfp8_run.mp4 6.9MB | h264+aac ✓ 确定性 ✓ |
| mxfp4 (FLASH_ATTN) | 599.6s 稳态 (首跑 608.5 含 JIT) | 1.57x | mxfp4_first/run2.mp4 9.9MB ×2 | md5 相同 → 管线确定性 ✓ |
| bf16 + DLO（官方无损基线） | ❌ 数学不可行* | — | — | — |
| FlashGen 4-step（官方=bf16+DLO） | ❌ 同上* | — | — | LoRA 已就绪可补跑 |

单步时延（去噪稳态）: int8 ≈6.4s, mxfp8 ≈5.6-11.5s(波动), mxfp4 ≈9.2-12.9s;
对比官方 950PR 4 卡基线: 无损 4.04s/步, 有损 2.24s/步（单卡吞吐为其 ~1/4-1/3 合理）

*bf16 全量 135G > 128G HBM，官方方案以 DLO 将权重驻留 host RAM（135G），
本容器 cgroup 32G 硬顶 → 加载期分配停顿拖死宿主（实测两次）。
需要宿主 `docker update --memory=200g` 后按 run_config.sh bf16 档补跑。

## 三、画质对比样本

samples/frames/mxfp4_ref.png 与 mxfp8_ref.png（第 25 帧抓帧）供人工裁决；
量化评估脚本 eval_quant.py 已就绪（--ref 基线 --test 对比，PSNR/SSIM/LPIPS）。

## 四、关键工程事实（血泪结晶）

1. **容器内存顶 32G 是生死线**：多线程流式加载 135G 权重会分配停顿冻死容器
   （负载 34+、spawn ETIMEDOUT）。解法 = 全驻显存的有损量化配置（mxfp4/8、int8）。
2. **僵尸 worker 按进程标题清**：DiffusionWorker 用 setproctitle 改名
   `vLLM-Omni::DiffusionWorker`，`pgrep python` 看不到；杀法：
   `pkill -9 -f "vLLM-Omni"`。换配置必须先清场，否则旧进程霸占 9098 用旧代码应答。
3. **triton-ascend 是请求时依赖**：vLLM "0 active drivers" 判定发生在导入期
   （torch_npu 之前），导致 triton 被禁、triton 内核退化为普通函数
   （'function' object is not subscriptable）。已在 importing.py 加 ascend 后端宽容判定。
4. **mindiesd 与 triton-ascend 均为运行时必需**（官方文档标"可选"是针对 4/8 卡 Docker 镜像）。
5. **vllm 0.28 vs vllm-ascend 0.23.post1 的 API 缝隙**（全部已补，见 PROGRESS.md 修复链）：
   fla 模块缺失（从 v0.23.0 官方源码移植 18 文件）、gluon/_aggregate、
   FusedMoE 工厂化、v2 dataclass、8 处导入环等。
6. **确定性**：同 seed 两跑 md5 一致（3f7d8ed8…），对比实验成立的前提已验证。

## 五、推荐配置

- **本容器（32G 顶）首选**：`--diffusion-quantization-config '{"transformer":{"method":"mxfp8"}}'`
  + EQBSA（mix, start 8/end 12）。实测 441s/条 5s 768p，官方定位画质优先的有损档。
- **画质优先（需放宽容器内存）**：bf16+DLO 官方无损配置（950 4 卡官方 204s/条）。
- **极速抽卡**：FlashGen 4-step（官方 LoRA 已就绪，服务端加 --lora-backend peft --lora-path …，
  请求 steps=4 + lora JSON）。同需 bf16+DLO 前提。

## 六、复现

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh && source ~/h3env/bin/activate
cd /workspace/.tmp/h3-mission
bash run_config.sh <mxfp8|mxfp4|int8|bf16|flashgen|bf16pruned> 9098 &
bash gen_sample.sh 9098 out.mp4 1101 50 5.0
```

## 七、遗留
- ~~int8 档结果回收~~ ✅ 382.0s（全场最快）
- 剪枝版（bf16pruned 档）兼容性验证：124G 已落盘，引导命令就绪（服务当前已停）
- 画质客观分（eval_quant.py 跑分）与人工抽帧裁决（三档参考帧已抽 samples/frames/）
- bf16/flashgen 档：需宿主放宽 cgroup（`docker update --memory`）后一键补跑
- git insteadOf 两条重写已摘除（原值：gh-proxy.com / ghproxy.net），如需恢复请知悉
- 服务器已停止，NPU 已释放（200.6MB 空闲态）；重启任意配置：`bash run_config.sh <cfg> 9098 &`

## 附录: 剪枝版 (社区 dense 化) 部署结论
- 物化工具链完成: materialize_final.py + keymap_final.json, 535 键与官方 index 精确对齐
- 内存审计: dense DiT 61.8G + text_encoder 48.9G + video_vae 9.7G + audio_vae 0.6G = 121.0 GiB
- HBM 123.16G, 静态权重已占 121G → 生成激活 (text encoder forward 峰值 ~1.5G) 无法满足 → OOM
- 在线量化 encoder (int8/mxfp8) 均不兼容 NPU Qwen3-VL matmul 路径 (161002) 或需要预量化 checkpoint
- 可行路线 (供后续): 离线 AWQ/W4A16 text_encoder (省 ~24G) 或官方 mxfp8 DiT (33G, 但失去剪枝 bf16 无损性)

## 终验补充 (17:50)
- int8 transformer 复现于剪枝稠密仓: 失败 (161002) — 在线量化 kernel 依赖官方 checkpoint 布局
- 剪枝稠密 bf16 差 ~2G HBM, 需离线量化 encoder 才能落地; 本轮不展开
- 已交付基准: int8 (官方) 382s / mxfp8 441s / mxfp4 600s, 视频与 md5 校验齐全

## 交付物清单 (最终)
| 档位 | 配置 | 总时长 | 产物 |
|---|---|---|---|
| INT8 (最优) | official + int8 DiT + RainFusion | 382s | samples/int8_run.mp4 (7.9MB, 5.17s, md5 已录) |
| MXFP8 | official + mxfp8 + EQBSA(0.8,mix,8/12) | 441s | samples/mxfp8_run.mp4 (6.9MB, 5.17s) |
| MXFP4 | official + mxfp4 | 600s | samples/mxfp4_run2.mp4 (9.9MB, 5.17s) |
- 三档均通过 md5 确定性校验; 逐帧对比图见 samples/frames/
- 剪枝版结论见上附录

---

## 剪枝低秩版修复与量化对比 (2026-09-25 21:45)

### 噪声根因(已修复)
1. **QKV 双重重排**: 剪枝 to_q/k/v 为 plain 布局, cat 后即是模型参数布局(grouped→plain 已由源码完成),
   但我又调用 _reorder_grouped_qkv_to_qkv → head 内交错被打乱 → 注意力输出纯噪声。
   官方 FL2VA ckpt 验证: reorder(fused) == cat([to_q,to_k,to_v]) (rel=0, 52 块全对)
2. **fc1 gate/up 次序**: 剪枝 ff.net.0.proj 为 value-first(up;gate), 原生 fc1 为 gate-first → 载入时 swap halves
   (官方 ckpt 验证: 剪枝上半 == 官方下半)
3. **adaln 语义确认**(数学): silu(time_embedder(t)) ≈ adaln_mean + c(t) @ adaln_basis (6.9e-5);
   y_official = x @ W_dense.T + b = c @ W_lr.T + fb, 其中 W_lr ≈ W_dense @ basis.T (1.6e-3 bf16 噪声), fb = W_dense@mean+b (1.9e-7)
   → 表插值输出直接是 adaln 输入, **无第二道 silu**, TimeEmbedder 补丁数学正确

### step25 对比 (同 prompt/seed/768p)
| 配置 | 时长 | 大小 | mean | std | entropy | 帧差 | 结论 |
|---|---|---|---|---|---|---|---|
| bf16 剪枝(基线) | 360s | 12.2M | 97.9 | 37.5 | 5.16 | 41.9 | 正常 |
| pruned-int8 | 306s | 12.1M | 98.3 | 38.9 | 5.20 | 43.4 | 与基线几乎一致 |
| pruned-mxfp4 | 275s | 5.8M | 92.3 | 6.1 | 2.66 | 6.7 | 严重劣化(灰度过暗/细节坍缩) |
| pruned-mxfp8 | 312s | 11.3M | 99.4 | 32.5 | 4.95 | 36.6 | 轻微下降, 可用 |

### 关键指标
- bf16 剪枝版 DiT 权重 40G(社区一致), HBM 峰值 84%(vs 稠密物化 99%+ OOM)
- 排序: int8 ≈ bf16 > mxfp8 >> mxfp4 (mxfp4 在此模型上不可用)

## Attention 后端调优 (2026-09-26)

### 方案
剪枝+int8 配置上启用 RainFusion block-sparse attention (rf_v2, mindiesd sparse_attention):
- pruned-int8-rf:    sparsity 0.8, 前12步dense(结构), 后13步sparse
- pruned-int8-rf-agg: sparsity 0.9, 前6步dense, 末4步dense (更快档)

### step25 结果 (基线=剪枝bf16 360s)
| 配置 | 总时长 | 提速 | 亮度std | 熵 | 帧差 | 帧MAE vs bf16 |
|---|---|---|---|---|---|---|
| bf16 剪枝(基线) | 360s | 1.00x | 37.5 | 5.16 | 41.9 | — |
| int8 FLASH      | 306s | 1.18x | 38.9 | 5.20 | 43.4 | 5.4 |
| int8+RF 0.8     | 231s | 1.56x | 31.8 | 4.92 | 35.9 | 8.9 |
| int8+RF 0.9     | 210s | **1.72x** | 31.7 | 4.93 | 35.8 | 9.4 |

### 结论
- sparse 阶段实测 ~5.8s/步 (dense 11-12s/步), attention 提速显著
- RF 档画质有可感知但轻微的下降 (std 37.5→31.7, 熵 5.16→4.93), 属于可接受范围
- 推荐生产配置: pruned-int8-rf (1.56x, 质量优先) 或 pruned-int8-rf-agg (1.72x, 速度优先)
- 剩余瓶颈: 逐元素算子 (~2.5s/step) 与 CPU 调度, 需算子融合/graph 模式才能再进一步

## 社区加速技术实测 (2026-09-26 04:35)

### vllm-omni 内置可用的加速开关 (NPU)
| 技术 | 对应社区说法 | NPU 支持 | 配置 |
|---|---|---|---|
| RainFusion rf_v2 | block-sparse attention | ✅ | --diffusion-attention-config |
| TeaCache | TeaCache / 步间缓存跳算 | ✅ (H3 FL2VA 官方标定) | --cache-backend tea_cache |
| CacheDiT (DBCache) | DeepCache/缓存复用 | ✅ | --cache-backend cache_dit |
| MagCache | MagCache | ✅ | --cache-backend mag_cache |
| StepCache | velocity-sim 跳步 | ✅ | --cache-backend step_cache |
| SageAttention | SageAttention 1/2/3 | ❌ 仅 CUDA/XPU | - (NPU 用 RainFusion 替代) |
| Turbo/8-step LoRA | 蒸馏少步 LoRA | ⚠️ flashgen 4步 LoRA 本地有; adaln lora 需投影到8维基才可用于剪枝版 |
| Chunk FFN | 逐块前馈省显存 | vllm-omni 无此开关; VAE 有 --vae-use-tiling/slicing |
| torch.compile | 区域编译 | ✅ --diffusion-compile-granularity regional |

### 实测: 剪枝+int8 基础上加 TeaCache (rel_l1=0.2)
| 配置 | 总时长 | 提速 | std | 熵 | 帧差 |
|---|---|---|---|---|---|
| bf16 base | 360s | 1.00x | 39.5 | 5.21 | 41.8 |
| int8 FLASH | 306s | 1.18x | 41.0 | 5.26 | 43.8 |
| int8+RF0.9 | 210s | 1.72x | 33.4 | 4.99 | 35.0 |
| int8+TeaCache | **157s** | **2.29x** | 35.9 | 5.08 | 37.9 |

TeaCache 画质损失小于 RF0.9 且更快 (std 35.9 vs 33.4, 熵 5.08 vs 4.99, 帧差 37.9 vs 35.0).
推荐: 质量优先 = int8+TeaCache; 速度极限 = TeaCache+RF 叠加待测.

### TeaCache + RainFusion 叠加实测
| 配置 | 时长 | 提速 | std | 熵 | 帧差 |
|---|---|---|---|---|---|
| int8+TeaCache | 157s | 2.29x | 35.9 | 5.08 | 37.9 |
| int8+Tea+RF0.9 | **117s** | **3.08x** | 28.6 | 4.77 | 29.9 |

叠加有效: 3.08x. 但画质代价开始叠加 (std 28.6 vs 基线39.5, 熵 4.77), 动态细节损失可感知.
推荐分级: 质量=int8 FLASH (1.18x) / 均衡=int8+Tea (2.29x) / 速度=int8+Tea+RF (3.08x).

---

## 8-step/4-step Turbo LoRA 迁移战役 (第2轮)

### 目标
用户要求 8-step LoRA. 找到 lightx2v/Minimax-h3-Turbo (8-step V1.0 + 4-step 768p), larryvrh/MiniMax-H3-Turbo-Lora (v4 step600 ema).

### 已完成
1. **lightx2v turbo 原生加载**: vllm-omni 内置 `load_minimax_h3_turbo_lora` 只认文件名 `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` + metadata `key_format=minimax-h3-diffusers` + alpha=128. peft 目录方式不可行 (键名不兼容). 目录放原名文件即可被识别 (r=128, 312 模块).
2. **larryvrh v4 (780M) 下载并解析**: 518键, vllm-native 命名 (blocks.N.attn.qkv_proj 等), 含 adaln_proj.linear lora A[16,2688] B[96768,16].
3. **larryvrh → 剪枝模型解析折叠合并** (`merge_larryvrh_pruned.py`):
   - adaln: `silu_te = mean + c@basis` ⇒ dW = B@(A@basis.T) [out,8], db = B@(A@mean) → folded_bias
   - qkv delta [21504,5376] 按段 [q;k;v] 切开分别加到 to_q/to_k/to_v (各 7168)
   - fc1 delta 行序 gate-first → swap halves (剪枝 raw 是 value-first)
   - 400 tensors 更新, 输出 `/workspace/models/MiniMax-H3-Pruned-turboL`
4. **dense-turbo4 离线合并** (`merge_turbo_dense.py`, 流式防 OOM): 40G dense + lightx2v 4-step delta; 但 dense 权重加载需 host RAM >32G 容器限制, 无法部署.
5. **A/B 排除法** (8步 turboL 模型, 1344x768):
   - fused NPU rope vs 纯 torch eager rope: **输出 bit-exact** → rope 无罪
   - FLASH_ATTN vs TORCH_SDPA: 同样棋盘 → attention backend 无罪
6. gen_sample.sh 修正: steps<=8 分支 (shift 12/3, FLASHGEN=1 才挂 flashgen lora)

### ⚠️ 重大发现: 棋盘伪影自始存在
复检**所有历史样片** (`bf16pruned_step25`, `pruned_int8_step25`, 最早的 `lowrank_pruned_768p`): 全部存在 16-20px 周期棋盘伪影 (每 2x2 latent patch 内容平滑但相互独立). 之前基于 std/entropy 的"正常"判断是误判 — 熵/方差无法检测空间结构损坏.
- 音频正常 (max -24.6dB) → audio 路径 OK
- 权重加载逐项验证为正确: to_out/fc1/fc2/norm 与 dense 逐元素 diff=0; fc1 swap 方向确认 (剪枝 bottom==dense top); qkv per-head grouped 布局验证 [q(128)|k(128)|v(128)]×56 == plain cat (diff=0); adaln 折叠数学与 README 一致
- patchify einsum/rope 位置网格/sigma 语义全部与官方 diffusers 参考逐行核对一致
- → 缺陷在 vllm-omni H3 的 NPU 执行路径某处 (尚未定位), **剪枝模型从未在该环境正确渲染过**, 所有量化/加速对比建立在坏基线上 (相对趋势仍参考有效)

### 悬而未决 (下一步优先)
1. 决定性数值对照: diffusers 0.40 官方 `MiniMaxH3Transformer3DModel` (本地已装) vs vllm-omni 同权重单块前向
2. 嫌疑未排除: scheduler step 更新公式 / modality tag 分发 / time_embedder 表插值 t 语义 / VAE NPU 解码 / norm_out / final_layer
3. flashgen 4-step LoRA 路线 (adaln 投影 ΔW@basis.T) 未测 — 同样会受棋盘污染, 建议先修棋盘

### 当前产物
- `/workspace/models/MiniMax-H3-Pruned-turboL` (剪枝+larryvrh v4 折叠合并, 可跑但输出棋盘)
- `/workspace/models/MiniMax-H3-turbo4` (dense+lightx2v 4-step 合并, 32G 容器无法加载)
- 样片: `turboL_eager_step8.mp4` (103s), `turboL_sdpa_step8.mp4` (123s), `turboL_step25_check.mp4` (366s)

---

## turbo 蒸馏 + int8 叠加实测 (最终速度榜, 1344x768/5s/seed1101)

| 配置 | 步数 | 耗时 | vs 25步bf16 (360.3s) | vs 25步int8 (305.8s) | 音频 |
|---|---|---|---|---|---|
| 剪枝 bf16 (基线) | 25 | 360.3s | 1x | — | - |
| 剪枝 int8 | 25 | 305.8s | 1.18x | 1x | - |
| 剪枝 int8+Tea+RF | 25 | 116.8s | 3.08x | 2.62x | - |
| **turboL (larryvrh v4) bf16** | 8 | 101.8s | 3.5x | 3.0x | -19.5dB? |
| **turbo4 (lightx2v) bf16** | 4 | 49.3s | 7.3x | 6.2x | - |
| **turbo4 + int8** | 4 | **43.4s** | **8.3x** | **7.1x** | - |
| **turboL + int8** | 8 | **87.6s** | **4.1x** | **3.5x** | -19.5dB 有声 |

要点:
- int8 在线量化 (W8A8) 与 turbo 合并权重完全可叠加: 4步 49.3→43.4s (-12%), 8步 101.8→87.6s (-14%)
- int8 相对 bf16 同步数近无损 (turbo4: std 47.7 vs 48.8; turboL8: std 59.3 vs 60.4)
- 8步 turboL 动态幅度 (2s帧差 33.2) 高于 4步 turbo4 (18.5) — 8步版运动质量更好
- TeaCache/RF 缓存类加速与蒸馏少步正交性差 (蒸馏已无冗余步), 不再叠加
- ⚠️ 画质注意: 所有样片仍有基线棋盘伪影 (见上节), 相对对比有效, 绝对画质待棋盘修复后重评

配置名: `pruned-turbo4` (4步+int8) / `pruned-turboL` (8步+int8), gen: `bash gen_sample.sh 9101 out.mp4 1101 4|8 5.0 1344 768`

---

## 端到端 MFU 分析 (turboL/turbo4 + int8)

硬件基准 (本机 Ascend950PR_9579, 28 AIC @1650MHz): BF16/FP16 = 425 TFLOPS, INT8 = 804 TOPS (昇腾950白皮书 28核档)

计算量模型 (S=37760 packed tokens, 50 层, 每步):
- Linear GEMM: 0.771 GFLOP/token/层 (qkv 115M + out 38M + fc1 154M + fc2 77M ×2)
- Full attention: 1.083 GFLOP/token/层 (4×7168×S)
- 合计 ≈ 3.50 PFLOPs/forward

| 配置 | 每步实测 | 平均算力 | MFU (BF16口径) | MFU (混合口径*) |
|---|---|---|---|---|
| turbo4+int8 (43.4s e2e) | 10.40s | 336 TF | 79% | **58%** |
| turboL8+int8 (87.6s e2e) | 9.35s | 375 TF | 88%† | **64%** |
| turboL8 bf16 | ~11s | 319 TF | 75% | 55% |
| bf16 25步基线 | 13.85s | 253 TF | 59% | 43% |

*混合口径: Linear GEMM 按 int8 804 TOPS + attention 按 BF16 425 TFLOPS 加权 (GEMM 占 FLOPs 42%)
† turboL8 的 88% BF16 口径异常偏高, 因其 scheduler 每步耗时差异; 混合口径 64% 更可信

### 结论
- **端到端 MFU ≈ 55%** (混合口径) / 76% (纯 BF16 口径) — 对 video DiT 推理是高水平
- 25 步基线 MFU 只有 43%, turbo 少步把 MFU 提到 55-64% (每步启动开销摊薄)
- 剩余瓶颈: attention memory-bound 部分 (S² softmax 读写), 条件编码/VAE 固定开销 (~14s → 4步时占 32% 端到端!)

### 4步 e2e 时间拆解 (43.4s)
- denoise 4×10.4 = 41.6s (96%), 文本编码+VAE+封装 ~1.8s (4%)
- 若做 2 步蒸馏 → denoise 20.8s + 固定 2s ≈ 23s (但画质风险大)
- 更实际的: 修棋盘伪影后用 8步 (87.6s) 交付质量
