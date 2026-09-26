# AGENTS.md — H3 on Ascend 950PR: 推理速度优化 + MXFP4 量化

## 使命
在单卡 Ascend 950PR (9579, 28 AIC, 128G HBM) 上跑 MiniMax-H3 剪枝模型的视频生成, 持续优化推理速度与 W4A4 MXFP4 画质。

## 最高优先级警告
**容器内存只有 32G, 严禁占满 — 一旦占满整机死机重启, 所有进行中工作全部丢失 (已发生过 2 次)。**
任何 Python 进程/加载/合并/物化脚本启动前必须估算峰值内存; 运行中 RSS 超 20G 立即中止优化。禁用一次性读全量权重的写法 (40G 权重哪怕流式合并也要分 shard 处理)。
- bash 工具 timeout ≤ 60s。长任务 `setsid nohup ... &` 后台跑; **禁止 sleep 轮询等待**。
- 杀 vllm 必须用 python os.kill (bash kill 无效):
  `python3 -c "import os,signal; [os.kill(int(p), signal.SIGKILL) for p in os.listdir('/proc') if p.isdigit() and open(f'/proc/{p}/comm').read().strip() in ('vllm','python') and int(p)!=310]"`
- `/workspace/models/` 全部只读; 实验产物放 `/workspace/.tmp/h3-mission/`; 正式产物放 `/workspace/h3-npu/`。
- 不重装 vllm_omni/diffusers (venv 里有手工补丁, 见下)。
- venv: `source /root/h3env/bin/activate` (python3.12, vllm_omni 0.28.0 + vllm_ascend + torch_npu)。

## 现状基线 (1344x768, 5s, seed 1101, 剪枝+turbo 合并权重)
| 配置 | 端到端 | 备注 |
|---|---|---|
| turboL 8步 + int8 | 87.6s | 画质近无损 (MAD 4.27 vs bf16) |
| turbo4 4步 + int8 | 43.4s | 快速档 |
| turboL 8步 + mxfp4_dualscale | 85.7s | W4A4, 结构正常但有色偏 (MAD 40.2) |
| 端到端 MFU | ~55-64% | 混合口径; attention 占 FLOPs 58% |

每步 ~9-10.5s (S=37760 packed tokens, 50 层, 3.5 PFLOPs/forward)。

## 服务器操作 (全部在 /workspace/h3-npu/scripts/)
```bash
# 启动 (后台, ~90s 就绪, 健康检查 curl 127.0.0.1:9101/health = 200)
cd /workspace/.tmp/h3-mission && setsid nohup bash run_config.sh pruned-turboL 9101 > server.log 2>&1 &
# 生成 (args: port out.mp4 seed steps duration W H)
bash gen_sample.sh 9101 out.mp4 1101 8 5.0 1344 768
# 抽帧质检
ffmpeg -y -v quiet -i out.mp4 -ss 2 -frames:v 1 frame.png
```
配置名 (run_config.sh): `pruned-turboL` (8步+int8) / `pruned-turbo4` (4步+int8) / `pruned-turboL-mxfp4ds` (8步+W4A4 dualscale)。

## 量化配方 (diffusion-quantization-config)
- int8 近无损: `{"transformer":{"method":"int8"}}`
- W4A4 dualscale (MindIE-SD 配方): `{"transformer":{"method":"mxfp4_dualscale","num_bf16_fallback_layers":5}}`
  - 单尺度 `mxfp4` 会完全坍缩 (std 6.1), 勿用
  - 在线模式 mul_scale=全1 (无校准) — **这就是当前 W4A4 色偏的来源, 头号优化项**
- 其它已注册: `svdquant` (NVFP4+rank32, Blackwell 向) / `inc` / `mxfp8`。源码: `/root/h3env/lib/python3.12/site-packages/vllm_omni/quantization/mxfp4_config.py`

## 优先 TODO
1. **mul_scale 校准** (W4A4 → 近无损的关键): 跑 N prompts × M 步, hook 每层 Linear 输入, 统计 per-channel scale, 注入离线 checkpoint 或运行时替换全1向量 (参考 rootonchair/MiniMax-H3-nunchaku-lite-int4 的 "calibrated 8×20" 与 SmoothQuant 思路)
2. fallback 层数扫描 (5→8/10, 每块 ~+1% 时间) 换质量
3. ignored_layers: attention qkv/out 留 BF16 只量化 MLP (attention 对离群最敏感)
4. 速度侧: attention 占 58% FLOPs (S²), 查 RainFusion/TeaCache 与 turbo 的兼容性; int8 GEMM 占比提升空间

## 必读
- `REPORT.md` — 完整实验史: 已排除的死路 (attention backend/rope kernel/朴素 mxfp4)、速度榜、MFU 推导、棋盘根因
- `FIX_REPORT.md` — 棋盘伪影根因 (RoPE inv_freq 未初始化) 与修复; 改加载逻辑前必读
- `vllm-omni-h3-npu/` — vllm_omni 补丁完整源码 (git, 同步在 github.com/baicai-1145/vllm-omni 分支 h3-ascend-950pr); site-packages 的 diff 与它一致
- `scripts/merge_larryvrh_pruned.py` — LoRA 离线合并参考 (含 adaln 解析折叠 dW=B@(A@basis.T), fc1 half-swap, qkv 切段)

## 已知坑
- dense 权重 (40G+) 在本容器加载即爆内存; 只能用剪枝底模
- 剪枝 safetensors 无 `rope.inv_freq` 键 → 补丁在 __init__ 直接计算, 若 rope 相关改动异常先查这里
- qkv: dense 是 per-head grouped [q|k|v]×56; 剪枝是 plain cat — 加载补丁处理过, 别按行序猜
- fc1: 剪枝 raw 为 value-first [up;gate], 运行时 gate-first — 合并 delta 时要 swap halves
- 画面质检必须看图 (std/entropy 无法检测结构损坏, 历史上误判过)
