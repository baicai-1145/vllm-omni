# MiniMax-H3 剪枝模型棋盘伪影根因定位与修复报告

## 一、分叉算子定位
- **模块**: `vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer.MiniMaxH3Rope`
- **位置**: 位于模型前向准备阶段 `prepare_rope_table` 及各 Transformer Block 注意力前向 RoPE 旋转
- **影响范围**: 全部 50 层 Transformer Blocks 的注意力机制（影响所有序列 token）
- **数值分叉量级**:
  - 理论正常 `inv_freq`: `[1.0000e+00, 5.6234e-01, 3.1623e-01, ..., 1.7783e-04]`（16 维单调平滑递减）
  - 剪枝模型实测未初始化 `inv_freq`:
    `tensor([1.0005e+19, 7.7770e+31, 1.6533e+19, 3.1654e-40, ..., 2.5353e+30])`
  - **分叉量级**: 误差高达 **10^35 倍**！高频分量数值高达 `1e+31`，低频分量接近 `0`。

## 二、根因解释
1. **权重源格式差异**:
   - 官方稠密权重仓（FL2VA）在 safetensors 索引中显式包含了 `"rope.inv_freq"` 键，vllm-omni 的 `load_weights` 会在加载稠密模型时将其覆盖。
   - 社区剪枝版本（MiniMax-H3-Pruned）继承自 Diffusers 官方标准，Diffusers 的 `MiniMaxH3RotaryPosEmbed` 在 `__init__` 中以公式动态生成 `inv_freq` 并设为 `persistent=False`，因此剪枝权重 safetensors 中**完全不包含** `"rope.inv_freq"` 键。
2. **未初始化垃圾内存**:
   - vllm-omni 补丁中的 `MiniMaxH3Rope.__init__` 使用了 `self.register_buffer("inv_freq", torch.empty(inv_freq_len, dtype=_FP32_DTYPE))`。
   - 当加载剪枝模型时，safetensors 无此键，`inv_freq` 从未被赋值，直接保留了操作系统分配的未初始化随机内存（数值在 `1e-45` 到 `1e+31` 之间）。
3. **空间结构破坏与棋盘伪影**:
   - 当 token 位置坐标乘以 `1e+31` 时，RoPE 旋转相位角剧烈震荡退化为伪随机数。每个 token（对应 16x16 像素的 2x2 latent patch）在注意力层中计算出的注意力权重完全丧失了空间连续性，相邻 token 之间空间语义解耦，仅保留 VAE 局部解码的平滑色彩，宏观表现即为严整的 16x16 像素棋盘伪影。
4. **历史误判排查复盘**:
   - 之前排查时比对“NPU fused RoPE vs eager RoPE”，两者均读取了同一张由垃圾 `inv_freq` 算出的 `rope_table`，输出当然 bit-exact，导致排查方向错误地将 RoPE 排除。

## 三、修复 Diff
修改文件: `/root/h3env/lib/python3.12/site-packages/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py`

```diff
--- /workspace/.tmp/h3-mission/minimax_h3_transformer.py.prev.bak
+++ /root/h3env/lib/python3.12/site-packages/vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py
@@ -99,6 +99,7 @@
     adaln_out_features: int = 18 * 5376
     final_adaln_out_features: int = 2 * 5376
     rope_inv_freq_len: int = 16
+    rope_theta: float = 10000.0
     norm_eps: float = 1e-5
     # Pruned-checkpoint AdaLN rank (0 = released dense layout). When > 0 the
     # time embedder is an interpolated coordinate table and every AdaLN
@@ -280,11 +281,14 @@
     with 16 frequencies per axis (inv_freq = base^-(arange(0,32,2)/32)).
     """
 
-    def __init__(self, inv_freq_len: int) -> None:
+    def __init__(self, inv_freq_len: int, rope_theta: float = 10000.0) -> None:
         super().__init__()
+        inv_freq = 1.0 / (
+            rope_theta ** (torch.arange(0, 2 * inv_freq_len, 2, dtype=_FP32_DTYPE) / (2 * inv_freq_len))
+        )
         self.register_buffer(
             "inv_freq",
-            torch.empty(inv_freq_len, dtype=_FP32_DTYPE),
+            inv_freq,
             persistent=True,
         )
 
@@ -1174,7 +1178,7 @@
             arch,
             prefix="time_embedder",
         )
-        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len)
+        self.rope = MiniMaxH3Rope(arch.rope_inv_freq_len, rope_theta=getattr(arch, "rope_theta", 10000.0))
         self.token_refiner = MiniMaxH3TokenRefiner(
             arch,
             quant_config,
```

## 四、验证结果
- **服务配置**: `pruned-turboL` (larryvrh 8 步合并 + INT8 在线量化, 单卡 NPU, 端口 9101)
- **生成命令**: `bash gen_sample.sh 9101 /workspace/.tmp/h3-mission/samples/fix_check.mp4 1101 8 5.0 1344 768`
- **生成产物**:
  - 视频: `/workspace/.tmp/h3-mission/samples/fix_check.mp4` (7.46MB, 5.18s, 1344x768 24fps, H.264 + AAC)
  - 抽帧: `/workspace/.tmp/h3-mission/samples/frames/fix_check_frame2.png`
- **视觉质量**:
  - 棋盘伪影完全消除！
  - 金毛犬在阳光草地奔跑的动态形态、毛发纹理、眼睛、嘴部、草地茎叶及逆光光斑清晰连贯，画面质量与官方稠密参考帧（`int8_ref.png`）结构完全一致。
