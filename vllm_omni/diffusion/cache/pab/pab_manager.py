# SPDX-License-Identifier: Apache-2.0
"""Pyramid Attention Broadcast (PAB) manager for MiniMax-H3 on Ascend NPU.

Reference:
- Paper: Real-Time Video Generation with Pyramid Attention Broadcast (ICLR 2025, arXiv:2408.12588)
- Implementation: Hooks into MiniMaxH3Attention forward to broadcast attention outputs
  across adjacent diffusion steps according to layer variance and pyramid scheduling.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from vllm.logger import init_logger
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)

_GLOBAL_PAB_MANAGER: PABManager | None = None
_CONTROL_FILE = "/workspace/.tmp/h3-mission/worker-pab/pab_control.json"


def is_pab_enabled() -> bool:
    """Fast check whether PAB is enabled via control file or env var."""
    if os.path.exists(_CONTROL_FILE):
        try:
            with open(_CONTROL_FILE, "r") as f:
                data = json.load(f)
            return bool(data.get("enabled", False))
        except Exception:
            pass
    val = os.environ.get("H3_ENABLE_PAB", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def is_pab_calibrate_enabled() -> bool:
    if os.path.exists(_CONTROL_FILE):
        try:
            with open(_CONTROL_FILE, "r") as f:
                data = json.load(f)
            return bool(data.get("calibrate", False))
        except Exception:
            pass
    val = os.environ.get("H3_PAB_CALIBRATE", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_pab_manager() -> PABManager:
    global _GLOBAL_PAB_MANAGER
    if _GLOBAL_PAB_MANAGER is None:
        _GLOBAL_PAB_MANAGER = PABManager()
    return _GLOBAL_PAB_MANAGER


class PABManager:
    """Manages attention broadcasting schedules and tensor caching across diffusion steps."""

    def __init__(self) -> None:
        self.enabled = is_pab_enabled()
        self.calibrate = is_pab_calibrate_enabled()

        # Scheduling configuration
        self.broadcast_range = int(os.environ.get("H3_PAB_RANGE", "2"))
        self.start_step = int(os.environ.get("H3_PAB_START_STEP", "2"))
        self.end_step = int(os.environ.get("H3_PAB_END_STEP", "1"))
        self.policy = os.environ.get("H3_PAB_POLICY", "pyramid").strip().lower()
        self.layers_config = os.environ.get("H3_PAB_LAYERS", "adaptive").strip()
        self.max_cached_layers = int(os.environ.get("H3_PAB_MAX_CACHED_LAYERS", "25"))

        # Explicit broadcast steps override (e.g. "2,4,6")
        custom_steps = os.environ.get("H3_PAB_BROADCAST_STEPS", "")
        self.custom_broadcast_steps: set[int] | None = None
        if custom_steps:
            self.custom_broadcast_steps = {
                int(s.strip()) for s in custom_steps.split(",") if s.strip().isdigit()
            }

        # Calibration output path
        self.calib_output = os.environ.get(
            "H3_PAB_CALIB_OUTPUT",
            "/workspace/.tmp/h3-mission/worker-pab/pab_layer_variances.json",
        )
        self.calib_file = os.environ.get("H3_PAB_CALIB_FILE", self.calib_output)

        # Tier ratios for adaptive grouping
        self.tier1_ratio = float(os.environ.get("H3_PAB_TIER1_RATIO", "0.40"))
        self.tier2_ratio = float(os.environ.get("H3_PAB_TIER2_RATIO", "0.20"))

        # Cache storage: layer_idx -> torch.Tensor (allocated on NPU)
        self.cache: dict[int, torch.Tensor] = {}

        # Calibration storage:
        # calib_prev: layer_idx -> torch.Tensor
        # calib_diffs: list of dicts {layer_idx: diff_val} per step
        self.calib_prev: dict[int, torch.Tensor] = {}
        self.calib_diffs: list[dict[int, float]] = []

        # Tracking state
        self.current_request_step: int | None = None
        self.last_seen_step: int | None = None
        self.broadcast_counts: dict[int, int] = {}
        self.total_attention_calls = 0
        self.broadcasted_attention_calls = 0

        # Layer group assignments (0 to 49)
        self.tier1_layers: set[int] = set()
        self.tier2_layers: set[int] = set()
        self.tier3_layers: set[int] = set()

        self._reload_control_file()
        self._init_layer_groups()

        logger.info(
            "PABManager initialized: enabled=%s, policy=%s, range=%d, "
            "start_step=%d, end_step=%d, layers=%s, max_cached=%d, calibrate=%s",
            self.enabled,
            self.policy,
            self.broadcast_range,
            self.start_step,
            self.end_step,
            self.layers_config,
            self.max_cached_layers,
            self.calibrate,
        )

    def _reload_control_file(self) -> None:
        """Dynamically reload parameters from control file if present."""
        if not os.path.exists(_CONTROL_FILE):
            return
        try:
            with open(_CONTROL_FILE, "r") as f:
                cfg = json.load(f)
            self.enabled = bool(cfg.get("enabled", self.enabled))
            self.calibrate = bool(cfg.get("calibrate", self.calibrate))
            if "range" in cfg:
                self.broadcast_range = int(cfg["range"])
            if "start_step" in cfg:
                self.start_step = int(cfg["start_step"])
            if "end_step" in cfg:
                self.end_step = int(cfg["end_step"])
            if "policy" in cfg:
                self.policy = str(cfg["policy"]).strip().lower()
            if "layers" in cfg:
                self.layers_config = str(cfg["layers"]).strip()
            if "max_cached_layers" in cfg:
                self.max_cached_layers = int(cfg["max_cached_layers"])
            if "broadcast_steps" in cfg:
                steps_val = cfg["broadcast_steps"]
                if isinstance(steps_val, list):
                    self.custom_broadcast_steps = set(steps_val)
                elif isinstance(steps_val, str) and steps_val:
                    self.custom_broadcast_steps = {
                        int(s.strip()) for s in steps_val.split(",") if s.strip().isdigit()
                    }
                else:
                    self.custom_broadcast_steps = None
            if "tier1_ratio" in cfg:
                self.tier1_ratio = float(cfg["tier1_ratio"])
            if "tier2_ratio" in cfg:
                self.tier2_ratio = float(cfg["tier2_ratio"])
            self._init_layer_groups()
            logger.info(
                "PAB control file reloaded: enabled=%s, policy=%s, range=%d, start_step=%d, end_step=%d",
                self.enabled,
                self.policy,
                self.broadcast_range,
                self.start_step,
                self.end_step,
            )
        except Exception as e:
            logger.warning("Failed to reload PAB control file: %s", e)

    def _init_layer_groups(self) -> None:
        """Initialize layer grouping into Tier 1 (high stability), Tier 2 (medium), Tier 3 (dense)."""
        num_layers = 50

        # If custom layer range is specified (e.g. "10-39" or "5,6,7")
        if self.layers_config not in ("adaptive", "all"):
            explicit_layers = self._parse_layer_range(self.layers_config, num_layers)
            self.tier1_layers = explicit_layers
            self.tier2_layers = set()
            self.tier3_layers = set(range(num_layers)) - explicit_layers
            logger.info("PAB explicit layers: %d layers in Tier 1", len(self.tier1_layers))
            return

        if self.layers_config == "all":
            self.tier1_layers = set(range(num_layers))
            self.tier2_layers = set()
            self.tier3_layers = set()
            logger.info("PAB all layers: 50 layers in Tier 1")
            return

        # Adaptive grouping: load from calibration file if available
        if os.path.exists(self.calib_file):
            try:
                with open(self.calib_file, "r") as f:
                    calib_data = json.load(f)
                ranked_layers = calib_data.get("ranked_layers_by_variance", [])
                if len(ranked_layers) == num_layers:
                    t1_cut = int(num_layers * self.tier1_ratio)
                    t2_cut = int(num_layers * (self.tier1_ratio + self.tier2_ratio))
                    self.tier1_layers = set(ranked_layers[:t1_cut])
                    self.tier2_layers = set(ranked_layers[t1_cut:t2_cut])
                    self.tier3_layers = set(ranked_layers[t2_cut:])
                    logger.info(
                        "PAB adaptive grouping loaded from %s: Tier 1=%d layers (%s), Tier 2=%d layers, Tier 3=%d layers",
                        self.calib_file,
                        len(self.tier1_layers),
                        sorted(list(self.tier1_layers))[:8],
                        len(self.tier2_layers),
                        len(self.tier3_layers),
                    )
                    return
            except Exception as e:
                logger.warning("Failed to load PAB calib file %s: %s. Using default grouping.", self.calib_file, e)

        # Default heuristic grouping if no calib file:
        self.tier1_layers = set(range(0, 20))  # 20 layers
        self.tier2_layers = set(range(20, 30))  # 10 layers
        self.tier3_layers = set(range(30, 50))  # 20 layers
        logger.info(
            "PAB heuristic grouping: Tier 1 (0-19, %d layers), Tier 2 (20-29, %d layers), Tier 3 (30-49, %d layers)",
            len(self.tier1_layers),
            len(self.tier2_layers),
            len(self.tier3_layers),
        )

    @staticmethod
    def _parse_layer_range(s: str, num_layers: int) -> set[int]:
        res = set()
        parts = s.split(",")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if "-" in p:
                start_s, end_s = p.split("-", 1)
                start, end = int(start_s), int(end_s)
                res.update(range(max(0, start), min(num_layers, end + 1)))
            elif p.isdigit():
                idx = int(p)
                if 0 <= idx < num_layers:
                    res.add(idx)
        return res

    def _sync_step(self) -> tuple[int | None, int | None]:
        """Query denoise step from forward context and detect request boundary."""
        if not is_forward_context_available():
            return None, None
        ctx = get_forward_context()
        step = ctx.denoise_step_idx
        total_steps = ctx.total_denoise_steps

        if step is not None:
            if self.last_seen_step is None or step < self.last_seen_step:
                # New diffusion generation request started!
                self._on_new_request(total_steps)
            self.last_seen_step = step
            self.current_request_step = step

        return step, total_steps

    def _on_new_request(self, total_steps: int | None) -> None:
        """Called at step 0 of a new generation pass."""
        self._reload_control_file()
        logger.info(
            "PAB: new generation request detected (total_steps=%s, enabled=%s, policy=%s). Resetting step counters.",
            total_steps,
            self.enabled,
            self.policy,
        )
        self.broadcast_counts = {}
        if self.calibrate:
            self.calib_prev = {}
            self.calib_diffs = []
        else:
            self.clear_cache()

    def should_broadcast(self, layer_idx: int) -> tuple[bool, torch.Tensor | None]:
        """Determine whether the attention calculation for layer_idx should be broadcast.

        Returns (should_broadcast, cached_tensor).
        """
        step, total_steps = self._sync_step()

        if not self.enabled:
            return False, None

        # In calibration mode, we never skip (always run full attention to collect diffs)
        if self.calibrate:
            return False, None

        if step is None:
            return False, None

        # Step 0 is always dense (warmup / initial structure)
        if step == 0:
            return False, None

        # Dense tail protection: keep the last `end_step` steps dense.
        # CRITICAL: Clean up cached tensors before VAE decode to prevent VAE OOM!
        if total_steps is not None and self.end_step > 0:
            if step >= total_steps - self.end_step:
                if self.cache:
                    self.clear_cache()
                return False, None

        # Don't broadcast before start_step
        if step < self.start_step:
            return False, None

        # If cache doesn't have this layer yet, must compute
        cached = self.cache.get(layer_idx)
        if cached is None:
            return False, None

        # Evaluate scheduling policy
        eligible = False

        if self.custom_broadcast_steps is not None:
            if step in self.custom_broadcast_steps and (
                layer_idx in self.tier1_layers or layer_idx in self.tier2_layers
            ):
                eligible = True
        elif self.policy == "pyramid":
            # Tier 1 (low variance): broadcast on steps 2, 4, 6 (or steps % 2 == 0)
            # Tier 2 (medium variance): broadcast on most stable middle steps (e.g. steps 3, 5)
            # Tier 3 (high variance): never broadcast (stays dense)
            if layer_idx in self.tier1_layers:
                if self.broadcast_range == 2:
                    eligible = (step % 2 == 0) if self.start_step % 2 == 0 else (step % 2 == 1)
                else:
                    eligible = (step % self.broadcast_range != 0)
            elif layer_idx in self.tier2_layers:
                mid_step = (total_steps or 8) // 2
                eligible = (step in (mid_step - 1, mid_step + 1)) and (step % 2 == 1)
            else:
                eligible = False
        elif self.policy == "periodic":
            if layer_idx in self.tier1_layers:
                eligible = (step % self.broadcast_range != 0)
            elif layer_idx in self.tier2_layers:
                eligible = (step % (self.broadcast_range * 2) != 0) and (step % self.broadcast_range != 0)
            else:
                eligible = False
        elif self.policy == "all_middle":
            # Broadcast all Tier 1 + Tier 2 layers in middle steps
            mid_start = self.start_step
            mid_end = (total_steps - self.end_step) if total_steps else 7
            if mid_start <= step < mid_end and (layer_idx in self.tier1_layers or layer_idx in self.tier2_layers):
                eligible = (step % self.broadcast_range != 0)
        else:
            eligible = (step % self.broadcast_range != 0) and (layer_idx in self.tier1_layers)

        self.total_attention_calls += 1
        if eligible:
            self.broadcasted_attention_calls += 1
            self.broadcast_counts[step] = self.broadcast_counts.get(step, 0) + 1
            return True, cached

        return False, None

    def record_attention(self, layer_idx: int, out: torch.Tensor) -> None:
        """Cache computed attention output and record calibration metrics if active."""
        if not self.enabled:
            return

        step, total_steps = self._sync_step()
        if step is None:
            return

        # Do not cache at end steps
        if total_steps is not None and self.end_step > 0:
            if step >= total_steps - self.end_step:
                return

        # Calibration recording
        if self.calibrate:
            self._record_calibration(layer_idx, out)
            return

        # Normal PAB: ONLY cache layers in Tier 1 or Tier 2 up to max_cached_layers!
        if layer_idx not in self.tier1_layers and layer_idx not in self.tier2_layers:
            return

        if layer_idx not in self.cache:
            if len(self.cache) >= self.max_cached_layers:
                return
            self.cache[layer_idx] = out.detach().clone()
        else:
            self.cache[layer_idx].copy_(out)

    def _record_calibration(self, layer_idx: int, out: torch.Tensor) -> None:
        step, total_steps = self._sync_step()
        if step is None:
            return

        if layer_idx in self.calib_prev:
            prev = self.calib_prev[layer_idx]
            # Compute relative L1 difference: ||out - prev||_1 / ||prev||_1
            with torch.no_grad():
                diff_norm = (out - prev).abs().sum().item()
                prev_norm = prev.abs().sum().item() + 1e-7
                rel_diff = diff_norm / prev_norm

            # Store in calib_diffs for the current step
            while len(self.calib_diffs) <= step:
                self.calib_diffs.append({})
            self.calib_diffs[step][layer_idx] = rel_diff

            # Update prev tensor in-place
            self.calib_prev[layer_idx].copy_(out)
        else:
            self.calib_prev[layer_idx] = out.detach().clone()

        # If this is the last layer of the last step, finalize calibration
        num_layers = 50
        max_step = (total_steps - 1) if total_steps is not None else 7
        if step == max_step and layer_idx == (num_layers - 1):
            self._finalize_calibration(num_layers, total_steps or 8)

    def _finalize_calibration(self, num_layers: int, total_steps: int) -> None:
        """Process collected step differences, rank layers by variance, and save to JSON."""
        logger.info("Finalizing PAB calibration data across %d steps...", len(self.calib_diffs))
        os.makedirs(os.path.dirname(self.calib_output), exist_ok=True)

        layer_stats = {}
        for l in range(num_layers):
            diffs = []
            for s in range(len(self.calib_diffs)):
                if l in self.calib_diffs[s]:
                    diffs.append(self.calib_diffs[s][l])
            if diffs:
                mean_diff = sum(diffs) / len(diffs)
                var_diff = sum((x - mean_diff) ** 2 for x in diffs) / len(diffs)
                layer_stats[l] = {
                    "mean_diff": mean_diff,
                    "variance": var_diff,
                    "step_diffs": diffs,
                }

        # Rank layers by variance ascending (lowest variance = most stable = best to broadcast)
        ranked_by_variance = sorted(layer_stats.keys(), key=lambda l: layer_stats[l]["variance"])
        ranked_by_mean = sorted(layer_stats.keys(), key=lambda l: layer_stats[l]["mean_diff"])

        # Compute per-step average difference across all layers to observe U-shaped curve
        step_averages = {}
        for s in range(len(self.calib_diffs)):
            vals = list(self.calib_diffs[s].values())
            step_averages[s] = sum(vals) / len(vals) if vals else 0.0

        calib_result = {
            "num_layers": num_layers,
            "total_steps": total_steps,
            "step_average_differences": step_averages,
            "ranked_layers_by_variance": ranked_by_variance,
            "ranked_layers_by_mean_diff": ranked_by_mean,
            "layer_stats": {str(k): v for k, v in layer_stats.items()},
        }

        with open(self.calib_output, "w") as f:
            json.dump(calib_result, f, indent=2)

        # Clear calib memory immediately!
        self.calib_prev.clear()
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()

        logger.info(
            "PAB calibration successfully saved to %s! Top 10 lowest-variance layers: %s",
            self.calib_output,
            ranked_by_variance[:10],
        )

    def clear_cache(self) -> None:
        """Clear cached tensors to free NPU memory."""
        self.cache.clear()
        self.calib_prev.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()

    def get_stats(self) -> dict[str, Any]:
        """Return execution summary statistics."""
        return {
            "enabled": self.enabled,
            "policy": self.policy,
            "total_calls": self.total_attention_calls,
            "broadcasted_calls": self.broadcasted_attention_calls,
            "broadcast_ratio": (
                self.broadcasted_attention_calls / self.total_attention_calls
                if self.total_attention_calls > 0
                else 0.0
            ),
            "step_broadcast_counts": dict(self.broadcast_counts),
        }
