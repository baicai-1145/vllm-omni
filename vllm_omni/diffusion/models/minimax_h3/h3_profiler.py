# SPDX-License-Identifier: Apache-2.0
"""Device-side region profiler for the MiniMax H3 denoising loop.

Why not msprof
---------------
msprof on this box is a dead end for fine-grained work: its raw trace is
schemaless binary, the installed CANN analysis package only ships the
communication analyzers, `--aic-metrics` rejects multi-value input, and
`--dynamic` mode hung for 54 min without producing data. Six attempts, no
usable operator breakdown.

What this does instead
----------------------
Times *regions of the model* with `torch.npu.Event`, i.e. device-side elapsed
time measured by the hardware, not by wall clock. That matters because the
denoising step launches a very large number of small ops: host wall time
includes dispatch and queueing, device event time does not. The split we care
about (attention core vs MLP GEMMs vs norm/RoPE/elementwise) is a
model-structure question, and instrumenting the model answers it directly
without any external profiler.

Overhead
--------
Two event records per region. Disabled by default (``H3_PROFILE`` unset) the
context manager is a no-op returning immediately.

Usage
-----
    H3_PROFILE=1 <launch server>   # then, from another shell:
    python -c "from vllm_omni.diffusion.models.minimax_h3.h3_profiler import dump; dump()"
or let the pipeline dump it: set ``H3_PROFILE_DUMP=1`` to print at exit.
"""
from __future__ import annotations

import atexit
import os
import time

_ENABLED = (os.environ.get("H3_PROFILE", "") or "").strip() not in ("", "0", "false", "False")

# name -> [call_count, total_ms, list_of_(start_event, end_event)]
_STATS: dict[str, list] = {}

# The server is torn down with SIGKILL, so atexit never runs. Write the file
# every N region exits instead; the last write before the kill is what we read.
_OUT = (os.environ.get("H3_PROFILE_OUT", "") or "").strip() or None
_WRITE_EVERY = max(1, int(os.environ.get("H3_PROFILE_EVERY", "500") or 500))
_TICK = [0]


def _write_json() -> None:
    """Serialize device times to H3_PROFILE_OUT. Best effort, never raises."""
    if not _OUT:
        return
    import json

    try:
        import torch

        torch.npu.synchronize()
        rows = {}
        for name, (calls, _, spans) in _STATS.items():
            total = 0.0
            for s, e in spans:
                try:
                    total += s.elapsed_time(e)
                except Exception:
                    pass
            rows[name] = {"calls": calls, "ms": round(total, 2)}
        tmp = f"{_OUT}.tmp"
        with open(tmp, "w") as fh:
            json.dump({"wall_clock": time.time(), "regions": rows}, fh, indent=2)
        os.replace(tmp, _OUT)  # atomic: never leaves a half-written file
    except Exception:
        pass


class _Region:
    """Records one device-timed region. Use via ``region(name)``."""

    __slots__ = ("name", "start", "end")

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = None
        self.end = None

    def __enter__(self) -> "_Region":
        import torch

        self.start = torch.npu.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import torch

        self.end = torch.npu.Event(enable_timing=True)
        self.end.record()
        slot = _STATS.get(self.name)
        if slot is None:
            _STATS[self.name] = [1, 0.0, [(self.start, self.end)]]
        else:
            slot[0] += 1
            slot[2].append((self.start, self.end))
        if _OUT is not None:
            _TICK[0] += 1
            if _TICK[0] % _WRITE_EVERY == 0:
                _write_json()
        return False


class _Noop:
    """Zero-cost stand-in used when profiling is disabled."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def region(name: str):
    """Context manager timing a device region. No-op unless H3_PROFILE is set."""
    if not _ENABLED:
        return _Noop()
    return _Region(name)


def reset() -> None:
    _STATS.clear()


def dump(top: int = 40) -> None:
    """Synchronize and print the accumulated device-time table."""
    if not _STATS:
        print("[h3_profiler] no regions recorded (is H3_PROFILE=1 set?)")
        return
    import torch

    torch.npu.synchronize()
    rows = []
    for name, (calls, _, spans) in _STATS.items():
        total = 0.0
        for s, e in spans:
            try:
                total += s.elapsed_time(e)
            except Exception:
                pass
        rows.append((total, calls, name))
    rows.sort(reverse=True)
    grand = sum(r[0] for r in rows) or 1.0
    print(f"\n{'=' * 66}")
    print(f"{'H3 region profiler (device time, torch.npu.Event)':^66}")
    print(f"{'=' * 66}")
    print(f"{'region':34s} {'calls':>8s} {'ms':>10s} {'pct':>7s}")
    print(f"{'-' * 66}")
    for total, calls, name in rows[:top]:
        print(f"{name[:34]:34s} {calls:8d} {total:10.1f} {total / grand * 100:6.1f}%")
    print(f"{'-' * 66}")
    print(f"{'TOTAL':34s} {sum(r[1] for r in rows):8d} {grand:10.1f}")
    print(f"{'=' * 66}\n", flush=True)


if _ENABLED and (os.environ.get("H3_PROFILE_DUMP", "") or "").strip() not in ("", "0"):
    atexit.register(dump)


def enabled() -> bool:
    return _ENABLED
