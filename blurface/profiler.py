"""
blurface.profiler — Lightweight per-phase timing.
"""
import time
from collections import defaultdict


class Profiler:
    """Accumulate per-phase wall-clock time.

    Usage:
        prof = Profiler()
        with prof("detect"):
            ...  # expensive work
        prof.summary(total_frames, total_wall)
    """

    def __init__(self):
        self.times = defaultdict(float)
        self._order = []

    def phase(self, name: str):
        """Context manager for timing a block."""
        if name not in self._order:
            self._order.append(name)
        return _PhaseTimer(self.times, name)

    def record(self, name: str, dt: float):
        if name not in self._order:
            self._order.append(name)
        self.times[name] += dt

    def summary(self, frames: int, total_wall: float):
        if frames == 0:
            return
        lines = []
        lines.append("Per-frame breakdown (avg):")
        for ph in self._order:
            avg_ms = (self.times[ph] / frames) * 1000
            pct = (self.times[ph] / total_wall * 100) if total_wall > 0 else 0
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            lines.append(f"  {ph:8s}  {avg_ms:7.2f}ms  ({pct:5.1f}%)  {bar}")
        return "\n".join(lines)


class _PhaseTimer:
    def __init__(self, times, name):
        self._times = times
        self._name = name

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        self._times[self._name] += time.time() - self._t0
