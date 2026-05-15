"""
Прослойка пакетного SSVEP: по LSL-маркерам Migalka решает, когда вызывать MSI.

Постоянный режим: classify_allowed всегда True (если буфер полон).
Пакетный: MSI только если в окне анализа достаточно времени с активной стимуляцией.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_LED_RE = re.compile(
    r"^LED\s+(\d+)\s+(\d+)\s+(ON|OFF)\s*$",
    re.IGNORECASE,
)


def parse_led_serial_line(line: str) -> Optional[Tuple[int, bool]]:
    """Строка прошивки: ``LED 2 12345 ON`` → (lamp_index, is_on)."""
    m = _LED_RE.match(line.strip())
    if not m:
        return None
    lamp = int(m.group(1))
    is_on = m.group(3).upper() == "ON"
    return lamp, is_on


def parse_lsl_marker(value: object) -> Optional[Tuple[int, bool]]:
    """``102|on`` / ``102|off`` → (lamp_index, is_on)."""
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        return None
    s = value.strip()
    if "|" not in s:
        return None
    left, right = s.split("|", 1)
    try:
        raw_id = int(left.strip())
    except ValueError:
        return None
    if raw_id < 100:
        return None
    lamp = raw_id - 100
    if not (0 <= lamp <= 8):
        return None
    ev = right.strip().lower()
    if ev == "on":
        return lamp, True
    if ev == "off":
        return lamp, False
    return None


@dataclass
class BurstGateConfig:
    window_sec: float = 2.0
    min_on_fraction: float = 0.70
    """Доля окна (по времени), когда хотя бы одна из active_lamps в ON."""
    min_on_sec: float = 1.4
    """Минимум секунд суммарного ON в окне (для вспышки ~2 с)."""


@dataclass
class _Interval:
    lamp: int
    t_on: float
    t_off: Optional[float] = None


@dataclass
class BurstGate:
    config: BurstGateConfig = field(default_factory=BurstGateConfig)
    active_lamps: Tuple[int, ...] = ()
    _intervals: List[_Interval] = field(default_factory=list)
    _open: Dict[int, _Interval] = field(default_factory=dict)
    _max_intervals: int = 512

    def set_active_lamps(self, n_lamps: int) -> None:
        self.active_lamps = tuple(range(max(0, int(n_lamps))))

    def ingest_marker(self, lsl_time: float, value: object) -> None:
        parsed = parse_lsl_marker(value)
        if parsed is None:
            return
        lamp, is_on = parsed
        self._apply_lamp_state(lamp, is_on, lsl_time)

    def ingest_led_line(self, line: str, lsl_time: float) -> None:
        parsed = parse_led_serial_line(line)
        if parsed is None:
            return
        lamp, is_on = parsed
        self._apply_lamp_state(lamp, is_on, lsl_time)

    def _apply_lamp_state(self, lamp: int, is_on: bool, t: float) -> None:
        t = float(t)
        if is_on:
            if lamp in self._open:
                return
            iv = _Interval(lamp=lamp, t_on=t)
            self._open[lamp] = iv
            self._intervals.append(iv)
        else:
            iv = self._open.pop(lamp, None)
            if iv is not None:
                iv.t_off = t
        if len(self._intervals) > self._max_intervals:
            drop = len(self._intervals) - self._max_intervals
            self._intervals = self._intervals[drop:]
            self._open = {iv.lamp: iv for iv in self._intervals if iv.t_off is None}

    def _lamp_on_at(self, lamp: int, t: float) -> bool:
        if lamp not in self.active_lamps:
            return False
        for iv in self._intervals:
            if iv.lamp != lamp:
                continue
            t_end = iv.t_off if iv.t_off is not None else float("inf")
            if iv.t_on <= t <= t_end:
                return True
        return False

    def classify_allowed(
        self,
        buf_times: np.ndarray,
        *,
        now: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        buf_times: метки LSL для каждого сэмпла в rolling-буфере (длина = EEG).
        """
        if buf_times.size == 0:
            return False, "нет меток времени EEG"

        t_end = float(now if now is not None else buf_times[-1])
        t_start = t_end - self.config.window_sec
        mask = (buf_times >= t_start) & (buf_times <= t_end)
        times = buf_times[mask]
        if times.size < 8:
            return False, "мало сэмплов в окне"

        if not self.active_lamps:
            return False, "не заданы активные лампы"

        on_samples = 0
        for t in times:
            if any(self._lamp_on_at(lamp, float(t)) for lamp in self.active_lamps):
                on_samples += 1

        frac = on_samples / float(times.size)
        dt = float(times[-1] - times[0]) if times.size > 1 else 0.0
        on_sec = frac * max(dt, self.config.window_sec * 0.99)

        if frac < self.config.min_on_fraction:
            return False, f"стимул {frac:.0%} окна (нужно ≥{self.config.min_on_fraction:.0%})"
        if on_sec < self.config.min_on_sec:
            return False, f"ON {on_sec:.2f} с (нужно ≥{self.config.min_on_sec:.1f} с)"
        return True, f"стимул {frac:.0%} окна, ON≈{on_sec:.2f} с"

    def intervals_in_range(
        self,
        t_min: float,
        t_max: float,
    ) -> List[Tuple[int, float, float]]:
        """Интервалы ON для диаграммы Ганта: (lamp_index, t_start, t_end)."""
        if t_max <= t_min:
            return []
        out: List[Tuple[int, float, float]] = []
        for iv in self._intervals:
            if iv.lamp not in self.active_lamps:
                continue
            t_end = iv.t_off if iv.t_off is not None else t_max
            t_start = max(float(iv.t_on), t_min)
            t_end_clip = min(float(t_end), t_max)
            if t_start < t_end_clip:
                out.append((iv.lamp, t_start, t_end_clip))
        return out


def append_chunk_timestamps(
    prev_t: np.ndarray,
    chunk_ts: Sequence[float],
    n_new: int,
    fs: float,
) -> np.ndarray:
    """Дополнить вектор LSL-времени для новых строк EEG."""
    if n_new <= 0:
        return prev_t
    ts = list(chunk_ts)
    if len(ts) == n_new:
        new_t = np.asarray(ts, dtype=np.float64)
    elif len(ts) == 1:
        step = 1.0 / max(fs, 1.0)
        t0 = float(ts[0]) - (n_new - 1) * step
        new_t = t0 + step * np.arange(n_new, dtype=np.float64)
    elif len(ts) > 1:
        new_t = np.linspace(float(ts[0]), float(ts[-1]), n_new, dtype=np.float64)
    else:
        step = 1.0 / max(fs, 1.0)
        t0 = float(prev_t[-1]) + step if prev_t.size else 0.0
        new_t = t0 + step * np.arange(n_new, dtype=np.float64)
    if prev_t.size == 0:
        return new_t
    return np.concatenate([prev_t, new_t])
