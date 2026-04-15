"""Разбор строковых маркеров LSL (плитки, trial_start)."""

from __future__ import annotations

import re
from typing import Any, Optional, Tuple

import numpy as np


def marker_value_to_stim_key(marker_value: Any) -> Optional[str]:
    """
    Ключ класса эпохи, например «стимул_3».

    GUI шлёт ``f"{tile_id}|{event}"``: ``5|on``, ``5|off``,
    а также ``-1|trial_start|target=...``, ``-2|trial_end``.

    Для P300 берём только вспышку ``|on``; ``|off`` и служебные id<0 пропускаем.
    """
    mv = marker_value

    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]

    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")

    if isinstance(mv, (int, np.integer)):
        return f"стимул_{int(mv)}"

    if isinstance(mv, (float, np.floating)):
        return f"стимул_{int(round(float(mv)))}"

    if isinstance(mv, str):
        s = mv.strip()
        if not s:
            return None
        if "|" in s:
            left, right = s.split("|", 1)
            left, right = left.strip(), right.strip()
            try:
                tile_id = int(left)
            except ValueError:
                tile_id = None
            if tile_id is not None and tile_id < 0:
                return None
            first_seg = right.split("|", 1)[0].strip()
            if tile_id is not None:
                if first_seg == "on":
                    return f"стимул_{tile_id}"
                if first_seg == "off":
                    return None
                if right.startswith("trial_start") or right.startswith("trial_end"):
                    return None
                return None
        m = re.search(r"(\d+)", s)
        if m:
            return f"стимул_{int(m.group(1))}"
        return s

    return str(mv)


def parse_trial_target_tile_id(marker_value: Any) -> Optional[int]:
    """Из маркера ``-1|trial_start|target=N`` извлекает N (id плитки 0..8)."""
    mv = marker_value
    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]
    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")
    if not isinstance(mv, str):
        return None
    s = mv.strip()
    if "trial_start" not in s:
        return None
    m = re.search(r"target[=:](\d+)", s)
    if not m:
        return None
    return int(m.group(1))


def stim_key_sort_key(stim_key: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", stim_key)
    if m:
        return int(m.group(1)), stim_key
    return 10**9, stim_key


def stim_key_to_tile_digit(stim_key: str) -> int:
    m = re.search(r"(\d+)", stim_key)
    return int(m.group(1)) if m else -1
