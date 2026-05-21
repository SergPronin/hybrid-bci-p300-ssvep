"""Частоты ламп мигалки и дефолты MSI (без Qt — для protocol_runner_gui и ssvep_analyzer)."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

MSI_DEFAULT_FS = 250.0
MSI_DEFAULT_WINDOW_SEC = 2.0
CHANNEL_CB_COLUMNS = 4

_LAMP_FREQ_CHOICES: List[Tuple[str, float]] = []


def lamp_frequency_choices() -> List[Tuple[str, float]]:
    """500 дискретных частот: 1000/i Гц, i = 1..500; подпись с десятичной точкой."""
    if not _LAMP_FREQ_CHOICES:
        for i in range(1, 501):
            v = 1000.0 / float(i)
            s = f"{v}".replace(",", ".")
            _LAMP_FREQ_CHOICES.append((s, v))
    return _LAMP_FREQ_CHOICES


def lamp_frequency_closest_index(target_hz: float) -> int:
    arr = np.array([v for _, v in lamp_frequency_choices()], dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(target_hz))))
