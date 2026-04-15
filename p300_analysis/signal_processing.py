"""Baseline correction и интеграция (AUC) для ERP."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def baseline_correction(raw: np.ndarray, time_ms: np.ndarray, baseline_ms: int) -> np.ndarray:
    """
    Baseline correction: corrected = raw - mean(raw[:baseline_idx]).

    raw: shape (..., n_time)
    """
    if raw.ndim < 1:
        raise ValueError("raw must have at least 1 dimension")
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be a 1D array")
    if raw.shape[-1] != time_ms.shape[0]:
        raise ValueError("raw and time_ms length mismatch")

    dt_ms = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    baseline_idx = int(round(float(baseline_ms) / dt_ms))
    baseline_idx = max(1, min(baseline_idx, time_ms.shape[0]))

    baseline_mean = np.mean(raw[..., :baseline_idx], axis=-1, keepdims=True)
    return raw - baseline_mean


def integrated_cumsum(
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Интеграция положительной части ERP: cumsum(max(corrected, 0))."""
    if corrected.ndim < 1:
        raise ValueError("corrected must have at least 1 dimension")
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be 1D array")
    if corrected.shape[-1] != time_ms.shape[0]:
        raise ValueError("corrected and time_ms length mismatch")

    dt_ms = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    x_idx = int(round(float(window_x_ms) / dt_ms))
    y_idx = int(round(float(window_y_ms) / dt_ms)) + 1

    x_idx = max(0, min(x_idx, time_ms.shape[0] - 1))
    y_idx = max(x_idx + 1, min(y_idx, time_ms.shape[0]))

    # Для P300 информативен положительный отклик в окне, а не модуль сигнала:
    # large negative deflections should not increase the winner score.
    segment = np.clip(corrected[..., x_idx:y_idx], a_min=0.0, a_max=None)
    integrated = np.cumsum(segment, axis=-1)
    time_crop = time_ms[x_idx:y_idx]
    return integrated, time_crop
