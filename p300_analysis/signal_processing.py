"""Baseline correction, фильтрация и детекция плохих каналов для ERP."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def bandpass_filter(
    X: np.ndarray,
    fs: float,
    lo: float = 0.5,
    hi: float = 20.0,
    order: int = 4,
) -> np.ndarray:
    """Полосовой фильтр Баттерворта (filtfilt, нулевой сдвиг фазы).

    X: (n_samples,) или (n_samples, n_channels)
    fs: частота дискретизации в Гц
    Возвращает массив той же формы.
    Если длина сигнала слишком мала для filtfilt — возвращает X без изменений.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        return X

    n = X.shape[0]
    min_len = 3 * (order + 1) * 2  # padlen для filtfilt ≈ 3*(order*2)
    if n < min_len:
        return X

    nyq = fs / 2.0
    lo_n = max(lo / nyq, 1e-5)
    hi_n = min(hi / nyq, 1.0 - 1e-5)
    if lo_n >= hi_n:
        return X

    b, a = butter(order, [lo_n, hi_n], btype="band")
    if X.ndim == 1:
        return filtfilt(b, a, X).astype(X.dtype)
    return filtfilt(b, a, X, axis=0).astype(X.dtype)


def detect_bad_channels(
    X: np.ndarray,
    std_thresh: float = 4.0,
    abs_thresh: float = 3.0,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Обнаруживает каналы с аномальным шумом.

    X: (n_samples, n_channels)
    Возвращает:
      bad_indices  — список индексов плохих каналов (0-based)
      abs_means    — среднее |x| по каждому каналу
      stds         — std по каждому каналу
    Критерий: канал считается плохим, если его std > std_thresh * median(stds)
              ИЛИ abs_mean > abs_thresh * median(abs_means).
    """
    if X.ndim != 2 or X.shape[1] == 0:
        return [], np.array([]), np.array([])

    abs_means = np.mean(np.abs(X), axis=0)
    stds = np.std(X, axis=0)

    med_abs = float(np.median(abs_means))
    med_std = float(np.median(stds))

    bad_std = stds > std_thresh * med_std if med_std > 0 else np.zeros(X.shape[1], dtype=bool)
    bad_abs = abs_means > abs_thresh * med_abs if med_abs > 0 else np.zeros(X.shape[1], dtype=bool)
    bad_mask = bad_std | bad_abs

    bad_indices = [int(i) for i in np.where(bad_mask)[0]]
    return bad_indices, abs_means, stds


def baseline_correction(raw: np.ndarray, time_ms: np.ndarray, baseline_ms: int) -> np.ndarray:
    """Baseline correction: corrected = raw - median(raw[:baseline_idx]).

    Использует median вместо mean для устойчивости к артефактам в pre-stimulus периоде.
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

    baseline_val = np.median(raw[..., :baseline_idx], axis=-1, keepdims=True)
    return raw - baseline_val


def integrated_cumsum(
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
) -> tuple:
    """Интеграция ERP по модулю: cumsum(abs(corrected[x_idx:y_idx]))."""
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

    segment = corrected[..., x_idx:y_idx]
    integrated = np.cumsum(np.abs(segment), axis=-1)
    time_crop = time_ms[x_idx:y_idx]
    return integrated, time_crop
