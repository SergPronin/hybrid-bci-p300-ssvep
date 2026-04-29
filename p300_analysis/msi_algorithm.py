"""MSI-like алгоритм оценки класса-победителя."""

from __future__ import annotations

import numpy as np


def compute_msi_like_scores(
    *,
    corr_win: np.ndarray,
    abs_auc_values: np.ndarray,
) -> np.ndarray:
    """MSI-like score: энергия окна + сходство с P300-подобным шаблоном.

    Возвращает вектор формы (n_stim,), где большее значение означает лучший кандидат.
    """
    n_stim = abs_auc_values.shape[0]
    if n_stim == 0:
        return np.zeros(0, dtype=np.float64)
    if corr_win.size == 0 or corr_win.shape[1] == 0:
        return abs_auc_values.astype(np.float64, copy=True)

    win_len = corr_win.shape[1]
    t = np.linspace(-1.0, 1.0, win_len, dtype=np.float64)
    template = np.exp(-0.5 * (t / 0.35) ** 2)
    template -= np.mean(template)
    template_norm = float(np.linalg.norm(template)) + 1e-12
    template /= template_norm

    rows = corr_win - np.mean(corr_win, axis=1, keepdims=True)
    row_norms = np.linalg.norm(rows, axis=1) + 1e-12
    corr_similarity = (rows @ template) / row_norms
    corr_similarity = np.clip(corr_similarity, -1.0, 1.0)

    auc_max = float(np.max(abs_auc_values)) if abs_auc_values.size else 0.0
    if auc_max <= 1e-12:
        auc_norm = np.zeros_like(abs_auc_values, dtype=np.float64)
    else:
        auc_norm = abs_auc_values / auc_max

    similarity_positive = np.maximum(corr_similarity, 0.0)
    return 0.7 * auc_norm + 0.3 * similarity_positive
