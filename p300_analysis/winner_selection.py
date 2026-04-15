"""Режимы выбора класса-победителя по ERP."""

from __future__ import annotations

from typing import Tuple

import numpy as np

WINNER_MODE_SIGNED_MEAN = "signed_mean"
WINNER_MODE_PEAK_RAW = "peak_raw"
WINNER_MODE_PEAK_CORRECTED = "peak_corrected"
WINNER_MODE_AUC = "auc"

MODE_SHORT_LABELS = {
    WINNER_MODE_SIGNED_MEAN: "signed",
    WINNER_MODE_PEAK_RAW: "raw_peak",
    WINNER_MODE_PEAK_CORRECTED: "corr_peak",
    WINNER_MODE_AUC: "auc",
}


def pick_winner_by_mode(
    mode: str,
    raw_win: np.ndarray,
    corr_win: np.ndarray,
    auc_final_per_class: np.ndarray,
) -> Tuple[int, str]:
    """Возвращает (индекс класса среди текущей матрицы, имя режима)."""
    if mode == WINNER_MODE_PEAK_CORRECTED:
        return int(np.argmax(np.max(corr_win, axis=1))), WINNER_MODE_PEAK_CORRECTED
    if mode == WINNER_MODE_SIGNED_MEAN:
        return int(np.argmax(np.mean(corr_win, axis=1))), WINNER_MODE_SIGNED_MEAN
    if mode == WINNER_MODE_AUC:
        return int(np.argmax(auc_final_per_class)), WINNER_MODE_AUC
    return int(np.argmax(np.max(raw_win, axis=1))), WINNER_MODE_PEAK_RAW


def mode_to_short_label(mode_used: str) -> str:
    return MODE_SHORT_LABELS.get(mode_used, mode_used)
