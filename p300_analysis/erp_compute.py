"""Усреднение эпох и подготовка данных для отображения победителя."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from p300_analysis.constants import MIN_EPOCHS_TO_DECIDE
from p300_analysis.marker_parsing import stim_key_sort_key, stim_key_to_tile_digit
from p300_analysis.signal_processing import baseline_correction, integrated_cumsum
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_SIGNED_MEAN


def build_averaged_erp(
    epochs_data: Dict[str, List[np.ndarray]],
    epoch_len: int,
) -> Tuple[List[str], np.ndarray]:
    stim_keys = [k for k, v in epochs_data.items() if v]
    stim_keys.sort(key=stim_key_sort_key)
    n_stim = len(stim_keys)
    raw_averaged = np.zeros((n_stim, epoch_len), dtype=np.float64)
    for i, key in enumerate(stim_keys):
        epochs = epochs_data.get(key, [])
        if not epochs:
            continue
        stack = np.stack([e[:epoch_len] for e in epochs], axis=0)
        raw_averaged[i, :] = np.mean(stack, axis=0)
    return stim_keys, raw_averaged


def compute_corrected_and_integrated(
    raw_averaged: np.ndarray,
    time_ms: np.ndarray,
    baseline_ms: int,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    wy = window_y_ms if window_y_ms > window_x_ms else window_x_ms + 1
    corrected = baseline_correction(raw_averaged, time_ms, baseline_ms=baseline_ms)
    integrated, time_crop = integrated_cumsum(
        corrected,
        time_ms,
        window_x_ms=window_x_ms,
        window_y_ms=wy,
    )
    return corrected, integrated, time_crop, window_x_ms, wy


def check_can_decide(stim_keys: List[str], epochs_data: Dict[str, List[np.ndarray]]) -> Tuple[bool, int]:
    if not stim_keys:
        return False, 0
    counts = [len(epochs_data.get(k, [])) for k in stim_keys]
    min_n = min(counts)
    can = all(c >= MIN_EPOCHS_TO_DECIDE for c in counts)
    return can, min_n


def compute_winner_metrics(
    stim_keys: List[str],
    raw_averaged: np.ndarray,
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
    winner_mode: str = WINNER_MODE_AUC,
) -> Tuple[int, str, Dict[str, Any]]:
    """Возвращает winner_idx, mode_used, data для debug_ndjson (winner_compare)."""
    dt_m = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    xi0 = int(round(float(window_x_ms) / dt_m))
    xi1 = int(round(float(window_y_ms) / dt_m)) + 1
    xi0 = max(0, min(xi0, time_ms.shape[0] - 1))
    xi1 = max(xi0 + 1, min(xi1, time_ms.shape[0]))
    corr_win = corrected[:, xi0:xi1]
    pos_win = np.clip(corr_win, a_min=0.0, a_max=None)
    positive_auc_values = np.sum(pos_win, axis=1) * dt_m
    abs_auc_values = np.sum(np.abs(corr_win), axis=1) * dt_m
    signed_mean_values = np.mean(corr_win, axis=1) if corr_win.size else np.zeros(len(stim_keys))
    positive_peak_values = np.max(corr_win, axis=1) if corr_win.size else np.zeros(len(stim_keys))

    if winner_mode == WINNER_MODE_SIGNED_MEAN:
        final_metric_values = signed_mean_values
        mode_used = WINNER_MODE_SIGNED_MEAN
    else:
        final_metric_values = positive_auc_values
        mode_used = WINNER_MODE_AUC

    winner_idx = int(np.argmax(final_metric_values))
    auc_winner_idx = int(np.argmax(positive_auc_values))
    abs_max_values = np.max(np.abs(corr_win), axis=1) if corr_win.size else np.zeros(len(stim_keys))
    debug_payload = {
        "winner_rule": mode_used,
        "chosen_winner_idx": winner_idx,
        "chosen_winner_key": stim_keys[winner_idx],
        "stim_keys": stim_keys,
        "final_metric_values": [float(x) for x in final_metric_values],
        "signed_mean_final": [float(x) for x in signed_mean_values],
        "positive_auc_values": [float(x) for x in positive_auc_values],
        "abs_auc_values": [float(x) for x in abs_auc_values],
        "auc_winner_idx": auc_winner_idx,
        "auc_winner_key": stim_keys[auc_winner_idx],
        "positive_peak_values": [float(x) for x in positive_peak_values],
        "window_index": [xi0, xi1],
        "dt_ms": float(dt_m),
        "window_ms": [window_x_ms, window_y_ms],
        "corr_abs_max": [float(x) for x in abs_max_values],
        "corr_mean_in_window": [float(x) for x in signed_mean_values],
        "corr_window_shape": [int(corr_win.shape[0]), int(corr_win.shape[1])],
    }
    return winner_idx, mode_used, debug_payload


def winner_display_lines(
    winner_key: str,
    mode_short: str,
    lsl_cue_target_id: Optional[int],
) -> Tuple[List[str], int, bool]:
    win_digit = stim_key_to_tile_digit(winner_key)
    lines = ["РЕЗУЛЬТАТ:", f"ПЛИТКА {win_digit}", f"режим: {mode_short}"]
    if lsl_cue_target_id is not None:
        lines.append(f"цель LSL: {lsl_cue_target_id}")
    match_lsl = lsl_cue_target_id is None or win_digit == lsl_cue_target_id
    return lines, win_digit, match_lsl
