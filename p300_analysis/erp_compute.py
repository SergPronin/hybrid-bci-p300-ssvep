"""Усреднение эпох и подготовка данных для отображения победителя."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from p300_analysis.constants import MIN_EPOCHS_TO_DECIDE
from p300_analysis.marker_parsing import stim_key_sort_key, stim_key_to_tile_digit
from p300_analysis.signal_processing import baseline_correction, integrated_cumsum
from p300_analysis.winner_selection import WINNER_MODE_SIGNED_MEAN, pick_winner_by_mode


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
    integrated: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
    winner_mode: str,
) -> Tuple[int, str, Dict[str, Any]]:
    """Возвращает winner_idx, mode_used, data для debug_ndjson (winner_compare)."""
    final_auc_values = integrated[:, -1]
    dt_m = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    xi0 = int(round(float(window_x_ms) / dt_m))
    xi1 = int(round(float(window_y_ms) / dt_m)) + 1
    xi0 = max(0, min(xi0, time_ms.shape[0] - 1))
    xi1 = max(xi0 + 1, min(xi1, time_ms.shape[0]))
    raw_win = raw_averaged[:, xi0:xi1]
    corr_win = corrected[:, xi0:xi1]
    mode = winner_mode if isinstance(winner_mode, str) else WINNER_MODE_SIGNED_MEAN
    winner_idx, mode_used = pick_winner_by_mode(mode, raw_win, corr_win, final_auc_values)
    auc_winner_idx = int(np.argmax(final_auc_values))
    peak_idx = int(np.argmax(np.max(raw_win, axis=1)))
    signed_idx = int(np.argmax(np.mean(corr_win, axis=1)))
    peak_vals = np.max(raw_win, axis=1).tolist()
    signed_vals = np.mean(corr_win, axis=1).tolist()
    debug_payload = {
        "winner_rule": mode_used,
        "chosen_winner_idx": winner_idx,
        "chosen_winner_key": stim_keys[winner_idx],
        "stim_keys": stim_keys,
        "auc_final": [float(x) for x in final_auc_values],
        "auc_winner_idx": auc_winner_idx,
        "auc_winner_key": stim_keys[auc_winner_idx],
        "peak_in_window_idx": peak_idx,
        "peak_in_window_key": stim_keys[peak_idx],
        "signed_mean_idx": signed_idx,
        "signed_mean_key": stim_keys[signed_idx],
        "window_ms": [window_x_ms, window_y_ms],
        "peak_vals": peak_vals,
        "signed_vals": signed_vals,
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
