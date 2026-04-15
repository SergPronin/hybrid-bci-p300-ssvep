from __future__ import annotations

import numpy as np

from p300_analysis.erp_compute import compute_corrected_and_integrated, compute_winner_metrics
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_SIGNED_MEAN


def test_auc_mode_uses_absolute_integral_and_matches_integrated_plot() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    raw = np.array(
        [
            [0.0, 3.0, -8.0, 0.0],
            [0.0, 2.0, 2.0, 0.0],
        ],
        dtype=np.float64,
    )

    corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
        raw,
        time_ms,
        baseline_ms=100,
        window_x_ms=100,
        window_y_ms=300,
    )
    winner_idx, mode_used, dbg = compute_winner_metrics(
        stim_keys,
        raw_averaged=raw,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=wx,
        window_y_ms=wy,
        winner_mode=WINNER_MODE_AUC,
    )

    assert mode_used == WINNER_MODE_AUC
    assert winner_idx == 0
    np.testing.assert_allclose(dbg["abs_auc_values"], integrated[:, -1] * 100.0)
    assert dbg["chosen_winner_key"] == "стимул_0"


def test_signed_mean_mode_remains_available_for_offline_comparison() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    corrected = np.array(
        [
            [0.0, 12.0, -10.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    winner_idx, mode_used, _ = compute_winner_metrics(
        stim_keys,
        raw_averaged=corrected,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=0,
        window_y_ms=300,
        winner_mode=WINNER_MODE_SIGNED_MEAN,
    )

    assert mode_used == WINNER_MODE_SIGNED_MEAN
    assert winner_idx == 1
