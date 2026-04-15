from __future__ import annotations

import numpy as np

from p300_analysis.erp_compute import compute_winner_metrics
from p300_analysis.signal_processing import integrated_cumsum
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_SIGNED_MEAN


def test_auc_mode_prefers_positive_p300_area_over_signed_mean() -> None:
    stim_keys = ["стимул_0", "стимул_1"]
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)
    corrected = np.array(
        [
            [0.0, 12.0, -10.0, 0.0],
            [2.0, 2.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )

    auc_winner_idx, auc_mode, auc_dbg = compute_winner_metrics(
        stim_keys,
        raw_averaged=corrected,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=0,
        window_y_ms=300,
        winner_mode=WINNER_MODE_AUC,
    )
    mean_winner_idx, mean_mode, _ = compute_winner_metrics(
        stim_keys,
        raw_averaged=corrected,
        corrected=corrected,
        time_ms=time_ms,
        window_x_ms=0,
        window_y_ms=300,
        winner_mode=WINNER_MODE_SIGNED_MEAN,
    )

    assert auc_mode == WINNER_MODE_AUC
    assert auc_winner_idx == 0
    assert auc_dbg["chosen_winner_key"] == "стимул_0"
    assert auc_dbg["positive_auc_values"][0] > auc_dbg["positive_auc_values"][1]

    assert mean_mode == WINNER_MODE_SIGNED_MEAN
    assert mean_winner_idx == 1


def test_integrated_cumsum_ignores_negative_lobes() -> None:
    corrected = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float64)
    time_ms = np.array([0.0, 100.0, 200.0, 300.0], dtype=np.float64)

    integrated, cropped_time = integrated_cumsum(
        corrected,
        time_ms,
        window_x_ms=0,
        window_y_ms=300,
    )

    np.testing.assert_allclose(cropped_time, time_ms)
    np.testing.assert_allclose(integrated, np.array([[1.0, 1.0, 4.0, 4.0]], dtype=np.float64))
