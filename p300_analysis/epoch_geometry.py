"""Шаблон времени эпохи и поиск индекса начала эпохи по t_eff."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from pylsl import StreamInlet

from p300_analysis.constants import EPOCH_DURATION_MS


class EpochGeometry:
    """Хранит dt, длину эпохи и сетку времени в мс; ищет start_idx для вырезания ERP."""

    def __init__(self) -> None:
        self._dt_ms: Optional[float] = None
        self._epoch_len: Optional[int] = None
        self._time_ms_template: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._dt_ms = None
        self._epoch_len = None
        self._time_ms_template = None

    @property
    def dt_ms(self) -> Optional[float]:
        return self._dt_ms

    @property
    def epoch_len(self) -> Optional[int]:
        return self._epoch_len

    @property
    def time_ms_template(self) -> Optional[np.ndarray]:
        return self._time_ms_template

    def ensure_template(
        self,
        inlet_eeg: Optional[StreamInlet],
        eeg_times: List[float],
    ) -> None:
        if self._time_ms_template is not None and self._epoch_len is not None and self._dt_ms is not None:
            return

        dt_ms: Optional[float] = None
        try:
            if inlet_eeg is not None:
                srate = float(inlet_eeg.info().nominal_srate())
                if srate > 0:
                    dt_ms = 1000.0 / srate
        except Exception:
            dt_ms = None

        if dt_ms is None and len(eeg_times) >= 100:
            times = np.asarray(eeg_times[-200:], dtype=np.float64)
            diffs = np.diff(times)
            diffs = diffs[diffs > 0]
            if diffs.size:
                dt_ms = float(np.median(diffs) * 1000.0)

        if dt_ms is None or dt_ms <= 0:
            return

        self._dt_ms = dt_ms
        self._epoch_len = int(round(EPOCH_DURATION_MS / dt_ms)) + 1
        self._time_ms_template = np.arange(self._epoch_len, dtype=np.float64) * dt_ms

    def compute_start_index(self, time_arr: np.ndarray, t_eff: float) -> Optional[int]:
        if self._dt_ms is None or self._epoch_len is None:
            return None
        n = int(time_arr.shape[0])
        el = int(self._epoch_len)
        if n < el:
            return None
        dt_s = float(self._dt_ms) / 1000.0
        t0 = float(time_arr[0])

        def refine_window(center: int) -> int:
            lo = max(0, center - 30)
            hi = min(n - el, center + 30)
            return int(
                min(range(lo, hi + 1), key=lambda j: (abs(float(time_arr[j]) - t_eff), j))
            )

        i_nom = int(np.round((t_eff - t0) / dt_s))
        i_nom = max(0, min(i_nom, n - el))
        start_idx = refine_window(i_nom)
        err = abs(float(time_arr[start_idx]) - t_eff)
        if err > 1.0:
            i_se = int(np.searchsorted(time_arr, t_eff, side="left"))
            i_se = max(0, min(i_se, n - el))
            start_idx2 = refine_window(i_se)
            if abs(float(time_arr[start_idx2]) - t_eff) < err:
                start_idx = start_idx2
        return int(start_idx)
