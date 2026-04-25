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
        self._baseline_ms: Optional[int] = None

    def reset(self) -> None:
        self._dt_ms = None
        self._epoch_len = None
        self._time_ms_template = None
        self._baseline_ms = None

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
        baseline_ms: int = 0,
    ) -> None:
        baseline_ms = max(0, int(baseline_ms))
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
        self._baseline_ms = baseline_ms
        epoch_total_ms = float(EPOCH_DURATION_MS + baseline_ms)
        self._epoch_len = int(round(epoch_total_ms / dt_ms)) + 1
        self._time_ms_template = np.arange(self._epoch_len, dtype=np.float64) * dt_ms - baseline_ms

    def compute_start_index(self, time_arr: np.ndarray, t_eff: float) -> Optional[int]:
        """Возвращает индекс начала эпохи в time_arr, ближайший к t_eff.

        Использует номинальный dt (от srate потока), а НЕ фактические timestamp'ы,
        потому что многие LSL-драйверы (в т.ч. Neurospect) отдают timestamp'ы
        с разрешением 1 с — все сэмплы внутри секунды получают одну и ту же метку.
        """
        if self._dt_ms is None or self._epoch_len is None:
            return None
        n = int(time_arr.shape[0])
        el = int(self._epoch_len)
        if n < el:
            return None
        dt_s = float(self._dt_ms) / 1000.0
        t0 = float(time_arr[0])

        i_nom = int(np.round((t_eff - t0) / dt_s))
        i_nom = max(0, min(i_nom, n - el))
        return i_nom
