"""SSVEPOnlineEngine: burst gate получает массив времён, не скаляр."""

from __future__ import annotations

import numpy as np

from ssvep_analysis.online_engine import SSVEPOnlineEngine, SSVEPParams


def test_can_classify_burst_passes_buf_times_array() -> None:
    eng = SSVEPOnlineEngine()
    eng.reset(
        params=SSVEPParams(
            fs_hz=250.0,
            window_sec=2.0,
            freqs_hz=(10.0, 12.0),
            mode="burst",
        )
    )
    eng._burst_gate.set_active_lamps(1)
    eng._burst_gate.ingest_marker(0.0, "100|on")
    n = eng._n_samples
    eng._buf_t = list(np.linspace(0.0, 1.9, n))
    eng._buf_x = [np.zeros(2) for _ in range(n)]
    assert eng.can_classify() in (True, False)
