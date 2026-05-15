"""Тесты диаграммы Ганта SSVEP."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ssvep_analysis.burst_gate import BurstGate  # noqa: E402
from ssvep_analysis.gantt_timeline import GanttExtraRow, StimGanttChart  # noqa: E402


def test_gantt_update_smoke() -> None:
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    chart, plot = StimGanttChart.create_plot(span_sec=10.0)
    gate = BurstGate()
    gate.set_active_lamps(2)
    gate.ingest_marker(0.0, "100|on")
    gate.ingest_marker(5.0, "100|off")
    chart.update(
        t_now=10.0,
        n_lamps=2,
        intervals=gate.intervals_in_range(0.0, 10.0),
        extra_rows=[
            GanttExtraRow(label="MSI разрешён", intervals=[(2.0, 8.0)], color="#43a047"),
        ],
        msi_window_sec=2.0,
        gate_allowed=True,
        msi_events=[(9.5, 1)],
    )
    assert len(chart._tracks) >= 3  # 2 лампы + 1 extra
    assert len(chart._bars) >= 1
    plot.close()
