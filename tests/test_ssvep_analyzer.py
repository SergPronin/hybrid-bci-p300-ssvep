"""Тесты ssvep_analyzer (без LSL/железа)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ssvep_analysis.burst_gate import BurstGate, BurstGateConfig  # noqa: E402

# Импорт после QT env
import ssvep_analyzer as sa  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestLampFrequencyHelpers:
    def test_choices_count(self) -> None:
        choices = sa.lamp_frequency_choices()
        assert len(choices) == 500
        assert choices[0][1] == pytest.approx(1000.0)

    def test_closest_index(self) -> None:
        idx = sa.lamp_frequency_closest_index(10.0)
        _, hz = sa.lamp_frequency_choices()[idx]
        assert hz == pytest.approx(10.0, rel=0.02)


class TestCoefToStrings:
    def test_count_listing(self) -> None:
        msi = MagicMock()
        coef = MagicMock()
        coef.Count = 2
        coef.__getitem__ = lambda _s, i: 0.5 + i * 0.1
        msi.Coef = coef
        lines = sa._coef_to_strings(msi, [8.0, 12.0])
        assert len(lines) == 2
        assert "8" in lines[0] and "12" in lines[1]


def _fake_channel_cb() -> MagicMock:
    cb = MagicMock()
    cb.isChecked.return_value = True
    return cb


class TestSSVEPAnalyzerWindow:
    def test_window_construct(self, qapp: QApplication) -> None:
        w = sa.SSVEPAnalyzerWindow()
        assert w._cb_stim_mode.count() == 2
        assert w._cb_stim_mode.currentData() == "continuous"
        w.close()

    def test_burst_mode_flag(self, qapp: QApplication) -> None:
        w = sa.SSVEPAnalyzerWindow()
        w._cb_stim_mode.setCurrentIndex(1)
        assert w._is_burst_mode()
        w._cb_stim_mode.setCurrentIndex(0)
        assert not w._is_burst_mode()
        w.close()

    def test_classify_burst_gated_skips_msi(self, qapp: QApplication) -> None:
        w = sa.SSVEPAnalyzerWindow()
        w._cb_stim_mode.setCurrentIndex(1)
        w._freqs_hz = [10.0, 12.0]
        w._burst_gate.set_active_lamps(2)
        w._n_template = 100
        w._nominal_fs = 250.0
        w._buf = np.random.randn(500, 2)
        w._buf_t = np.linspace(0.0, 2.0, 500)
        w._msi = MagicMock()
        w._ch_checkboxes = [_fake_channel_cb(), _fake_channel_cb()]
        w._marker_inlet = MagicMock()
        with patch.object(sa.tme, "numpy_to_double_matrix2d"):
            w._on_classify()
        w._msi.MSIExec.assert_not_called()
        assert "ПАУЗА" in w._lbl_winner.text()
        w.close()

    def test_classify_continuous_calls_msi(self, qapp: QApplication) -> None:
        w = sa.SSVEPAnalyzerWindow()
        w._cb_stim_mode.setCurrentIndex(0)
        w._freqs_hz = [10.0]
        w._n_template = 50
        w._buf = np.random.randn(100, 1)
        w._buf_t = np.linspace(0.0, 1.0, 100)
        w._msi = MagicMock()
        w._msi.MSIExec.return_value = 1
        w._ch_checkboxes = [_fake_channel_cb()]
        with patch.object(sa.tme, "numpy_to_double_matrix2d", return_value=MagicMock()):
            w._on_classify()
        w._msi.MSIExec.assert_called_once()
        w.close()

    def test_classify_burst_allowed_calls_msi(self, qapp: QApplication) -> None:
        w = sa.SSVEPAnalyzerWindow()
        w._cb_stim_mode.setCurrentIndex(1)
        w._freqs_hz = [10.0]
        w._n_template = 50
        w._nominal_fs = 250.0
        w._buf = np.random.randn(500, 1)
        w._buf_t = np.linspace(0.0, 2.0, 500)
        w._burst_gate = BurstGate(BurstGateConfig(window_sec=2.0, min_on_fraction=0.5, min_on_sec=0.4))
        w._burst_gate.set_active_lamps(1)
        w._burst_gate.ingest_marker(0.0, "100|on")
        w._msi = MagicMock()
        w._msi.MSIExec.return_value = 1
        w._ch_checkboxes = [_fake_channel_cb()]
        w._marker_inlet = MagicMock()
        with patch.object(sa.tme, "numpy_to_double_matrix2d", return_value=MagicMock()):
            w._on_classify()
        w._msi.MSIExec.assert_called_once()
        w.close()


class TestMigalkaFreqList:
    def test_migalka_labels_match_count(self) -> None:
        labels = sa.lamp_frequency_choices()
        assert len(labels) == 500
        assert labels[0][1] == pytest.approx(1000.0)
