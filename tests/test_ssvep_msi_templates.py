"""Тесты шаблонов MSI (numpy) и опционально MSIExec."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
for _p in (str(_SCRIPTS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_msi_exec as tme  # noqa: E402


class TestGenerateModelSignals:
    def test_shape_sin_cos(self) -> None:
        freqs = [8.0, 12.0]
        models = tme.generate_model_signals(freqs, srate=250.0, duration=2.0)
        assert len(models) == 2
        assert models[0].shape == (2, 500)
        assert models[1].shape == (2, 500)

    def test_frequency_content(self) -> None:
        f0 = 10.0
        fs = 250.0
        dur = 1.0
        m = tme.generate_model_signals([f0], fs, dur)[0]
        t = np.arange(m.shape[1]) / fs
        ref = np.sin(2 * np.pi * f0 * t)
        corr = np.corrcoef(m[0], ref)[0, 1]
        assert corr > 0.99

    def test_raises_short_duration(self) -> None:
        with pytest.raises(ValueError):
            tme.generate_model_signals([10.0], 250.0, 0.001)


class TestGenerateSineSignal:
    def test_channels_samples(self) -> None:
        x = tme.generate_sine_signal(12.0, 250.0, 2.0, channels=3)
        assert x.shape == (3, 500)


def _load_msi_or_skip():
    try:
        return tme.load_msi_runtime()
    except Exception as e:
        pytest.skip(f"MSI runtime unavailable: {e}")


class TestNumpyToManaged:
    def test_roundtrip_shape(self) -> None:
        _load_msi_or_skip()
        arr = np.random.randn(2, 100)
        managed = tme.numpy_to_double_matrix2d(arr, verbose=False)
        assert managed.GetLength(0) == 2
        assert managed.GetLength(1) == 100


@pytest.mark.integration
class TestMsiExecSynthetic:
    def test_picks_matching_frequency(self) -> None:
        msi, _, _ = _load_msi_or_skip()

        freqs = [8.0, 12.0, 15.0]
        fs = 250.0
        dur = 2.0
        target_hz = 12.0
        eeg = tme.generate_sine_signal(target_hz, fs, dur, channels=2)
        models = tme.generate_model_signals(freqs, fs, dur)
        msi.ModelSignal = tme.build_model_signal_list(msi, models, verbose=False)
        managed = tme.numpy_to_double_matrix2d(eeg, verbose=False)
        winner = int(msi.MSIExec(managed))
        assert winner == 2  # 1-based индекс 12 Hz
