"""
MSI realtime: rolling буфер + классификатор поверх test_msi_exec / test_msi_import.

Загрузка DLL и построение ModelSignal — только через ``import test_msi_exec`` (без копипасты CoreCLR).

Кратко о SSVEP vs P300
----------------------
* **P300**: событийный потенциал, ответ на редкий стимул во времени; классификация по форме ERP.
* **SSVEP**: непрерывная модуляция ЭЭГ на частоте (и гармониках) **ритма визуального мигания**;
  классификатор ищет, какая из эталонных частот лучше всего объясняет наблюдаемый сигнал.

**MSI** здесь — готовый managed-класс: ему задают эталоны ``ModelSignal`` и окно ЭЭГ ``double[,]``,
возвращается индекс победителя (в демо мы используем 1-based как в MSIController).

Rolling window
--------------
Классификатору нужно **фиксированное число отсчётов** (например 2 с при 250 Hz → 500 сэмплов).
``RollingEEGBuffer`` накапливает поток по каналам и отдаёт последний прямоугольный кусок
``(n_channels, window_samples)`` в хронологическом порядке (старый → новый по времени).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"


def _ensure_scripts_on_path() -> None:
    for p in (str(_SCRIPTS), str(_REPO_ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)


class RollingEEGBuffer:
    """
    Кольцевой буфер ЭЭГ: форма (n_channels, max_samples).

    * ``append`` принимает куски (ch, n_new) по времени слева направо.
    * ``get_window(n)`` возвращает последние n отсчётов, столбцы от старого к новому, или None если данных мало.
    """

    def __init__(self, n_channels: int, max_samples: int) -> None:
        if n_channels < 1 or max_samples < 2:
            raise ValueError("n_channels>=1, max_samples>=2")
        self._n_ch = n_channels
        self._cap = max_samples
        self._buf = np.zeros((n_channels, max_samples), dtype=np.float64)
        self._pos = 0  # следующая позиция записи [0, cap)
        self._total = 0  # всего принятых сэмплов (монотонно)

    @property
    def n_channels(self) -> int:
        return self._n_ch

    @property
    def total_samples(self) -> int:
        return self._total

    def clear(self) -> None:
        self._buf.fill(0.0)
        self._pos = 0
        self._total = 0

    def append(self, chunk: np.ndarray) -> None:
        """
        chunk: shape (n_channels, n_new), float-подобный; копируется в буфер по времени.
        """
        if chunk.ndim != 2:
            raise ValueError(f"append: ожидали 2D, shape={chunk.shape}")
        ch, n_new = chunk.shape
        if ch != self._n_ch:
            raise ValueError(f"append: channels {ch} != buffer {self._n_ch}")
        for j in range(n_new):
            self._buf[:, self._pos] = np.asarray(chunk[:, j], dtype=np.float64)
            self._pos = (self._pos + 1) % self._cap
            self._total += 1

    def get_window(self, n_samples: int) -> np.ndarray | None:
        """Последние n_samples (старые → новые). None если ещё не накопили."""
        if n_samples < 1 or n_samples > self._cap:
            raise ValueError("invalid n_samples")
        if self._total < n_samples:
            return None
        out = np.empty((self._n_ch, n_samples), dtype=np.float64)
        for j in range(n_samples):
            age = n_samples - 1 - j
            idx = (self._pos - 1 - age) % self._cap
            out[:, j] = self._buf[:, idx]
        return out

    def latest_columns(self, n_samples: int) -> np.ndarray | None:
        """Последние min(n_samples, total, capacity) столбцов (для графика)."""
        n = int(min(n_samples, self._total, self._cap))
        if n < 2:
            return None
        return self.get_window(n)


class MSIRealtimeClassifier:
    """
    Обёртка над MSI: один раз грузит runtime, строит ModelSignal (10/12/15/20 Hz sin/cos),
    на каждом окне вызывает MSIExec.
    """

    FREQS_HZ = (10.0, 12.0, 15.0, 20.0)

    def __init__(
        self,
        srate: float = 250.0,
        window_sec: float = 2.0,
        n_channels: int = 2,
    ) -> None:
        if n_channels != 2:
            # MSI в smoke-тесте проверялся на 2 каналах; иначе нужна сверка с контрактом DLL.
            raise ValueError("В демо зафиксировано n_channels=2 (совместимость с MSIExec).")
        self.srate = float(srate)
        self.window_sec = float(window_sec)
        self.n_channels = int(n_channels)
        self.window_samples = int(round(self.srate * self.window_sec))
        if self.window_samples < 8:
            raise ValueError("Слишком короткое окно")

        _ensure_scripts_on_path()
        import test_msi_exec as tme  # noqa: WPS433

        self._tme = tme
        self._msi, self._msi_res, self._dotnet_root = tme.load_msi_runtime()
        np_models = tme.generate_model_signals(self.FREQS_HZ, self.srate, self.window_sec)
        model_list = tme.build_model_signal_list(self._msi, np_models, verbose=False)
        self._msi.ModelSignal = model_list

    @property
    def freqs_hz(self) -> tuple[float, ...]:
        return self.FREQS_HZ

    def predict(self, window_ch_samples: np.ndarray) -> dict[str, Any]:
        """
        window_ch_samples: (2, window_samples), float64.

        Возвращает dict: winner (1-based), freq_hz, raw_winner, coef_repr.
        """
        if window_ch_samples.shape != (self.n_channels, self.window_samples):
            raise ValueError(
                f"predict: ожидали shape ({self.n_channels}, {self.window_samples}), "
                f"получили {window_ch_samples.shape}"
            )
        x = np.ascontiguousarray(window_ch_samples, dtype=np.float64)
        managed = self._tme.numpy_to_double_matrix2d(x, verbose=False)
        raw = int(self._msi.MSIExec(managed))
        freq = None
        if 1 <= raw <= len(self.FREQS_HZ):
            freq = self.FREQS_HZ[raw - 1]
        coef_repr = self._tme._format_coef_for_log(self._msi)
        return {
            "winner_1based": raw,
            "freq_hz": freq,
            "coef_repr": coef_repr,
        }
