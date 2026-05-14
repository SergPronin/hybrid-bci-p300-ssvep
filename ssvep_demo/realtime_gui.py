#!/usr/bin/env python3
"""
Realtime GUI для SSVEP-демо: LSL EEG → rolling buffer → MSI → предсказание частоты.

* **PyQt6** — окно и таймер.
* **pyqtgraph** — быстрый осциллограммный график (мультиканал).

Авто-поиск LSL-потоков ``type == 'EEG'``. Если потоков нет — подсказка запустить ``fake_eeg_lsl.py``.

Не импортирует ``qt_window`` и не трогает P300 pipeline.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("SSVEP_DEMO_LAUNCHED") == "1":
    _ve = os.environ.get("VIRTUAL_ENV", "")
    _in = bool(_ve) or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    print(f"[realtime_gui] sys.executable={sys.executable!r}", flush=True)
    print(f"[realtime_gui] resolved={os.path.normpath(os.path.realpath(sys.executable))!r}", flush=True)
    print(f"[realtime_gui] sys.prefix={sys.prefix!r} VIRTUAL_ENV={_ve!r} venv-like={_in}", flush=True)
    print(f"[realtime_gui] sys.path[:10]={sys.path[:10]!r}", flush=True)

import time
import traceback
from pathlib import Path

import numpy as np

# pyqtgraph: явно выбираем Qt6 до импорта pyqtgraph
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import pyqtgraph as pg  # noqa: E402
from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

try:
    from ssvep_demo.msi_realtime import MSIRealtimeClassifier, RollingEEGBuffer  # noqa: E402
except ImportError:
    from msi_realtime import MSIRealtimeClassifier, RollingEEGBuffer  # noqa: E402

try:
    from pylsl import StreamInlet, resolve_byprop, resolve_streams
except ImportError as e:
    raise SystemExit(
        f"Нужен pylsl (pip install pylsl). sys.executable={sys.executable!r} "
        f"prefix={sys.prefix!r} VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '')!r}"
    ) from e
try:
    from pylsl import LostError
except ImportError:
    # Новые сборки pylsl не реэкспортируют LostError из корня пакета.
    from pylsl.util import LostError


SRATE = 250.0
WINDOW_SEC = 2.0
N_CHANNELS = 2
BUF_CAP = 3000
PLOT_COLS = 1250
PULL_MS = 160


def _resolve_eeg_candidates() -> list:
    found = resolve_byprop("type", "EEG", minimum=1, timeout=1.2)
    if found:
        return list(found)
    alls = resolve_streams(wait_time=0.8)
    return [s for s in alls if (s.type() or "").upper() == "EEG"]


class RealtimeDemoWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP + MSI realtime (LSL)")
        self._inlet: StreamInlet | None = None
        self._buf = RollingEEGBuffer(N_CHANNELS, BUF_CAP)
        self._clf: MSIRealtimeClassifier | None = None
        self._stream_infos: list = []
        self._last_pred = "---"
        self._last_winner = -1
        self._t0 = time.perf_counter()
        self._updates = 0
        self._last_debug = 0.0

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)

        row = QHBoxLayout()
        self._combo = QComboBox()
        self._combo.setMinimumWidth(320)
        b_ref = QPushButton("Обновить список")
        b_ref.clicked.connect(self._refresh_streams)
        b_conn = QPushButton("Подключиться")
        b_conn.clicked.connect(self._connect)
        row.addWidget(QLabel("LSL EEG:"))
        row.addWidget(self._combo, stretch=1)
        row.addWidget(b_ref)
        row.addWidget(b_conn)
        lay.addLayout(row)

        self._hint = QLabel()
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color:#9ab; font-size:12px;")
        self._hint.setText(
            "Нет потоков? Запустите: python -m ssvep_demo.fake_eeg_lsl\n"
            "Смена частоты синтетики (UDP 17391): printf '1' | nc -u -w0 127.0.0.1 17391  (1..4)"
        )
        lay.addWidget(self._hint)

        self._stream_label = QLabel("Поток: —")
        self._stream_label.setStyleSheet("color:#ccd;")
        lay.addWidget(self._stream_label)

        pred = QLabel(">>>  — Hz  <<<")
        pred.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pred.setFont(pred.font())
        f = pred.font()
        f.setPointSize(22)
        f.setBold(True)
        pred.setFont(f)
        pred.setStyleSheet("color:#6f6; padding:16px;")
        self._pred_label = pred
        lay.addWidget(pred)

        self._meta = QLabel("winner: —  |  coef: —")
        self._meta.setStyleSheet("color:#aab;")
        lay.addWidget(self._meta)

        self._plot = pg.PlotWidget(title="EEG (последние отсчёты)")
        self._plot.showGrid(x=True, y=True, alpha=0.35)
        self._plot.setBackground("#1a1a1e")
        self._curves = [
            self._plot.plot(pen=pg.mkPen("#7af", width=1)),
            self._plot.plot(pen=pg.mkPen("#fa7", width=1)),
        ]
        lay.addWidget(self._plot, stretch=1)

        self._status = QLabel("Инициализация MSI…")
        self._status.setStyleSheet("color:#889;")
        lay.addWidget(self._status)

        QTimer.singleShot(50, self._init_msi)
        self._timer = QTimer(self)
        self._timer.setInterval(PULL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        self._refresh_streams()

    def _log(self, msg: str) -> None:
        print(f"[realtime_gui] {msg}", flush=True)

    def _init_msi(self) -> None:
        try:
            self._clf = MSIRealtimeClassifier(
                srate=SRATE, window_sec=WINDOW_SEC, n_channels=N_CHANNELS
            )
            self._status.setText(
                f"MSI готов | окно {WINDOW_SEC}s = {self._clf.window_samples} отсчётов | fs={SRATE} Hz"
            )
            self._log("MSI initialized OK")
        except Exception as e:
            self._status.setText(f"Ошибка MSI: {e}")
            self._log(f"MSI init FAILED: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "MSI", str(e))

    def _refresh_streams(self) -> None:
        self._combo.clear()
        try:
            self._stream_infos = _resolve_eeg_candidates()
        except Exception as e:
            self._log(f"resolve error: {e}")
            self._stream_infos = []
        for info in self._stream_infos:
            name = info.name() or "(no name)"
            rate = info.nominal_srate() or 0.0
            self._combo.addItem(f"{name}  @ {rate:.0f} Hz")
        self._log(f"LSL EEG candidates: {self._combo.count()}")

    def _connect(self) -> None:
        if self._combo.currentIndex() < 0 or not self._stream_infos:
            QMessageBox.warning(self, "LSL", "Нет потоков в списке.")
            return
        try:
            if self._inlet is not None:
                try:
                    self._inlet.close_stream()
                except Exception:
                    pass
            idx = self._combo.currentIndex()
            if idx < 0 or idx >= len(self._stream_infos):
                QMessageBox.warning(self, "LSL", "Некорректный выбор.")
                return
            picked = self._stream_infos[idx]
            self._inlet = StreamInlet(picked, max_buflen=8)
            self._buf.clear()
            fs = picked.nominal_srate() or 0.0
            self._stream_label.setText(
                f"Поток: {picked.name()} | заявлено {fs:.1f} Hz (классификатор ожидает {SRATE})"
            )
            if abs(fs - SRATE) > 3:
                self._stream_label.setText(
                    self._stream_label.text()
                    + "  ⚠ несовпадение fs с MSI (250) — демо может врать"
                )
            self._log(f"connected inlet name={picked.name()} fs={fs}")
        except Exception as e:
            QMessageBox.critical(self, "LSL", str(e))
            self._log(traceback.format_exc())

    def _tick(self) -> None:
        self._updates += 1
        if self._inlet is None:
            return
        try:
            chunk, ts = self._inlet.pull_chunk(max_samples=256, timeout=0.0)
        except LostError:
            self._log("LSL stream lost")
            self._inlet = None
            return
        except Exception as e:
            self._log(f"pull_chunk error: {e}")
            return
        if not chunk:
            return
        a = np.asarray(chunk, dtype=np.float64)
        if a.ndim != 2 or a.shape[0] == 0:
            return
        # pylsl: (n_samples, n_channels)
        if a.shape[1] != N_CHANNELS:
            self._log(f"skip chunk shape {a.shape} (need {N_CHANNELS} channels)")
            return
        x = a.T.copy()
        try:
            self._buf.append(x)
        except ValueError as e:
            self._log(f"buffer append: {e}")
            return

        win = self._buf.get_window(self._clf.window_samples) if self._clf else None
        if win is not None and self._clf is not None:
            try:
                pr = self._clf.predict(win)
                hz = pr.get("freq_hz")
                w1 = int(pr["winner_1based"])
                self._last_pred = f"{hz:g} Hz" if hz is not None else f"winner {w1}"
                self._last_winner = w1
                self._pred_label.setText(f">>>  {self._last_pred}  <<<")
                self._meta.setText(
                    f"winner (1-based): {w1}  |  coef: {pr.get('coef_repr', '—')}"
                )
            except Exception as e:
                self._meta.setText(f"MSIExec error: {e}")
                self._log(traceback.format_exc())

        plot_data = self._buf.latest_columns(PLOT_COLS)
        if plot_data is not None:
            t_axis = np.arange(plot_data.shape[1], dtype=np.float64) / SRATE
            for c in range(min(N_CHANNELS, plot_data.shape[0])):
                self._curves[c].setData(t_axis, plot_data[c])

        now = time.perf_counter()
        if now - self._last_debug > 2.0:
            self._last_debug = now
            rate = self._updates / (now - self._t0)
            self._log(
                f"buf_total={self._buf.total_samples} pred={self._last_pred!r} "
                f"winner={self._last_winner} ui_timer_hz~{rate * (PULL_MS / 1000):.1f}"
            )


def main() -> int:
    app = QApplication(sys.argv)
    w = RealtimeDemoWindow()
    w.resize(980, 640)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
