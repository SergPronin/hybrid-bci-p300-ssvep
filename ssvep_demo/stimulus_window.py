#!/usr/bin/env python3
"""
Отдельное окно SSVEP-стимуляции: 4 квадрата с **квадратной** модуляцией яркости на 10/12/15/20 Hz.

Фаза берётся от ``time.perf_counter()`` — частота не «уплывает» из-за дрейфа QTimer.

Полноэкранный режим: **F11** (переключение). **Esc** — выход из полноэкранного.

Не использует PsychoPy и не связан с P300-стимулятором проекта.
"""

from __future__ import annotations

import os
import sys

if os.environ.get("SSVEP_DEMO_LAUNCHED") == "1":
    _ve = os.environ.get("VIRTUAL_ENV", "")
    _in = bool(_ve) or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    print(f"[stimulus_window] sys.executable={sys.executable!r}", flush=True)
    print(f"[stimulus_window] resolved={os.path.normpath(os.path.realpath(sys.executable))!r}", flush=True)
    print(f"[stimulus_window] sys.prefix={sys.prefix!r} VIRTUAL_ENV={_ve!r} venv-like={_in}", flush=True)
    print(f"[stimulus_window] sys.path[:10]={sys.path[:10]!r}", flush=True)

import argparse
import math
import time
from typing import Sequence

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QFrame, QGridLayout, QLabel, QWidget


class FlickerFrame(QFrame):
    """Квадрат: яркость high/low по sign(sin(2π f t))."""

    def __init__(self, freq_hz: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._freq = float(freq_hz)
        self._t0 = time.perf_counter()
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(2)
        self.setMinimumSize(160, 160)
        self._update_color()

    def tick(self) -> None:
        self._update_color()

    def _update_color(self) -> None:
        t = time.perf_counter() - self._t0
        phase = 2.0 * math.pi * self._freq * t
        on = math.sin(phase) >= 0.0
        bg = QColor(235, 235, 235) if on else QColor(25, 25, 28)
        self.setStyleSheet(f"background-color: {bg.name()}; border-radius: 8px;")


class StimulusWindow(QWidget):
    def __init__(self, freqs: Sequence[float] = (10.0, 12.0, 15.0, 20.0)) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP demo — flicker (10 / 12 / 15 / 20 Hz)")
        self.setStyleSheet("background-color: #121214;")
        self._fullscreen = False

        grid = QGridLayout(self)
        grid.setSpacing(24)
        self._frames: list[FlickerFrame] = []
        labels: list[str] = [f"{int(f)} Hz" for f in freqs]

        for i, (fhz, lab) in enumerate(zip(freqs, labels)):
            cell = QWidget()
            v = QGridLayout(cell)
            fr = FlickerFrame(fhz)
            lb = QLabel(lab)
            lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lb.setStyleSheet("color: #c8c8d0; font-weight: bold;")
            lb.setFont(QFont("Sans Serif", 14))
            v.addWidget(fr, 0, 0)
            v.addWidget(lb, 1, 0)
            grid.addWidget(cell, 0, i)
            self._frames.append(fr)

        # ~120 Hz обновление фазы (достаточно для стабильного восприятия мерцания)
        self._timer = QTimer(self)
        self._timer.setInterval(8)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("F11"), self, activated=self._toggle_fullscreen)

    def _on_tick(self) -> None:
        for fr in self._frames:
            fr.tick()

    def _toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        if self._fullscreen:
            self.showFullScreen()
        else:
            self.showNormal()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    w = StimulusWindow()
    w.resize(920, 260)
    if args.fullscreen:
        w.showFullScreen()
    else:
        w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
