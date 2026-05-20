"""Полноэкранная подсказка перед блоком ССВП (номер эксперимента и лампа-диод)."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class SsvepCueOverlay(QWidget):
    """Серый fullscreen, как оверлей паузы в PsychoPy-стимуляторе."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet("background-color: #202020;")
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 40, 40, 40)
        root.addStretch(2)

        self.lbl_title = QLabel("")
        self.lbl_title.setAlignment(Qt.AlignCenter)
        self.lbl_title.setStyleSheet("color: white;")
        self.lbl_title.setFont(QFont("", 42, QFont.Bold))
        root.addWidget(self.lbl_title)

        self.lbl_mode = QLabel("")
        self.lbl_mode.setAlignment(Qt.AlignCenter)
        self.lbl_mode.setStyleSheet("color: #cccccc;")
        self.lbl_mode.setFont(QFont("", 22))
        root.addWidget(self.lbl_mode)

        self.lbl_sub = QLabel("Смотрите на лампу (диод):")
        self.lbl_sub.setAlignment(Qt.AlignCenter)
        self.lbl_sub.setStyleSheet("color: white;")
        self.lbl_sub.setFont(QFont("", 28))
        root.addWidget(self.lbl_sub)

        self.lbl_lamp = QLabel("")
        self.lbl_lamp.setAlignment(Qt.AlignCenter)
        self.lbl_lamp.setStyleSheet("color: #ffdd44;")
        self.lbl_lamp.setFont(QFont("", 96, QFont.Bold))
        root.addWidget(self.lbl_lamp)

        self.lbl_hz = QLabel("")
        self.lbl_hz.setAlignment(Qt.AlignCenter)
        self.lbl_hz.setStyleSheet("color: #ffdd44;")
        self.lbl_hz.setFont(QFont("", 32))
        root.addWidget(self.lbl_hz)

        root.addStretch(3)
        self.hide()

    def show_cue(
        self,
        *,
        experiment_index: int,
        experiment_total: int,
        lamp_1based: int,
        freq_hz: float | None = None,
        mode_label: str = "",
    ) -> None:
        self.lbl_title.setText(f"Эксперимент №{int(experiment_index)} из {int(experiment_total)}")
        self.lbl_mode.setText(str(mode_label) if mode_label else "ССВП")
        self.lbl_lamp.setText(str(int(lamp_1based)))
        if freq_hz is not None and float(freq_hz) > 0:
            self.lbl_hz.setText(f"≈ {float(freq_hz):g} Гц")
            self.lbl_hz.show()
        else:
            self.lbl_hz.hide()
        if not self.isVisible():
            self.showFullScreen()
        self.raise_()
        self.activateWindow()

    def hide_overlay(self) -> None:
        self.hide()
