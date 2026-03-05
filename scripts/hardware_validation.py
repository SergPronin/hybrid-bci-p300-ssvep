#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аппаратная валидация ЭЭГ: УЛЬТРА-ПРОИЗВОДИТЕЛЬНАЯ ВЕРСИЯ.

Оптимизации для мощных ПК (60 FPS):
- Умное масштабирование оси Y (перерисовка осей только при выходе за границы).
- Отключено сглаживание (Antialiasing=False) для максимального FPS.
- Используется Qt.PreciseTimer для жестких 16ms (60 кадров/сек).
- Отключена обработка событий мыши на графиках.
"""

import sys
import time
import logging
import numpy as np
from typing import Optional, List

from pylsl import StreamInlet, StreamInfo, resolve_byprop

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QGroupBox, QInputDialog
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
EEG_STREAM_TYPES = ("EEG", "Signal")
WINDOW_SEC = 1.0  # скользящее окно (сек)
COV_UPDATE_INTERVAL = 5.0  # интервал вывода матрицы ковариаций
UPDATE_INTERVAL_MS = 16  # СТРОГО 60 FPS для графиков (1000 / 60 = 16.6)
STATS_INTERVAL_MS = 500  # 2 FPS для текстовой статистики (чтобы не моргали цифры)

SIMULATOR_NAME = "EEG_Simulator"
SIMULATOR_SOURCE_ID = "eeg-simulator-neurospectr"
NEUROSPECTR_MARKER = "neuro"

# НАСТРОЙКИ МАКСИМАЛЬНОЙ ПРОИЗВОДИТЕЛЬНОСТИ
# OpenGL ускоряет отрисовку, сглаживание (antialias) ВЫКЛЮЧАЕМ для стабильных 60 FPS
pg.setConfigOptions(useOpenGL=True, antialias=False)


# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================
def setup_logging():
    logger = logging.getLogger("EEG_Validation")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                  datefmt='%H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("hardware_validation.log", mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


log = setup_logging()


# ==============================================================================
# ПОИСК ПОТОКОВ И МЕТАДАННЫЕ
# ==============================================================================
def _is_allowed_stream(info: StreamInfo) -> bool:
    try:
        name = (info.name() or "").strip().lower()
        sid = (info.source_id() or "").strip().lower()
    except Exception:
        return False
    if name == SIMULATOR_NAME.lower() or SIMULATOR_SOURCE_ID in sid:
        return True
    if NEUROSPECTR_MARKER in name or NEUROSPECTR_MARKER in sid:
        return True
    return False


def find_eeg_streams(timeout: float = 3.0) -> List[StreamInfo]:
    log.info("Поиск LSL потоков ЭЭГ...")
    all_streams = []
    for stream_type in EEG_STREAM_TYPES:
        streams = resolve_byprop("type", stream_type, timeout=timeout)
        all_streams.extend(streams)

    allowed = [s for s in all_streams if _is_allowed_stream(s)]
    log.info(f"Найдено подходящих потоков: {len(allowed)}")
    return allowed


def get_channel_names(info: StreamInfo, n_channels: int) -> List[str]:
    channels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(n_channels):
            name = ch.child_value("label")
            if not name:
                name = ch.child_value("name")
            if not name:
                name = ch.child_value("type")

            channels.append(name.strip() if name else f"Ch {len(channels) + 1}")
            ch = ch.next_sibling()
    except Exception as e:
        log.warning(f"XML с описанием каналов пуст или не прочитан: {e}")

    while len(channels) < n_channels:
        channels.append(f"Ch {len(channels) + 1}")

    return channels[:n_channels]


# ==============================================================================
# GUI КОМПОНЕНТЫ
# ==============================================================================
class ChannelWidget(QWidget):

    def __init__(self, channel_id: int, channel_name: str, sampling_rate: float):
        super().__init__()
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.sampling_rate = sampling_rate
        self.buffer_size = int(WINDOW_SEC * sampling_rate)

        self.y_data = np.zeros(self.buffer_size, dtype=np.float32)
        self.x_data = np.linspace(-WINDOW_SEC, 0, self.buffer_size, dtype=np.float32)
        self.filled = 0

        # Переменные для умного масштабирования
        self.current_y_min = -1.0
        self.current_y_max = 1.0

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)

        header = QLabel(f"[{self.channel_id + 1}] {self.channel_name}")
        header.setFont(QFont("Arial", 9, QFont.Bold))
        header.setStyleSheet("color: white; background-color: #3d3d3d; padding: 2px 5px; border-radius: 3px;")
        layout.addWidget(header)

        self.plot_widget = pg.PlotWidget()

        # Хардкорные оптимизации виджета
        self.plot_widget.setDownsampling(mode=None)
        self.plot_widget.setClipToView(True)
        self.plot_widget.hideButtons()  # Скрываем кнопку 'A'
        self.plot_widget.setMouseEnabled(x=False, y=False)  # Отключаем мышь (экономит процессорное время)
        self.plot_widget.setMenuEnabled(False)

        self.plot_widget.setLabel('left', 'мкВ', color='white', fontsize=7)
        self.plot_widget.setBackground('black')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setXRange(-WINDOW_SEC, 0)

        colors = pg.intColor(self.channel_id, hues=15, values=1, maxValue=255, minValue=150)
        self.signal_line = self.plot_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(colors, width=1.2))

        layout.addWidget(self.plot_widget)

        self.stats_label = QLabel("Ожидание данных...")
        self.stats_label.setFont(QFont("Courier", 8))
        self.stats_label.setStyleSheet("color: #a8ff9e; background-color: #1e1e1e;")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    def push_chunk(self, new_data: np.ndarray):
        n = len(new_data)
        if n == 0:
            return

        self.filled = min(self.buffer_size, self.filled + n)

        if n >= self.buffer_size:
            self.y_data[:] = new_data[-self.buffer_size:]
        else:
            self.y_data[:-n] = self.y_data[n:]
            self.y_data[-n:] = new_data

    def update_plot(self):
        if self.filled == 0:
            return

        # 1. Быстрое обновление линии
        self.signal_line.setData(self.x_data, self.y_data)

        # 2. Умное масштабирование (Smart Y-scaling)
        # Ищем экстремумы
        y_min, y_max = np.min(self.y_data), np.max(self.y_data)
        margin = (y_max - y_min) * 0.15 if y_max != y_min else 10.0

        target_min = y_min - margin
        target_max = y_max + margin

        # Меняем масштаб оси ТОЛЬКО если сигнал вылез за границы
        # или если амплитуда сигнала уменьшилась более чем в 2 раза (чтобы график не был плоской полосой)
        curr_span = self.current_y_max - self.current_y_min
        targ_span = target_max - target_min

        if (target_min < self.current_y_min) or (target_max > self.current_y_max) or (curr_span > targ_span * 2.5):
            self.current_y_min = target_min
            self.current_y_max = target_max
            self.plot_widget.setYRange(self.current_y_min, self.current_y_max, padding=0)

    def update_stats(self):
        if self.filled < 10:
            return

        data = self.y_data[-self.filled:]
        mean_val = np.mean(data)
        std_val = np.std(data)
        rms_val = np.sqrt(np.mean(data ** 2))
        p2p = np.max(data) - np.min(data)

        self.stats_label.setText(
            f"Mean:{mean_val:6.1f} | STD:{std_val:6.1f} | RMS:{rms_val:6.1f} | P2P:{p2p:6.1f}"
        )


# ==============================================================================
# ГЛАВНОЕ ОКНО
# ==============================================================================
class HardwareValidationWindow(QMainWindow):

    def __init__(self, inlet: StreamInlet, full_info: StreamInfo, channel_names: List[str]):
        super().__init__()
        self.inlet = inlet
        self.stream_name = full_info.name() or "EEG"
        self.channel_names = channel_names
        self.n_channels = full_info.channel_count()
        self.sampling_rate = full_info.nominal_srate()

        self.cov_window_samples = int(WINDOW_SEC * self.sampling_rate)
        self.channel_widgets: List[ChannelWidget] = []

        self.start_time = time.time()
        self.last_cov_time = self.start_time
        self.sample_count = 0

        self._setup_ui()
        self._setup_timers()
        log.info("GUI инициализировано, запускаем таймеры.")

    def _setup_ui(self):
        self.setWindowTitle(f"LSL Аппаратная валидация — {self.stream_name}")
        self.setStyleSheet("background-color: #121212; color: white;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        info_panel = QHBoxLayout()
        info_panel.addWidget(QLabel(f"<b>Поток:</b> {self.stream_name}"))
        info_panel.addWidget(QLabel(f"<b>Каналов:</b> {self.n_channels}"))
        info_panel.addWidget(QLabel(f"<b>Частота:</b> {self.sampling_rate} Гц"))
        main_layout.addLayout(info_panel)

        n_cols = 3
        grid_layout = QGridLayout()
        grid_layout.setSpacing(4)

        for ch in range(self.n_channels):
            cw = ChannelWidget(ch, self.channel_names[ch], self.sampling_rate)
            self.channel_widgets.append(cw)
            grid_layout.addWidget(cw, ch // n_cols, ch % n_cols)

        main_layout.addLayout(grid_layout)
        self.resize(1500, 1000)

    def _setup_timers(self):
        self.plot_timer = QTimer()
        self.plot_timer.setTimerType(Qt.PreciseTimer)  # ЖЕСТКИЙ РЕЖИМ РЕАЛЬНОГО ВРЕМЕНИ
        self.plot_timer.timeout.connect(self._pull_and_plot)
        self.plot_timer.start(UPDATE_INTERVAL_MS)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_text_stats)
        self.stats_timer.start(STATS_INTERVAL_MS)

    def _pull_and_plot(self):
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)
        if chunk:
            arr = np.array(chunk)
            if arr.shape[1] != self.n_channels and arr.shape[0] == self.n_channels:
                arr = arr.T

            self.sample_count += len(arr)

            for ch in range(self.n_channels):
                self.channel_widgets[ch].push_chunk(arr[:, ch])

        for cw in self.channel_widgets:
            cw.update_plot()

        now = time.time()
        if now - self.last_cov_time >= COV_UPDATE_INTERVAL:
            self.last_cov_time = now
            self._calculate_covariance()

    def _update_text_stats(self):
        for cw in self.channel_widgets:
            cw.update_stats()

    def _calculate_covariance(self):
        data_matrix = np.column_stack([cw.y_data for cw in self.channel_widgets])
        data_matrix = data_matrix - np.mean(data_matrix, axis=0)
        cov_matrix = np.cov(data_matrix.T)

        log.info(f"--- МАТРИЦА КОВАРИАЦИЙ (Сэмплов: {self.sample_count}) ---")

        col_w = 11
        header = " " * col_w + "".join([f"{name[:col_w]:>{col_w}}" for name in self.channel_names])
        log.info(header)

        for i in range(self.n_channels):
            row_str = f"{self.channel_names[i][:col_w]:>{col_w}}"
            for j in range(self.n_channels):
                row_str += f"{cov_matrix[i, j]:{col_w}.2f}"
            log.info(row_str)

        np.fill_diagonal(cov_matrix, 0)
        max_idx = np.unravel_index(np.argmax(np.abs(cov_matrix)), cov_matrix.shape)
        max_val = cov_matrix[max_idx]

        ch1, ch2 = self.channel_names[max_idx[0]], self.channel_names[max_idx[1]]
        log.info(f"Максимальная взаимная наводка: {ch1} ↔ {ch2} = {max_val:.2f}\n")


def select_stream_gui(streams: List[StreamInfo]):
    items = []
    for info in streams:
        name = info.name() or "Unknown"
        stype = info.type() or "EEG"
        items.append(f"{name} (type={stype}, ch={info.channel_count()})")

    item, ok = QInputDialog.getItem(
        None, "Выбор LSL‑потока", "Выберите поток:", items, 0, False
    )
    return streams[items.index(item)] if ok else None


def main():
    log.info("=== СТАРТ АППАРАТНОЙ ВАЛИДАЦИИ ===")
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    streams = find_eeg_streams()
    if not streams:
        log.error("Потоки LSL не найдены. Выход.")
        return

    eeg_info = streams[0] if len(streams) == 1 else select_stream_gui(streams)
    if not eeg_info:
        log.warning("Отменено пользователем.")
        return

    inlet = StreamInlet(eeg_info)

    log.info("Запрашиваем полные метаданные потока (XML)...")
    full_info = inlet.info()
    n_channels = full_info.channel_count()

    channel_names = get_channel_names(full_info, n_channels)
    log.info(f"Имена каналов: {', '.join(channel_names)}")

    window = HardwareValidationWindow(inlet, full_info, channel_names)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()