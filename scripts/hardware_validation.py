#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аппаратная валидация ЭЭГ: высокопроизводительная визуализация каналов и матрица ковариаций.

Оптимизации в этой версии:
- NumPy кольцевые буферы (отказ от deque), нет аллокации памяти в главном цикле.
- Векторизованная загрузка чанков из LSL (убран двойной цикл).
- Разделение частоты обновления графиков (60 FPS) и статистики текста (4 FPS).
- Извлечение имен каналов из метаданных XML LSL потока.
- Встроенная система подробного логирования.
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
WINDOW_SEC = 3.0          # скользящее окно (сек)
COV_UPDATE_INTERVAL = 2.0 # интервал вывода матрицы ковариаций
UPDATE_INTERVAL_MS = 16   # ~60 FPS для графиков
STATS_INTERVAL_MS = 250   # 4 FPS для обновления текстовой статистики (СКО, среднее)

SIMULATOR_NAME = "EEG_Simulator"
SIMULATOR_SOURCE_ID = "eeg-simulator-neurospectr"
NEUROSPECTR_MARKER = "neuro"

# Настройка pyqtgraph для макс. производительности
pg.setConfigOptions(useOpenGL=True, antialias=True)

# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================
def setup_logging():
    logger = logging.getLogger("EEG_Validation")
    logger.setLevel(logging.DEBUG)
    
    # Форматтер: время - уровень - сообщение
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', 
                                  datefmt='%H:%M:%S')
    
    # Вывод в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Вывод в файл
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
    """Парсинг XML-описания LSL для получения имен каналов."""
    channels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(n_channels):
            name = ch.child_value("label")
            if not name:
                name = ch.child_value("name")
            channels.append(name if name else f"Ch {len(channels)+1}")
            ch = ch.next_sibling()
    except Exception as e:
        log.warning(f"Ошибка при извлечении имен каналов: {e}")
        
    # Если распарсили меньше чем нужно, дополняем дефолтными
    while len(channels) < n_channels:
        channels.append(f"Ch {len(channels)+1}")
        
    return channels[:n_channels]

# ==============================================================================
# GUI КОМПОНЕНТЫ
# ==============================================================================
class ChannelWidget(QWidget):
    """Оптимизированный виджет для отображения канала."""
    
    def __init__(self, channel_id: int, channel_name: str, sampling_rate: float):
        super().__init__()
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.sampling_rate = sampling_rate
        self.buffer_size = int(WINDOW_SEC * sampling_rate)
        
        # NumPy Ring Buffers (предвыделенная память)
        self.y_data = np.zeros(self.buffer_size, dtype=np.float32)
        # Шкала времени X предрассчитывается ОДИН раз
        self.x_data = np.linspace(-WINDOW_SEC, 0, self.buffer_size, dtype=np.float32)
        
        self.filled = 0 # Сколько реальных сэмплов получено
        
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.setSpacing(2)
        
        # Заголовок с названием канала из LSL
        header = QLabel(f"[{self.channel_id + 1}] {self.channel_name}")
        header.setFont(QFont("Arial", 9, QFont.Bold))
        header.setStyleSheet("color: white; background-color: #3d3d3d; padding: 2px 5px; border-radius: 3px;")
        layout.addWidget(header)
        
        # График
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setDownsampling(mode=None)
        self.plot_widget.setClipToView(True) # Ускоряет рендеринг
        self.plot_widget.setAutoVisible(y=True)
        self.plot_widget.setLabel('left', 'мкВ', color='white', fontsize=7)
        self.plot_widget.setBackground('black')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setXRange(-WINDOW_SEC, 0)
        
        # Линия
        colors = pg.intColor(self.channel_id, hues=15, values=1, maxValue=255, minValue=150)
        self.signal_line = self.plot_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(colors, width=1.2))
        
        layout.addWidget(self.plot_widget)
        
        # Статистика
        self.stats_label = QLabel("Ожидание данных...")
        self.stats_label.setFont(QFont("Courier", 8))
        self.stats_label.setStyleSheet("color: #a8ff9e; background-color: #1e1e1e;")
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def push_chunk(self, new_data: np.ndarray):
        """Векторизованное добавление данных сдвигом массива."""
        n = len(new_data)
        if n == 0:
            return
            
        self.filled = min(self.buffer_size, self.filled + n)
        
        if n >= self.buffer_size:
            self.y_data[:] = new_data[-self.buffer_size:]
        else:
            # Сдвигаем старые данные влево и записываем новые в конец
            self.y_data[:-n] = self.y_data[n:]
            self.y_data[-n:] = new_data
            
    def update_plot(self):
        """Перерисовка графиков (быстрая)."""
        if self.filled == 0:
            return
            
        self.signal_line.setData(self.x_data, self.y_data)
        
        # Умное автомасштабирование (раз в N кадров или с запасом)
        y_min, y_max = np.min(self.y_data), np.max(self.y_data)
        if y_max != y_min:
            margin = (y_max - y_min) * 0.15
            self.plot_widget.setYRange(y_min - margin, y_max + margin, padding=0)
        else:
            self.plot_widget.setYRange(y_min - 10, y_max + 10, padding=0)

    def update_stats(self):
        """Тяжелые расчеты (вызывать реже)."""
        if self.filled < 10:
            return
            
        data = self.y_data[-self.filled:]
        mean_val = np.mean(data)
        std_val = np.std(data)
        rms_val = np.sqrt(np.mean(data**2))
        p2p = np.max(data) - np.min(data)
        
        self.stats_label.setText(
            f"Mean:{mean_val:6.1f} | STD:{std_val:6.1f} | RMS:{rms_val:6.1f} | P2P:{p2p:6.1f}"
        )

# ==============================================================================
# ГЛАВНОЕ ОКНО
# ==============================================================================
class HardwareValidationWindow(QMainWindow):
    
    def __init__(self, inlet: StreamInlet, stream_name: str, channel_names: List[str]):
        super().__init__()
        self.inlet = inlet
        self.stream_name = stream_name
        self.channel_names = channel_names
        
        info = inlet.info()
        self.n_channels = info.channel_count()
        self.sampling_rate = info.nominal_srate()
        
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
        
        # Инфо панель
        info_panel = QHBoxLayout()
        info_panel.addWidget(QLabel(f"<b>Поток:</b> {self.stream_name}"))
        info_panel.addWidget(QLabel(f"<b>Каналов:</b> {self.n_channels}"))
        info_panel.addWidget(QLabel(f"<b>Частота:</b> {self.sampling_rate} Гц"))
        main_layout.addLayout(info_panel)
        
        # Сетка каналов (авторасчет строк)
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
        # Быстрый таймер для отрисовки графиков и приема данных
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self._pull_and_plot)
        self.plot_timer.start(UPDATE_INTERVAL_MS)
        
        # Медленный таймер для текста статистики
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_text_stats)
        self.stats_timer.start(STATS_INTERVAL_MS)
    
    def _pull_and_plot(self):
        # Пулл чанка (сразу массив)
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)
        if chunk:
            arr = np.array(chunk) # Формат LSL: [samples, channels]
            
            # Защита от транспонированных данных
            if arr.shape[1] != self.n_channels and arr.shape[0] == self.n_channels:
                arr = arr.T
                
            self.sample_count += len(arr)
            
            # Векторизованно раздаем данные по каналам
            for ch in range(self.n_channels):
                self.channel_widgets[ch].push_chunk(arr[:, ch])
                
        # Перерисовка
        for cw in self.channel_widgets:
            cw.update_plot()
            
        # Логирование ковариации
        now = time.time()
        if now - self.last_cov_time >= COV_UPDATE_INTERVAL:
            self.last_cov_time = now
            self._calculate_covariance()

    def _update_text_stats(self):
        for cw in self.channel_widgets:
            cw.update_stats()

    def _calculate_covariance(self):
        # Сбор матрицы данных из виджетов
        # Размер матрицы: (samples, channels)
        data_matrix = np.column_stack([cw.y_data for cw in self.channel_widgets])
        
        # Центрирование
        data_matrix = data_matrix - np.mean(data_matrix, axis=0)
        # Матрица ковариаций (numpy ожидает [channels, samples], поэтому .T)
        cov_matrix = np.cov(data_matrix.T)
        
        # Логирование
        log.info(f"--- МАТРИЦА КОВАРИАЦИЙ (Сэмплов: {self.sample_count}) ---")
        
        # Форматирование шапки с именами каналов
        header = " " * 8 + "".join([f"{name[:6]:>8}" for name in self.channel_names])
        log.info(header)
        
        for i in range(self.n_channels):
            row_str = f"{self.channel_names[i][:6]:>8}"
            for j in range(self.n_channels):
                row_str += f"{cov_matrix[i, j]:8.2f}"
            log.info(row_str)
            
        # Анализ максимальной наводки (вне диагонали)
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
    stream_name = eeg_info.name() or "EEG"
    n_channels = eeg_info.channel_count()
    srate = eeg_info.nominal_srate()
    
    log.info(f"Подключение к: {stream_name} (Каналов: {n_channels}, Частота: {srate} Гц)")
    
    # Получение названий каналов
    channel_names = get_channel_names(eeg_info, n_channels)
    log.info(f"Имена каналов: {', '.join(channel_names)}")

    window = HardwareValidationWindow(inlet, stream_name, channel_names)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()