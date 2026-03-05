#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аппаратная валидация ЭЭГ: EXTREME PERFORMANCE + DYNAMIC UI + MANUAL SCALING.
Оптимизации:
- Интерактивный сайдбар для включения/выключения каналов.
- Глобальный тумблер "Автомасштаб Y" для возможности ручного зума (колесико мыши).
- Жесткая защита от мусорных данных (NaN, Inf), предотвращающая краш C++ ядра Qt.
- Оптимизированный рендеринг (120+ FPS) скрытых и активных каналов.
"""

import sys
import time
import logging
import numpy as np
from typing import Optional, List

from pylsl import StreamInlet, StreamInfo, resolve_byprop

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QGroupBox, QInputDialog, QCheckBox, 
    QPushButton, QScrollArea, QFrame
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
EEG_STREAM_TYPES = ("EEG", "Signal")
WINDOW_SEC = 0.2       
COV_UPDATE_INTERVAL = 10.0 
UPDATE_INTERVAL_MS = 100   # ~125 FPS
STATS_INTERVAL_MS = 500   

SIMULATOR_NAME = "EEG_Simulator"
SIMULATOR_SOURCE_ID = "eeg-simulator-neurospectr"
NEUROSPECTR_MARKER = "neuro"

pg.setConfigOptions(useOpenGL=False, antialias=False, useCupy=False)

# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================
def setup_logging():
    logger = logging.getLogger("EEG_Validation")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
    
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
            name = ch.child_value("label") or ch.child_value("name") or ch.child_value("type")
            channels.append(name.strip() if name else f"Ch {len(channels)+1}")
            ch = ch.next_sibling()
    except Exception:
        pass
    while len(channels) < n_channels:
        channels.append(f"Ch {len(channels)+1}")
    return channels[:n_channels]

# ==============================================================================
# GUI КОМПОНЕНТЫ
# ==============================================================================
class ChannelWidget(QFrame):
    
    def __init__(self, channel_id: int, channel_name: str, sampling_rate: float):
        super().__init__()
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.sampling_rate = sampling_rate
        self.buffer_size = int(WINDOW_SEC * sampling_rate)
        
        self.y_data = np.zeros(self.buffer_size, dtype=np.float32)
        self.x_data = np.linspace(-WINDOW_SEC, 0, self.buffer_size, dtype=np.float32)
        self.filled = 0
        
        self.current_y_min = -10.0
        self.current_y_max = 10.0
        
        self.is_active = True 
        
        self._setup_ui()
        
    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; }")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        
        header = QLabel(f"[{self.channel_id + 1}] {self.channel_name}")
        header.setFont(QFont("Arial", 8, QFont.Bold))
        header.setStyleSheet("color: white; background-color: transparent; border: none; padding: 2px;")
        layout.addWidget(header)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setDownsampling(mode=None)
        self.plot_widget.setClipToView(True)
        
        # Разрешаем зум мышью только по оси Y (чтобы шкала времени не ломалась)
        self.plot_widget.setMouseEnabled(x=False, y=True) 
        self.plot_widget.setMenuEnabled(True) # Включаем меню по правому клику
        
        self.plot_widget.getViewBox().setLimits(yMin=-1000000, yMax=1000000)
        
        self.plot_widget.setLabel('left', 'мкВ', color='white', fontsize=6)
        self.plot_widget.setBackground('#000000')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setXRange(-WINDOW_SEC, 0, padding=0)
        
        colors = pg.intColor(self.channel_id, hues=15, values=1, maxValue=255, minValue=150)
        self.signal_line = self.plot_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(colors, width=1.0), autoDownsample=False)
        
        layout.addWidget(self.plot_widget)
        
        self.stats_label = QLabel("Ожидание...")
        self.stats_label.setFont(QFont("Courier", 7))
        self.stats_label.setStyleSheet("color: #a8ff9e; background-color: transparent; border: none;")
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def push_chunk(self, new_data: np.ndarray):
        if not self.is_active:
            return

        n = len(new_data)
        if n == 0: return
            
        try:
            # Защита от переполнения и NaN
            clean_data = np.nan_to_num(new_data, nan=0.0, posinf=0.0, neginf=0.0)
            clean_data = np.clip(clean_data, -250000.0, 250000.0) 
        except Exception:
            clean_data = np.zeros(n, dtype=np.float32)
            
        self.filled = min(self.buffer_size, self.filled + n)
        
        if n >= self.buffer_size:
            self.y_data[:] = clean_data[-self.buffer_size:]
        else:
            self.y_data[:-n] = self.y_data[n:]
            self.y_data[-n:] = clean_data
            
    def update_plot(self, auto_scale: bool):
        if not self.is_active or self.filled == 0 or not self.isVisible():
            return
            
        try:
            self.signal_line.setData(self.x_data, self.y_data, skipFiniteCheck=True)
            
            # Применяем умное масштабирование, ТОЛЬКО если включена галочка
            if auto_scale:
                y_min, y_max = float(np.min(self.y_data)), float(np.max(self.y_data))
                
                span = y_max - y_min
                if span < 20.0:
                    center = (y_max + y_min) / 2.0
                    target_min, target_max = center - 10.0, center + 10.0
                else:
                    margin = span * 0.15
                    target_min, target_max = y_min - margin, y_max + margin

                if abs(target_min - self.current_y_min) > 2.0 or abs(target_max - self.current_y_max) > 2.0:
                    self.current_y_min, self.current_y_max = target_min, target_max
                    self.plot_widget.setYRange(self.current_y_min, self.current_y_max, padding=0, update=False)
        except Exception:
            pass

    def update_stats(self):
        if not self.is_active or self.filled < 10 or not self.isVisible():
            return
            
        try:
            data = self.y_data[-self.filled:]
            mean_val, std_val = float(np.mean(data)), float(np.std(data))
            rms_val = float(np.sqrt(np.mean(data**2)))
            p2p = float(np.max(data) - np.min(data))
            
            if p2p > 200000:
                self.stats_label.setStyleSheet("color: #ff4d4d; font-weight: bold; border: none;")
                self.stats_label.setText("⚠ ОШИБКА КОНТАКТА ⚠")
            else:
                self.stats_label.setStyleSheet("color: #a8ff9e; border: none;")
                self.stats_label.setText(f"Mean:{mean_val:6.1f} | STD:{std_val:6.1f} | RMS:{rms_val:6.1f} | P2P:{p2p:6.1f}")
        except Exception:
            pass

# ==============================================================================
# ГЛАВНОЕ ОКНО С САЙДБАРОМ
# ==============================================================================
class HardwareValidationWindow(QMainWindow):
    
    def __init__(self, inlet: StreamInlet, full_info: StreamInfo, channel_names: List[str]):
        super().__init__()
        self.inlet = inlet
        self.stream_name = full_info.name() or "EEG"
        self.channel_names = channel_names
        self.n_channels = full_info.channel_count()
        self.sampling_rate = full_info.nominal_srate()
        
        self.channel_widgets: List[ChannelWidget] = []
        self.checkboxes: List[QCheckBox] = []
        
        self.start_time = time.time()
        self.last_cov_time = self.start_time
        self.sample_count = 0
        
        self._setup_ui()
        self._setup_timers()
        log.info("GUI готово. Режим DYNAMIC GRID активирован.")
    
    def _setup_ui(self):
        self.setWindowTitle(f"LSL Validation — {self.stream_name}")
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a0a; color: white; }
            QLabel { color: white; }
            QCheckBox { color: white; spacing: 5px; font-size: 13px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QPushButton { background-color: #333; color: white; border-radius: 3px; padding: 5px; }
            QPushButton:hover { background-color: #444; }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget) 
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # 1. ЛЕВАЯ ПАНЕЛЬ (САЙДБАР)
        sidebar = QWidget()
        sidebar.setFixedWidth(180)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        
        # Кнопка автомасштаба
        self.cb_autoscale = QCheckBox("Автомасштаб Y")
        self.cb_autoscale.setChecked(True)
        self.cb_autoscale.setStyleSheet("QCheckBox { color: #5bc0be; font-weight: bold; margin-bottom: 10px; }")
        sidebar_layout.addWidget(self.cb_autoscale)
        
        lbl = QLabel("ОТОБРАЖЕНИЕ:")
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(lbl)
        
        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Все")
        btn_all.clicked.connect(lambda: self._set_all_channels(True))
        btn_none = QPushButton("Сброс")
        btn_none.clicked.connect(lambda: self._set_all_channels(False))
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        sidebar_layout.addLayout(btn_layout)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        cb_container = QWidget()
        cb_container.setStyleSheet("background-color: transparent;")
        self.cb_layout = QVBoxLayout(cb_container)
        self.cb_layout.setSpacing(6)
        
        for i, name in enumerate(self.channel_names):
            cb = QCheckBox(f"[{i+1}] {name}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._rebuild_grid)
            self.checkboxes.append(cb)
            self.cb_layout.addWidget(cb)
            
        self.cb_layout.addStretch()
        scroll.setWidget(cb_container)
        sidebar_layout.addWidget(scroll)
        
        # 2. ПРАВАЯ ПАНЕЛЬ (ГРАФИКИ)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        info_panel = QHBoxLayout()
        info_panel.addWidget(QLabel(f"<b>Поток:</b> {self.stream_name}"))
        info_panel.addWidget(QLabel(f"<b>Частота:</b> {self.sampling_rate} Гц"))
        right_layout.addLayout(info_panel)
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(0,0,0,0)
        
        for ch in range(self.n_channels):
            cw = ChannelWidget(ch, self.channel_names[ch], self.sampling_rate)
            self.channel_widgets.append(cw)
            
        right_layout.addWidget(self.grid_widget, stretch=1)
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(right_panel, stretch=1)
        
        self._rebuild_grid()
        self.resize(1600, 1000) 
        
    def _set_all_channels(self, state: bool):
        for cb in self.checkboxes:
            cb.blockSignals(True) 
            cb.setChecked(state)
            cb.blockSignals(False)
        self._rebuild_grid() 

    def _rebuild_grid(self):
        for cw in self.channel_widgets:
            self.grid_layout.removeWidget(cw)
            cw.setVisible(False)
            cw.is_active = False
            
        active_indices = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
        n_active = len(active_indices)
        
        if n_active == 0:
            return
            
        n_cols = 1 if n_active == 1 else (2 if n_active <= 4 else 3)
        
        for grid_idx, ch_idx in enumerate(active_indices):
            cw = self.channel_widgets[ch_idx]
            cw.is_active = True
            cw.setVisible(True)
            row, col = grid_idx // n_cols, grid_idx % n_cols
            self.grid_layout.addWidget(cw, row, col)

    def _setup_timers(self):
        self.plot_timer = QTimer()
        self.plot_timer.setTimerType(Qt.PreciseTimer)
        self.plot_timer.timeout.connect(self._pull_and_plot)
        self.plot_timer.start(UPDATE_INTERVAL_MS)
        
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_text_stats)
        self.stats_timer.start(STATS_INTERVAL_MS)
    
    def _pull_and_plot(self):
        try:
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=1024)
            if chunk:
                arr = np.array(chunk) 
                if arr.shape[1] != self.n_channels and arr.shape[0] == self.n_channels:
                    arr = arr.T
                    
                self.sample_count += len(arr)
                
                for ch in range(self.n_channels):
                    self.channel_widgets[ch].push_chunk(arr[:, ch])
                    
            auto_scale_enabled = self.cb_autoscale.isChecked()
            for cw in self.channel_widgets:
                cw.update_plot(auto_scale_enabled)
                
            now = time.time()
            if now - self.last_cov_time >= COV_UPDATE_INTERVAL:
                self.last_cov_time = now
                self._calculate_covariance()
        except Exception:
            pass

    def _update_text_stats(self):
        for cw in self.channel_widgets:
            cw.update_stats()

    def _calculate_covariance(self):
        try:
            active_cws = [cw for cw in self.channel_widgets if cw.is_active]
            
            if len(active_cws) < 2:
                return 
                
            data_matrix = np.column_stack([cw.y_data for cw in active_cws])
            data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            data_matrix = data_matrix - np.mean(data_matrix, axis=0)
            cov_matrix = np.cov(data_matrix.T)
            
            names = [cw.channel_name for cw in active_cws]
            n_active = len(names)
            
            log.info(f"--- МАТРИЦА КОВАРИАЦИЙ ({n_active} активных, сэмплов: {self.sample_count}) ---")
            col_w = 10
            header = " " * col_w + "".join([f"{name[:col_w]:>{col_w}}" for name in names])
            log.info(header)
            
            for i in range(n_active):
                row_str = f"{names[i][:col_w]:>{col_w}}"
                for j in range(n_active):
                    val = cov_matrix[i, j]
                    if np.isnan(val) or np.isinf(val): val = 0.0
                    row_str += f"{val:{col_w}.2f}"
                log.info(row_str)
                
            np.fill_diagonal(cov_matrix, 0)
            max_idx = np.unravel_index(np.argmax(np.abs(cov_matrix)), cov_matrix.shape)
            max_val = cov_matrix[max_idx]
            
            ch1, ch2 = names[max_idx[0]], names[max_idx[1]]
            log.info(f"Макс. наводка: {ch1} ↔ {ch2} = {max_val:.2f}\n")
        except Exception:
            pass


def select_stream_gui(streams: List[StreamInfo]):
    items = []
    for info in streams:
        name = info.name() or "Unknown"
        stype = info.type() or "EEG"
        items.append(f"{name} (type={stype}, ch={info.channel_count()})")

    item, ok = QInputDialog.getItem(None, "Выбор LSL‑потока", "Выберите поток:", items, 0, False)
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
        return

    inlet = StreamInlet(eeg_info)
    full_info = inlet.info()
    n_channels = full_info.channel_count()
    channel_names = get_channel_names(full_info, n_channels)

    window = HardwareValidationWindow(inlet, full_info, channel_names)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()