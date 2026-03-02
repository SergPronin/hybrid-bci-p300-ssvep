#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Профессиональный BCI-монитор (Дашборд) для P300-проекта.

Интерфейс состоит из трех главных зон:
1. ЗОНА 1: Непрерывный сигнал (Верхняя половина окна) - бегущая волна ЭЭГ
2. ЗОНА 2: Сетка когерентного накопления P300 (Нижняя левая часть) - 9 графиков 3x3
3. ЗОНА 3: Панель управления и статуса (Нижняя правая часть) - метрики и кнопки

Запуск:
    python scripts/bci_monitor_dashboard.py

Перед запуском убедитесь, что:
    - Запущено приложение стимуляции (app.main)
    - Запущен поток ЭЭГ (например, Нейроспектр)
    - Оба потока транслируются через LSL
"""

import sys
import time
import numpy as np
from collections import deque
from typing import Dict, Optional
from datetime import datetime
import os

from scipy import signal
from pylsl import StreamInlet, resolve_byprop

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QGroupBox
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QColor
import pyqtgraph as pg

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from eeg_epoch_processor import (
    EpochProcessor, EEGBuffer, find_streams, parse_marker,
    EPOCH_PRE_STIM, EPOCH_POST_STIM, EPOCH_TOTAL,
    FILTER_LOW, FILTER_HIGH, NUM_TILES, BUFFER_DURATION,
    MARKER_STREAM_NAME, EEG_STREAM_TYPE
)

CONTINUOUS_WINDOW_SEC = 10.0
CONTINUOUS_UPDATE_MS = 33
P300_SEARCH_START = 0.25
P300_SEARCH_END = 0.55
P300_Y_MIN = -20.0
P300_Y_MAX = 20.0
SIGNAL_QUALITY_THRESHOLD = 15.0


class ContinuousSignalWidget(QWidget):
    """Виджет для отображения непрерывного ЭЭГ-сигнала (бегущая волна)."""
    
    def __init__(self, sampling_rate: float, num_channels: int):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        
        self.buffer_size = int(CONTINUOUS_WINDOW_SEC * sampling_rate)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.raw_data_buffer = deque(maxlen=self.buffer_size)
        self.filtered_data_buffer = deque(maxlen=self.buffer_size)
        
        self.show_filtered = False
        self.markers = deque(maxlen=100)
        self.filter_state = None
        self.auto_scale = True
        
        self._create_filter()
        self._setup_ui()
    
    def _create_filter(self):
        """Создаёт фильтр для непрерывного сигнала."""
        nyquist = self.sampling_rate / 2.0
        low = FILTER_LOW / nyquist
        high = FILTER_HIGH / nyquist
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        self.filter_state = signal.lfilter_zi(self.b, self.a)
    
    def _setup_ui(self):
        """Создаёт интерфейс виджета."""
        layout = QVBoxLayout()
        
        control_panel = QHBoxLayout()
        self.filter_checkbox = QCheckBox("Фильтр 0.5-10 Гц")
        self.filter_checkbox.setChecked(False)
        self.filter_checkbox.stateChanged.connect(self._on_filter_changed)
        self.filter_checkbox.setStyleSheet("color: white; font-size: 12px;")
        control_panel.addWidget(self.filter_checkbox)
        control_panel.addStretch()
        layout.addLayout(control_panel)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'Амплитуда (мкВ)', color='white')
        self.plot_widget.setLabel('bottom', 'Время (сек)', color='white')
        self.plot_widget.setBackground('black')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        self.signal_line = self.plot_widget.plot([], [], pen=pg.mkPen('cyan', width=2))
        self.marker_lines = []
        self.marker_labels = []
        
        layout.addWidget(self.plot_widget)
        self.setLayout(layout)
    
    def _on_filter_changed(self, state):
        """Обработчик изменения состояния фильтра."""
        self.show_filtered = (state == Qt.Checked)
        self._update_plot()
    
    def add_sample(self, timestamp: float, data: np.ndarray):
        """Добавляет новый сэмпл в буфер."""
        if self.num_channels > 1:
            mean_data = np.mean(data)
        else:
            mean_data = data[0] if len(data) > 0 else 0.0
        
        self.raw_data_buffer.append(mean_data)
        
        try:
            filtered_val, self.filter_state = signal.lfilter(
                self.b, self.a, [mean_data], zi=self.filter_state
            )
            self.filtered_data_buffer.append(filtered_val[0])
        except Exception:
            self.filtered_data_buffer.append(mean_data)
        
        self.time_buffer.append(timestamp)
    
    def add_marker(self, timestamp: float, tile_id: int):
        """Добавляет маркер стимула."""
        self.markers.append((timestamp, tile_id))
    
    def _update_plot(self):
        """Обновляет график."""
        if len(self.time_buffer) < 1:
            return
        
        times = np.array(self.time_buffer)
        
        if self.show_filtered:
            data = np.array(self.filtered_data_buffer)
        else:
            data = np.array(self.raw_data_buffer)
        
        if len(times) != len(data):
            min_len = min(len(times), len(data))
            times = times[:min_len]
            data = data[:min_len]
        
        if len(times) == 0:
            return
        
        now_lsl = times[-1]
        relative_times = times - now_lsl
        
        self.signal_line.setData(relative_times, data)
        self.plot_widget.setXRange(-CONTINUOUS_WINDOW_SEC, 0)
        
        if self.auto_scale and len(data) > 0:
            y_min, y_max = np.min(data), np.max(data)
            if y_max != y_min:
                margin = (y_max - y_min) * 0.2
                self.plot_widget.setYRange(y_min - margin, y_max + margin)
            else:
                self.plot_widget.setYRange(y_min - 10, y_max + 10)
        
        self._update_markers(now_lsl)
    
    def _update_markers(self, now_lsl: float):
        """Обновляет маркеры на графике."""
        for line in self.marker_lines:
            self.plot_widget.removeItem(line)
        for label in self.marker_labels:
            self.plot_widget.removeItem(label)
        self.marker_lines.clear()
        self.marker_labels.clear()
        
        for marker_time, tile_id in self.markers:
            relative_time = marker_time - now_lsl
            if -CONTINUOUS_WINDOW_SEC <= relative_time <= 0:
                line = pg.InfiniteLine(
                    pos=relative_time, angle=90,
                    pen=pg.mkPen('yellow', width=2)
                )
                self.plot_widget.addItem(line)
                self.marker_lines.append(line)
                
                y_range = self.plot_widget.viewRange()[1]
                y_top = y_range[1]
                label = pg.TextItem(
                    f"#{tile_id}", color='yellow', anchor=(0.5, 1)
                )
                label.setPos(relative_time, y_top * 0.95)
                self.plot_widget.addItem(label)
                self.marker_labels.append(label)
    
    def update_display(self):
        """Обновляет отображение (вызывается таймером)."""
        self._update_plot()


class P300GridWidget(QWidget):
    """Виджет для отображения сетки 3x3 графиков P300."""
    
    def __init__(self, sampling_rate: float, num_channels: int):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        
        self.time_vector = np.linspace(
            -EPOCH_PRE_STIM, EPOCH_POST_STIM,
            int(EPOCH_TOTAL * sampling_rate)
        )
        
        self.processor: Optional[EpochProcessor] = None
        self.show_butterfly = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Создаёт интерфейс виджета."""
        layout = QVBoxLayout()
        
        self.butterfly_checkbox = QCheckBox("Butterfly Plot (показать все эпохи)")
        self.butterfly_checkbox.setChecked(False)
        self.butterfly_checkbox.stateChanged.connect(self._on_butterfly_changed)
        layout.addWidget(self.butterfly_checkbox)
        
        grid_layout = QGridLayout()
        self.plots = {}
        self.avg_lines = {}
        self.butterfly_lines = {}
        
        for tile_id in range(NUM_TILES):
            row = tile_id // 3
            col = tile_id % 3
            
            plot_widget = pg.PlotWidget(title=f'Плитка {tile_id}')
            plot_widget.setLabel('left', 'Амплитуда (мкВ)')
            plot_widget.setLabel('bottom', 'Время (сек)')
            plot_widget.setBackground('black')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setYRange(P300_Y_MIN, P300_Y_MAX)
            plot_widget.setXRange(-EPOCH_PRE_STIM, EPOCH_POST_STIM)
            
            zero_line = pg.InfiniteLine(
                pos=0.0, angle=90,
                pen=pg.mkPen('red', width=2, style=Qt.SolidLine)
            )
            plot_widget.addItem(zero_line)
            
            p300_region = pg.LinearRegionItem(
                [P300_SEARCH_START, P300_SEARCH_END],
                brush=pg.mkBrush(QColor(0, 255, 0, 50)),
                pen=pg.mkPen(QColor(0, 255, 0, 100))
            )
            plot_widget.addItem(p300_region)
            
            avg_line = plot_widget.plot(
                [], [], pen=pg.mkPen('white', width=2.5)
            )
            
            text_item = pg.TextItem('Эпох: 0', color='white', anchor=(0, 1))
            text_item.setPos(-EPOCH_PRE_STIM + 0.05, P300_Y_MAX * 0.9)
            plot_widget.addItem(text_item)
            
            self.plots[tile_id] = plot_widget
            self.avg_lines[tile_id] = avg_line
            self.butterfly_lines[tile_id] = []
            self.plots[tile_id].text_item = text_item
            
            grid_layout.addWidget(plot_widget, row, col)
        
        layout.addLayout(grid_layout)
        self.setLayout(layout)
    
    def _on_butterfly_changed(self, state):
        """Обработчик изменения состояния butterfly plot."""
        self.show_butterfly = (state == Qt.Checked)
        self.update_display()
    
    def set_processor(self, processor: EpochProcessor):
        """Устанавливает процессор эпох."""
        self.processor = processor
    
    def update_display(self):
        """Обновляет отображение графиков."""
        if self.processor is None:
            return
        
        averaged = self.processor.get_averaged_epochs()
        counts = self.processor.get_epoch_counts()
        
        for tile_id in range(NUM_TILES):
            plot_widget = self.plots[tile_id]
            avg_line = self.avg_lines[tile_id]
            
            for line in self.butterfly_lines[tile_id]:
                plot_widget.removeItem(line)
            self.butterfly_lines[tile_id].clear()
            
            if averaged[tile_id] is not None:
                epoch_data = averaged[tile_id]
                
                if self.num_channels > 1:
                    mean_signal = np.mean(epoch_data, axis=0)
                else:
                    mean_signal = epoch_data[0, :]
                
                avg_line.setData(self.time_vector, mean_signal)
                
                if self.show_butterfly and tile_id in self.processor.epochs:
                    epochs_list = self.processor.epochs[tile_id]
                    for epoch in epochs_list:
                        if self.num_channels > 1:
                            epoch_mean = np.mean(epoch, axis=0)
                        else:
                            epoch_mean = epoch[0, :]
                        
                        butterfly_line = plot_widget.plot(
                            self.time_vector, epoch_mean,
                            pen=pg.mkPen('cyan', width=0.5, style=Qt.DashLine)
                        )
                        self.butterfly_lines[tile_id].append(butterfly_line)
            else:
                avg_line.setData([], [])
            
            count = counts[tile_id]
            plot_widget.text_item.setText(f'Эпох: {count}')


class ControlPanelWidget(QWidget):
    """Виджет для панели управления и статуса."""
    
    start_clicked = pyqtSignal()
    reset_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        """Создаёт интерфейс виджета."""
        layout = QVBoxLayout()
        
        status_group = QGroupBox("Статус")
        status_layout = QVBoxLayout()
        
        self.epochs_label = QLabel("Всего эпох: 0")
        self.epochs_label.setStyleSheet("font-size: 14px; color: white;")
        status_layout.addWidget(self.epochs_label)
        
        self.quality_label = QLabel("Качество сигнала: Проверка...")
        self.quality_label.setStyleSheet("font-size: 14px; color: white;")
        status_layout.addWidget(self.quality_label)
        
        self.result_label = QLabel("РЕЗУЛЬТАТ КЛАССИФИКАТОРА:\nФокус внимания: -")
        self.result_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: yellow;"
        )
        self.result_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.result_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        control_group = QGroupBox("Управление")
        control_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: green; color: white;"
        )
        self.start_button.clicked.connect(self.start_clicked.emit)
        control_layout.addWidget(self.start_button)
        
        self.reset_button = QPushButton("Reset Buffers")
        self.reset_button.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: orange; color: white;"
        )
        self.reset_button.clicked.connect(self.reset_clicked.emit)
        control_layout.addWidget(self.reset_button)
        
        self.stop_button = QPushButton("Stop & Save")
        self.stop_button.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: red; color: white;"
        )
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        control_layout.addWidget(self.stop_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_epochs(self, total: int):
        """Обновляет количество эпох."""
        self.epochs_label.setText(f"Всего эпох: {total}")
    
    def update_quality(self, variance: float, has_artifacts: bool):
        """Обновляет качество сигнала."""
        if has_artifacts:
            self.quality_label.setText(
                f"Качество сигнала: ВНИМАНИЕ: АРТЕФАКТЫ! (σ={variance:.1f} мкВ)"
            )
            self.quality_label.setStyleSheet("font-size: 14px; color: red; font-weight: bold;")
        else:
            self.quality_label.setText(
                f"Качество сигнала: Шум в норме (σ={variance:.1f} мкВ)"
            )
            self.quality_label.setStyleSheet("font-size: 14px; color: green;")
    
    def update_result(self, tile_id: Optional[int]):
        """Обновляет результат классификатора."""
        if tile_id is not None:
            self.result_label.setText(
                f"РЕЗУЛЬТАТ КЛАССИФИКАТОРА:\nФокус внимания: ПЛИТКА № {tile_id}"
            )
            self.result_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: yellow;"
            )
        else:
            self.result_label.setText("РЕЗУЛЬТАТ КЛАССИФИКАТОРА:\nФокус внимания: -")
            self.result_label.setStyleSheet(
                "font-size: 24px; font-weight: bold; color: gray;"
            )


class P300Classifier:
    """Простой классификатор для определения плитки с максимальным P300."""
    
    def __init__(self, sampling_rate: float):
        self.sampling_rate = sampling_rate
        self.search_start_idx = int((P300_SEARCH_START + EPOCH_PRE_STIM) * sampling_rate)
        self.search_end_idx = int((P300_SEARCH_END + EPOCH_PRE_STIM) * sampling_rate)
    
    def classify(self, averaged_epochs: Dict[int, Optional[np.ndarray]]) -> Optional[int]:
        """
        Определяет плитку с максимальным P300.
        
        Args:
            averaged_epochs: Словарь усреднённых эпох
        
        Returns:
            ID плитки с максимальным P300 или None
        """
        max_amplitude = -np.inf
        best_tile = None
        
        for tile_id, epoch_data in averaged_epochs.items():
            if epoch_data is None:
                continue
            
            if epoch_data.ndim > 1 and epoch_data.shape[0] > 1:
                mean_signal = np.mean(epoch_data, axis=0)
            else:
                mean_signal = epoch_data[0, :] if epoch_data.ndim > 1 else epoch_data
            
            if len(mean_signal) > self.search_end_idx:
                p300_window = mean_signal[self.search_start_idx:self.search_end_idx]
                max_amp = np.max(p300_window)
                
                if max_amp > 20.0:
                    continue
                
                if max_amp > max_amplitude:
                    max_amplitude = max_amp
                    best_tile = tile_id
        
        if best_tile is not None and max_amplitude > 2.0:
            return best_tile
        
        return None


class BCIMonitorDashboard(QMainWindow):
    """Главное окно BCI-монитора с тремя зонами."""
    
    def __init__(self):
        super().__init__()
        
        self.buffer: Optional[EEGBuffer] = None
        self.processor: Optional[EpochProcessor] = None
        self.classifier: Optional[P300Classifier] = None
        self.marker_inlet: Optional[StreamInlet] = None
        self.eeg_inlet: Optional[StreamInlet] = None
        
        self.is_processing = False
        self.eeg_first_time: Optional[float] = None
        self.processed_markers = 0
        self.skipped_markers = 0
        self.raw_signal_buffer = deque(maxlen=1000)
        
        self._setup_ui()
        
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.read_lsl_data)
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        
        self.quality_timer = QTimer()
        self.quality_timer.timeout.connect(self.update_quality)
    
    def _setup_ui(self):
        """Создаёт интерфейс главного окна."""
        self.setWindowTitle("BCI Monitor Dashboard - P300")
        self.setGeometry(100, 100, 1600, 1000)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()
        
        self.control_panel = ControlPanelWidget()
        self.control_panel.start_clicked.connect(self.start_processing)
        self.control_panel.reset_clicked.connect(self.reset_buffers)
        self.control_panel.stop_clicked.connect(self.stop_processing)
        
        bottom_layout.addWidget(self.control_panel, 3)
        main_layout.addLayout(bottom_layout, 1)
        
        central_widget.setLayout(main_layout)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
    
    def initialize_streams(self):
        """Инициализирует подключение к LSL потокам."""
        print("=" * 70)
        print("ИНИЦИАЛИЗАЦИЯ BCI MONITOR DASHBOARD")
        print("=" * 70)
        
        marker_info, eeg_info = find_streams()
        if marker_info is None or eeg_info is None:
            print("\nНе удалось найти необходимые потоки.")
            return False
        
        print("\nПодключение к потокам...")
        self.marker_inlet = StreamInlet(marker_info)
        self.eeg_inlet = StreamInlet(eeg_info)
        
        eeg_sampling_rate = eeg_info.nominal_srate()
        eeg_num_channels = eeg_info.channel_count()
        
        if eeg_sampling_rate == 0:
            print("ОШИБКА: Частота дискретизации ЭЭГ не определена!")
            return False
        
        print(f"  ✓ Подключено к потоку маркеров")
        print(f"  ✓ Подключено к потоку ЭЭГ ({eeg_num_channels} каналов, {eeg_sampling_rate} Гц)")
        
        print("\nИнициализация компонентов обработки...")
        self.buffer = EEGBuffer(eeg_sampling_rate, eeg_num_channels)
        self.processor = EpochProcessor(eeg_sampling_rate, eeg_num_channels)
        self.classifier = P300Classifier(eeg_sampling_rate)
        
        central_widget = self.centralWidget()
        main_layout = central_widget.layout()
        
        self.continuous_widget = ContinuousSignalWidget(eeg_sampling_rate, eeg_num_channels)
        main_layout.insertWidget(0, self.continuous_widget, 1)
        
        self.p300_widget = P300GridWidget(eeg_sampling_rate, eeg_num_channels)
        self.p300_widget.set_processor(self.processor)
        
        bottom_layout = main_layout.itemAt(1).layout()
        bottom_layout.insertWidget(0, self.p300_widget, 7)
        
        print("  ✓ Все компоненты инициализированы")
        print("\n" + "=" * 70)
        print("ГОТОВ К РАБОТЕ")
        print("Нажмите 'Start Processing' для начала мониторинга")
        print("=" * 70 + "\n")
        
        return True
    
    def start_processing(self):
        """Запускает обработку данных."""
        if self.is_processing:
            return
        
        if self.buffer is None or self.processor is None:
            print("ОШИБКА: Компоненты не инициализированы!")
            return
        
        print("\n" + "=" * 70)
        print("НАЧАЛО ОБРАБОТКИ")
        print("=" * 70 + "\n")
        
        print("Накопление данных в буфере...")
        min_buffer_duration = EPOCH_POST_STIM + 0.5
        buffer_ready = False
        start_time = time.time()
        samples_count = 0
        
        while not buffer_ready:
            QApplication.processEvents()
            chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.1, max_samples=200)
            if chunk and timestamps:
                chunk_array = np.array(chunk)
                if chunk_array.shape[1] == self.buffer.num_channels:
                    pass
                elif chunk_array.shape[0] == self.buffer.num_channels:
                    chunk_array = chunk_array.T
                else:
                    continue
                
                for i, ts in enumerate(timestamps):
                    self.buffer.add_sample(ts, chunk_array[i, :])
                    self.continuous_widget.add_sample(ts, chunk_array[i, :])
                    if self.buffer.num_channels > 1:
                        mean_sample = np.mean(chunk_array[i, :])
                    else:
                        mean_sample = chunk_array[i, 0]
                    self.raw_signal_buffer.append(mean_sample)
                    samples_count += 1
            
            buffer_info = self.buffer.get_buffer_info()
            if buffer_info['duration'] >= min_buffer_duration:
                buffer_ready = True
                self.eeg_first_time = self.buffer.get_first_sample_time()
                print(f"  ✓ Буфер готов: {buffer_info['duration']:.2f} сек данных, {samples_count} сэмплов")
            elif time.time() - start_time > 10:
                print(f"  ⚠ Буфер частично готов: {buffer_info['duration']:.2f} сек данных, {samples_count} сэмплов")
                buffer_ready = True
                self.eeg_first_time = self.buffer.get_first_sample_time()
        
        self.is_processing = True
        self.control_panel.start_button.setEnabled(False)
        self.control_panel.start_button.setText("Обработка...")
        
        self.data_timer.start(10)
        self.update_timer.start(CONTINUOUS_UPDATE_MS)
        self.quality_timer.start(500)
        
        print("Обработка запущена! Графики должны обновляться в реальном времени.")
    
    def reset_buffers(self):
        """Сбрасывает все накопленные эпохи."""
        if self.processor is None:
            return
        
        print("\nСброс буферов...")
        self.processor.epochs.clear()
        self.update_displays()
        print("  ✓ Буферы сброшены")
    
    def stop_processing(self):
        """Останавливает обработку и сохраняет результаты."""
        if not self.is_processing:
            return
        
        print("\n" + "=" * 70)
        print("ОСТАНОВКА ОБРАБОТКИ")
        print("=" * 70)
        
        self.is_processing = False
        
        self.data_timer.stop()
        self.update_timer.stop()
        self.quality_timer.stop()
        
        self.control_panel.start_button.setEnabled(True)
        self.control_panel.start_button.setText("Start Processing")
        
        if self.processor is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            filename = os.path.join(results_dir, f"eeg_epochs_{timestamp}.pkl")
            
            self.processor.save_results(filename)
            
            print(f"\nРезультаты сохранены в: {filename}")
        
        print("\nОбработка остановлена.")
    
    def read_lsl_data(self):
        """Читает данные из LSL потоков."""
        if not self.is_processing:
            return
        
        if self.marker_inlet is None or self.eeg_inlet is None:
            return
        
        eeg_first_time = self.buffer.get_first_sample_time()
        if eeg_first_time is None:
            return
        
        if self.eeg_first_time is None:
            self.eeg_first_time = eeg_first_time
        
        while True:
            marker_sample, marker_timestamp = self.marker_inlet.pull_sample(timeout=0.0)
            if marker_sample is None:
                break
            
            raw_marker = marker_sample[0] if isinstance(marker_sample, (list, tuple)) else marker_sample
            
            parsed = parse_marker(raw_marker)
            if parsed is None:
                continue
            
            tile_id, event = parsed
            if event != "on":
                continue
            
            if not (0 <= tile_id < NUM_TILES):
                continue
            
            if not hasattr(self, '_marker_time_correction'):
                if marker_timestamp < eeg_first_time:
                    self._marker_time_correction = eeg_first_time - marker_timestamp
                else:
                    self._marker_time_correction = 0.0
            
            corrected_marker_time = marker_timestamp + self._marker_time_correction
            marker_relative_time = corrected_marker_time - eeg_first_time
            
            if marker_relative_time < 0:
                self.skipped_markers += 1
                continue
            
            self.continuous_widget.add_marker(corrected_marker_time, tile_id)
            
            is_ready, _ = self.buffer.is_ready(
                corrected_marker_time, EPOCH_PRE_STIM, EPOCH_POST_STIM, eeg_first_time
            )
            
            if not is_ready:
                self.skipped_markers += 1
                continue
            
            epoch_result = self.buffer.extract_epoch(
                corrected_marker_time, EPOCH_PRE_STIM, EPOCH_POST_STIM, eeg_first_time
            )
            
            if epoch_result is None:
                self.skipped_markers += 1
                continue
            
            epoch_data, time_vector = epoch_result
            self.processor.add_epoch(tile_id, epoch_data)
            self.processed_markers += 1
        
        chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=100)
        if chunk and timestamps:
            chunk_array = np.array(chunk)
            if chunk_array.shape[1] == self.buffer.num_channels:
                pass
            elif chunk_array.shape[0] == self.buffer.num_channels:
                chunk_array = chunk_array.T
            else:
                return
            
            for i, ts in enumerate(timestamps):
                self.buffer.add_sample(ts, chunk_array[i, :])
                self.continuous_widget.add_sample(ts, chunk_array[i, :])
                
                if self.buffer.num_channels > 1:
                    mean_sample = np.mean(chunk_array[i, :])
                else:
                    mean_sample = chunk_array[i, 0]
                self.raw_signal_buffer.append(mean_sample)
    
    def update_displays(self):
        """Обновляет все отображения."""
        if not self.is_processing:
            return
        
        self.continuous_widget.update_display()
        self.p300_widget.update_display()
        
        if self.processor is not None:
            counts = self.processor.get_epoch_counts()
            total_epochs = sum(counts.values())
            self.control_panel.update_epochs(total_epochs)
            
            averaged = self.processor.get_averaged_epochs()
            predicted_tile = self.classifier.classify(averaged)
            self.control_panel.update_result(predicted_tile)
    
    def update_quality(self):
        """Обновляет качество сигнала."""
        if len(self.raw_signal_buffer) < 100:
            return
        
        signal_array = np.array(self.raw_signal_buffer)
        variance = np.std(signal_array)
        has_artifacts = variance > SIGNAL_QUALITY_THRESHOLD
        
        self.control_panel.update_quality(variance, has_artifacts)
    
    def closeEvent(self, event):
        """Обработчик закрытия окна."""
        self.stop_processing()
        event.accept()


def main():
    """Главная функция: запускает BCI Monitor Dashboard."""
    app = QApplication(sys.argv)
    
    dashboard = BCIMonitorDashboard()
    
    if not dashboard.initialize_streams():
        print("\nНе удалось инициализировать потоки. Выход.")
        return 1
    
    dashboard.show()
    exit_code = app.exec_()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
