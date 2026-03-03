#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аппаратная валидация ЭЭГ: изолированная визуализация каналов и матрица ковариаций.

Одноразовая утилита для лабораторной проверки независимости каналов ЭЭГ-усилителя
с помощью генератора стандартных сигналов.

Функциональность:
1. Автоматически находит LSL-поток с типом EEG или Signal и подключается к нему.
2. Динамически создаёт столько графиков (subplots), сколько каналов в потоке.
   На график 1 — строго канал 1, на график 2 — канал 2 и т.д. Данные НЕ усредняются.
3. Скользящее окно 3 секунды, обновление в реальном времени.
4. Автомасштабирование оси Y для каждого графика отдельно.
5. Каждые 2 секунды: буфер за последние 3 сек → матрица ковариаций (numpy.cov) → вывод в консоль.

Ожидаемое поведение при подаче генератора на 1 канал:
  x11 — большое (мощность на канале 1), x12, x21, x22 — стремятся к нулю (нет наводки).

Запуск:
    python scripts/hardware_validation.py

Перед запуском: включите Нейроспектр (или другой источник ЭЭГ) с LSL-трансляцией.
"""

import sys
import time
import numpy as np
from collections import deque
from typing import Optional, List, Tuple

from pylsl import StreamInlet, resolve_byprop, local_clock

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QScrollArea, QGroupBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# Параметры по ТЗ
EEG_STREAM_TYPES = ("EEG", "Signal")
WINDOW_SEC = 3.0          # скользящее окно на экране и для ковариации (сек)
COV_UPDATE_INTERVAL = 2.0  # вывод матрицы ковариаций каждые 2 секунды
UPDATE_INTERVAL_MS = 33    # обновление графиков каждые 33 мс (~30 FPS)


def find_eeg_stream():
    """Находит поток ЭЭГ в LSL (тип EEG или Signal)."""
    for stream_type in EEG_STREAM_TYPES:
        streams = resolve_byprop("type", stream_type, timeout=3)
        if streams:
            return streams[0]
    return None


class ChannelWidget(QWidget):
    """Виджет для отображения одного канала ЭЭГ."""
    
    def __init__(self, channel_id: int, sampling_rate: float):
        super().__init__()
        self.channel_id = channel_id
        self.sampling_rate = sampling_rate
        self.buffer_size = int(WINDOW_SEC * sampling_rate)
        
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.first_sample_time: Optional[float] = None
        self.last_sample_time: Optional[float] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Создаёт интерфейс виджета."""
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Заголовок канала
        header = QLabel(f"Канал {self.channel_id + 1}")
        header.setFont(QFont("Arial", 10, QFont.Bold))
        header.setStyleSheet("color: white; background-color: #2b2b2b; padding: 5px; border-radius: 3px;")
        layout.addWidget(header)
        
        # График
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', 'мкВ', color='white', fontsize=9)
        self.plot_widget.setLabel('bottom', 'Время (сек)', color='white', fontsize=9)
        self.plot_widget.setBackground('black')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        self.plot_widget.setMinimumHeight(150)
        
        # Линия сигнала
        colors = pg.intColor(self.channel_id, hues=10, values=1, maxValue=255, minValue=150, maxHue=360)
        self.signal_line = self.plot_widget.plot([], [], pen=pg.mkPen(colors, width=1.5))
        
        layout.addWidget(self.plot_widget)
        
        # Статистика
        self.stats_label = QLabel("Ожидание данных...")
        self.stats_label.setFont(QFont("Courier", 8))
        self.stats_label.setStyleSheet("color: #ffffcc; background-color: #1e1e1e; padding: 5px; border-radius: 3px;")
        self.stats_label.setWordWrap(False)
        layout.addWidget(self.stats_label)
        
        self.setLayout(layout)
    
    def add_sample(self, timestamp: float, value: float):
        """Добавляет новый сэмпл в буфер."""
        if self.first_sample_time is None:
            self.first_sample_time = timestamp
        
        self.last_sample_time = timestamp
        relative_time = timestamp - self.first_sample_time
        
        self.data_buffer.append(value)
        self.time_buffer.append(relative_time)
    
    def update_plot(self):
        """Обновляет график и статистику."""
        if len(self.time_buffer) < 2 or self.first_sample_time is None:
            return
        
        times = np.array(self.time_buffer)
        data = np.array(self.data_buffer)
        
        if len(times) != len(data):
            min_len = min(len(times), len(data))
            times = times[:min_len]
            data = data[:min_len]
        
        if len(times) == 0:
            return
        
        # Вычисляем текущее относительное время
        if self.last_sample_time is not None:
            current_relative_time = self.last_sample_time - self.first_sample_time
        else:
            current_relative_time = local_clock() - self.first_sample_time
        
        # Преобразуем времена в систему координат графика (отрицательные = секунды назад)
        plot_times = times - current_relative_time
        
        # Фильтруем данные для окна отображения
        mask = (plot_times >= -WINDOW_SEC) & (plot_times <= 0.1)
        if np.any(mask):
            plot_times = plot_times[mask]
            plot_data = data[mask]
        else:
            if len(plot_times) > 0:
                plot_times = plot_times
                plot_data = data
            else:
                plot_times = np.array([])
                plot_data = np.array([])
        
        # Обновляем график
        self.signal_line.setData(plot_times, plot_data)
        self.plot_widget.setXRange(-WINDOW_SEC, 0)
        
        # Автомасштабирование
        if len(plot_data) > 0:
            y_min, y_max = np.nanmin(plot_data), np.nanmax(plot_data)
            if y_max != y_min:
                margin = (y_max - y_min) * 0.15
                if margin == 0:
                    margin = 1.0
                self.plot_widget.setYRange(y_min - margin, y_max + margin)
            else:
                self.plot_widget.setYRange(y_min - 10, y_max + 10)
        
        # Обновляем статистику
        if len(data) > 0:
            mean_val = np.nanmean(data)
            std_val = np.nanstd(data)
            rms_val = np.sqrt(np.nanmean(data**2))
            peak_to_peak = np.nanmax(data) - np.nanmin(data)
            
            stats_text = (
                f"Среднее: {mean_val:8.2f} мкВ  |  "
                f"СКО: {std_val:8.2f} мкВ  |  "
                f"СКЗ: {rms_val:8.2f} мкВ  |  "
                f"Размах: {peak_to_peak:8.2f} мкВ"
            )
            self.stats_label.setText(stats_text)
    
    def get_recent_data(self, window_samples: int) -> np.ndarray:
        """Возвращает последние window_samples сэмплов."""
        if len(self.data_buffer) >= window_samples:
            return np.array(list(self.data_buffer)[-window_samples:])
        else:
            return np.array(list(self.data_buffer))


class HardwareValidationWindow(QMainWindow):
    """Главное окно приложения для аппаратной валидации ЭЭГ."""
    
    def __init__(self, stream_name: str, n_channels: int, sampling_rate: float):
        super().__init__()
        self.stream_name = stream_name
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.cov_window_samples = int(WINDOW_SEC * sampling_rate)
        
        self.channel_widgets: List[ChannelWidget] = []
        self.inlet: Optional[StreamInlet] = None
        
        self.start_time = time.time()
        self.last_cov_time = time.time()
        self.sample_count = 0
        
        self._setup_ui()
        self._setup_timer()
    
    def _setup_ui(self):
        """Создаёт интерфейс окна."""
        self.setWindowTitle(f"Аппаратная валидация ЭЭГ — {self.stream_name}")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Заголовок
        title = QLabel(f"Аппаратная валидация ЭЭГ — {self.stream_name}")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: white; padding: 10px; background-color: #2b2b2b; border-radius: 5px;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)
        
        # Информационная панель
        info_panel = QGroupBox("Информация о потоке")
        info_panel.setStyleSheet("QGroupBox { color: white; font-weight: bold; }")
        info_layout = QHBoxLayout()
        info_layout.addWidget(QLabel(f"Каналов: {self.n_channels}"))
        info_layout.addWidget(QLabel(f"Частота: {self.sampling_rate} Гц"))
        info_layout.addWidget(QLabel(f"Окно: {WINDOW_SEC} сек"))
        info_layout.addStretch()
        info_panel.setLayout(info_layout)
        main_layout.addWidget(info_panel)
        
        # Прокручиваемая область с графиками
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("background-color: #1e1e1e; border: none;")
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setContentsMargins(5, 5, 5, 5)
        scroll_layout.setSpacing(10)
        
        # Создаём виджеты для каждого канала
        for ch in range(self.n_channels):
            channel_widget = ChannelWidget(ch, self.sampling_rate)
            self.channel_widgets.append(channel_widget)
            scroll_layout.addWidget(channel_widget)
        
        scroll_layout.addStretch()
        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        
        main_layout.addWidget(scroll_area)
        
        central_widget.setLayout(main_layout)
        
        # Устанавливаем размер окна
        self.resize(1400, 900)
    
    def _setup_timer(self):
        """Настраивает таймер для обновления графиков."""
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_all_plots)
        self.update_timer.start(UPDATE_INTERVAL_MS)
    
    def set_inlet(self, inlet: StreamInlet):
        """Устанавливает LSL inlet для получения данных."""
        self.inlet = inlet
    
    def _update_all_plots(self):
        """Обновляет все графики и обрабатывает новые данные."""
        if self.inlet is None:
            return
        
        # Получаем новые данные из LSL
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=256)
        if chunk and timestamps:
            arr = np.array(chunk)
            # LSL может отдавать (samples, channels) или (channels, samples)
            if arr.shape[1] != self.n_channels:
                if arr.shape[0] == self.n_channels:
                    arr = arr.T
                else:
                    return
            
            # Добавляем данные в буферы каждого канала
            for i in range(len(arr)):
                timestamp = timestamps[i]
                for ch in range(self.n_channels):
                    self.channel_widgets[ch].add_sample(timestamp, arr[i, ch])
                self.sample_count += 1
        
        # Обновляем все графики
        for widget in self.channel_widgets:
            widget.update_plot()
        
        # Проверяем, нужно ли вывести матрицу ковариаций
        now = time.time()
        if now - self.last_cov_time >= COV_UPDATE_INTERVAL:
            self.last_cov_time = now
            self._print_covariance_matrix()
    
    def _print_covariance_matrix(self):
        """Выводит матрицу ковариаций в консоль."""
        # Собираем данные со всех каналов
        all_data = []
        for widget in self.channel_widgets:
            channel_data = widget.get_recent_data(self.cov_window_samples)
            all_data.append(channel_data)
        
        # Проверяем, что все каналы имеют данные
        min_len = min(len(d) for d in all_data if len(d) > 0)
        if min_len < 2:
            return
        
        # Обрезаем до минимальной длины и формируем матрицу (samples, channels)
        data_matrix = np.zeros((min_len, self.n_channels))
        for ch in range(self.n_channels):
            channel_data = all_data[ch]
            if len(channel_data) >= min_len:
                data_matrix[:, ch] = channel_data[-min_len:]
        
        # Центрируем по каналам для ковариации
        data_matrix = data_matrix - np.mean(data_matrix, axis=0)
        cov_matrix = np.cov(data_matrix.T)
        
        # Улучшенный вывод матрицы ковариаций
        elapsed_time = time.time() - self.start_time
        print("\n" + "=" * 80)
        print(f"МАТРИЦА КОВАРИАЦИЙ | Время работы: {elapsed_time:.1f} сек | Сэмплов: {self.sample_count}")
        print("=" * 80)
        print("Интерпретация:")
        print("  • Диагональ (x11, x22, ...): мощность (дисперсия) на канале")
        print("  • Вне диагонали (x12, x21, ...): наводка между каналами")
        print("  • При сигнале на 1 канале: x11 большое, остальные → 0")
        print("-" * 80)
        
        # Красивое форматирование матрицы
        np.set_printoptions(precision=4, suppress=True, linewidth=100)
        print("\nМатрица ковариаций:")
        print("     ", end="")
        for ch in range(self.n_channels):
            print(f"Ch{ch+1:2d}    ", end="")
        print()
        
        for i in range(self.n_channels):
            print(f"Ch{i+1:2d} ", end="")
            for j in range(self.n_channels):
                val = cov_matrix[i, j]
                print(f"{val:8.4f} ", end="")
            print()
        
        # Анализ наводок
        print("\nАнализ наводок:")
        max_off_diag = 0
        max_off_diag_pair = (0, 0)
        for i in range(self.n_channels):
            for j in range(self.n_channels):
                if i != j:
                    abs_val = abs(cov_matrix[i, j])
                    if abs_val > max_off_diag:
                        max_off_diag = abs_val
                        max_off_diag_pair = (i+1, j+1)
        
        diag_mean = np.mean(np.diag(cov_matrix))
        if max_off_diag > 0:
            ratio = max_off_diag / diag_mean if diag_mean > 0 else 0
            print(f"  Максимальная наводка: Ch{max_off_diag_pair[0]} ↔ Ch{max_off_diag_pair[1]} = {max_off_diag:.4f}")
            print(f"  Отношение к средней мощности: {ratio*100:.2f}%")
            if ratio > 0.1:
                print(f"  ⚠ ВНИМАНИЕ: Обнаружена значительная наводка!")
            else:
                print(f"  ✓ Наводки в пределах нормы")
        print("=" * 80)


def main():
    print("=" * 60)
    print("АППАРАТНАЯ ВАЛИДАЦИЯ ЭЭГ (проверка независимости каналов)")
    print("=" * 60)
    print("\nПоиск LSL-потока ЭЭГ/Signal...")
    
    eeg_info = find_eeg_stream()
    if eeg_info is None:
        print("ОШИБКА: Поток ЭЭГ не найден. Запустите Нейроспектр с LSL.")
        return
    
    inlet = StreamInlet(eeg_info)
    srate = eeg_info.nominal_srate()
    n_channels = eeg_info.channel_count()
    try:
        name = eeg_info.name()
        stream_name = name if name else "EEG"
    except UnicodeDecodeError:
        stream_name = "EEG"
    
    if srate <= 0:
        print("ОШИБКА: Частота дискретизации не определена.")
        return
    
    print(f"  ✓ Подключено: {stream_name}")
    print(f"    Каналов: {n_channels}, частота: {srate} Гц")
    print(f"\nВизуализация: последние {WINDOW_SEC} сек (изолированно по каналам)")
    print(f"Ковариация: окно {WINDOW_SEC} сек, вывод каждые {COV_UPDATE_INTERVAL} сек")
    print("Закройте окно для выхода\n")
    
    # Создаём приложение Qt
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Создаём главное окно
    window = HardwareValidationWindow(stream_name, n_channels, srate)
    window.set_inlet(inlet)
    window.show()
    
    # Запускаем главный цикл
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
