#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обработки ЭЭГ данных в реальном времени.

Этот скрипт:
1. Подключается к LSL потокам маркеров и ЭЭГ
2. Извлекает эпохи ЭЭГ при каждом событии "on" (загорание плитки)
3. Фильтрует сигнал (0.5-10 Гц) и вычитает baseline
4. Накопляет эпохи для каждой из 9 плиток отдельно
5. Показывает усреднённые сигналы в реальном времени
6. Сохраняет результаты после остановки

Запуск:
    python scripts/eeg_epoch_processor.py

Перед запуском убедитесь, что:
    - Запущено приложение стимуляции (app.main)
    - Запущен поток ЭЭГ (например, Нейроспектр)
    - Оба потока транслируются через LSL
"""

import time
import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle
import os
import hashlib

from scipy import signal
from pylsl import StreamInlet, resolve_byprop, local_clock

# PyQt5 и pyqtgraph для событийно-ориентированной визуализации
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QGridLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor
import pyqtgraph as pg

# ============================================================================
# КОНСТАНТЫ И ПАРАМЕТРЫ
# ============================================================================

# Параметры эпохи (в секундах)
EPOCH_PRE_STIM = 0.2   # Время до стимула: -200 мс
EPOCH_POST_STIM = 0.8  # Время после стимула: +800 мс
EPOCH_TOTAL = EPOCH_PRE_STIM + EPOCH_POST_STIM  # Общая длина эпохи: 1.0 сек

# Параметры фильтрации
FILTER_LOW = 0.5   # Нижняя частота среза: 0.5 Гц (убирает медленные дрейфы)
FILTER_HIGH = 10.0  # Верхняя частота среза: 10 Гц (фокус на P300, который ~3 Гц)

# Параметры baseline correction
BASELINE_START = -0.2  # Начало baseline периода (относительно стимула)
BASELINE_END = 0.0     # Конец baseline периода (момент стимула)

# Количество плиток в сетке
NUM_TILES = 9  # Сетка 3x3, плитки с ID от 0 до 8

# Параметры буфера данных
BUFFER_DURATION = 30.0  # Храним последние 30 секунд ЭЭГ данных
UPDATE_INTERVAL_MS = 33  # Обновление графика каждые 33 мс (~30 FPS для плавности)
STATUS_UPDATE_INTERVAL = 1.0  # Обновление статуса в консоли каждую секунду

# Имена потоков LSL
MARKER_STREAM_NAME = "BCI_StimMarkers"
EEG_STREAM_TYPE = "EEG"  # Или "Signal" в зависимости от вашего оборудования


# ============================================================================
# КЛАСС ДЛЯ ХРАНЕНИЯ И ОБРАБОТКИ ЭПОХ
# ============================================================================

class EpochProcessor:
    """
    Класс для накопления и обработки эпох ЭЭГ.
    
    Хранит эпохи для каждой плитки отдельно и вычисляет усреднённые сигналы.
    """
    
    def __init__(self, sampling_rate: float, num_channels: int):
        """
        Инициализация процессора эпох.
        
        Args:
            sampling_rate: Частота дискретизации ЭЭГ (например, 250 Гц)
            num_channels: Количество каналов ЭЭГ
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        
        # Вычисляем количество точек в эпохе
        self.epoch_samples_pre = int(EPOCH_PRE_STIM * sampling_rate)
        self.epoch_samples_post = int(EPOCH_POST_STIM * sampling_rate)
        self.epoch_samples_total = self.epoch_samples_pre + self.epoch_samples_post
        
        # Вычисляем индексы для baseline correction
        self.baseline_start_idx = int((BASELINE_START - EPOCH_PRE_STIM) * sampling_rate)
        self.baseline_end_idx = int((BASELINE_END - EPOCH_PRE_STIM) * sampling_rate)
        
        # Хранилище эпох: для каждой плитки (0-8) список эпох
        # Каждая эпоха - это массив формы (num_channels, epoch_samples_total)
        self.epochs: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # Создаём фильтр для обработки сигнала
        # Используем Butterworth фильтр - стандартный для ЭЭГ
        self._create_filter()
        
        print(f"Инициализирован процессор эпох:")
        print(f"  Частота дискретизации: {sampling_rate} Гц")
        print(f"  Каналов: {num_channels}")
        print(f"  Длина эпохи: {self.epoch_samples_total} точек ({EPOCH_TOTAL} сек)")
        print(f"  Baseline: от {BASELINE_START} до {BASELINE_END} сек")
    
    def _create_filter(self):
        """
        Создаёт bandpass фильтр для частот 0.5-10 Гц.
        
        Butterworth фильтр - это стандартный тип фильтра для ЭЭГ,
        который имеет плоскую частотную характеристику в полосе пропускания.
        """
        # Нормализуем частоты относительно частоты Найквиста (половина sampling_rate)
        nyquist = self.sampling_rate / 2.0
        low = FILTER_LOW / nyquist
        high = FILTER_HIGH / nyquist
        
        # Создаём фильтр 4-го порядка (более крутой срез, меньше артефактов)
        self.b, self.a = signal.butter(4, [low, high], btype='band')
        
        # Создаём начальные условия для фильтра (чтобы избежать переходных процессов)
        # Используем нулевые начальные условия
        self.zi = signal.lfilter_zi(self.b, self.a)
        # Расширяем для всех каналов
        self.zi = np.tile(self.zi[:, np.newaxis], (1, self.num_channels)).T
    
    def add_epoch(self, tile_id: int, epoch_data: np.ndarray):
        """
        Добавляет новую эпоху для указанной плитки.
        
        Args:
            tile_id: ID плитки (0-8)
            epoch_data: Массив формы (num_channels, epoch_samples_total)
        """
        # Проверяем размерность данных
        if epoch_data.shape != (self.num_channels, self.epoch_samples_total):
            print(f"ОШИБКА: Неверный размер эпохи. Ожидается "
                  f"({self.num_channels}, {self.epoch_samples_total}), "
                  f"получено {epoch_data.shape}")
            return
        
        # Применяем фильтрацию
        filtered_epoch = self._filter_epoch(epoch_data)
        
        # Вычитаем baseline (среднее значение в период до стимула)
        baseline_corrected = self._baseline_correction(filtered_epoch)
        
        # Добавляем обработанную эпоху в хранилище
        self.epochs[tile_id].append(baseline_corrected)
        
        # Выводим информацию о накопленных эпохах
        num_epochs = len(self.epochs[tile_id])
        if num_epochs % 10 == 0:  # Каждые 10 эпох
            print(f"Плитка {tile_id}: накоплено {num_epochs} эпох")
    
    def _filter_epoch(self, epoch_data: np.ndarray) -> np.ndarray:
        """
        Применяет bandpass фильтр к эпохе.
        
        Args:
            epoch_data: Массив формы (num_channels, epoch_samples_total)
        
        Returns:
            Отфильтрованная эпоха той же формы
        """
        # Применяем фильтр к каждому каналу отдельно
        filtered = np.zeros_like(epoch_data)
        for ch in range(self.num_channels):
            # Используем filtfilt для двунаправленной фильтрации
            # (убирает фазовые искажения)
            filtered[ch, :] = signal.filtfilt(self.b, self.a, epoch_data[ch, :])
        
        return filtered
    
    def _baseline_correction(self, epoch_data: np.ndarray) -> np.ndarray:
        """
        Вычитает среднее значение baseline периода из всей эпохи.
        
        Baseline - это период ДО стимула (обычно -200 до 0 мс).
        Это нужно, чтобы убрать постоянную составляющую и дрейфы.
        
        Args:
            epoch_data: Массив формы (num_channels, epoch_samples_total)
        
        Returns:
            Эпоха с вычтенным baseline
        """
        corrected = epoch_data.copy()
        
        # Для каждого канала вычисляем среднее в baseline период
        for ch in range(self.num_channels):
            baseline_mean = np.mean(epoch_data[ch, self.baseline_start_idx:self.baseline_end_idx])
            # Вычитаем это среднее из всего канала
            corrected[ch, :] -= baseline_mean
        
        return corrected
    
    def get_averaged_epochs(self) -> Dict[int, Optional[np.ndarray]]:
        """
        Возвращает усреднённые эпохи для каждой плитки.
        
        Returns:
            Словарь: tile_id -> усреднённая эпоха (num_channels, epoch_samples_total)
                     или None, если для плитки нет эпох
        """
        averaged = {}
        for tile_id in range(NUM_TILES):
            if len(self.epochs[tile_id]) > 0:
                # Усредняем все эпохи для этой плитки
                # np.array(self.epochs[tile_id]) создаёт массив формы
                # (num_epochs, num_channels, epoch_samples_total)
                # np.mean(axis=0) усредняет по первой оси (по эпохам)
                averaged[tile_id] = np.mean(self.epochs[tile_id], axis=0)
            else:
                averaged[tile_id] = None
        
        return averaged
    
    def get_epoch_counts(self) -> Dict[int, int]:
        """Возвращает количество накопленных эпох для каждой плитки."""
        return {tile_id: len(self.epochs[tile_id]) for tile_id in range(NUM_TILES)}
    
    def save_results(self, filename: str):
        """
        Сохраняет все накопленные эпохи в файл.
        
        Args:
            filename: Путь к файлу для сохранения
        """
        results = {
            'epochs': dict(self.epochs),  # Все эпохи
            'averaged_epochs': self.get_averaged_epochs(),  # Усреднённые
            'sampling_rate': self.sampling_rate,
            'num_channels': self.num_channels,
            'epoch_counts': self.get_epoch_counts(),
            'filter_params': {'low': FILTER_LOW, 'high': FILTER_HIGH},
            'baseline_params': {'start': BASELINE_START, 'end': BASELINE_END},
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\nРезультаты сохранены в: {filename}")
        print(f"Всего эпох по плиткам:")
        for tile_id, count in self.get_epoch_counts().items():
            if count > 0:
                print(f"  Плитка {tile_id}: {count} эпох")


# ============================================================================
# КЛАСС ДЛЯ БУФЕРИЗАЦИИ ЭЭГ ДАННЫХ
# ============================================================================

class EEGBuffer:
    """
    Класс для хранения скользящего буфера ЭЭГ данных.

    Хранит последние N секунд данных по относительному времени от первого сэмпла ЭЭГ.
    Использует относительное время для синхронизации с маркерами.
    """

    def __init__(self, sampling_rate: float, num_channels: int, duration: float = BUFFER_DURATION):
        """
        Args:
            sampling_rate: Частота дискретизации
            num_channels: Количество каналов
            duration: Длительность буфера в секундах
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.buffer_size = int(duration * sampling_rate)

        # Буфер: кортежи (relative_time, data) — относительное время от первого сэмпла
        self.buffer: deque = deque(maxlen=self.buffer_size)
        self.first_sample_time: Optional[float] = None  # Время первого сэмпла (LSL время)

        print(f"Инициализирован буфер ЭЭГ:")
        print(f"  Размер буфера: {self.buffer_size} точек ({duration} сек)")
        print(f"  Используется относительное время от первого сэмпла ЭЭГ для синхронизации")

    def add_sample(self, timestamp: float, data: np.ndarray) -> None:
        """
        Добавляет сэмпл в буфер.

        Args:
            timestamp: LSL-таймстемп из pull_chunk
            data: Массив формы (num_channels,)
        """
        if data.shape != (self.num_channels,):
            print(f"ОШИБКА: Неверный размер данных. Ожидается ({self.num_channels},), "
                  f"получено {data.shape}")
            return
        
        # Запоминаем время первого сэмпла
        if self.first_sample_time is None:
            self.first_sample_time = timestamp
        
        # Вычисляем относительное время от первого сэмпла
        relative_time = timestamp - self.first_sample_time
        
        self.buffer.append((relative_time, data.copy()))
    
    def get_first_sample_time(self) -> Optional[float]:
        """Возвращает время первого сэмпла (для синхронизации маркеров)."""
        return self.first_sample_time

    def is_ready(self, marker_timestamp: float, pre_stim: float, post_stim: float, eeg_first_time: Optional[float] = None) -> Tuple[bool, str]:
        """
        Проверяет, достаточно ли данных для извлечения эпохи.

        Args:
            marker_timestamp: LSL-таймстемп маркера из pull_sample
            pre_stim: Время до стимула, сек
            post_stim: Время после стимула, сек
            eeg_first_time: Время первого сэмпла ЭЭГ (для синхронизации)

        Returns:
            (is_ready, message)
        """
        if len(self.buffer) < 2:
            return False, f"Буфер пуст (нужно минимум 2 точки, есть {len(self.buffer)})"
        
        if self.first_sample_time is None:
            return False, "Буфер не инициализирован (нет первого сэмпла)"
        
        # Конвертируем маркер в относительное время
        if eeg_first_time is None:
            eeg_first_time = self.first_sample_time
        
        marker_relative_time = marker_timestamp - eeg_first_time
        
        epoch_start = marker_relative_time - pre_stim
        epoch_end = marker_relative_time + post_stim

        buffer_times = np.array([ts for ts, _ in self.buffer])
        buffer_start = buffer_times[0]
        buffer_end = buffer_times[-1]
        buffer_duration = buffer_end - buffer_start

        if buffer_start > epoch_start:
            return False, (
                f"Недостаточно данных ДО маркера (нужно {pre_stim:.3f} сек до, "
                f"буфер начинается на {buffer_start - epoch_start:.3f} сек позже)"
            )
        if buffer_end < epoch_end:
            return False, (
                f"Недостаточно данных ПОСЛЕ маркера (нужно {post_stim:.3f} сек после, "
                f"буфер заканчивается на {epoch_end - buffer_end:.3f} сек раньше)"
            )
        return True, f"Буфер готов (длительность: {buffer_duration:.2f} сек, точек: {len(self.buffer)})"

    def get_buffer_info(self) -> dict:
        """Возвращает состояние буфера (относительное время)."""
        if len(self.buffer) < 2:
            return {'size': len(self.buffer), 'duration': 0.0, 'start_time': None, 'end_time': None}
        buffer_times = np.array([ts for ts, _ in self.buffer])
        return {
            'size': len(self.buffer),
            'duration': buffer_times[-1] - buffer_times[0],
            'start_time': buffer_times[0],
            'end_time': buffer_times[-1],
            'first_sample_time': self.first_sample_time,  # Абсолютное время первого сэмпла
        }

    def extract_epoch(
        self, marker_timestamp: float, pre_stim: float, post_stim: float, eeg_first_time: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Извлекает эпоху вокруг маркера по относительному времени.

        Args:
            marker_timestamp: LSL-таймстемп маркера
            pre_stim: Время до стимула, сек
            post_stim: Время после стимула, сек
            eeg_first_time: Время первого сэмпла ЭЭГ (для синхронизации)

        Returns:
            (epoch_data, time_vector) или None
        """
        if eeg_first_time is None:
            eeg_first_time = self.first_sample_time
        
        is_ready, _ = self.is_ready(marker_timestamp, pre_stim, post_stim, eeg_first_time)
        if not is_ready:
            return None

        num_samples_pre = int(pre_stim * self.sampling_rate)
        num_samples_post = int(post_stim * self.sampling_rate)
        num_samples_total = num_samples_pre + num_samples_post

        epoch_data = np.zeros((self.num_channels, num_samples_total))
        time_vector = np.linspace(-pre_stim, post_stim, num_samples_total)

        buffer_times = np.array([ts for ts, _ in self.buffer])
        buffer_data = np.array([data for _, data in self.buffer])

        # Конвертируем маркер в относительное время
        marker_relative_time = marker_timestamp - eeg_first_time
        
        # Относительные времена точек эпохи
        target_times = time_vector + marker_relative_time

        for ch in range(self.num_channels):
            epoch_data[ch, :] = np.interp(
                target_times,
                buffer_times,
                buffer_data[:, ch],
            )
        return epoch_data, time_vector


# ============================================================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С LSL
# ============================================================================

def find_streams():
    """
    Находит потоки маркеров и ЭЭГ в сети LSL.
    
    Returns:
        Кортеж (marker_stream_info, eeg_stream_info) или (None, None) если не найдены
    """
    print("\nПоиск LSL потоков...")
    
    # Ищем поток маркеров
    print(f"  Поиск потока маркеров '{MARKER_STREAM_NAME}'...")
    marker_streams = resolve_byprop("name", MARKER_STREAM_NAME, timeout=5)
    if not marker_streams:
        print(f"  ОШИБКА: Поток маркеров '{MARKER_STREAM_NAME}' не найден!")
        print(f"  Убедитесь, что приложение стимуляции запущено и нажата кнопка START.")
        return None, None
    
    marker_info = marker_streams[0]
    print(f"  ✓ Найден поток маркеров: {marker_info.name()}")
    
    # Ищем поток ЭЭГ
    print(f"  Поиск потока ЭЭГ (тип '{EEG_STREAM_TYPE}')...")
    eeg_streams = resolve_byprop("type", EEG_STREAM_TYPE, timeout=5)
    if not eeg_streams:
        # Пробуем альтернативные типы
        for alt_type in ["Signal", "EEG"]:
            eeg_streams = resolve_byprop("type", alt_type, timeout=2)
            if eeg_streams:
                break
    
    if not eeg_streams:
        print(f"  ОШИБКА: Поток ЭЭГ не найден!")
        print(f"  Убедитесь, что ЭЭГ оборудование транслирует данные через LSL.")
        return None, None
    
    eeg_info = eeg_streams[0]
    try:
        eeg_name = eeg_info.name()
    except:
        eeg_name = "Unknown"
    
    print(f"  ✓ Найден поток ЭЭГ: {eeg_name}")
    print(f"     Каналов: {eeg_info.channel_count()}")
    print(f"     Частота: {eeg_info.nominal_srate()} Гц")
    
    return marker_info, eeg_info


def parse_marker(marker_string: str) -> Optional[Tuple[int, str]]:
    """
    Парсит строку маркера в формате "tile_id|event".
    
    Args:
        marker_string: Строка маркера, например "3|on"
    
    Returns:
        Кортеж (tile_id, event) или None при ошибке
    """
    try:
        parts = marker_string.split('|')
        if len(parts) != 2:
            return None
        tile_id = int(parts[0])
        event = parts[1].strip().lower()
        return tile_id, event
    except (ValueError, AttributeError):
        return None


# ============================================================================
# ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ (PyQt5 + pyqtgraph)
# ============================================================================

class RealtimePlotter(QMainWindow):
    """
    Класс для отображения графиков в реальном времени на основе PyQt5 и pyqtgraph.
    
    Показывает усреднённые эпохи для всех 9 плиток в виде сетки графиков.
    Использует событийно-ориентированную модель с QTimer для плавного обновления.
    """
    
    def __init__(self, sampling_rate: float, num_channels: int, processor: Optional['EpochProcessor'] = None):
        """
        Инициализация визуализации.
        
        Args:
            sampling_rate: Частота дискретизации
            num_channels: Количество каналов ЭЭГ
            processor: Экземпляр EpochProcessor (будет установлен позже)
        """
        super().__init__()
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.processor = processor
        
        # Создаём вектор времени для эпохи
        self.time_vector = np.linspace(-EPOCH_PRE_STIM, EPOCH_POST_STIM, 
                                       int(EPOCH_TOTAL * sampling_rate))
        
        # Инициализируем графики для каждой плитки
        self.plots = {}  # pyqtgraph PlotWidget для каждой плитки
        self.lines = {}  # Линии графиков
        self.text_labels = {}  # QLabel для отображения количества эпох
        self.highlight_times = {}  # Время последней подсветки
        self.last_epoch_counts = {}  # Последнее количество эпох
        self.last_data_hash = {}  # Хеш последних данных
        
        # Настройка pyqtgraph
        pg.setConfigOptions(antialias=True, background='black', foreground='white')
        
        # Создаём центральный виджет и сетку
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QGridLayout(central_widget)
        
        # Создаём графики для каждой плитки
        for tile_id in range(NUM_TILES):
            row = tile_id // 3
            col = tile_id % 3
            
            # Создаём PlotWidget
            plot_widget = pg.PlotWidget(title=f'Плитка {tile_id}')
            plot_widget.setLabel('left', 'Амплитуда (мкВ)')
            plot_widget.setLabel('bottom', 'Время (сек)')
            # Фиксированный диапазон по X: от -0.2 до +0.8 секунды (длина эпохи)
            # Это нормально - каждая эпоха имеет фиксированную длину для ERP анализа
            plot_widget.setXRange(-EPOCH_PRE_STIM, EPOCH_POST_STIM)
            plot_widget.setYRange(-50, 50)
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            
            # Вертикальная линия на моменте стимула
            stim_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', style=Qt.DashLine, width=2))
            plot_widget.addItem(stim_line)
            
            # Создаём линию для данных
            if num_channels > 1:
                # Для нескольких каналов будем показывать средний
                line = plot_widget.plot([], [], pen=pg.mkPen('w', width=1.5))
            else:
                # Для одного канала
                line = plot_widget.plot([], [], pen=pg.mkPen('w', width=1.5))
            
            self.plots[tile_id] = plot_widget
            self.lines[tile_id] = line
            
            # Создаём метку для количества эпох (добавляем на график через TextItem)
            # Используем относительные координаты через ViewBox
            text_item = pg.TextItem('Эпох: 0', color='white', anchor=(0, 1))
            # Позиция в координатах данных: x в секундах, y в мкВ
            text_item.setPos(-EPOCH_PRE_STIM + 0.05, 45)  # Позиция в координатах графика
            plot_widget.addItem(text_item)
            self.text_labels[tile_id] = text_item
            
            # Добавляем в layout
            layout.addWidget(plot_widget, row, col)
            
            # Инициализация состояний
            self.highlight_times[tile_id] = 0
            self.last_epoch_counts[tile_id] = 0
            self.last_data_hash[tile_id] = None
        
        self.setWindowTitle('Усреднённые эпохи ЭЭГ по плиткам (обновление в реальном времени)')
        self.resize(1500, 1000)
        
        # Сохраняем состояние окна
        self.window_open = True
        
    
    def set_processor(self, processor: 'EpochProcessor'):
        """Устанавливает процессор эпох для обновления графиков."""
        self.processor = processor
    
    def update(self, processor: Optional['EpochProcessor'] = None):
        """
        Обновляет графики на основе текущих данных процессора.
        
        Args:
            processor: Экземпляр EpochProcessor с накопленными данными (опционально, если не указан, используется self.processor)
        """
        if not self.window_open or not self.isVisible():
            return
        
        if processor is None:
            processor = self.processor
        
        if processor is None:
            return
        
        try:
            averaged = processor.get_averaged_epochs()
            counts = processor.get_epoch_counts()
            current_time = time.time()
            highlight_duration = 0.15  # Длительность подсветки в секундах
            
            
            for tile_id in range(NUM_TILES):
                plot_widget = self.plots[tile_id]
                line = self.lines[tile_id]
                
                data_changed = False
                count_changed = False
                
                if averaged[tile_id] is not None:
                    epoch_data = averaged[tile_id]
                    
                    # Вычисляем хеш данных для определения реальных изменений
                    data_hash = hashlib.md5(epoch_data.tobytes()).hexdigest()
                    
                    if self.last_data_hash[tile_id] != data_hash:
                        data_changed = True
                        self.last_data_hash[tile_id] = data_hash
                    
                    # Если несколько каналов, показываем средний по каналам
                    if self.num_channels > 1:
                        mean_signal = np.mean(epoch_data, axis=0)
                        line.setData(self.time_vector, mean_signal)
                    else:
                        # Один канал
                        line.setData(self.time_vector, epoch_data[0, :])
                    
                    # Автоматически подстраиваем масштаб по Y
                    if data_changed:
                        y_data = epoch_data.flatten()
                        if len(y_data) > 0:
                            y_min, y_max = np.min(y_data), np.max(y_data)
                            if y_max != y_min:
                                margin = (y_max - y_min) * 0.1
                                plot_widget.setYRange(y_min - margin, y_max + margin)
                            else:
                                plot_widget.setYRange(y_min - 10, y_max + 10)
                else:
                    # Нет данных - очищаем график
                    line.setData([], [])
                    plot_widget.setYRange(-50, 50)  # Возвращаем стандартный диапазон
                
                # Обновляем текст с количеством эпох
                count = counts[tile_id]
                if self.last_epoch_counts[tile_id] != count:
                    count_changed = True
                    self.last_epoch_counts[tile_id] = count
                    self.text_labels[tile_id].setText(f'Эпох: {count}')
                
                # Подсветка графика при изменении данных
                if data_changed or count_changed:
                    self.highlight_times[tile_id] = current_time
                    # Временно меняем цвет линии на зелёный для подсветки
                    line.setPen(pg.mkPen('lime', width=2))
                else:
                    # Плавно возвращаем обычный цвет
                    time_since_highlight = current_time - self.highlight_times[tile_id]
                    if time_since_highlight < highlight_duration:
                        progress = time_since_highlight / highlight_duration
                        # Плавный переход от зелёного к белому
                        if progress < 0.5:
                            line.setPen(pg.mkPen('lime', width=2))
                        else:
                            line.setPen(pg.mkPen('w', width=1.5))
                    else:
                        line.setPen(pg.mkPen('w', width=1.5))
            
        except Exception as e:
            if not hasattr(self, '_error_shown'):
                print(f"⚠ Ошибка при обновлении графиков: {e}")
                self._error_shown = True
    
    def closeEvent(self, event):
        """Обработчик закрытия окна."""
        self.window_open = False
        # Вызываем shutdown через QApplication
        app = QApplication.instance()
        if app and hasattr(app, 'shutdown'):
            app.shutdown()
        event.accept()
    
    def close(self):
        """Закрывает окно графика."""
        self.window_open = False
        super().close()


# ============================================================================
# КЛАСС ПРИЛОЖЕНИЯ ДЛЯ СОБЫТИЙНО-ОРИЕНТИРОВАННОЙ МОДЕЛИ
# ============================================================================

class EEGProcessorApp(QApplication):
    """
    Главное приложение для обработки ЭЭГ в реальном времени.
    Использует событийно-ориентированную модель с QTimer.
    """
    
    def __init__(self, argv):
        super().__init__(argv)
        
        # Компоненты обработки (будут инициализированы позже)
        self.buffer: Optional[EEGBuffer] = None
        self.processor: Optional[EpochProcessor] = None
        self.plotter: Optional[RealtimePlotter] = None
        self.marker_inlet: Optional[StreamInlet] = None
        self.eeg_inlet: Optional[StreamInlet] = None
        
        # Статистика
        self.processed_markers = 0
        self.skipped_markers = 0
        self.start_time = time.time()
        self.last_status_time = time.time()
        self.last_eeg_log_time = 0
        
        # Таймеры для событийно-ориентированной модели
        self.update_timer = QTimer()  # Таймер для обновления графиков
        self.update_timer.timeout.connect(self.update_plots)
        
        self.data_timer = QTimer()  # Таймер для чтения данных LSL
        self.data_timer.timeout.connect(self.read_lsl_data)
        
        self.status_timer = QTimer()  # Таймер для вывода статуса
        self.status_timer.timeout.connect(self.print_status)
        
        # Очередь маркеров
        self.marker_queue = []
        
    def initialize(self):
        """Инициализация компонентов обработки."""
        print("=" * 70)
        print("ОБРАБОТКА ЭЭГ ДАННЫХ В РЕАЛЬНОМ ВРЕМЕНИ")
        print("=" * 70)
        print("\nЭтот скрипт будет:")
        print("  1. Подключаться к потокам маркеров и ЭЭГ")
        print("  2. Извлекать эпохи при каждом событии 'on'")
        print("  3. Фильтровать и обрабатывать данные")
        print("  4. Показывать усреднённые сигналы в реальном времени")
        print("  5. Сохранять результаты при остановке\n")
        
        # Шаг 1: Находим потоки
        marker_info, eeg_info = find_streams()
        if marker_info is None or eeg_info is None:
            print("\nНе удалось найти необходимые потоки. Выход.")
            return False
        
        # Шаг 2: Создаём inlets для чтения данных
        print("\nПодключение к потокам...")
        self.marker_inlet = StreamInlet(marker_info)
        self.eeg_inlet = StreamInlet(eeg_info)
        
        # Получаем параметры потока ЭЭГ
        eeg_sampling_rate = eeg_info.nominal_srate()
        eeg_num_channels = eeg_info.channel_count()
        
        if eeg_sampling_rate == 0:
            print("ОШИБКА: Частота дискретизации ЭЭГ не определена!")
            return False
        
        print(f"  ✓ Подключено к потоку маркеров")
        print(f"  ✓ Подключено к потоку ЭЭГ ({eeg_num_channels} каналов, {eeg_sampling_rate} Гц)")
        print("\n  Синхронизация: относительное время от первого сэмпла ЭЭГ")
        
        # Шаг 3: Инициализируем компоненты обработки
        print("\nИнициализация компонентов обработки...")
        self.buffer = EEGBuffer(eeg_sampling_rate, eeg_num_channels)
        self.processor = EpochProcessor(eeg_sampling_rate, eeg_num_channels)
        self.plotter = RealtimePlotter(eeg_sampling_rate, eeg_num_channels, self.processor)
        
        print("  ✓ Все компоненты инициализированы")
        
        # Шаг 3.5: Накапливаем данные в буфере перед началом обработки
        min_buffer_duration = EPOCH_POST_STIM + 0.5
        print(f"\nНакопление данных в буфере (минимум {min_buffer_duration:.1f} сек для эпох)...")
        buffer_ready = False
        start_time = time.time()
        
        while not buffer_ready:
            self.processEvents()  # Обрабатываем события Qt
            chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.1, max_samples=100)
            if chunk and timestamps:
                chunk_array = np.array(chunk)
                if chunk_array.shape[1] == eeg_num_channels:
                    pass
                elif chunk_array.shape[0] == eeg_num_channels:
                    chunk_array = chunk_array.T
                else:
                    continue
                
                for i, ts in enumerate(timestamps):
                    self.buffer.add_sample(ts, chunk_array[i, :])
            
            marker_sample, marker_timestamp = self.marker_inlet.pull_sample(timeout=0.0)
            if marker_sample is not None:
                parsed = parse_marker(marker_sample[0])
                if parsed is not None:
                    tile_id, event = parsed
                    if event == "on" and 0 <= tile_id < NUM_TILES:
                        self.marker_queue.append((tile_id, marker_timestamp))
                        if len(self.marker_queue) == 1:
                            print(f"  Получен первый маркер (плитка {tile_id}), ждём накопления данных...")
            
            buffer_info = self.buffer.get_buffer_info()
            if buffer_info['duration'] >= min_buffer_duration:
                buffer_ready = True
                eeg_first_time = self.buffer.get_first_sample_time()
                if eeg_first_time:
                    print(f"  ✓ Буфер готов: {buffer_info['duration']:.2f} сек данных, {buffer_info['size']} точек")
                    print(f"  Время первого сэмпла ЭЭГ: {eeg_first_time:.3f}")
                else:
                    print(f"  ✓ Буфер готов: {buffer_info['duration']:.2f} сек данных, {buffer_info['size']} точек")
                if len(self.marker_queue) > 0:
                    print(f"  В очереди {len(self.marker_queue)} маркеров для обработки")
            elif time.time() - start_time > 15:
                print(f"  ⚠ Буфер частично готов: {buffer_info['duration']:.2f} сек данных")
                print(f"     Продолжаем, но некоторые маркеры могут быть пропущены")
                buffer_ready = True
        
        # Обрабатываем маркеры из очереди
        self.process_marker_queue()
        
        print("\n" + "=" * 70)
        print("НАЧАЛО ОБРАБОТКИ")
        print("Закройте окно для остановки и сохранения результатов")
        print("=" * 70 + "\n")
        
        # Запускаем таймеры
        self.update_timer.start(UPDATE_INTERVAL_MS)  # Обновление графиков
        self.data_timer.start(10)  # Чтение данных LSL каждые 10 мс
        self.status_timer.start(int(STATUS_UPDATE_INTERVAL * 1000))  # Статус каждую секунду
        
        # Показываем окно
        self.plotter.show()
        self.plotter.raise_()
        self.plotter.activateWindow()
        QApplication.processEvents()
        
        return True
    
    def process_marker_queue(self):
        """Обрабатывает маркеры из очереди."""
        if not self.marker_queue:
            return
        
        # Получаем время первого сэмпла ЭЭГ для синхронизации
        eeg_first_time = self.buffer.get_first_sample_time()
        if eeg_first_time is None:
            print("⚠ Предупреждение: время первого сэмпла ЭЭГ не определено, пропускаем очередь маркеров")
            self.marker_queue.clear()
            return
        
        print(f"\n{'='*70}")
        print(f"ОБРАБОТКА {len(self.marker_queue)} МАРКЕРОВ ИЗ ОЧЕРЕДИ")
        print(f"{'='*70}")
        
        # Фильтруем маркеры: оставляем только те, которые пришли ПОСЛЕ первого сэмпла ЭЭГ
        valid_markers = []
        skipped_before_start = 0
        
        for tile_id, marker_ts in self.marker_queue:
            marker_relative_time = marker_ts - eeg_first_time
            if marker_relative_time < 0:
                skipped_before_start += 1
                continue  # Пропускаем маркеры, которые пришли до первого сэмпла ЭЭГ
            valid_markers.append((tile_id, marker_ts))
        
        if skipped_before_start > 0:
            print(f"⚠ Пропущено {skipped_before_start} маркеров, которые пришли до первого сэмпла ЭЭГ")
        
        if not valid_markers:
            print("Нет валидных маркеров для обработки (все пришли до начала записи ЭЭГ)")
            self.marker_queue.clear()
            return
        
        for idx, (tile_id, marker_ts) in enumerate(valid_markers, 1):
            is_ready, ready_message = self.buffer.is_ready(
                marker_ts, EPOCH_PRE_STIM, EPOCH_POST_STIM, eeg_first_time
            )
            marker_relative_time = marker_ts - eeg_first_time
            print(f"\n[{idx}/{len(valid_markers)}] Маркер плитки {tile_id} (относительное время: {marker_relative_time:.3f} сек)")
            
            if is_ready:
                epoch_result = self.buffer.extract_epoch(
                    marker_ts, EPOCH_PRE_STIM, EPOCH_POST_STIM, eeg_first_time
                )
                
                if epoch_result is not None:
                    epoch_data, time_vector = epoch_result
                    self.processor.add_epoch(tile_id, epoch_data)
                    self.processed_markers += 1
                    epoch_counts = self.processor.get_epoch_counts()
                    count_for_tile = epoch_counts[tile_id]
                    print(f"  ✓ Обработана успешно | Эпох для плитки: {count_for_tile}")
                else:
                    self.skipped_markers += 1
                    print(f"  ✗ Пропущена: ошибка извлечения эпохи")
            else:
                self.skipped_markers += 1
                buffer_info = self.buffer.get_buffer_info()
                print(f"  ✗ Пропущена: {ready_message}")
        
        print(f"\n{'='*70}")
        print(f"Обработано из очереди: {self.processed_markers}, пропущено: {self.skipped_markers + skipped_before_start}")
        print(f"{'='*70}\n")
        
        self.marker_queue.clear()
    
    def read_lsl_data(self):
        """Читает данные из LSL потоков (вызывается таймером)."""
        if self.marker_inlet is None or self.eeg_inlet is None:
            return
        
        # Получаем время первого сэмпла ЭЭГ для синхронизации
        eeg_first_time = self.buffer.get_first_sample_time()
        if eeg_first_time is None:
            return  # Ещё нет данных ЭЭГ
        
        # Читаем маркеры (может быть несколько в очереди)
        while True:
            marker_sample, marker_timestamp = self.marker_inlet.pull_sample(timeout=0.0)
            if marker_sample is None:
                break  # Больше нет маркеров
            
            raw_marker = marker_sample[0] if isinstance(marker_sample, (list, tuple)) else marker_sample
            
            parsed = parse_marker(raw_marker)
            if parsed is None:
                continue
            
            tile_id, event = parsed
            if event != "on":
                continue
            
            if not (0 <= tile_id < NUM_TILES):
                continue
            
            # Вычисляем коррекцию времени для синхронизации маркеров с ЭЭГ
            if not hasattr(self, '_marker_time_correction'):
                if marker_timestamp < eeg_first_time:
                    # Маркер использует локальное время LSL, а ЭЭГ - абсолютное
                    # Вычисляем смещение для синхронизации
                    self._marker_time_correction = eeg_first_time - marker_timestamp
                else:
                    self._marker_time_correction = 0.0
            
            # Применяем коррекцию времени к маркеру
            corrected_marker_time = marker_timestamp + self._marker_time_correction
            marker_relative_time = corrected_marker_time - eeg_first_time
            
            if marker_relative_time < 0:
                # Маркер пришёл до начала записи ЭЭГ - пропускаем
                self.skipped_markers += 1
                continue
            
            # Используем скорректированное время для проверки готовности
            is_ready, ready_message = self.buffer.is_ready(
                corrected_marker_time, EPOCH_PRE_STIM, EPOCH_POST_STIM, eeg_first_time
            )
            
            if not is_ready:
                self.skipped_markers += 1
                continue
            
            # Пытаемся извлечь эпоху (используем скорректированное время)
            epoch_result = self.buffer.extract_epoch(
                corrected_marker_time, EPOCH_PRE_STIM, EPOCH_POST_STIM, eeg_first_time
            )
            
            if epoch_result is None:
                self.skipped_markers += 1
                continue
            
            epoch_data, time_vector = epoch_result
            self.processor.add_epoch(tile_id, epoch_data)
            self.processed_markers += 1
        
        # Читаем данные ЭЭГ
        chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0, max_samples=100)
        if chunk and timestamps:
            chunk_array = np.array(chunk)
            if chunk_array.shape[1] == self.buffer.num_channels:
                pass
            elif chunk_array.shape[0] == self.buffer.num_channels:
                chunk_array = chunk_array.T
            else:
                return
            
            samples_added = 0
            for i, ts in enumerate(timestamps):
                self.buffer.add_sample(ts, chunk_array[i, :])
                samples_added += 1
            
            # Периодически выводим информацию о полученных данных
            current_time = time.time()
            if current_time - self.last_eeg_log_time >= 5.0:
                buffer_info = self.buffer.get_buffer_info()
                print(f"📊 Получено {samples_added} сэмплов ЭЭГ | "
                      f"Буфер: {buffer_info['size']} точек, {buffer_info['duration']:.2f} сек")
                self.last_eeg_log_time = current_time
    
    def update_plots(self):
        """Обновляет графики (вызывается таймером)."""
        if self.plotter is not None and self.processor is not None:
            self.plotter.update(self.processor)
    
    def print_status(self):
        """Выводит статус обработки (вызывается таймером)."""
        if self.buffer is None or self.processor is None:
            return
        
        current_time = time.time()
        buffer_info = self.buffer.get_buffer_info()
        epoch_counts = self.processor.get_epoch_counts()
        total_epochs = sum(epoch_counts.values())
        
        print(f"\n{'='*70}")
        print(f"СТАТУС [t={current_time - self.start_time:.1f}s работы]")
        print(f"  Буфер ЭЭГ: {buffer_info['size']} точек, "
              f"{buffer_info['duration']:.2f} сек данных")
        if buffer_info['start_time'] is not None:
            print(f"    Временной диапазон: {buffer_info['start_time']:.3f} - {buffer_info['end_time']:.3f} сек")
        print(f"  Обработано маркеров: {self.processed_markers} | Пропущено: {self.skipped_markers}")
        if total_epochs > 0:
            success_rate = self.processed_markers / (self.processed_markers + self.skipped_markers) * 100 if (self.processed_markers + self.skipped_markers) > 0 else 0
            print(f"  Успешность: {success_rate:.1f}%")
            print(f"  Всего эпох накоплено: {total_epochs}")
            print(f"  Эпохи по плиткам:")
            for tile_id in range(NUM_TILES):
                count = epoch_counts[tile_id]
                if count > 0:
                    print(f"    Плитка {tile_id}: {count} эпох")
        print(f"{'='*70}\n")
    
    def shutdown(self):
        """Останавливает обработку и сохраняет результаты."""
        print("\n\n" + "=" * 70)
        print("ОСТАНОВКА ОБРАБОТКИ")
        print("=" * 70)
        
        # Останавливаем таймеры
        self.update_timer.stop()
        self.data_timer.stop()
        self.status_timer.stop()
        
        # Выводим статистику
        print(f"\nСтатистика обработки:")
        print(f"  Обработано маркеров: {self.processed_markers}")
        print(f"  Пропущено маркеров: {self.skipped_markers}")
        if self.processed_markers + self.skipped_markers > 0:
            success_rate = self.processed_markers / (self.processed_markers + self.skipped_markers) * 100
            print(f"  Процент успешной обработки: {success_rate:.1f}%")
        
        # Информация о буфере
        if self.buffer is not None:
            buffer_info = self.buffer.get_buffer_info()
            print(f"\nСостояние буфера:")
            print(f"  Размер: {buffer_info['size']} точек")
            print(f"  Длительность: {buffer_info['duration']:.2f} сек")
        
        # Сохраняем результаты
        if self.processor is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            filename = os.path.join(results_dir, f"eeg_epochs_{timestamp}.pkl")
            
            self.processor.save_results(filename)
            
            print("\nОбработка завершена. Результаты сохранены.")
            print(f"Для загрузки результатов используйте:")
            print(f"  import pickle")
            print(f"  with open('{filename}', 'rb') as f:")
            print(f"      results = pickle.load(f)")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция: запускает обработку ЭЭГ в реальном времени.
    Использует событийно-ориентированную модель с PyQt5 и pyqtgraph.
    """
    import sys
    
    # Создаём приложение Qt
    app = EEGProcessorApp(sys.argv)
    
    # Инициализируем компоненты
    if not app.initialize():
        return 1
    
    # Подключаем обработчик закрытия окна через сигналы Qt
    app.aboutToQuit.connect(app.shutdown)
    
    # Запускаем главный цикл событий Qt
    exit_code = app.exec_()
    
    # Сохраняем результаты при выходе
    app.shutdown()
    
    return exit_code


if __name__ == "__main__":
    import sys
    sys.exit(main())
