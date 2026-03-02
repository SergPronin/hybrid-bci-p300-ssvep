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

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
from pylsl import StreamInlet, resolve_byprop, local_clock

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
UPDATE_INTERVAL = 0.05  # Обновление графика каждые 50 мс (более плавно)
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
    
    Хранит последние N секунд данных, чтобы можно было извлечь эпоху
    при получении маркера.
    
    Использует относительное время от момента получения первого сэмпла
    для синхронизации с маркерами.
    """
    
    def __init__(self, sampling_rate: float, num_channels: int, duration: float = BUFFER_DURATION):
        """
        Инициализация буфера.
        
        Args:
            sampling_rate: Частота дискретизации
            num_channels: Количество каналов
            duration: Длительность буфера в секундах
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        self.buffer_size = int(duration * sampling_rate)
        
        # Буфер данных: список кортежей (relative_time, data_array)
        # relative_time - относительное время в секундах от первого сэмпла
        # data_array имеет форму (num_channels,)
        self.buffer: deque = deque(maxlen=self.buffer_size)
        
        # Время первого сэмпла (используем системное время для синхронизации)
        self.first_sample_time: Optional[float] = None
        
        print(f"Инициализирован буфер ЭЭГ:")
        print(f"  Размер буфера: {self.buffer_size} точек ({duration} сек)")
        print(f"  Используется относительное время от первого сэмпла")
    
    def add_sample(self, timestamp: float, data: np.ndarray, local_time: float):
        """
        Добавляет новый сэмпл в буфер.
        
        Args:
            timestamp: Временная метка LSL (не используется, только для совместимости)
            data: Массив формы (num_channels,)
            local_time: Локальное системное время (time.time() или local_clock())
        """
        if data.shape != (self.num_channels,):
            print(f"ОШИБКА: Неверный размер данных. Ожидается ({self.num_channels},), "
                  f"получено {data.shape}")
            return
        
        # Запоминаем время первого сэмпла
        if self.first_sample_time is None:
            self.first_sample_time = local_time
        
        # Вычисляем относительное время от первого сэмпла
        relative_time = local_time - self.first_sample_time
        
        self.buffer.append((relative_time, data.copy()))
    
    def get_first_sample_time(self) -> Optional[float]:
        """Возвращает время первого сэмпла (для синхронизации с маркерами)."""
        return self.first_sample_time
    
    def is_ready(self, marker_relative_time: float, pre_stim: float, post_stim: float) -> Tuple[bool, str]:
        """
        Проверяет, достаточно ли данных в буфере для извлечения эпохи.
        
        Args:
            marker_relative_time: Относительное время маркера (от момента первого сэмпла ЭЭГ)
            pre_stim: Время до стимула в секундах
            post_stim: Время после стимула в секундах
        
        Returns:
            Кортеж (is_ready, message) - готов ли буфер и сообщение о состоянии
        """
        if len(self.buffer) < 2:
            return False, f"Буфер пуст (нужно минимум 2 точки, есть {len(self.buffer)})"
        
        if self.first_sample_time is None:
            return False, "Буфер не инициализирован (нет первого сэмпла)"
        
        # Вычисляем временные границы эпохи (в относительном времени)
        epoch_start_time = marker_relative_time - pre_stim
        epoch_end_time = marker_relative_time + post_stim
        
        # Получаем временные границы буфера
        buffer_times = np.array([ts for ts, _ in self.buffer])
        buffer_start = buffer_times[0]
        buffer_end = buffer_times[-1]
        buffer_duration = buffer_end - buffer_start
        
        # Проверяем наличие данных до маркера
        if buffer_start > epoch_start_time:
            needed_before = epoch_start_time - buffer_start
            return False, f"Недостаточно данных ДО маркера (нужно {pre_stim:.3f} сек до, буфер начинается на {needed_before:.3f} сек позже)"
        
        # Проверяем наличие данных после маркера
        if buffer_end < epoch_end_time:
            needed_after = epoch_end_time - buffer_end
            return False, f"Недостаточно данных ПОСЛЕ маркера (нужно {post_stim:.3f} сек после, буфер заканчивается на {needed_after:.3f} сек раньше)"
        
        return True, f"Буфер готов (длительность: {buffer_duration:.2f} сек, точек: {len(self.buffer)})"
    
    def get_buffer_info(self) -> dict:
        """Возвращает информацию о текущем состоянии буфера."""
        if len(self.buffer) < 2:
            return {
                'size': len(self.buffer),
                'duration': 0.0,
                'start_time': None,
                'end_time': None,
            }
        
        buffer_times = np.array([ts for ts, _ in self.buffer])
        return {
            'size': len(self.buffer),
            'duration': buffer_times[-1] - buffer_times[0],
            'start_time': buffer_times[0],
            'end_time': buffer_times[-1],
        }
    
    def extract_epoch(self, marker_relative_time: float, pre_stim: float, post_stim: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Извлекает эпоху вокруг маркера.
        
        Args:
            marker_relative_time: Относительное время маркера (от момента первого сэмпла ЭЭГ)
            pre_stim: Время до стимула в секундах
            post_stim: Время после стимула в секундах
        
        Returns:
            Кортеж (epoch_data, time_vector) или None, если данных недостаточно
            epoch_data: массив формы (num_channels, num_samples)
            time_vector: массив времени в секундах относительно стимула
        """
        # Проверяем готовность буфера
        is_ready, message = self.is_ready(marker_relative_time, pre_stim, post_stim)
        if not is_ready:
            return None
        
        # Вычисляем количество точек
        num_samples_pre = int(pre_stim * self.sampling_rate)
        num_samples_post = int(post_stim * self.sampling_rate)
        num_samples_total = num_samples_pre + num_samples_post
        
        # Создаём массив для эпохи
        epoch_data = np.zeros((self.num_channels, num_samples_total))
        time_vector = np.linspace(-pre_stim, post_stim, num_samples_total)
        
        # Заполняем эпоху данными из буфера
        # Используем линейную интерполяцию для точного извлечения
        buffer_times = np.array([ts for ts, _ in self.buffer])
        buffer_data = np.array([data for _, data in self.buffer])
        
        # Интерполируем данные для каждого канала
        # Используем относительное время
        for ch in range(self.num_channels):
            epoch_data[ch, :] = np.interp(
                time_vector + marker_relative_time,  # Относительное время
                buffer_times,  # Временные метки в буфере (тоже относительные)
                buffer_data[:, ch]  # Данные канала
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
# ФУНКЦИИ ДЛЯ ВИЗУАЛИЗАЦИИ
# ============================================================================

class RealtimePlotter:
    """
    Класс для отображения графиков в реальном времени.
    
    Показывает усреднённые эпохи для всех 9 плиток в виде сетки графиков.
    """
    
    def __init__(self, sampling_rate: float, num_channels: int):
        """
        Инициализация визуализации.
        
        Args:
            sampling_rate: Частота дискретизации
            num_channels: Количество каналов ЭЭГ
        """
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels
        
        # Создаём фигуру с сеткой 3x3 для 9 плиток
        self.fig, self.axes = plt.subplots(3, 3, figsize=(15, 12))
        self.fig.suptitle('Усреднённые эпохи ЭЭГ по плиткам (обновление в реальном времени)', 
                          fontsize=14, fontweight='bold')
        
        # Создаём вектор времени для эпохи
        self.time_vector = np.linspace(-EPOCH_PRE_STIM, EPOCH_POST_STIM, 
                                       int(EPOCH_TOTAL * sampling_rate))
        
        # Инициализируем графики для каждой плитки
        self.lines = {}
        self.texts = {}
        for tile_id in range(NUM_TILES):
            row = tile_id // 3
            col = tile_id % 3
            ax = self.axes[row, col]
            
            # Создаём пустые линии для каждого канала
            lines = []
            for ch in range(num_channels):
                line, = ax.plot([], [], label=f'Канал {ch+1}', alpha=0.7, linewidth=1.5)
                lines.append(line)
            
            self.lines[tile_id] = lines
            
            # Настраиваем оси
            ax.set_xlim(-EPOCH_PRE_STIM, EPOCH_POST_STIM)
            ax.set_ylim(-50, 50)  # Начальные пределы, будут автоматически подстраиваться
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Стимул')
            ax.set_xlabel('Время (сек)', fontsize=9)
            ax.set_ylabel('Амплитуда (мкВ)', fontsize=9)
            ax.set_title(f'Плитка {tile_id}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='upper right')
            
            # Текст с количеством эпох
            text = ax.text(0.02, 0.98, 'Эпох: 0', transform=ax.transAxes,
                          fontsize=8, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            self.texts[tile_id] = text
        
        plt.tight_layout()
        plt.ion()  # Интерактивный режим
        plt.show()
    
    def update(self, processor: EpochProcessor):
        """
        Обновляет графики на основе текущих данных процессора.
        
        Args:
            processor: Экземпляр EpochProcessor с накопленными данными
        """
        averaged = processor.get_averaged_epochs()
        counts = processor.get_epoch_counts()
        
        for tile_id in range(NUM_TILES):
            row = tile_id // 3
            col = tile_id % 3
            ax = self.axes[row, col]
            
            if averaged[tile_id] is not None:
                # Обновляем данные для каждого канала
                epoch_data = averaged[tile_id]
                
                # Если несколько каналов, показываем средний по каналам
                if self.num_channels > 1:
                    mean_signal = np.mean(epoch_data, axis=0)
                    self.lines[tile_id][0].set_data(self.time_vector, mean_signal)
                    # Скрываем остальные линии, если они есть
                    for i in range(1, len(self.lines[tile_id])):
                        self.lines[tile_id][i].set_data([], [])
                else:
                    # Один канал
                    self.lines[tile_id][0].set_data(self.time_vector, epoch_data[0, :])
                
                # Автоматически подстраиваем масштаб по Y
                y_data = epoch_data.flatten()
                if len(y_data) > 0:
                    y_min, y_max = np.min(y_data), np.max(y_data)
                    margin = (y_max - y_min) * 0.1
                    ax.set_ylim(y_min - margin, y_max + margin)
            else:
                # Нет данных - очищаем график
                for line in self.lines[tile_id]:
                    line.set_data([], [])
            
            # Обновляем текст с количеством эпох
            count = counts[tile_id]
            self.texts[tile_id].set_text(f'Эпох: {count}')
        
        # Обновляем отображение
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Закрывает окно графика."""
        plt.close(self.fig)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """
    Главная функция: запускает обработку ЭЭГ в реальном времени.
    """
    print("=" * 70)
    print("ОБРАБОТКА ЭЭГ ДАННЫХ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 70)
    print("\nЭтот скрипт будет:")
    print("  1. Подключаться к потокам маркеров и ЭЭГ")
    print("  2. Извлекать эпохи при каждом событии 'on'")
    print("  3. Фильтровать и обрабатывать данные")
    print("  4. Показывать усреднённые сигналы в реальном времени")
    print("  5. Сохранять результаты при остановке (Ctrl+C)\n")
    
    # Шаг 1: Находим потоки
    marker_info, eeg_info = find_streams()
    if marker_info is None or eeg_info is None:
        print("\nНе удалось найти необходимые потоки. Выход.")
        return
    
    # Шаг 2: Создаём inlets для чтения данных
    print("\nПодключение к потокам...")
    marker_inlet = StreamInlet(marker_info)
    eeg_inlet = StreamInlet(eeg_info)
    
    # Получаем параметры потока ЭЭГ
    eeg_sampling_rate = eeg_info.nominal_srate()
    eeg_num_channels = eeg_info.channel_count()
    
    if eeg_sampling_rate == 0:
        print("ОШИБКА: Частота дискретизации ЭЭГ не определена!")
        return
    
    print(f"  ✓ Подключено к потоку маркеров")
    print(f"  ✓ Подключено к потоку ЭЭГ ({eeg_num_channels} каналов, {eeg_sampling_rate} Гц)")
    
    print("\n  ВАЖНО: Используется относительное время для синхронизации")
    print("         Временные метки будут синхронизированы относительно момента получения первого сэмпла ЭЭГ")
    
    # Шаг 3: Инициализируем компоненты обработки
    print("\nИнициализация компонентов обработки...")
    buffer = EEGBuffer(eeg_sampling_rate, eeg_num_channels)
    processor = EpochProcessor(eeg_sampling_rate, eeg_num_channels)
    plotter = RealtimePlotter(eeg_sampling_rate, eeg_num_channels)
    
    print("  ✓ Все компоненты инициализированы")
    
    # Шаг 3.5: Накапливаем данные в буфере перед началом обработки
    # Нужно накопить достаточно данных, чтобы можно было извлечь эпоху после маркера
    # Эпоха требует EPOCH_POST_STIM секунд после маркера, поэтому нужно накопить минимум столько
    min_buffer_duration = EPOCH_POST_STIM + 0.5  # Добавляем запас 0.5 сек
    print(f"\nНакопление данных в буфере (минимум {min_buffer_duration:.1f} сек для эпох)...")
    buffer_ready = False
    start_time = time.time()
    
    # Также накапливаем маркеры в очереди, чтобы обработать их позже
    marker_queue = []
    
    while not buffer_ready:
        # Читаем данные ЭЭГ
        chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1, max_samples=100)
        if chunk and timestamps:
            chunk_array = np.array(chunk)
            if chunk_array.shape[1] == eeg_num_channels:
                pass
            elif chunk_array.shape[0] == eeg_num_channels:
                chunk_array = chunk_array.T
            else:
                continue
            
            # Используем локальное время для синхронизации
            current_local_time = local_clock()
            
            for i, ts in enumerate(timestamps):
                sample = chunk_array[i, :]
                # Передаём локальное время для синхронизации
                buffer.add_sample(ts, sample, current_local_time)
        
        # Читаем маркеры, но пока не обрабатываем - сохраняем в очередь
        marker_sample, marker_timestamp = marker_inlet.pull_sample(timeout=0.0)
        if marker_sample is not None:
            parsed = parse_marker(marker_sample[0])
            if parsed is not None:
                tile_id, event = parsed
                if event == "on" and 0 <= tile_id < NUM_TILES:
                    marker_local_time = local_clock()
                    marker_queue.append((tile_id, marker_local_time))
                    if len(marker_queue) == 1:
                        print(f"  Получен первый маркер (плитка {tile_id}), ждём накопления данных...")
        
        # Проверяем готовность буфера
        buffer_info = buffer.get_buffer_info()
        if buffer_info['duration'] >= min_buffer_duration:
            buffer_ready = True
            print(f"  ✓ Буфер готов: {buffer_info['duration']:.2f} сек данных, {buffer_info['size']} точек")
            if len(marker_queue) > 0:
                print(f"  В очереди {len(marker_queue)} маркеров для обработки")
        elif time.time() - start_time > 15:
            # Если прошло 15 секунд, но буфер не готов, продолжаем с предупреждением
            print(f"  ⚠ Буфер частично готов: {buffer_info['duration']:.2f} сек данных")
            print(f"     Продолжаем, но некоторые маркеры могут быть пропущены")
            buffer_ready = True
    
    # Получаем время первого сэмпла ЭЭГ для синхронизации маркеров
    eeg_first_time = buffer.get_first_sample_time()
    if eeg_first_time is None:
        print("ОШИБКА: Не удалось получить время первого сэмпла ЭЭГ!")
        return
    
    print("\n" + "=" * 70)
    print("НАЧАЛО ОБРАБОТКИ")
    print("Нажмите Ctrl+C для остановки и сохранения результатов")
    print("=" * 70 + "\n")
    
    # Шаг 4: Главный цикл обработки
    last_update_time = time.time()
    skipped_markers = 0  # Счётчик пропущенных маркеров
    processed_markers = 0  # Счётчик обработанных маркеров
    main.start_time = time.time()  # Время начала работы для статуса
    main.last_status_time = time.time()  # Время последнего статуса
    
    # Если есть маркеры в очереди, ждём накопления данных после последнего маркера
    if marker_queue:
        print(f"\nОжидание накопления данных для обработки {len(marker_queue)} маркеров из очереди...")
        last_marker_time = max(marker_local_time for _, marker_local_time in marker_queue)
        last_marker_relative_time = last_marker_time - eeg_first_time
        
        # Нужно накопить данные до момента: последний_маркер + EPOCH_POST_STIM
        required_buffer_end = last_marker_relative_time + EPOCH_POST_STIM + 0.1  # +0.1 сек запас
        
        wait_start = time.time()
        while True:
            # Продолжаем читать данные ЭЭГ
            chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1, max_samples=100)
            if chunk and timestamps:
                chunk_array = np.array(chunk)
                if chunk_array.shape[1] == eeg_num_channels:
                    pass
                elif chunk_array.shape[0] == eeg_num_channels:
                    chunk_array = chunk_array.T
                else:
                    continue
                
                current_local_time = local_clock()
                for i, ts in enumerate(timestamps):
                    sample = chunk_array[i, :]
                    buffer.add_sample(ts, sample, current_local_time)
            
            # Проверяем, достаточно ли данных
            buffer_info = buffer.get_buffer_info()
            if buffer_info['end_time'] is not None and buffer_info['end_time'] >= required_buffer_end:
                print(f"  ✓ Данных достаточно (буфер до {buffer_info['end_time']:.2f} сек, требуется до {required_buffer_end:.2f} сек)")
                break
            elif time.time() - wait_start > 10:
                print(f"  ⚠ Таймаут ожидания. Продолжаем с текущим буфером (до {buffer_info.get('end_time', 0):.2f} сек)")
                break
    
    # Обрабатываем маркеры из очереди (накопленные во время инициализации)
    if marker_queue:
        print(f"\n{'='*70}")
        print(f"ОБРАБОТКА {len(marker_queue)} МАРКЕРОВ ИЗ ОЧЕРЕДИ")
        print(f"{'='*70}")
        for idx, (tile_id, marker_local_time) in enumerate(marker_queue, 1):
            marker_relative_time = marker_local_time - eeg_first_time
            is_ready, ready_message = buffer.is_ready(
                marker_relative_time, 
                EPOCH_PRE_STIM, 
                EPOCH_POST_STIM
            )
            
            print(f"\n[{idx}/{len(marker_queue)}] Маркер плитки {tile_id} (t={marker_relative_time:.3f}s)")
            
            if is_ready:
                epoch_result = buffer.extract_epoch(
                    marker_relative_time, 
                    EPOCH_PRE_STIM, 
                    EPOCH_POST_STIM
                )
                
                if epoch_result is not None:
                    epoch_data, time_vector = epoch_result
                    processor.add_epoch(tile_id, epoch_data)
                    processed_markers += 1
                    epoch_counts = processor.get_epoch_counts()
                    count_for_tile = epoch_counts[tile_id]
                    print(f"  ✓ Обработана успешно | Эпох для плитки: {count_for_tile}")
                else:
                    skipped_markers += 1
                    print(f"  ✗ Пропущена: ошибка извлечения эпохи")
            else:
                skipped_markers += 1
                buffer_info = buffer.get_buffer_info()
                print(f"  ✗ Пропущена: {ready_message}")
                if buffer_info['duration'] > 0:
                    print(f"    Буфер: {buffer_info['duration']:.2f} сек, "
                          f"от {buffer_info.get('start_time', 0):.3f} до {buffer_info.get('end_time', 0):.3f} сек")
        
        print(f"\n{'='*70}")
        print(f"Обработано из очереди: {processed_markers}, пропущено: {skipped_markers}")
        print(f"{'='*70}\n")
    
    try:
        while True:
            # Читаем новые маркеры
            marker_sample, marker_timestamp = marker_inlet.pull_sample(timeout=0.0)
            if marker_sample is not None:
                # Парсим маркер
                parsed = parse_marker(marker_sample[0])
                if parsed is not None:
                    tile_id, event = parsed
                    
                    # Обрабатываем только события "on" (загорание плитки)
                    if event == "on" and 0 <= tile_id < NUM_TILES:
                        # Конвертируем временную метку маркера в относительное время
                        # Используем локальное время для синхронизации
                        marker_local_time = local_clock()
                        marker_relative_time = marker_local_time - eeg_first_time
                        
                        # Проверяем готовность буфера перед извлечением
                        is_ready, ready_message = buffer.is_ready(
                            marker_relative_time, 
                            EPOCH_PRE_STIM, 
                            EPOCH_POST_STIM
                        )
                        
                        if is_ready:
                            # Извлекаем эпоху из буфера
                            epoch_result = buffer.extract_epoch(
                                marker_relative_time, 
                                EPOCH_PRE_STIM, 
                                EPOCH_POST_STIM
                            )
                            
                            if epoch_result is not None:
                                epoch_data, time_vector = epoch_result
                                # Добавляем эпоху в процессор
                                processor.add_epoch(tile_id, epoch_data)
                                processed_markers += 1
                                
                                # Получаем количество эпох для этой плитки
                                epoch_counts = processor.get_epoch_counts()
                                count_for_tile = epoch_counts[tile_id]
                                
                                # Подробный вывод для каждого маркера
                                print(f"✓ [t={marker_relative_time:.3f}s] Маркер плитки {tile_id} обработана | "
                                      f"Эпох для плитки: {count_for_tile} | "
                                      f"Всего обработано: {processed_markers}, пропущено: {skipped_markers}")
                            else:
                                skipped_markers += 1
                                print(f"✗ [t={marker_relative_time:.3f}s] Маркер плитки {tile_id} пропущен: ошибка извлечения эпохи")
                        else:
                            skipped_markers += 1
                            # Показываем все пропуски с подробной информацией
                            buffer_info = buffer.get_buffer_info()
                            print(f"✗ [t={marker_relative_time:.3f}s] Маркер плитки {tile_id} пропущен: {ready_message}")
                            if buffer_info['duration'] > 0:
                                print(f"    Буфер: {buffer_info['duration']:.2f} сек, "
                                      f"от {buffer_info.get('start_time', 0):.3f} до {buffer_info.get('end_time', 0):.3f} сек")
            
            # Читаем новые данные ЭЭГ и добавляем в буфер
            chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.0, max_samples=100)
            if chunk and timestamps:
                # chunk - это список списков, каждый внутренний список - один сэмпл
                # Преобразуем в numpy массив формы (num_samples, num_channels)
                chunk_array = np.array(chunk)
                
                # Если данные в формате (num_channels, num_samples), транспонируем
                if chunk_array.shape[1] == eeg_num_channels:
                    # Формат правильный: (num_samples, num_channels)
                    pass
                elif chunk_array.shape[0] == eeg_num_channels:
                    # Нужно транспонировать
                    chunk_array = chunk_array.T
                else:
                    print(f"ОШИБКА: Неожиданная форма данных: {chunk_array.shape}")
                    continue
                
                # Используем локальное время для синхронизации
                current_local_time = local_clock()
                
                # Добавляем каждый сэмпл в буфер
                samples_added = 0
                for i, ts in enumerate(timestamps):
                    sample = chunk_array[i, :]  # Форма (num_channels,)
                    buffer.add_sample(ts, sample, current_local_time)
                    samples_added += 1
                
                # Периодически выводим информацию о полученных данных
                if not hasattr(main, 'last_eeg_log_time'):
                    main.last_eeg_log_time = 0
                
                current_time = time.time()
                if current_time - main.last_eeg_log_time >= 5.0:  # Каждые 5 секунд
                    buffer_info = buffer.get_buffer_info()
                    print(f"📊 Получено {samples_added} сэмплов ЭЭГ | "
                          f"Буфер: {buffer_info['size']} точек, {buffer_info['duration']:.2f} сек")
                    main.last_eeg_log_time = current_time
            
            # Обновляем график периодически (часто для плавности)
            current_time = time.time()
            if current_time - last_update_time >= UPDATE_INTERVAL:
                plotter.update(processor)
                last_update_time = current_time
            
            # Выводим подробный статус периодически
            if not hasattr(main, 'last_status_time'):
                main.last_status_time = current_time
            
            if current_time - main.last_status_time >= STATUS_UPDATE_INTERVAL:
                # Подробная информация о состоянии
                buffer_info = buffer.get_buffer_info()
                epoch_counts = processor.get_epoch_counts()
                total_epochs = sum(epoch_counts.values())
                
                print(f"\n{'='*70}")
                print(f"СТАТУС [t={current_time - start_time:.1f}s работы]")
                print(f"  Буфер ЭЭГ: {buffer_info['size']} точек, "
                      f"{buffer_info['duration']:.2f} сек данных")
                if buffer_info['start_time'] is not None:
                    print(f"    Временной диапазон: {buffer_info['start_time']:.3f} - {buffer_info['end_time']:.3f} сек")
                print(f"  Обработано маркеров: {processed_markers} | Пропущено: {skipped_markers}")
                if total_epochs > 0:
                    success_rate = processed_markers / (processed_markers + skipped_markers) * 100 if (processed_markers + skipped_markers) > 0 else 0
                    print(f"  Успешность: {success_rate:.1f}%")
                    print(f"  Всего эпох накоплено: {total_epochs}")
                    print(f"  Эпохи по плиткам:")
                    for tile_id in range(NUM_TILES):
                        count = epoch_counts[tile_id]
                        if count > 0:
                            print(f"    Плитка {tile_id}: {count} эпох")
                print(f"{'='*70}\n")
                
                main.last_status_time = current_time
            
            # Небольшая задержка, чтобы не нагружать CPU
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("ОСТАНОВКА ОБРАБОТКИ")
        print("=" * 70)
        
        # Выводим статистику
        print(f"\nСтатистика обработки:")
        print(f"  Обработано маркеров: {processed_markers}")
        print(f"  Пропущено маркеров: {skipped_markers}")
        if processed_markers + skipped_markers > 0:
            success_rate = processed_markers / (processed_markers + skipped_markers) * 100
            print(f"  Процент успешной обработки: {success_rate:.1f}%")
        
        # Информация о буфере
        buffer_info = buffer.get_buffer_info()
        print(f"\nСостояние буфера:")
        print(f"  Размер: {buffer_info['size']} точек")
        print(f"  Длительность: {buffer_info['duration']:.2f} сек")
        
        # Сохраняем результаты
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(results_dir, f"eeg_epochs_{timestamp}.pkl")
        
        processor.save_results(filename)
        
        # Закрываем график
        plotter.close()
        
        print("\nОбработка завершена. Результаты сохранены.")
        print(f"Для загрузки результатов используйте:")
        print(f"  import pickle")
        print(f"  with open('{filename}', 'rb') as f:")
        print(f"      results = pickle.load(f)")


if __name__ == "__main__":
    main()
