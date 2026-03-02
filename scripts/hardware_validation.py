#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аппаратная валидация ЭЭГ: сырой сигнал и матрица ковариаций.

Скрипт для проверки качества оборудования:
1. Читает LSL-поток ЭЭГ
2. Отрисовывает сырой сигнал бегущей волной в реальном времени (без усреднений)
3. Выводит матрицу ковариаций (x11, x22, x12 и т.д.) для проверки отсутствия
   аппаратных наводок между каналами

Запуск:
    python scripts/hardware_validation.py

Перед запуском: включите Нейроспектр (или другой источник ЭЭГ) с LSL-трансляцией.
Для проверки каналов: подайте синусоиду на один канал — остальные должны быть
независимы (низкая ковариация между каналами).
"""

import time
import numpy as np
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop

# Параметры
EEG_STREAM_TYPES = ("EEG", "Signal")
DISPLAY_DURATION = 5.0   # секунд на экране
COV_UPDATE_INTERVAL = 1.0  # вывод матрицы ковариаций каждую секунду
COV_WINDOW = 3.0  # окно для расчёта ковариации (сек)


def find_eeg_stream():
    """Находит поток ЭЭГ в LSL."""
    for stream_type in EEG_STREAM_TYPES:
        streams = resolve_byprop("type", stream_type, timeout=3)
        if streams:
            return streams[0]
    return None


def main():
    print("=" * 60)
    print("АППАРАТНАЯ ВАЛИДАЦИЯ ЭЭГ")
    print("=" * 60)
    print("\nПоиск LSL-потока ЭЭГ...")

    eeg_info = find_eeg_stream()
    if eeg_info is None:
        print("ОШИБКА: Поток ЭЭГ не найден. Запустите Нейроспектр с LSL.")
        return

    inlet = StreamInlet(eeg_info)
    srate = eeg_info.nominal_srate()
    n_channels = eeg_info.channel_count()
    stream_name = eeg_info.name() if eeg_info.name() else "EEG"

    if srate <= 0:
        print("ОШИБКА: Частота дискретизации не определена.")
        return

    buffer_size = int(DISPLAY_DURATION * srate)
    cov_window_samples = int(COV_WINDOW * srate)

    # Буфер: (num_samples, num_channels)
    buffer = deque(maxlen=buffer_size)

    print(f"  ✓ Подключено: {stream_name}")
    print(f"    Каналов: {n_channels}, частота: {srate} Гц")
    print(f"\nОтрисовка: последние {DISPLAY_DURATION} сек | Ковариация: окно {COV_WINDOW} сек")
    print("Ctrl+C — выход\n")

    # График
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, DISPLAY_DURATION)
    ax.set_xlabel("Время (сек)")
    ax.set_ylabel("Амплитуда (мкВ)")
    ax.set_title(f"Сырой сигнал ЭЭГ — {stream_name} (без усреднений)")
    ax.grid(True, alpha=0.3)

    # Линии для каждого канала (с вертикальным смещением для читаемости)
    time_axis = np.linspace(0, DISPLAY_DURATION, buffer_size)
    lines = []
    offsets = np.linspace(0, (n_channels - 1) * 80, n_channels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_channels, 10)))
    for ch in range(n_channels):
        line, = ax.plot([], [], color=colors[ch % 10], label=f"Ch{ch+1}", linewidth=0.8)
        lines.append(line)
    ax.legend(loc="upper right", fontsize=8, ncol=min(n_channels, 5))
    plt.ion()
    plt.show()

    last_cov_time = time.time()
    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=0.05, max_samples=256)
            if chunk and timestamps:
                arr = np.array(chunk)
                if arr.shape[1] != n_channels:
                    arr = arr.T if arr.shape[0] == n_channels else arr
                for i in range(len(arr)):
                    buffer.append(arr[i, :])

            if len(buffer) < 2:
                plt.pause(0.02)
                continue

            data = np.array(buffer)
            t = np.linspace(
                DISPLAY_DURATION - len(data) / srate,
                DISPLAY_DURATION,
                len(data),
            )

            # Смещение каналов для визуализации
            for ch in range(n_channels):
                lines[ch].set_data(t, data[:, ch] + offsets[ch])

            # Автомасштаб по Y
            y_min = data.min() + offsets.min()
            y_max = data.max() + offsets.max()
            margin = (y_max - y_min) * 0.1 or 1
            ax.set_ylim(y_min - margin, y_max + margin)

            # Матрица ковариаций
            now = time.time()
            if now - last_cov_time >= COV_UPDATE_INTERVAL:
                last_cov_time = now
                cov_data = data[-cov_window_samples:] if len(data) >= cov_window_samples else data
                cov_data = cov_data - np.mean(cov_data, axis=0)
                cov_matrix = np.cov(cov_data.T)

                print("\n" + "-" * 50)
                print("МАТРИЦА КОВАРИАЦИЙ (проверка наводок между каналами)")
                print("-" * 50)
                if n_channels == 1:
                    print(f"  x11 (дисперсия Ch1): {cov_matrix[0, 0]:.2f}")
                elif n_channels == 2:
                    x11, x22 = cov_matrix[0, 0], cov_matrix[1, 1]
                    x12 = cov_matrix[0, 1]
                    print(f"  x11 (дисперсия Ch1): {x11:.2f}")
                    print(f"  x22 (дисперсия Ch2): {x22:.2f}")
                    print(f"  x12 (ковариация Ch1-Ch2): {x12:.2f}")
                    if x11 > 0 and x22 > 0:
                        corr = x12 / np.sqrt(x11 * x22)
                        print(f"  Корреляция Ch1-Ch2: {corr:.4f}")
                        print("  (≈0 при независимых каналах, ≈±1 при наводках)")
                else:
                    np.set_printoptions(precision=2, suppress=True)
                    print(cov_matrix)
                    print("  Диагональ — дисперсии каналов, вне диагонали — ковариации.")
                print("-" * 50)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.02)

    except KeyboardInterrupt:
        print("\nВыход.")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
