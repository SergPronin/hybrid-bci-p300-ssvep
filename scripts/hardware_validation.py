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

import time
import numpy as np
from collections import deque
from typing import Optional

import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop

# Параметры по ТЗ
EEG_STREAM_TYPES = ("EEG", "Signal")
WINDOW_SEC = 3.0          # скользящее окно на экране и для ковариации (сек)
COV_UPDATE_INTERVAL = 2.0  # вывод матрицы ковариаций каждые 2 секунды


def find_eeg_stream():
    """Находит поток ЭЭГ в LSL (тип EEG или Signal)."""
    for stream_type in EEG_STREAM_TYPES:
        streams = resolve_byprop("type", stream_type, timeout=3)
        if streams:
            return streams[0]
    return None


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
    stream_name = eeg_info.name() if eeg_info.name() else "EEG"

    if srate <= 0:
        print("ОШИБКА: Частота дискретизации не определена.")
        return

    buffer_size = int(WINDOW_SEC * srate)
    cov_window_samples = int(WINDOW_SEC * srate)

    # Буфер: список строк, каждая строка — сэмпл по всем каналам (num_samples, num_channels)
    buffer = deque(maxlen=buffer_size)

    print(f"  ✓ Подключено: {stream_name}")
    print(f"    Каналов: {n_channels}, частота: {srate} Гц")
    print(f"\nВизуализация: последние {WINDOW_SEC} сек (изолированно по каналам)")
    print(f"Ковариация: окно {WINDOW_SEC} сек, вывод каждые {COV_UPDATE_INTERVAL} сек")
    print("Ctrl+C — выход\n")

    # Изолированная визуализация: N subplots по вертикали, по одному на канал
    fig, axes = plt.subplots(n_channels, 1, sharex=True, figsize=(12, max(2 * n_channels, 6)))
    if n_channels == 1:
        axes = [axes]
    fig.suptitle(f"Сырой сигнал ЭЭГ — {stream_name} (канал → свой график, без усреднения)")
    fig.subplots_adjust(hspace=0.35)

    time_axis = np.linspace(0, WINDOW_SEC, buffer_size)
    lines = []
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_channels, 10)))
    for ch in range(n_channels):
        ax = axes[ch]
        line, = ax.plot([], [], color=colors[ch % 10], linewidth=0.8)
        lines.append(line)
        ax.set_ylabel(f"Ch{ch + 1} (мкВ)")
        ax.grid(True, alpha=0.3)
        ax.autoscale(enable=True, axis="y")
    axes[-1].set_xlabel("Время (сек)")

    plt.ion()
    plt.show()

    last_cov_time = time.time()
    try:
        while True:
            chunk, timestamps = inlet.pull_chunk(timeout=0.05, max_samples=256)
            if chunk and timestamps:
                arr = np.array(chunk)
                # LSL может отдавать (samples, channels) или (channels, samples)
                if arr.shape[1] != n_channels:
                    if arr.shape[0] == n_channels:
                        arr = arr.T
                    else:
                        plt.pause(0.02)
                        continue
                for i in range(len(arr)):
                    buffer.append(arr[i, :])

            if len(buffer) < 2:
                plt.pause(0.02)
                continue

            data = np.array(buffer)
            t = np.linspace(
                WINDOW_SEC - len(data) / srate,
                WINDOW_SEC,
                len(data),
            )

            # Каждый график — только свой канал (изолированная визуализация)
            for ch in range(n_channels):
                lines[ch].set_data(t, data[:, ch])
                ax = axes[ch]
                y = data[:, ch]
                if len(y) > 0:
                    margin = (np.nanmax(y) - np.nanmin(y)) * 0.1
                    if margin == 0:
                        margin = 1.0
                    ax.set_ylim(np.nanmin(y) - margin, np.nanmax(y) + margin)

            # Матрица ковариаций каждые 2 секунды (буфер за последние 3 сек)
            now = time.time()
            if now - last_cov_time >= COV_UPDATE_INTERVAL:
                last_cov_time = now
                cov_data = data[-cov_window_samples:] if len(data) >= cov_window_samples else data
                # Центрируем по каналам для ковариации
                cov_data = cov_data - np.mean(cov_data, axis=0)
                cov_matrix = np.cov(cov_data.T)

                print("\n" + "-" * 60)
                print("МАТРИЦА КОВАРИАЦИЙ X (проверка наводок между каналами)")
                print("-" * 60)
                print("Диагональ (x11, x22, ...): мощность (дисперсия) на канале.")
                print("Вне диагонали (x12, x21, ...): наводка между каналами.")
                print("При сигнале на 1 канале: x11 большое, остальные → 0.\n")
                np.set_printoptions(precision=4, suppress=True)
                print(cov_matrix)
                print("-" * 60)

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.02)

    except KeyboardInterrupt:
        print("\nВыход.")
    finally:
        plt.close()


if __name__ == "__main__":
    main()
