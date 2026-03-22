#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, resolve_byprop, StreamInlet

# --- Настройки "Идеального мозга" ---
TARGET_TILE_ID = "4"  # Плитка, на которую "смотрит" наш виртуальный человек (ID 4 - центр)
SRATE = 250  # Частота дискретизации (Гц)
N_CHANNELS = 21  # Количество доступных каналов ЭЭГ
NOISE_AMP = 10.0  # Амплитуда фонового шума ЭЭГ
P300_AMP = 15.0  # Амплитуда целевого отклика P300
P300_LATENCY_S = 0.3  # Задержка пика (300 мс)
P300_WIDTH_S = 0.05  # Ширина пика (50 мс)


def main():
    print("=== ЗАПУСК СИМУЛЯТОРА P300 ===")

    # 1. Создаем поток ЭЭГ, который будет читать анализатор
    # Имя 'EEG_Simulator' прописано в разрешенных потоках вашего анализатора
    info = StreamInfo("EEG_Simulator", "EEG", N_CHANNELS, SRATE, "float32", "eeg-simulator-neurospectr")
    outlet = StreamOutlet(info)
    print(f"LSL Outlet создан: EEG_Simulator ({N_CHANNELS} каналов, {SRATE} Гц)")

    # 2. Ищем поток маркеров от PsychoPy
    print("\nОжидаю запуск стимуляции (PsychoPy)...")
    marker_streams = resolve_byprop("type", "Markers")
    inlet_markers = StreamInlet(marker_streams[0])
    print(f"Поток маркеров найден! Начинаю трансляцию ЭЭГ.")
    print(f"Целевая плитка для P300: № {TARGET_TILE_ID}\n")

    # 3. Генерируем идеальный шаблон P300 (длиной 1 секунда)
    t = np.linspace(0, 1.0, SRATE)
    p300_template = P300_AMP * np.exp(-0.5 * ((t - P300_LATENCY_S) / P300_WIDTH_S) ** 2)
    template_len = len(p300_template)

    # Буфер для "будущего" сигнала (чтобы P300 разворачивался во времени)
    buffer_len = SRATE * 2
    upcoming_signal = np.zeros(buffer_len)

    chunk_size = int(SRATE / 20)  # Отправляем данные кусочками по 50 мс
    sleep_time = chunk_size / SRATE

    while True:
        # А) Проверяем маркеры (без зависания)
        marker, ts = inlet_markers.pull_sample(timeout=0.0)
        if marker is not None:
            marker_val = marker[0]
            # PsychoPy отправляет маркеры в формате "ID|event", например "4|on"
            if marker_val.endswith("|on"):
                tile_id = marker_val.split("|")[0]
                if tile_id == TARGET_TILE_ID:
                    print(f"🔥 Вспышка целевой плитки ({marker_val})! Генерирую P300...")
                    # Накладываем шаблон P300 на текущий шум в буфере
                    upcoming_signal[:template_len] += p300_template
                else:
                    print(f"Вспышка фоновой плитки ({marker_val}). Игнорирую.")

        # Б) Берем следующий кусок сигнала, добавляем шум и отдаем по всем 21 каналам
        chunk_base = upcoming_signal[:chunk_size]
        chunk = np.tile(chunk_base, (N_CHANNELS, 1)).T
        chunk += np.random.normal(0, NOISE_AMP, chunk.shape)  # Добавляем белый шум

        # В) Сдвигаем буфер
        upcoming_signal[:-chunk_size] = upcoming_signal[chunk_size:]
        upcoming_signal[-chunk_size:] = 0

        # Г) Отправляем в LSL и ждем
        outlet.push_chunk(chunk.tolist())
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()