from pylsl import StreamInlet, resolve_bypred, resolve_byprop
import numpy as np
import time

print("Поиск потоков...")

# Ищем поток маркеров от вашей программы
marker_streams = resolve_byprop('name', 'BCI_StimMarkers', timeout=5)
if not marker_streams:
    print("Поток маркеров не найден!")
    exit()
marker_inlet = StreamInlet(marker_streams[0])
print(f"✓ Найден поток маркеров: {marker_streams[0].name()}")

# Ищем поток ЭЭГ от Нейроспектра (название может отличаться)
print("Поиск потока ЭЭГ от Нейроспектра...")
eeg_streams = resolve_bypred("type='EEG'", timeout=10)  # или name='Neurospectr'
if not eeg_streams:
    print("Поток ЭЭГ не найден!")
    print("Убедитесь, что в ПО Нейроспектра включена LSL-трансляция")
    exit()

eeg_inlet = StreamInlet(eeg_streams[0])
print(f"✓ Найден поток ЭЭГ: {eeg_streams[0].name()}")
print(f"  Частота дискретизации: {eeg_streams[0].nominal_srate()} Гц")
print(f"  Количество каналов: {eeg_streams[0].channel_count()}")

print("\nНачинаю прием синхронизированных данных...")
print("-" * 50)

# Буфер для ЭЭГ
eeg_buffer = []
marker_buffer = []

try:
    while True:
        # Получаем блок ЭЭГ (по 100 мс)
        eeg_samples, eeg_timestamps = eeg_inlet.pull_chunk(timeout=0.0, max_samples=25)

        # Получаем все новые маркеры
        markers = []
        while True:
            marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
            if marker is None:
                break
            markers.append((marker_ts, marker[0]))

        # Выводим маркеры по мере поступления
        for ts, marker in markers:
            print(f"⏱️ {ts:.3f} | Маркер: {marker}")

        # Если есть ЭЭГ, можно что-то с ним делать
        if eeg_samples and len(eeg_samples) > 0:
            # Здесь будет ваш анализ
            pass

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nОстановлено пользователем")