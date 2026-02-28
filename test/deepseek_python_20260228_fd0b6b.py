import time

from pylsl import resolve_streams, resolve_byprop, StreamInlet, resolve_bypred

import time
import numpy as np

print("🔍 Поиск LSL потоков...")
print("=" * 50)

# Показываем все доступные потоки
print("📋 Текущие LSL потоки в системе:")
streams = resolve_streams(wait_time=2.0)
if streams:
    for i, s in enumerate(streams):
        print(f"  {i + 1}. {s.name()} (тип: {s.type()})")
else:
    print("  Нет активных потоков")

print("\n" + "=" * 50)

# 1. Ищем поток маркеров
print("📌 Поиск потока маркеров 'BCI_StimMarkers'...")
marker_streams = resolve_byprop('name', 'BCI_StimMarkers', timeout=5)

if marker_streams:
    print(f"✅ Найден поток маркеров: {marker_streams[0].name()}")
    marker_inlet = StreamInlet(marker_streams[0])
else:
    print("❌ Поток маркеров не найден")
    marker_inlet = None
    exit()

# 2. Ищем поток ЭЭГ по ТИПУ (type='EEG'), а не по имени
print("\n📌 Поиск потока ЭЭГ от Нейроспектра (по типу 'EEG')...")
print("   (ожидание до 10 секунд...)")

eeg_streams = resolve_bypred("type='EEG'", timeout=10)

if eeg_streams:
    print(f"✅ Найден поток ЭЭГ: {eeg_streams[0].name()}")
    print(f"   Частота: {eeg_streams[0].nominal_srate()} Гц")
    print(f"   Каналов: {eeg_streams[0].channel_count()}")
    print(f"   Тип: {eeg_streams[0].type()}")

    eeg_inlet = StreamInlet(eeg_streams[0])

    print("\n🚀 Оба потока найдены! Начинаю прием данных...")
    print("-" * 50)

    try:
        marker_count = 0
        eeg_count = 0
        start_time = time.time()

        while True:
            # Получаем маркеры
            marker, marker_ts = marker_inlet.pull_sample(timeout=0.0)
            if marker:
                marker_count += 1
                print(f"⏱️ {marker_ts:.3f} | Маркер: {marker[0]}")

            # Получаем ЭЭГ
            eeg_samples, eeg_timestamps = eeg_inlet.pull_chunk(timeout=0.0, max_samples=50)
            if eeg_samples and len(eeg_samples) > 0:
                eeg_count += len(eeg_samples)

            # Статистика каждые 5 секунд
            if time.time() - start_time >= 5:
                print(f"\n📊 Статистика за 5 сек:")
                print(f"   Маркеров: {marker_count}")
                print(f"   Сэмплов ЭЭГ: {eeg_count}")
                print(f"   Частота ЭЭГ: {eeg_count / 5:.1f} Гц")
                print("-" * 50)

                marker_count = 0
                eeg_count = 0
                start_time = time.time()

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n🛑 Остановлено пользователем")
else:
    print("❌ Поток ЭЭГ не найден!")
    print("\n📋 Возможные причины:")
    print("1. ПО Нейроспектра не запущено")
    print("2. В ПО Нейроспектра не начата запись/мониторинг")
    print("3. В настройках Нейроспектра отключена LSL-трансляция")
    print("\n📋 Текущие потоки в системе:")
    for i, s in enumerate(streams):
        print(f"   {i + 1}. {s.name()} (тип: {s.type()})")
