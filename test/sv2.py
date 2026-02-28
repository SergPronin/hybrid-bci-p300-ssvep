#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSL Stream Explorer - подробный анализ всех LSL потоков
"""

from pylsl import StreamInlet, resolve_streams, resolve_bypred, local_clock
import numpy as np
import time
import json
from datetime import datetime

def print_header(text):
    """Красивый вывод заголовка"""
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)

def print_stream_info(stream, index=None):
    """Детальный вывод информации о потоке"""
    prefix = f"Поток {index}: " if index else "Поток: "
    print(f"\n{prefix}{stream.name()}")
    print("-" * 60)
    
    # Основная информация
    info = {
        "Тип": stream.type(),
        "ID источника": stream.source_id(),
        "Количество каналов": stream.channel_count(),
        "Частота дискретизации (Гц)": stream.nominal_srate(),
        "Формат данных": stream.channel_format(),
        "UID потока": stream.uid(),
        "Версия": stream.version(),
        "Создатель": stream.created_at(),
        "Хост": stream.hostname(),
        "Сессия ID": stream.session_id(),
    }
    
    for key, value in info.items():
        print(f"  {key}: {value}")

def inspect_stream_data(inlet, duration=5, stream_name=""):
    """Анализ данных потока за указанное время"""
    print(f"\n📊 Анализ данных потока '{stream_name}' ({duration} секунд)...")
    print("-" * 60)
    
    # Получаем информацию о потоке
    info = inlet.info()
    n_channels = info.channel_count()
    srate = info.nominal_srate()
    
    print(f"  Каналов: {n_channels}")
    print(f"  Частота: {srate} Гц")
    print(f"  Ожидаемое количество сэмплов за {duration}с: ~{int(srate * duration)}")
    
    # Сбор данных
    all_samples = []
    all_timestamps = []
    start_time = time.time()
    sample_count = 0
    
    print(f"\n  ⏳ Сбор данных...")
    
    while time.time() - start_time < duration:
        samples, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=1000)
        if samples and len(samples) > 0:
            all_samples.extend(samples)
            all_timestamps.extend(timestamps)
            sample_count += len(samples)
            
            if sample_count % 100 == 0:
                print(f"     Получено {sample_count} сэмплов...")
    
    print(f"\n  ✅ Собрано {len(all_samples)} сэмплов")
    
    if len(all_samples) == 0:
        print("  ⚠️ Нет данных для анализа!")
        return None
    
    # Преобразуем в numpy для анализа
    data = np.array(all_samples)
    timestamps = np.array(all_timestamps)
    
    # Статистика
    print(f"\n  📈 Статистика данных:")
    print(f"     Форма данных: {data.shape}")
    print(f"     Диапазон времени: {timestamps[0]:.3f} - {timestamps[-1]:.3f}")
    print(f"     Длительность: {timestamps[-1] - timestamps[0]:.3f} сек")
    print(f"     Реальная частота: {len(data) / (timestamps[-1] - timestamps[0]):.1f} Гц")
    
    # Статистика по каналам
    if n_channels <= 10:  # Для небольшого числа каналов показываем все
        print(f"\n  📊 Статистика по каналам:")
        for ch in range(n_channels):
            ch_data = data[:, ch]
            print(f"     Канал {ch}:")
            print(f"        Мин: {np.min(ch_data):.3f}")
            print(f"        Макс: {np.max(ch_data):.3f}")
            print(f"        Среднее: {np.mean(ch_data):.3f}")
            print(f"        Стд: {np.std(ch_data):.3f}")
    else:
        # Для многих каналов показываем общую статистику
        print(f"\n  📊 Общая статистика (первые 5 каналов):")
        for ch in range(min(5, n_channels)):
            ch_data = data[:, ch]
            print(f"     Канал {ch}: мин={np.min(ch_data):.3f}, макс={np.max(ch_data):.3f}, "
                  f"сред={np.mean(ch_data):.3f}, стд={np.std(ch_data):.3f}")
    
    # Проверка на наличие пропусков
    time_diffs = np.diff(timestamps)
    print(f"\n  ⏱️  Анализ временных меток:")
    print(f"     Средний интервал: {np.mean(time_diffs)*1000:.2f} мс")
    print(f"     Мин интервал: {np.min(time_diffs)*1000:.2f} мс")
    print(f"     Макс интервал: {np.max(time_diffs)*1000:.2f} мс")
    print(f"     Стд интервала: {np.std(time_diffs)*1000:.2f} мс")
    
    # Проверка на выбросы (возможные артефакты)
    if n_channels <= 20:
        for ch in range(min(5, n_channels)):
            ch_data = data[:, ch]
            z_scores = np.abs((ch_data - np.mean(ch_data)) / np.std(ch_data))
            outliers = np.sum(z_scores > 3)
            if outliers > 0:
                print(f"     Канал {ch}: {outliers} выбросов (>3σ)")
    
    return data, timestamps

def monitor_markers(inlet, duration=10):
    """Мониторинг потока маркеров"""
    print(f"\n🏷️  Мониторинг маркеров ({duration} секунд)...")
    print("-" * 60)
    
    markers = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        marker, timestamp = inlet.pull_sample(timeout=0.1)
        if marker:
            markers.append((timestamp, marker[0]))
            print(f"  ⏱️ {timestamp:.3f} | {marker[0]}")
    
    print(f"\n  ✅ Получено {len(markers)} маркеров")
    
    if markers:
        # Анализ маркеров
        unique_markers = set([m[1] for m in markers])
        print(f"  📋 Уникальные маркеры: {sorted(unique_markers)}")
        
        # Частота маркеров
        if len(markers) > 1:
            total_time = markers[-1][0] - markers[0][0]
            print(f"  📊 Частота маркеров: {len(markers)/total_time:.2f} Гц")
    
    return markers

def main():
    print_header("🔍 LSL STREAM EXPLORER - ДЕТАЛЬНЫЙ АНАЛИЗ ПОТОКОВ")
    print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Поиск всех потоков
    print_header("1. ПОИСК ВСЕХ LSL ПОТОКОВ")
    print("Поиск потоков в сети...")
    
    streams = resolve_streams(wait_time=5.0)
    
    if not streams:
        print("❌ Ни одного LSL потока не найдено!")
        print("\nВозможные причины:")
        print("  • ПО Нейроспектр не запущено")
        print("  • Ваша программа с маркерами не запущена")
        print("  • Брандмауэр блокирует LSL (порты 22345, 22346)")
        return
    
    print(f"\n✅ Найдено {len(streams)} потоков:")
    
    # 2. Детальная информация о каждом потоке
    for i, stream in enumerate(streams):
        print_stream_info(stream, i+1)
    
    # 3. Выбор потока для детального анализа
    print_header("2. ДЕТАЛЬНЫЙ АНАЛИЗ ПОТОКОВ")
    
    for i, stream in enumerate(streams):
        print(f"\n🔍 Анализ потока {i+1}: {stream.name()}")
        print("  1 - Пропустить")
        print("  2 - Анализировать")
        print("  3 - Подробный анализ с данными")
        
        choice = input(f"  Выберите действие для потока {i+1} (1/2/3): ").strip()
        
        if choice in ['2', '3']:
            try:
                # Создаем inlet
                inlet = StreamInlet(stream)
                
                # Получаем детальную информацию
                info = inlet.info()
                print(f"\n  📋 Параметры подключения:")
                print(f"     Макс буфер: {info.buffer_max_seconds()} сек")
                print(f"     Формат: {info.channel_format()}")
                print(f"     Производитель: {info.manufacturer()}")
                
                if stream.type() == 'Markers':
                    # Для маркеров - мониторинг
                    markers = monitor_markers(inlet, duration=10)
                else:
                    # Для других типов - анализ данных
                    if choice == '3':
                        result = inspect_stream_data(inlet, duration=10, stream_name=stream.name())
                    else:
                        print(f"\n  ⏩ Пропускаем сбор данных для потока {i+1}")
                        
            except Exception as e:
                print(f"  ❌ Ошибка при анализе потока: {e}")
    
    # 4. Поиск специфических потоков
    print_header("3. ПОИСК СПЕЦИФИЧЕСКИХ ПОТОКОВ")
    
    # Поиск EEG потоков
    print("\n🔍 Поиск потоков типа 'EEG'...")
    eeg_streams = resolve_bypred("type='EEG'", timeout=2)
    if eeg_streams:
        print(f"✅ Найдено {len(eeg_streams)} EEG потоков:")
        for i, s in enumerate(eeg_streams):
            print(f"   {i+1}. {s.name()} (каналов: {s.channel_count()}, частота: {s.nominal_srate()} Гц)")
    else:
        print("❌ EEG потоки не найдены")
    
    # Поиск маркерных потоков
    print("\n🔍 Поиск потоков типа 'Markers'...")
    marker_streams = resolve_bypred("type='Markers'", timeout=2)
    if marker_streams:
        print(f"✅ Найдено {len(marker_streams)} маркерных потоков:")
        for i, s in enumerate(marker_streams):
            print(f"   {i+1}. {s.name()}")
    else:
        print("❌ Маркерные потоки не найдены")
    
    # Поиск вашего конкретного потока
    print("\n🔍 Поиск потока 'BCI_StimMarkers'...")
    bci_streams = resolve_bypred("name='BCI_StimMarkers'", timeout=2)
    if bci_streams:
        print(f"✅ Найден поток: {bci_streams[0].name()}")
    else:
        print("❌ Поток BCI_StimMarkers не найден")
    
    # 5. Дополнительная информация о сети
    print_header("4. ИНФОРМАЦИЯ О СЕТИ LSL")
    
    try:
        from pylsl import get_version, library_version, protocol_version
        print(f"📦 Версия LSL: {get_version()}")
        print(f"📚 Версия библиотеки: {library_version()}")
        print(f"🌐 Версия протокола: {protocol_version()}")
    except:
        pass
    
    # 6. Сохранение отчета
    print_header("5. СОХРАНЕНИЕ ОТЧЕТА")
    
    filename = f"lsl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("LSL STREAM EXPLORER REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Время: {datetime.now()}\n\n")
            
            f.write(f"Найдено потоков: {len(streams)}\n\n")
            
            for i, stream in enumerate(streams):
                f.write(f"Поток {i+1}: {stream.name()}\n")
                f.write(f"  Тип: {stream.type()}\n")
                f.write(f"  Каналов: {stream.channel_count()}\n")
                f.write(f"  Частота: {stream.nominal_srate()} Гц\n")
                f.write(f"  ID: {stream.source_id()}\n")
                f.write(f"  UID: {stream.uid()}\n")
                f.write(f"  Хост: {stream.hostname()}\n")
                f.write(f"  Сессия: {stream.session_id()}\n")
                f.write("\n")
        
        print(f"✅ Отчет сохранен в файл: {filename}")
    except Exception as e:
        print(f"❌ Ошибка при сохранении отчета: {e}")
    
    print_header("✅ АНАЛИЗ ЗАВЕРШЕН")
    print("Для более детального анализа конкретного потока:")
    print("  1. Убедитесь, что источник данных активен")
    print("  2. Запустите этот скрипт снова")
    print("  3. Выберите опцию 3 для сбора данных")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Скрипт остановлен пользователем")
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()