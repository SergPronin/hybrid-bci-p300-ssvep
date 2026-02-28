from pylsl import resolve_streams

print("🔍 Поиск всех LSL потоков...")
streams = resolve_streams(wait_time=5.0)

if not streams:
    print("❌ Ни одного LSL потока не найдено")
else:
    print(f"✅ Найдено {len(streams)} потоков:")
    for i, stream in enumerate(streams):
        print(f"\n--- Поток {i+1} ---")
        print(f"  Имя: {stream.name()}")
        print(f"  Тип: {stream.type()}")
        print(f"  ID источника: {stream.source_id()}")
        print(f"  Каналов: {stream.channel_count()}")
        print(f"  Частота: {stream.nominal_srate()} Гц")
