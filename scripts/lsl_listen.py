#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Слушает все доступные LSL-стримы в сети:
  - маркеры из нашей программы (BCI_StimMarkers);
  - ЭЭГ и другие потоки из Нейроспектра или иных программ.

Запуск из корня проекта с активированным venv:
  python scripts/lsl_listen.py

Перед запуском включите приложение стимуляции и/или Нейроспектр с LSL-трансляцией.
"""

import time
from typing import List, Tuple

from pylsl import StreamInlet, resolve_byprop

TIMEOUT = 5
POLL_INTERVAL = 0.02


def discover_streams() -> List:
    """Находит все LSL-стримы в сети (Markers, EEG и др.)."""
    seen_uid = set()
    streams = []
    for stream_type in ("Markers", "EEG", "Signal"):
        found = resolve_byprop("type", stream_type, timeout=2)
        for s in found:
            uid = s.uid()
            if uid not in seen_uid:
                seen_uid.add(uid)
                streams.append(s)
    return streams


def main() -> None:
    print("Поиск LSL-стримов (маркеры, ЭЭГ и др.)...")
    streams = discover_streams()
    if not streams:
        print("Стримы не найдены. Запустите приложение стимуляции и/или Нейроспектр.")
        return

    inlets: List[Tuple[str, str, StreamInlet]] = []
    for s in streams:
        inlet = StreamInlet(s)
        inlets.append((s.name(), s.type(), inlet))
        print(f"  Подключено: {s.name()} (тип: {s.type()}, каналов: {s.channel_count()})")

    print("\nПриём данных (Ctrl+C — выход):\n")
    try:
        while True:
            for name, stype, inlet in inlets:
                if stype == "Markers":
                    sample, ts = inlet.pull_sample(timeout=0.0)
                    if sample is not None:
                        print(f"[{name}] {ts:.3f}  маркер: {sample[0]}")
                else:
                    chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
                    if chunk and timestamps:
                        n = len(chunk)
                        if n == 1:
                            print(f"[{name}] {timestamps[0]:.3f}  сэмпл: {chunk[0]}")
                        else:
                            print(f"[{name}] {timestamps[0]:.3f}  получено {n} сэмплов (ЭЭГ)")
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nВыход.")


if __name__ == "__main__":
    main()
