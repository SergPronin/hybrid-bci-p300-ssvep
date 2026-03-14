#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Минимальная запись LSL без GUI — нулевая задержка от отрисовки и Qt.
Только pylsl + numpy: поиск потока, открытие inlet, цикл pull_chunk → буфер → CSV.

Запуск:
  python lsl_record_minimal.py                    # запись до Ctrl+C
  python lsl_record_minimal.py --duration 60     # запись 60 секунд
  python lsl_record_minimal.py --out my_rec.csv  # свой файл
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from pylsl import StreamInlet, resolve_byprop

# Имена каналов 10–20 (как в hardware_validation)
DEFAULT_CHANNEL_NAMES_21 = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]
LSL_MAX_BUFFERED_SEC = 600
PULL_TIMEOUT_S = 0.01
MAX_SAMPLES_PER_CHUNK = 8192
STREAM_TYPES = ("EEG", "Signal")


def _is_allowed_stream(info) -> bool:
    try:
        name = (info.name() or "").strip().lower()
        sid = (info.source_id() or "").strip().lower()
    except Exception:
        return False
    if "neuro" in name or "neuro" in sid:
        return True
    if "eeg-simulator" in sid or name == "eeg_simulator":
        return True
    return False


def find_stream(timeout: float = 2.0):
    for st in STREAM_TYPES:
        try:
            streams = resolve_byprop("type", st, timeout=timeout)
        except TypeError:
            streams = resolve_byprop("type", st)
        for s in streams:
            if _is_allowed_stream(s):
                return s
    return None


def main():
    parser = argparse.ArgumentParser(description="Запись LSL без GUI (минимальная задержка)")
    parser.add_argument("--duration", "-d", type=float, default=None, help="Длительность записи в секундах (по умолчанию — до Ctrl+C)")
    parser.add_argument("--out", "-o", type=str, default=None, help="Путь к CSV (по умолчанию: saved_data/eeg_YYYY-MM-DD_HH-MM-SS.csv)")
    parser.add_argument("--resolve-timeout", type=float, default=2.0, help="Таймаут поиска потока, сек")
    args = parser.parse_args()

    print("Поиск LSL потока...")
    info = find_stream(timeout=args.resolve_timeout)
    if info is None:
        print("Поток не найден. Запустите NeuroSpectrum или симулятор.", file=sys.stderr)
        sys.exit(1)

    n_channels = info.channel_count()
    srate = info.nominal_srate()
    stream_name = info.name() or "EEG"
    print(f"Найден: {stream_name}, каналов={n_channels}, Гц={srate}")

    try:
        inlet = StreamInlet(info, max_buffered=LSL_MAX_BUFFERED_SEC)
    except TypeError:
        inlet = StreamInlet(info)
    try:
        inlet.open_stream(timeout=1.0)
    except Exception:
        pass

    channel_names = [
        DEFAULT_CHANNEL_NAMES_21[i] if i < len(DEFAULT_CHANNEL_NAMES_21) else f"Ch{i+1}"
        for i in range(n_channels)
    ]
    buffers = [[] for _ in range(n_channels)]
    stop = [False]

    def on_stop(sig=None, frame=None):
        stop[0] = True

    signal.signal(signal.SIGINT, on_stop)
    signal.signal(signal.SIGTERM, on_stop)

    print("Запись... (Ctrl+C для остановки)")
    start_time = time.time()

    while not stop[0]:
        if args.duration is not None and (time.time() - start_time) >= args.duration:
            break
        try:
            chunk, _ = inlet.pull_chunk(timeout=PULL_TIMEOUT_S, max_samples=MAX_SAMPLES_PER_CHUNK)
        except Exception:
            chunk = None
        if chunk:
            arr = np.asarray(chunk)
            if arr.ndim == 2:
                if arr.shape[1] != n_channels and arr.shape[0] == n_channels:
                    arr = arr.T
                for ch in range(min(n_channels, arr.shape[1])):
                    col = np.nan_to_num(arr[:, ch].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                    buffers[ch].extend(col.tolist())

    total_samples = len(buffers[0]) if buffers else 0
    print(f"Записано сэмплов: {total_samples}")

    if total_samples == 0:
        print("Нет данных для сохранения.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path("saved_data")
    out_dir.mkdir(exist_ok=True)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = out_dir / f"eeg_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# sampling_rate={srate}\n")
        f.write(",".join(channel_names) + "\n")
        for i in range(total_samples):
            row = []
            for ch in range(n_channels):
                val = buffers[ch][i] if i < len(buffers[ch]) else 0.0
                row.append(f"{val:.3f}".replace(".", ","))
            f.write(",".join(row) + "\n")

    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
