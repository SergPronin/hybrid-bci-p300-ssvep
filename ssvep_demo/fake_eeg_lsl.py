#!/usr/bin/env python3
"""
Синтетический EEG для SSVEP-демо через LSL (pylsl StreamOutlet).

Сигнал: на каждом канале доминирующая синусоида на ``current_freq_hz`` + слабый 1/f-подобный шум
и белый шум (чтобы MSI не получал «идеальную» математику без артефактов сети).

Управление частотой (UDP, порт по умолчанию 17391), байт ASCII ``1``..``4``:
  1 → 10 Hz, 2 → 12 Hz, 3 → 15 Hz, 4 → 20 Hz

Пример (macOS/Linux)::

    printf '1' | nc -u -w0 127.0.0.1 17391

Запуск ``python -m ssvep_demo.fake_eeg_lsl`` из корня репозитория (или ``python fake_eeg_lsl.py`` из ssvep_demo/).
"""

from __future__ import annotations

import os
import sys

if os.environ.get("SSVEP_DEMO_LAUNCHED") == "1":
    _ve = os.environ.get("VIRTUAL_ENV", "")
    _in = bool(_ve) or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    print(f"[fake_eeg_lsl] sys.executable={sys.executable!r}", flush=True)
    print(f"[fake_eeg_lsl] resolved={os.path.normpath(os.path.realpath(sys.executable))!r}", flush=True)
    print(f"[fake_eeg_lsl] sys.prefix={sys.prefix!r} VIRTUAL_ENV={_ve!r} venv-like={_in}", flush=True)
    print(f"[fake_eeg_lsl] sys.path[:10]={sys.path[:10]!r}", flush=True)

import argparse
import math
import socket
import time
from pathlib import Path

import numpy as np

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock
except ImportError as e:
    raise SystemExit(
        f"Нужен pylsl (pip install pylsl). sys.executable={sys.executable!r} "
        f"prefix={sys.prefix!r} VIRTUAL_ENV={os.environ.get('VIRTUAL_ENV', '')!r}"
    ) from e

FREQ_MAP = {1: 10.0, 2: 12.0, 3: 15.0, 4: 20.0}
DEFAULT_UDP = 17391


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def generate_chunk(
    n_samples: int,
    srate: float,
    start_t: float,
    freq_hz: float,
    n_channels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Возвращает массив формы (n_samples, n_channels) float32 — порядок для LSL push_chunk.
    """
    t = start_t + np.arange(n_samples, dtype=np.float64) / srate
    w = 2.0 * math.pi * freq_hz * t
    carrier = np.sin(w, dtype=np.float64)[:, None]
    sig = np.repeat(carrier, n_channels, axis=1)
    # лёгкая межканальная декорреляция
    for c in range(n_channels):
        sig[:, c] *= 0.85 + 0.15 * np.sin(2 * math.pi * (0.7 + 0.1 * c) * t)
    noise = 0.08 * rng.standard_normal((n_samples, n_channels))
    slow = 0.03 * np.cumsum(rng.standard_normal((n_samples, n_channels)), axis=0)
    out = (sig + noise + slow).astype(np.float32)
    return out


def run_outlet(
    *,
    name: str,
    srate: float,
    n_channels: int,
    chunk_samples: int,
    udp_port: int,
) -> None:
    rng = np.random.default_rng(42)
    current_key = 1
    current_freq = FREQ_MAP[current_key]

    info = StreamInfo(
        name=name,
        type="EEG",
        channel_count=n_channels,
        nominal_srate=srate,
        channel_format="float32",
        source_id="ssvep-demo-synthetic-eeg",
    )
    outlet = StreamOutlet(info, chunk_size=chunk_samples, max_buffered=360)

    sock: socket.socket | None = None
    if udp_port > 0:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(udp_port)))
        sock.setblocking(False)
        print(
            f"[fake_eeg] stream='{name}' {n_channels}ch @ {srate} Hz | "
            f"UDP {udp_port}: send 1-4 to switch 10/12/15/20 Hz",
            flush=True,
        )
    else:
        print(
            f"[fake_eeg] stream='{name}' {n_channels}ch @ {srate} Hz (UDP control disabled)",
            flush=True,
        )

    t_stream = 0.0
    start_wall = time.perf_counter()

    while True:
        if sock is not None:
            try:
                while True:
                    data, _addr = sock.recvfrom(8)
                    if not data:
                        continue
                    key = data[0]
                    if key in (49, 50, 51, 52):  # '1'..'4'
                        k = key - 48
                        if k in FREQ_MAP:
                            current_key = k
                            current_freq = FREQ_MAP[k]
                            print(f"[fake_eeg] switched to {current_freq} Hz (key {k})", flush=True)
            except BlockingIOError:
                pass

        chunk = generate_chunk(
            chunk_samples, srate, t_stream, current_freq, n_channels, rng
        )
        ts = local_clock()
        outlet.push_chunk(chunk, timestamp=ts)
        t_stream += chunk_samples / srate

        # держим темп ~realtime
        target = start_wall + t_stream
        dt = target - time.perf_counter()
        if dt > 0:
            time.sleep(dt)
        elif dt < -2.0:
            # сильно отстаём — сброс якоря
            start_wall = time.perf_counter()
            t_stream = 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic SSVEP-like EEG LSL outlet")
    ap.add_argument("--name", default="SSVEP-Demo-EEG", help="LSL stream name")
    ap.add_argument("--srate", type=float, default=250.0)
    ap.add_argument("--channels", type=int, default=2, choices=(2,))
    ap.add_argument("--chunk", type=int, default=25, help="samples per push_chunk")
    ap.add_argument("--udp-port", type=int, default=DEFAULT_UDP, help="0 to disable UDP control")
    args = ap.parse_args()

    if args.chunk < 1:
        return 2

    print(f"[fake_eeg] cwd={Path.cwd()} repo_hint={_repo_root()}", flush=True)
    try:
        run_outlet(
            name=args.name,
            srate=float(args.srate),
            n_channels=int(args.channels),
            chunk_samples=int(args.chunk),
            udp_port=int(args.udp_port),
        )
    except KeyboardInterrupt:
        print("[fake_eeg] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
