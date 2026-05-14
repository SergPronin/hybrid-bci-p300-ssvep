#!/usr/bin/env python3
"""
Лончер демо: поднимает три процесса

1) ``<venv>/bin/python -m ssvep_demo.fake_eeg_lsl`` (или тот же интерпретатор, что у лончера) — LSL EEG
2) ``… -m ssvep_demo.stimulus_window`` — мигание
3) ``… -m ssvep_demo.realtime_gui`` — MSI GUI

Лончер ``run_demo.py`` всегда порождает subprocess с ``sys.executable`` текущего процесса (см. ``SSVEP_DEMO_LAUNCHED`` и логи).

Закройте окно **realtime_gui** — лончер завершит остальные процессы.

Смена доминирующей частоты в синтетике (UDP **17391**)::

    printf '1' | nc -u -w0 127.0.0.1 17391   # 10 Hz
    printf '2' | nc -u -w0 127.0.0.1 17391   # 12 Hz
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _child_argv(module: str) -> list[str]:
    """Всегда [sys.executable, -m, module] — никаких python/python3 из PATH."""
    return [sys.executable, "-m", module]


def _print_launcher_diagnostics() -> None:
    print("[run_demo] launcher sys.executable =", sys.executable, flush=True)
    print("[run_demo] sys.prefix =", sys.prefix, flush=True)
    print("[run_demo] sys.base_prefix =", getattr(sys, "base_prefix", sys.prefix), flush=True)
    ve = os.environ.get("VIRTUAL_ENV", "")
    print("[run_demo] VIRTUAL_ENV =", repr(ve), flush=True)
    in_venv = bool(ve) or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
    print("[run_demo] venv-like (VIRTUAL_ENV or prefix mismatch) =", in_venv, flush=True)
    print("[run_demo] sys.path (first 12 entries):", flush=True)
    for i, p in enumerate(sys.path[:12]):
        print(f"  [{i}] {p}", flush=True)
    if len(sys.path) > 12:
        print(f"  ... ({len(sys.path) - 12} more)", flush=True)
    if not in_venv:
        print(
            "[run_demo] WARN: похоже, лончер не в venv — subprocess получит тот же sys.executable; "
            "если нет pylsl, запустите: .venv/bin/python ssvep_demo/run_demo.py",
            flush=True,
        )


def _child_env() -> dict[str, str]:
    """Копия окружения + PYTHONPATH для репо + маркер для дочерних debug-принтов + PATH для venv bin."""
    e = os.environ.copy()
    sep = ";" if os.name == "nt" else ":"
    e["PYTHONPATH"] = sep.join([str(_REPO), str(_REPO / "scripts")])
    # Дочерние процессы печатают интерпретатор до тяжёлых импортов (pylsl / Qt).
    e["SSVEP_DEMO_LAUNCHED"] = "1"
    # Чтобы дочерние процессы видели те же CLI-утилиты, что и venv (не влияет на import pylsl).
    ve = e.get("VIRTUAL_ENV", "").strip()
    if ve:
        bindir = Path(ve) / ("Scripts" if os.name == "nt" else "bin")
        if bindir.is_dir():
            e["PATH"] = str(bindir) + os.pathsep + e.get("PATH", "")
    return e


def main() -> int:
    print("=== SSVEP + MSI + LSL demo launcher ===", flush=True)
    _print_launcher_diagnostics()
    print(f"repo={_REPO}", flush=True)
    print(
        "Нужно: msi-res (MSIController + deps + alglib nupkg), .NET 8 runtime, "
        "pythonnet, pylsl, numpy, pyqtgraph, PyQt6",
        flush=True,
    )
    print("Частота синтетики (UDP): printf '1' | nc -u -w0 127.0.0.1 17391  (1..4)", flush=True)

    env = _child_env()
    print("[run_demo] subprocess executable =", sys.executable, flush=True)

    procs: list[subprocess.Popen] = []
    try:
        procs.append(
            subprocess.Popen(
                _child_argv("ssvep_demo.fake_eeg_lsl"),
                cwd=str(_REPO),
                env=env,
                executable=sys.executable,
            )
        )
        print("[run_demo] fake_eeg_lsl PID", procs[-1].pid, flush=True)
        time.sleep(1.8)

        procs.append(
            subprocess.Popen(
                _child_argv("ssvep_demo.stimulus_window"),
                cwd=str(_REPO),
                env=env,
                executable=sys.executable,
            )
        )
        print("[run_demo] stimulus_window PID", procs[-1].pid, flush=True)

        procs.append(
            subprocess.Popen(
                _child_argv("ssvep_demo.realtime_gui"),
                cwd=str(_REPO),
                env=env,
                executable=sys.executable,
            )
        )
        print("[run_demo] realtime_gui PID", procs[-1].pid, flush=True)
        print("[run_demo] Закройте окно realtime_gui для выхода.", flush=True)

        procs[-1].wait()
    except KeyboardInterrupt:
        print("[run_demo] interrupt", flush=True)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        time.sleep(0.35)
        for p in procs:
            if p.poll() is None:
                p.kill()
        print("[run_demo] stopped all", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
