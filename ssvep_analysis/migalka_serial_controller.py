from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import serial

from ssvep_analysis.burst_gate import parse_led_serial_line
from ssvep_analysis.migalka_lsl import MigalkaLslSender

try:
    from experiment_protocol.protocol_log import error as _log_error
    from experiment_protocol.protocol_log import info as _log_info
except Exception:  # pragma: no cover

    def _log_info(msg: str) -> None:
        print(f"[migalka] {msg}", flush=True)

    def _log_error(msg: str) -> None:
        print(f"[migalka] ERROR {msg}", flush=True)


@dataclass(frozen=True)
class MigalkaConfig:
    port: str
    baudrate: int = 115200
    timeout_s: float = 0.1
    # Migalka protocol: "M <mode>" where 0=continuous, 1=burst (as in migalka.py)
    mode: int = 0
    # six lamp frequency labels (strings), e.g. "10.0" or "0"
    freqs: Tuple[str, str, str, str, str, str] = ("0", "0", "0", "0", "0", "0")


LampEvent = Tuple[int, bool, str]  # (lamp_index, is_on, raw_line)


class MigalkaSerialController:
    """Direct COM control for Arduino Due Migalka + optional LSL marker mirroring."""

    def __init__(
        self,
        *,
        mirror_lsl: bool = True,
        on_event: Optional[Callable[[LampEvent], None]] = None,
    ) -> None:
        self._ser: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_event = on_event
        self._lsl = MigalkaLslSender() if mirror_lsl else None
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        return self._ser is not None and bool(getattr(self._ser, "is_open", False))

    def open_and_start(self, cfg: MigalkaConfig) -> None:
        with self._lock:
            if self.is_open():
                _log_info(
                    f"COM уже открыт — перенастройка mode={cfg.mode} "
                    f"({'постоянный' if int(cfg.mode) == 0 else 'пакетный'}), freqs={cfg.freqs}"
                )
                self._send_mode(cfg.mode)
                time.sleep(0.05)
                self._send_freqs(cfg.freqs)
                return
            _log_info(
                f"открываем COM {cfg.port!r} baud={cfg.baudrate}, mode={cfg.mode} "
                f"({'постоянный' if int(cfg.mode) == 0 else 'пакетный'}), freqs={cfg.freqs}"
            )
            self._ser = serial.Serial(cfg.port, cfg.baudrate, timeout=cfg.timeout_s)
            time.sleep(0.5)
            self._running = True
            self._send_mode(cfg.mode)
            self._send_freqs(cfg.freqs)
            _log_info(f"команды отправлены: M {cfg.mode}, freqs={' '.join(cfg.freqs)}")
            self._thread = threading.Thread(target=self._read_loop, name="MigalkaSerialReader", daemon=True)
            self._thread.start()
            _log_info("read_loop запущен")

    def stop_and_close(self) -> None:
        with self._lock:
            _log_info("stop_and_close")
            self._running = False
            try:
                self._send_freqs(("0", "0", "0", "0", "0", "0"))
                time.sleep(0.2)
            except Exception:
                pass
            try:
                if self._ser is not None:
                    self._ser.close()
            except Exception:
                pass
            self._ser = None

    def set_mode(self, mode: int) -> None:
        self._send_mode(mode)

    def set_freqs(self, freqs: Sequence[str]) -> None:
        tup = tuple(str(x) for x in freqs)
        if len(tup) != 6:
            raise ValueError("Migalka expects 6 frequency values")
        self._send_freqs(tup)  # type: ignore[arg-type]

    def _write_line(self, line: str) -> None:
        if not self.is_open():
            return
        assert self._ser is not None
        self._ser.write((line.strip() + "\n").encode("utf-8"))

    def _send_mode(self, mode: int) -> None:
        if not self.is_open():
            return
        m = int(mode)
        m = 0 if m <= 0 else 1
        self._write_line(f"M {m}")

    def _send_freqs(self, freqs: Sequence[str]) -> None:
        if not self.is_open():
            return
        vals = [str(x).strip().replace(",", ".") for x in freqs]
        if len(vals) != 6:
            raise ValueError("Migalka expects 6 frequency values")
        self._write_line(" ".join(vals))

    def _emit_event(self, lamp: int, is_on: bool, raw_line: str) -> None:
        ev: LampEvent = (int(lamp), bool(is_on), str(raw_line))
        if self._lsl is not None:
            # LSL uses 100+lamp|on/off
            self._lsl.send_lamp_event(int(lamp), "on" if is_on else "off")
        if self._on_event is not None:
            try:
                self._on_event(ev)
            except Exception:
                pass

    def _read_loop(self) -> None:
        while True:
            if not self._running or not self.is_open():
                return
            try:
                assert self._ser is not None
                line = self._ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                parsed = parse_led_serial_line(line)
                if parsed is None:
                    continue
                lamp, is_on = parsed
                self._emit_event(lamp, is_on, line)
            except Exception:
                # Never crash protocol runner due to serial hiccups.
                time.sleep(0.01)

