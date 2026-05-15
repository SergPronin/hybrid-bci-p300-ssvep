#!/usr/bin/env python3
"""
Migalka — управление Arduino Due (лампы SSVEP) + LSL-маркеры для ssvep_analyzer.
"""

from __future__ import annotations

import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional

import serial
from serial.tools import list_ports

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ssvep_analysis.burst_gate import parse_led_serial_line  # noqa: E402
from ssvep_analysis.migalka_lsl import MigalkaLslSender  # noqa: E402

try:
    from pylsl import local_clock
except ImportError:
    local_clock = time.time  # type: ignore[misc, assignment]


def lamp_frequency_labels() -> List[str]:
    out: List[str] = ["0"]
    for i in range(1, 501):
        v = 1000.0 / float(i)
        out.append(f"{v}".replace(",", "."))
    return out


class MigalkaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Migalka")
        self.geometry("520x480")

        self._labels = lamp_frequency_labels()
        self._ser: Optional[serial.Serial] = None
        self._running = False
        self._lsl = MigalkaLslSender()

        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        self.cb_mode = ttk.Combobox(
            top,
            values=("Постоянный", "Пакетный"),
            state="readonly",
        )
        self.cb_mode.current(0)
        self.cb_mode.grid(row=0, column=0)
        self.cb_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.btn_run = ttk.Button(top, text="Задать", command=self._on_run_stop)
        self.btn_run.grid(row=1, column=0, pady=(8, 0))

        self.cb_port = ttk.Combobox(top, width=20, state="readonly")
        self.cb_port.grid(row=0, column=1, padx=(8, 0))

        ttk.Button(top, text="Обновить", command=self._refresh_ports).grid(row=0, column=2, padx=(4, 0))

        self._lbl_lsl = ttk.Label(top, text="LSL: MigalkaStimMarkers")
        self._lbl_lsl.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=(8, 0), pady=(8, 0))

        freq_frame = ttk.LabelFrame(self, text="Частоты (6 каналов, 0 = выкл)", padding=6)
        freq_frame.pack(fill=tk.X, padx=6, pady=6)

        self.freq_boxes: List[ttk.Combobox] = []
        for i in range(6):
            row = ttk.Frame(freq_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{i + 1}", width=3).pack(side=tk.LEFT)
            cb = ttk.Combobox(row, values=self._labels, state="readonly", width=18)
            cb.set("0")
            cb.pack(side=tk.LEFT)
            self.freq_boxes.append(cb)

        self.text_log = tk.Text(self, height=10)
        self.text_log.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        self._refresh_ports()

    def _is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _write_line(self, line: str) -> None:
        if self._is_open():
            assert self._ser is not None
            self._ser.write((line + "\n").encode())

    def _send_mode(self) -> None:
        if not self._is_open():
            return
        mode = self.cb_mode.current()
        cmd = f"M {mode}"
        self._write_line(cmd)
        self._append_log(f"=> {cmd}")

    def _on_mode_change(self, event=None) -> None:
        self._send_mode()
        if self._is_open():
            time.sleep(0.05)
            line = self._freq_line()
            self._write_line(line)
            self._append_log(f"=> {line}")

    def _handle_serial_line(self, line: str) -> None:
        self._append_log("< " + line)
        parsed = parse_led_serial_line(line)
        if parsed is None:
            return
        lamp, is_on = parsed
        self._lsl.send_lamp_event(lamp, "on" if is_on else "off")

    def _read_loop(self) -> None:
        if not self._running or not self._is_open():
            return
        try:
            assert self._ser is not None
            line = self._ser.readline().decode(errors="replace").strip()
            if line:
                self._handle_serial_line(line)
        except Exception as e:
            self._append_log("ERR: " + str(e))
        self.after(10, self._read_loop)

    def _append_log(self, s: str) -> None:
        self.text_log.insert(tk.END, s + "\n")
        self.text_log.see(tk.END)

    def _freq_line(self) -> str:
        return " ".join(cb.get() for cb in self.freq_boxes)

    def _refresh_ports(self) -> None:
        ports = [p.device for p in list_ports.comports()]
        self.cb_port["values"] = ports
        if ports:
            self.cb_port.current(0)

    def _on_run_stop(self) -> None:
        if not self._is_open():
            port = (self.cb_port.get() or "").strip()
            if not port:
                messagebox.showwarning("Порт", "Задайте COM-порт")
                return
            try:
                self._ser = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(0.5)
                self._send_mode()
                line = self._freq_line()
                self._write_line(line)
                self._append_log(f"=> {line}")
                self._running = True
                self.after(50, self._read_loop)
                self.btn_run["text"] = "Стоп"
                self._append_log("LSL: MigalkaStimMarkers (запустите ssvep_analyzer в пакетном режиме)")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                self._ser = None
        else:
            self._running = False
            try:
                self._write_line("0 0 0 0 0 0")
                time.sleep(0.2)
                assert self._ser is not None
                self._ser.close()
            except Exception:
                pass
            self._ser = None
            self.btn_run["text"] = "Задать"

    def destroy(self) -> None:
        self._running = False
        if self._is_open():
            try:
                self._write_line("0 0 0 0 0 0")
                time.sleep(0.1)
                assert self._ser is not None
                self._ser.close()
            except Exception:
                pass
        super().destroy()


def main() -> None:
    MigalkaApp().mainloop()


if __name__ == "__main__":
    main()
