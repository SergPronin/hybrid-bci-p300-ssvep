#!/usr/bin/env python3

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional

import serial
from serial.tools import list_ports


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

        # ===== UI =====
        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        self.cb_mode = ttk.Combobox(
            top,
            values=("Постоянный", "Пакетный"),
            state="readonly"
        )
        self.cb_mode.current(0)
        self.cb_mode.grid(row=0, column=0)

        # 🔥 ВАЖНО — обработчик
        self.cb_mode.bind("<<ComboboxSelected>>", self._on_mode_change)

        self.btn_run = ttk.Button(top, text="Задать", command=self._on_run_stop)
        self.btn_run.grid(row=1, column=0)

        self.cb_port = ttk.Combobox(top, width=20, state="readonly")
        self.cb_port.grid(row=0, column=1)

        ttk.Button(top, text="Обновить", command=self._refresh_ports).grid(row=0, column=2)

        # ===== частоты =====
        self.freq_boxes: List[ttk.Combobox] = []
        for i in range(6):
            cb = ttk.Combobox(self, values=self._labels, state="readonly")
            cb.set("0")
            cb.pack()
            self.freq_boxes.append(cb)

        # ===== лог =====
        self.text_log = tk.Text(self, height=10)
        self.text_log.pack(fill=tk.BOTH, expand=True)

        self._refresh_ports()

    # ================= SERIAL =================

    def _is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _write_line(self, line: str) -> None:
        if self._is_open():
            self._ser.write((line + "\n").encode())

    # ================= РЕЖИМ =================

    def _send_mode(self):
        if not self._is_open():
            return

        mode = self.cb_mode.current()
        cmd = f"M {mode}"
        self._write_line(cmd)
        self._append_log(f"=> {cmd}")

    def _on_mode_change(self, event=None):
        self._send_mode()
        if self._is_open():
            time.sleep(0.05)
            line = self._freq_line()
            self._write_line(line)
            self._append_log(f"=> {line}")

    # ================= ЧТЕНИЕ =================

    def _read_loop(self):
        if not self._running or not self._is_open():
            return

        try:
            line = self._ser.readline().decode(errors="replace").strip()
            if line:
                self._append_log("< " + line)
        except Exception as e:
            self._append_log("ERR: " + str(e))

        self.after(10, self._read_loop)

    # ================= UI =================

    def _append_log(self, s: str):
        self.text_log.insert(tk.END, s + "\n")
        self.text_log.see(tk.END)

    def _freq_line(self):
        return " ".join(cb.get() for cb in self.freq_boxes)

    def _refresh_ports(self):
        ports = [p.device for p in list_ports.comports()]
        self.cb_port["values"] = ports
        if ports:
            self.cb_port.current(0)

    # ================= КНОПКА =================

    def _on_run_stop(self):
        if not self._is_open():
            port = self.cb_port.get()

            try:
                # 115200 — иначе при потоке LED ON/OFF loop на Due не успевает (мигание «плывёт»)
                self._ser = serial.Serial(port, 115200, timeout=0.1)
                time.sleep(0.5)

                # 🔥 СНАЧАЛА режим
                self._send_mode()

                # ПОТОМ частоты (прошивка перезапускает таймеры под выбранный режим)
                line = self._freq_line()
                self._write_line(line)
                self._append_log(f"=> {line}")

                # запуск чтения
                self._running = True
                self.after(50, self._read_loop)

                self.btn_run["text"] = "Стоп"

            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                self._ser = None

        else:
            self._running = False

            try:
                self._write_line("0 0 0 0 0 0")
                time.sleep(0.2)
                self._ser.close()
            except:
                pass

            self._ser = None
            self.btn_run["text"] = "Задать"

    # ================= ЗАКРЫТИЕ =================

    def destroy(self):
        self._running = False
        if self._is_open():
            try:
                self._ser.close()
            except:
                pass
        super().destroy()


def main():
    app = MigalkaApp()
    app.mainloop()


if __name__ == "__main__":
    main()