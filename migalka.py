#!/usr/bin/env python3
"""
Migalka — порт оригинального WinForms-приложения (Migalka/MyForm.cs) на Python.

COM-порт, шесть значений частоты в одной строке (как в прошивке), режим «Постоянный» / «Пакетный»
(M 0 / M 1), сохранение в Settings.xml рядом со скриптом (или в текущей рабочей директории).
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox, ttk
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional

try:
    import serial
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Нужен пакет pyserial: pip install pyserial"
    ) from exc

# Как в C#: имя файла относительно рабочей директории процесса.
INIT_FILE = Path("Settings.xml")


def lamp_frequency_labels() -> List[str]:
    """Как в C#: '0' и (1000.0f / i).ToString() для i = 1 .. 500."""
    out: List[str] = ["0"]
    for i in range(1, 501):
        v = 1000.0 / float(i)
        out.append(f"{v}".replace(",", "."))
    return out


def _nearest_label(requested: str, labels: List[str]) -> str:
    """Подобрать значение из списка по числу из XML (форматы могут отличаться)."""
    requested = (requested or "").strip()
    if not requested or requested == "0":
        return "0"
    try:
        target = float(requested.replace(",", "."))
    except ValueError:
        return "0"
    best = "0"
    best_d = abs(target)
    for lab in labels[1:]:
        try:
            d = abs(float(lab.replace(",", ".")) - target)
        except ValueError:
            continue
        if d < best_d:
            best_d = d
            best = lab
    return best


class MigalkaApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Migalka")
        self.geometry("520x480")
        self.resizable(True, True)

        self._labels = lamp_frequency_labels()
        self._ser: Optional[serial.Serial] = None

        top = ttk.Frame(self, padding=6)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Режим:").grid(row=0, column=0, sticky=tk.W, padx=(0, 4))
        self.cb_mode = ttk.Combobox(
            top,
            width=14,
            state="readonly",
            values=("Постоянный", "Пакетный"),
        )
        self.cb_mode.grid(row=0, column=1, sticky=tk.W)
        self.cb_mode.current(0)
        self.cb_mode.bind("<<ComboboxSelected>>", self._on_mode_changed)

        self.btn_run = ttk.Button(top, text="Задать", command=self._on_run_stop)
        self.btn_run.grid(row=1, column=0, padx=(0, 8), pady=(8, 0), sticky=tk.W)

        ttk.Label(top, text="COM-порт:").grid(row=0, column=2, padx=(16, 4), sticky=tk.E)
        self.cb_port = ttk.Combobox(top, width=22, state="readonly")
        self.cb_port.grid(row=0, column=3, sticky=tk.W)
        self.cb_port.bind("<<ComboboxSelected>>", self._on_port_pick_changed)

        ttk.Button(top, text="…", width=3, command=self._refresh_ports).grid(
            row=0, column=4, padx=(4, 0), sticky=tk.W
        )

        table_frame = ttk.LabelFrame(self, text="Частоты (6 каналов)", padding=6)
        table_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.freq_boxes: List[ttk.Combobox] = []
        for i in range(6):
            ttk.Label(table_frame, text=f"{i + 1}", width=3).grid(row=i, column=0, sticky=tk.W)
            cb = ttk.Combobox(
                table_frame,
                width=18,
                state="readonly",
                values=self._labels,
            )
            cb.grid(row=i, column=1, sticky=tk.W, pady=2)
            cb.set("0")
            self.freq_boxes.append(cb)

        log_frame = ttk.LabelFrame(self, text="Обмен", padding=4)
        log_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=6, pady=(0, 6))
        self.text_log = tk.Text(log_frame, height=7, wrap=tk.WORD, state=tk.DISABLED)
        self.text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(log_frame, command=self.text_log.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_log["yscrollcommand"] = sb.set

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_ports()
        self._load_init()

    def _append_log(self, s: str) -> None:
        self.text_log["state"] = tk.NORMAL
        self.text_log.insert(tk.END, s)
        self.text_log.see(tk.END)
        self.text_log["state"] = tk.DISABLED

    def _is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def _freq_line(self) -> str:
        return " ".join(cb.get() for cb in self.freq_boxes)

    def _write_line(self, line: str) -> None:
        if not self._is_open():
            return
        assert self._ser is not None
        data = (line + "\n").encode("ascii", errors="replace")
        self._ser.write(data)

    def _read_existing(self) -> str:
        if not self._is_open():
            return ""
        assert self._ser is not None
        time.sleep(0.01)
        n = self._ser.in_waiting
        if n <= 0:
            return ""
        raw = self._ser.read(n)
        return raw.decode("ascii", errors="replace")

    def _refresh_ports(self) -> None:
        old = self.cb_port.get()
        names = [p.device for p in list_ports.comports()]
        self.cb_port["values"] = names
        if len(names) == 1:
            self.cb_port.current(0)
        elif old in names:
            self.cb_port.set(old)
        elif names:
            self.cb_port.current(0)
        else:
            self.cb_port.set("")

    def _load_init(self) -> None:
        path = INIT_FILE
        if not path.is_file():
            return
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            return
        root = tree.getroot()
        if root is None:
            return
        blink = root.find("Blink")
        if blink is None:
            return
        for i in range(6):
            el = blink.find(f"Blink{i}")
            if el is None:
                return
            freq = el.get("Frequency", "0")
            self.freq_boxes[i].set(_nearest_label(freq, self._labels))

    def _save_init(self) -> None:
        path = INIT_FILE
        xml_root: ET.Element
        if path.is_file():
            try:
                tree = ET.parse(path)
                root = tree.getroot()
                xml_root = root if root is not None else ET.Element("Settings")
            except ET.ParseError:
                xml_root = ET.Element("Settings")
        else:
            xml_root = ET.Element("Settings")

        blink = xml_root.find("Blink")
        if blink is None:
            blink = ET.SubElement(xml_root, "Blink")
            for i in range(6):
                ET.SubElement(blink, f"Blink{i}", Frequency="0")
        for i in range(6):
            el = blink.find(f"Blink{i}")
            if el is None:
                el = ET.SubElement(blink, f"Blink{i}")
            el.set("Frequency", self.freq_boxes[i].get())

        tree = ET.ElementTree(xml_root)
        ET.indent(tree, space="  ")
        tree.write(path, encoding="utf-8", xml_declaration=False)

    def _on_mode_changed(self, _event: object | None = None) -> None:
        if not self._is_open():
            return
        try:
            mode_cmd = "M 0" if self.cb_mode.current() == 0 else "M 1"
            self._write_line(mode_cmd)
            time.sleep(0.05)
            s = self._freq_line()
            self._write_line(s)
            self._append_log(f"\n=> {mode_cmd}\n=> {s}\n")
        except (OSError, serial.SerialException) as ex:
            messagebox.showerror("Ошибка", str(ex))

    def _on_run_stop(self) -> None:
        if not self._is_open():
            port = (self.cb_port.get() or "").strip()
            if not port:
                messagebox.showwarning("Порт", "Задайте порт")
                return
            try:
                ser = serial.Serial()
                ser.port = port
                ser.baudrate = 9600
                ser.bytesize = serial.EIGHTBITS
                ser.parity = serial.PARITY_NONE
                ser.stopbits = serial.STOPBITS_ONE
                ser.timeout = 0
                ser.write_timeout = 2
                ser.open()
                self._ser = ser
                time.sleep(0.5)
                s = self._freq_line()
                self._write_line(s)
                time.sleep(0.1)
                reply = self._read_existing()
                self.text_log["state"] = tk.NORMAL
                self.text_log.delete("1.0", tk.END)
                self.text_log["state"] = tk.DISABLED
                self._append_log(f"=> {s}\n<={reply}")
                self.cb_port["state"] = "disabled"
                self.btn_run["text"] = "Остановить"
            except (OSError, serial.SerialException, ValueError) as ex:
                self._ser = None
                messagebox.showerror("Ошибка", str(ex))
        else:
            try:
                self._write_line("0 0 0 0 0 0")
                time.sleep(0.2)
                if self._ser is not None:
                    self._ser.close()
            except (OSError, serial.SerialException) as ex:
                messagebox.showerror("Ошибка", str(ex))
            finally:
                self._ser = None
                self.cb_port["state"] = "readonly"
                self.btn_run["text"] = "Задать"

    def _on_port_pick_changed(self, _event: object | None = None) -> None:
        if self._is_open() and self._ser is not None:
            try:
                self._ser.close()
            except OSError:
                pass
            self._ser = None
            self.cb_port["state"] = "readonly"
            self.btn_run["text"] = "Задать"

    def _on_close(self) -> None:
        try:
            if self._is_open():
                self._write_line("0 0 0 0 0 0")
                time.sleep(0.2)
                if self._ser is not None:
                    self._ser.close()
        except (OSError, serial.SerialException) as ex:
            messagebox.showerror("Ошибка", str(ex))
        finally:
            self._ser = None
        self._save_init()
        self.destroy()


def main() -> None:
    MigalkaApp().mainloop()


if __name__ == "__main__":
    main()
