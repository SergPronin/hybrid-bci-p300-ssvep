#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone SSVEP realtime analyzer: LSL EEG → MSI (MSIController.dll) → GUI.

Использует MSI-хелперы из scripts/test_msi_exec.py без их дублирования.
Стиль интерфейса близок к P300 Analyzer (тёмная тема, pyqtgraph).
"""

from __future__ import annotations

import sys
import traceback
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QSignalBlocker, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pylsl import StreamInfo, StreamInlet

# scripts/ + корень репозитория — для import test_msi_exec
_SCRIPTS = Path(__file__).resolve().parent / "scripts"
_REPO = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_msi_exec as tme  # noqa: E402

from p300_analysis.lsl_streams import resolve_marker_streams, stream_inlet_with_buffer  # noqa: E402
from ssvep_analysis.burst_gate import (  # noqa: E402
    BurstGate,
    BurstGateConfig,
    append_chunk_timestamps,
)
from ssvep_analysis.gantt_timeline import GanttExtraRow, StimGanttChart  # noqa: E402
from ssvep_analysis.migalka_lsl import STREAM_NAME as MIGALKA_MARKER_STREAM  # noqa: E402

# Как в WinForms: for (int i = 1; i <= 500; i++) Items.Add((1000.0f / i).ToString().Replace(',', '.'))
_LAMP_FREQ_CHOICES: List[Tuple[str, float]] = []


def lamp_frequency_choices() -> List[Tuple[str, float]]:
    """500 дискретных частот: 1000/i Гц, i = 1..500; подпись с десятичной точкой."""
    if not _LAMP_FREQ_CHOICES:
        for i in range(1, 501):
            v = 1000.0 / float(i)
            s = f"{v}".replace(",", ".")
            _LAMP_FREQ_CHOICES.append((s, v))
    return _LAMP_FREQ_CHOICES


def lamp_frequency_closest_index(target_hz: float) -> int:
    arr = np.array([v for _, v in lamp_frequency_choices()], dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(target_hz))))


def _resolve_eeg_streams(timeout: float = 2.0) -> List[StreamInfo]:
    """Потоки типа EEG или Signal (как в P300), без фильтра по имени устройства."""
    from pylsl import resolve_byprop

    seen: set[Tuple[str, str]] = set()
    out: List[StreamInfo] = []
    for stype in ("EEG", "Signal"):
        try:
            batch = list(resolve_byprop("type", stype, timeout=float(timeout)))
        except Exception:
            batch = []
        for s in batch:
            try:
                key = (s.name() or "", s.session_id() or "")
            except Exception:
                key = (str(s), "")
            if key not in seen:
                seen.add(key)
                out.append(s)
    return out


def _stream_channel_labels(info: StreamInfo, count: int) -> List[str]:
    """Подписи каналов из LSL desc/XML (аналогично p300_analysis.qt_window)."""
    labels: List[str] = []
    try:
        channels = info.desc().child("channels")
        ch = channels.child("channel")
        for i in range(count):
            if ch is None:
                break
            label = (
                ch.child_value("label")
                or ch.child_value("name")
                or ch.child_value("channel")
                or ""
            )
            label = str(label).strip()
            labels.append(label if label else f"Ch {i + 1}")
            nxt = ch.next_sibling()
            if nxt is None:
                break
            ch = nxt
    except Exception:
        labels = []
    if len(labels) < count:
        try:
            root = ET.fromstring(info.as_xml())
            for ch_el in root.findall(".//channels/channel"):
                if len(labels) >= count:
                    break
                label = (
                    (ch_el.findtext("label") or "").strip()
                    or (ch_el.findtext("name") or "").strip()
                    or (ch_el.findtext("channel") or "").strip()
                )
                labels.append(label if label else f"Ch {len(labels) + 1}")
        except Exception:
            pass
    if len(labels) < count:
        labels.extend([f"Ch {i + 1}" for i in range(len(labels), count)])
    return labels[:count]


def _stream_label(info: StreamInfo) -> str:
    try:
        name = info.name() or "?"
        fs = info.nominal_srate() or 0.0
        nch = info.channel_count()
        return f"{name}  |  {fs:.1f} Hz  |  {nch} ch"
    except Exception:
        return repr(info)


def _coef_to_strings(msi, freqs_hz: Sequence[float]) -> List[str]:
    """Строки для отображения уверенности / коэффициентов MSI."""
    lines: List[str] = []
    try:
        c = msi.Coef
    except Exception as e:
        return [f"(Coef недоступен: {e})"]

    if c is None:
        return ["Coef: None"]

    cnt = getattr(c, "Count", None)
    if cnt is not None:
        n = int(cnt)
        for i in range(n):
            try:
                val = float(c[i])
            except Exception:
                val = c[i]
            hz = freqs_hz[i] if i < len(freqs_hz) else float(i)
            lines.append(f"{hz:g} Hz: {val}")
        return lines if lines else [f"Coef Count={n} (пусто)"]

    # скаляр или другой тип
    try:
        return [f"Coef (raw): {float(c)}"]
    except Exception:
        return [f"Coef (raw): {repr(c)}"]


class SSVEPAnalyzerWindow(QMainWindow):
    DEFAULT_FS = 250.0
    WINDOW_SEC = 2.0
    BUFFER_MARGIN = 1.15
    CLASSIFY_MS = 200
    PULL_MS = 40
    CHANNEL_PLOT_SEP = 100.0
    CHANNEL_CB_COLUMNS = 4
    MAX_LAMPS = 6
    GANTT_SPAN_SEC = 30.0
    GANTT_MSI_HISTORY = 200
    # Частоты по умолчанию для первых 4 ламп (как в Migalka / 1000/i)
    DEFAULT_LAMP_FREQS: Tuple[float, ...] = (
        10.0,
        1000.0 / 99.0,
        1000.0 / 87.0,
        1000.0 / 76.0,
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SSVEP Analyzer (LSL → MSI)")
        self.setMinimumSize(1000, 720)

        pg.setConfigOptions(useOpenGL=False, antialias=False)
        pg.setConfigOption("background", "#0a0a0a")
        pg.setConfigOption("foreground", "#E0E0E0")

        self._inlet: Optional[StreamInlet] = None
        self._marker_inlet: Optional[StreamInlet] = None
        self._stream_info: Optional[StreamInfo] = None
        self._stim_mode: str = "continuous"  # continuous | burst
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=self.WINDOW_SEC))
        self._buf_t: np.ndarray = np.zeros(0, dtype=np.float64)
        self._nominal_fs: float = self.DEFAULT_FS
        self._n_channels: int = 1

        self._msi = None
        self._freqs_hz: List[float] = []
        self._n_template: int = int(round(self.DEFAULT_FS * self.WINDOW_SEC))
        self._freq_row_widgets: List[QWidget] = []
        self._freq_combos: List[QComboBox] = []

        # rolling EEG (samples, channels)
        self._max_buf: int = int(self.DEFAULT_FS * self.WINDOW_SEC * self.BUFFER_MARGIN) + 64
        self._buf: np.ndarray = np.zeros((0, 1), dtype=np.float64)

        self._ch_labels: List[str] = []
        self._ch_checkboxes: List[QCheckBox] = []
        self._plot_curves: List[Any] = []
        self._ch_grid_host: Optional[QWidget] = None
        self._ch_grid_layout: Optional[QGridLayout] = None

        self._last_winner: Optional[int] = None
        self._last_gate_allowed: bool = True
        self._msi_events: Deque[Tuple[float, int]] = deque(maxlen=self.GANTT_MSI_HISTORY)
        self._gate_open_t: Optional[float] = None
        self._gate_intervals: List[Tuple[float, float]] = []
        self._gantt_chart: Optional[StimGanttChart] = None
        self._plot_gantt: Optional[pg.PlotWidget] = None
        self._plot_session: Optional[pg.PlotWidget] = None
        self._curve_session: Optional[Any] = None

        self._session_active = False
        self._session_t0: Optional[float] = None
        self._session_points: List[Tuple[float, int, float, bool]] = []

        self._timer_pull = QTimer(self)
        self._timer_pull.setInterval(self.PULL_MS)
        self._timer_pull.timeout.connect(self._on_pull)

        self._timer_class = QTimer(self)
        self._timer_class.setInterval(self.CLASSIFY_MS)
        self._timer_class.timeout.connect(self._on_classify)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Левая колонка: настройки (прокрутка) ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(300)
        left_panel = QWidget()
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(2, 2, 8, 2)

        stim_row = QHBoxLayout()
        stim_row.addWidget(QLabel("Режим стимула:"))
        self._cb_stim_mode = QComboBox()
        self._cb_stim_mode.addItem("Постоянный", "continuous")
        self._cb_stim_mode.addItem("Пакетный (Migalka)", "burst")
        self._cb_stim_mode.setToolTip(
            "Постоянный: MSI на каждом окне 2 с.\n"
            "Пакетный: MSI только при достаточной вспышке (LSL MigalkaStimMarkers)."
        )
        self._cb_stim_mode.currentIndexChanged.connect(self._on_stim_mode_changed)
        stim_row.addWidget(self._cb_stim_mode, stretch=1)
        left.addLayout(stim_row)

        self._lbl_burst_status = QLabel("Пакетный гейт: —")
        self._lbl_burst_status.setWordWrap(True)
        self._lbl_burst_status.setStyleSheet("color: #aaa; font-size: 12px;")
        left.addWidget(self._lbl_burst_status)

        left.addWidget(QLabel("LSL stream"))
        lsl_box = QWidget()
        lsl_l = QVBoxLayout(lsl_box)
        lsl_l.setContentsMargins(0, 0, 0, 0)
        lsl_btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh streams")
        self._btn_refresh.clicked.connect(self._on_refresh_streams)
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        lsl_btn_row.addWidget(self._btn_refresh)
        lsl_btn_row.addWidget(self._btn_connect)
        lsl_l.addLayout(lsl_btn_row)
        lsl_l.addWidget(QLabel("LSL EEG streams:"))
        self._list_streams = QListWidget()
        self._list_streams.setMinimumHeight(72)
        self._list_streams.setMaximumHeight(120)
        lsl_l.addWidget(self._list_streams)
        left.addWidget(lsl_box)

        self._lbl_stream_meta = QLabel("Not connected")
        self._lbl_stream_meta.setWordWrap(True)
        left.addWidget(self._lbl_stream_meta)

        left.addWidget(QLabel("Частоты SSVEP (лампы), MSI templates"))
        freq_top = QHBoxLayout()
        self._lbl_freq_slots = QLabel(f"Ламп: 0/{self.MAX_LAMPS}")
        freq_top.addWidget(self._lbl_freq_slots)
        freq_top.addStretch(1)
        self._btn_freq_add = QPushButton("+")
        self._btn_freq_add.setToolTip(f"Добавить частоту (ещё одна лампа), не более {self.MAX_LAMPS}.")
        self._btn_freq_add.clicked.connect(self._on_freq_add_row)
        freq_top.addWidget(self._btn_freq_add)
        self._btn_apply_freq = QPushButton("Apply")
        self._btn_apply_freq.setToolTip("Применить частоты к MSI templates")
        self._btn_apply_freq.clicked.connect(self._on_apply_frequencies)
        freq_top.addWidget(self._btn_apply_freq)
        left.addLayout(freq_top)
        self._freq_rows_host = QWidget()
        self._freq_rows_layout = QVBoxLayout(self._freq_rows_host)
        self._freq_rows_layout.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self._freq_rows_host)

        ch_head = QHBoxLayout()
        ch_head.addWidget(QLabel("Channels (plot + MSI)"))
        ch_head.addStretch(1)
        self._btn_ch_all_off = QPushButton("Снять")
        self._btn_ch_all_off.setToolTip(
            "Снять выбор со всех каналов (график пустой, MSI ждёт выбора)."
        )
        self._btn_ch_all_off.clicked.connect(self._on_channels_all_off)
        self._btn_ch_all_on = QPushButton("Все")
        self._btn_ch_all_on.setToolTip("Включить все каналы на графике и в MSI.")
        self._btn_ch_all_on.clicked.connect(self._on_channels_all_on)
        ch_head.addWidget(self._btn_ch_all_off)
        ch_head.addWidget(self._btn_ch_all_on)
        left.addLayout(ch_head)
        ch_scroll = QScrollArea()
        ch_scroll.setWidgetResizable(True)
        ch_scroll.setMaximumHeight(160)
        ch_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ch_grid_host = QWidget()
        self._ch_grid_layout = QGridLayout(self._ch_grid_host)
        ch_scroll.setWidget(self._ch_grid_host)
        left.addWidget(ch_scroll)

        sess_row = QHBoxLayout()
        self._btn_session_start = QPushButton("Старт")
        self._btn_session_start.setToolTip("Начать запись графика сессии (решения MSI по времени).")
        self._btn_session_start.clicked.connect(self._on_session_start)
        self._btn_session_stop = QPushButton("Стоп")
        self._btn_session_stop.setToolTip("Остановить запись и сохранить скриншот.")
        self._btn_session_stop.clicked.connect(self._on_session_stop)
        self._btn_session_reset = QPushButton("Сброс")
        self._btn_session_reset.setToolTip("Очистить график сессии, Gantt и текущий результат.")
        self._btn_session_reset.clicked.connect(self._on_session_reset)
        self._btn_session_save = QPushButton("PNG")
        self._btn_session_save.setToolTip("Сохранить график сессии и Gantt в PNG.")
        self._btn_session_save.clicked.connect(self._on_session_save_png)
        for w in (
            self._btn_session_start,
            self._btn_session_stop,
            self._btn_session_reset,
            self._btn_session_save,
        ):
            sess_row.addWidget(w)
        left.addLayout(sess_row)
        self._lbl_session = QLabel("Сессия: не записывается")
        self._lbl_session.setWordWrap(True)
        self._lbl_session.setStyleSheet("color: #888; font-size: 11px;")
        left.addWidget(self._lbl_session)

        left.addWidget(QLabel("Status / log"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(100)
        self._log.setMaximumHeight(200)
        left.addWidget(self._log)
        left.addStretch(1)

        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        # --- Правая колонка: графики + плитка победителя ---
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 0, 0)

        self._lbl_winner = QLabel("CURRENT TARGET:\n—")
        self._lbl_winner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_winner.setWordWrap(True)
        self._lbl_winner.setMinimumHeight(96)
        self._lbl_winner.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._lbl_winner.setStyleSheet(
            "font-size: 24px; font-weight: bold; padding: 12px 16px; "
            "background-color: #1a1a1a; border: 2px solid #555; border-radius: 6px;"
        )
        right.addWidget(self._lbl_winner)

        self._lbl_scores = QLabel("Scores / Coef:\n—")
        self._lbl_scores.setWordWrap(True)
        self._lbl_scores.setMaximumHeight(72)
        self._lbl_scores.setStyleSheet("font-family: monospace; font-size: 11px; color: #bbb;")
        right.addWidget(self._lbl_scores)

        right.addWidget(QLabel("Realtime EEG"))
        self._plot = pg.PlotWidget(title="EEG (rolling), all channels stacked")
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.setMinimumHeight(140)
        right.addWidget(self._plot, stretch=2)

        right.addWidget(QLabel("График сессии (MSI)"))
        self._plot_session = pg.PlotWidget(title="Сессия: лампа по времени")
        self._plot_session.showGrid(x=True, y=True, alpha=0.3)
        self._plot_session.setMinimumHeight(120)
        self._plot_session.setMaximumHeight(200)
        self._plot_session.setLabel("bottom", "Время сессии", units="s")
        self._plot_session.setLabel("left", "Лампа №")
        self._curve_session = self._plot_session.plot(
            pen=pg.mkPen("#7fd97f", width=2),
            symbol="o",
            symbolSize=7,
            symbolBrush=pg.mkBrush("#7fd97f"),
        )
        right.addWidget(self._plot_session, stretch=0)

        self._gantt_chart, self._plot_gantt = StimGanttChart.create_plot(
            span_sec=self.GANTT_SPAN_SEC
        )
        self._plot_gantt.setMinimumHeight(140)
        self._plot_gantt.setMaximumHeight(280)
        gantt_hint = QLabel(
            "Время → | События ↓ | серая дорожка = нет; цветной ■ = лампа ON; "
            "зелёный ■ = MSI разрешён; голубая зона = окно MSI; ◆ = решение"
        )
        gantt_hint.setWordWrap(True)
        gantt_hint.setStyleSheet("color: #999; font-size: 11px;")
        right.addWidget(gantt_hint)
        right.addWidget(self._plot_gantt, stretch=0)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 860])
        outer.addWidget(splitter)

        self._seed_default_lamps()
        self._log_line(
            "Заданы 4 лампы по умолчанию — Apply, Connect EEG, Старт для записи графика."
        )
        self._update_freq_slot_label()

    def _clear_channel_widgets(self) -> None:
        if self._ch_grid_layout is None:
            return
        while self._ch_grid_layout.count():
            item = self._ch_grid_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._ch_checkboxes.clear()
        for c in self._plot_curves:
            try:
                self._plot.removeItem(c)
            except Exception:
                pass
        self._plot_curves.clear()
        self._ch_labels.clear()

    def _rebuild_channel_controls(self) -> None:
        """Чекбоксы и кривые под текущее число каналов потока."""
        self._clear_channel_widgets()
        n = self._n_channels
        if n < 1 or self._ch_grid_layout is None:
            return
        info = self._stream_info
        if info is not None:
            self._ch_labels = _stream_channel_labels(info, n)
        else:
            self._ch_labels = [f"Ch {i + 1}" for i in range(n)]

        cols = self.CHANNEL_CB_COLUMNS
        for i in range(n):
            text = f"{i + 1}: {self._ch_labels[i]}"
            cb = QCheckBox(text)
            cb.setChecked(True)
            self._ch_checkboxes.append(cb)
            self._ch_grid_layout.addWidget(cb, i // cols, i % cols)

        nv = max(n, 8)
        for i in range(n):
            pen = pg.mkPen(pg.intColor(i, values=nv), width=1)
            self._plot_curves.append(self._plot.plot(pen=pen))

        self._log_line(
            f"Channel UI: {n} channel(s). Кнопки «Снять все» / «Выбрать все» — массовый выбор; "
            "если ни один канал не отмечен, MSI не вызывается."
        )

    def _set_all_channel_checkboxes(self, checked: bool) -> None:
        for cb in self._ch_checkboxes:
            with QSignalBlocker(cb):
                cb.setChecked(checked)

    def _on_channels_all_off(self) -> None:
        if not self._ch_checkboxes:
            return
        self._set_all_channel_checkboxes(False)
        self._log_line("Все каналы сняты. Включите нужные — без выбранных каналов MSI не считает.")

    def _on_channels_all_on(self) -> None:
        if not self._ch_checkboxes:
            return
        self._set_all_channel_checkboxes(True)
        self._log_line("Все каналы выбраны (график + MSI).")

    def _selected_channel_indices(self) -> List[int]:
        return [i for i, cb in enumerate(self._ch_checkboxes) if cb.isChecked()]

    def _gantt_lamp_labels(self) -> List[str]:
        labels: List[str] = []
        for i, hz in enumerate(self._freqs_hz):
            labels.append(f"L{i + 1} {hz:g} Hz")
        return labels

    def _clear_gate_history(self) -> None:
        self._gate_open_t = None
        self._gate_intervals.clear()

    def _record_gate_state(self, t_now: float, allowed: bool) -> None:
        """Накопить интервалы «MSI разрешён» для строки Gantt."""
        if allowed:
            if self._gate_open_t is None:
                self._gate_open_t = float(t_now)
        elif self._gate_open_t is not None:
            self._gate_intervals.append((self._gate_open_t, float(t_now)))
            self._gate_open_t = None
        span = self.GANTT_SPAN_SEC + 5.0
        cutoff = float(t_now) - span
        self._gate_intervals = [
            (a, b) for a, b in self._gate_intervals if b >= cutoff
        ]

    def _gate_intervals_for_gantt(self, t_now: float) -> List[Tuple[float, float]]:
        out = list(self._gate_intervals)
        if self._gate_open_t is not None:
            out.append((self._gate_open_t, float(t_now)))
        return out

    def _update_gantt(
        self,
        *,
        gate_allowed: Optional[bool] = None,
        record_msi: Optional[Tuple[int, bool]] = None,
    ) -> None:
        if self._gantt_chart is None or self._buf_t.size == 0:
            return
        t_now = float(self._buf_t[-1])
        n_lamps = len(self._freqs_hz)
        if n_lamps < 1:
            self._gantt_chart.clear()
            return

        if gate_allowed is not None:
            self._last_gate_allowed = bool(gate_allowed)
            if self._is_burst_mode():
                self._record_gate_state(t_now, self._last_gate_allowed)
        if record_msi is not None:
            winner, ok = record_msi
            if ok:
                self._msi_events.append((t_now, int(winner)))

        t_min = t_now - self.GANTT_SPAN_SEC
        intervals = self._burst_gate.intervals_in_range(t_min, t_now)
        extra: List[GanttExtraRow] = []
        if self._is_burst_mode():
            extra.append(
                GanttExtraRow(
                    label="MSI разрешён",
                    intervals=self._gate_intervals_for_gantt(t_now),
                    color="#43a047",
                )
            )
        self._gantt_chart.update(
            t_now=t_now,
            n_lamps=n_lamps,
            lamp_labels=self._gantt_lamp_labels(),
            intervals=intervals,
            extra_rows=extra,
            msi_window_sec=self.WINDOW_SEC,
            gate_allowed=self._last_gate_allowed,
            msi_events=list(self._msi_events),
        )

    def _log_line(self, msg: str) -> None:
        self._log.append(msg)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer_pull.stop()
        self._timer_class.stop()
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
            self._inlet = None
        if self._marker_inlet is not None:
            try:
                self._marker_inlet.close_stream()
            except Exception:
                pass
            self._marker_inlet = None
        super().closeEvent(event)

    def _is_burst_mode(self) -> bool:
        return self._cb_stim_mode.currentData() == "burst"

    def _on_stim_mode_changed(self) -> None:
        if self._is_burst_mode():
            self._log_line(
                "Пакетный режим: запустите migalka.py, затем Connect EEG. "
                f"Нужен LSL «{MIGALKA_MARKER_STREAM}»."
            )
            self._try_connect_markers()
        else:
            self._lbl_burst_status.setText("Пакетный гейт: выкл (постоянный режим)")
            self._lbl_burst_status.setStyleSheet("color: #888; font-size: 12px;")

    def _try_connect_markers(self) -> bool:
        if self._marker_inlet is not None:
            return True
        try:
            streams = resolve_marker_streams(timeout=1.5, attempts=2)
        except Exception as e:
            self._log_line(f"Markers resolve error: {e}")
            return False
        pick: Optional[StreamInfo] = None
        for s in streams:
            try:
                if (s.name() or "") == MIGALKA_MARKER_STREAM:
                    pick = s
                    break
            except Exception:
                continue
        if pick is None and streams:
            pick = streams[0]
        if pick is None:
            self._lbl_burst_status.setText(
                f"Маркеры: не найдены (запустите migalka.py → «{MIGALKA_MARKER_STREAM}»)"
            )
            self._lbl_burst_status.setStyleSheet("color: #d9534f; font-size: 12px;")
            return False
        try:
            self._marker_inlet = stream_inlet_with_buffer(pick, buffer_seconds=60)
            self._marker_inlet.open_stream(timeout=2.0)
            self._lbl_burst_status.setText(f"Маркеры: {pick.name()!r}")
            self._lbl_burst_status.setStyleSheet("color: #5cb85c; font-size: 12px;")
            self._log_line(f"Marker stream connected: {pick.name()!r}")
            return True
        except Exception as e:
            self._log_line(f"Marker connect failed: {e}")
            self._marker_inlet = None
            return False

    def _pull_markers(self) -> None:
        if self._marker_inlet is None:
            return
        try:
            try:
                chunk, stamps = self._marker_inlet.pull_chunk(timeout=0.0, max_samples=256)
            except TypeError:
                chunk, stamps = self._marker_inlet.pull_chunk(timeout=0.0)
        except Exception as e:
            self._log_line(f"Marker pull error: {e}")
            return
        if not chunk:
            return
        for sample, ts in zip(chunk, stamps):
            try:
                val = sample[0] if sample else ""
            except (TypeError, IndexError):
                val = sample
            self._burst_gate.ingest_marker(float(ts), val)

    def _on_refresh_streams(self) -> None:
        self._list_streams.clear()
        try:
            streams = _resolve_eeg_streams(timeout=2.0)
        except Exception as e:
            self._log_line(f"Refresh error: {e}")
            return
        for s in streams:
            it = QListWidgetItem(_stream_label(s))
            it.setData(Qt.ItemDataRole.UserRole, s)
            self._list_streams.addItem(it)
        self._log_line(f"Found {len(streams)} stream(s).")

    def _on_connect(self) -> None:
        row = self._list_streams.currentRow()
        if row < 0:
            QMessageBox.warning(self, "LSL", "Select a stream from the list.")
            return
        item = self._list_streams.item(row)
        info = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(info, StreamInfo):
            QMessageBox.warning(self, "LSL", "Invalid stream selection.")
            return

        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
            self._inlet = None

        try:
            self._inlet = stream_inlet_with_buffer(info, buffer_seconds=8)
        except Exception as e:
            self._log_line(f"Connect failed: {e}")
            QMessageBox.critical(self, "LSL", str(e))
            return

        self._stream_info = info
        fs = float(info.nominal_srate() or 0.0)
        self._nominal_fs = fs if fs > 1.0 else self.DEFAULT_FS
        self._n_channels = int(info.channel_count())
        if self._n_channels < 1:
            self._n_channels = 1

        self._max_buf = int(self._nominal_fs * self.WINDOW_SEC * self.BUFFER_MARGIN) + 64
        self._buf = np.zeros((0, self._n_channels), dtype=np.float64)
        self._buf_t = np.zeros(0, dtype=np.float64)
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=self.WINDOW_SEC))
        self._msi_events.clear()
        self._clear_gate_history()
        if self._gantt_chart is not None:
            self._gantt_chart.clear()

        meta = (
            f"Connected: {info.name()!r}\n"
            f"nominal_srate = {info.nominal_srate()!r} (using {self._nominal_fs:g} Hz for buffer/templates)\n"
            f"channels = {self._n_channels}"
        )
        self._lbl_stream_meta.setText(meta)
        self._log_line("Stream connected.")
        if fs <= 1.0:
            self._log_line("Warning: nominal_srate missing/low — using default 250 Hz for MSI window length.")

        self._rebuild_channel_controls()

        self._timer_pull.start()
        self._timer_class.start()

        if self._is_burst_mode():
            self._try_connect_markers()

        # пересобрать шаблоны под фактический fs, если MSI уже инициализирован
        if self._msi is not None and self._freqs_hz:
            self._apply_model_signals_to_msi()

    def _update_freq_slot_label(self) -> None:
        n = len(self._freq_combos)
        self._lbl_freq_slots.setText(f"Ламп: {n}/{self.MAX_LAMPS}")
        self._btn_freq_add.setEnabled(n < self.MAX_LAMPS)

    def _renumber_freq_rows(self) -> None:
        for idx, row in enumerate(self._freq_row_widgets):
            lay = row.layout()
            if lay is None or lay.count() < 1:
                continue
            w0 = lay.itemAt(0).widget()
            if isinstance(w0, QLabel):
                w0.setText(f"Лампа {idx + 1}")

    def _remove_freq_row(self, row: QWidget) -> None:
        try:
            idx = self._freq_row_widgets.index(row)
        except ValueError:
            return
        self._freq_row_widgets.pop(idx)
        self._freq_combos.pop(idx)
        self._freq_rows_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        self._renumber_freq_rows()
        self._update_freq_slot_label()
        self._log_line(f"Лампа удалена. Осталось частот: {len(self._freq_combos)}.")

    def _seed_default_lamps(self) -> None:
        for hz in self.DEFAULT_LAMP_FREQS:
            self._add_freq_row(initial_hz=float(hz), log=False)
        if self._freq_combos:
            self._log_line(
                "Лампы 1–4: "
                + ", ".join(f"{float(self.DEFAULT_LAMP_FREQS[i]):g} Hz" for i in range(len(self.DEFAULT_LAMP_FREQS)))
            )

    def _add_freq_row(self, *, initial_hz: float, log: bool = True) -> None:
        if len(self._freq_combos) >= self.MAX_LAMPS:
            return
        k = len(self._freq_combos)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel(f"Лампа {k + 1}"))
        combo = QComboBox()
        combo.setMinimumWidth(160)
        for text, val in lamp_frequency_choices():
            combo.addItem(text, val)
        combo.setCurrentIndex(lamp_frequency_closest_index(initial_hz))
        combo.setToolTip("Список частот: 1000/i Гц для i = 1 … 500.")
        h.addWidget(combo, stretch=1)
        rm = QPushButton("−")
        rm.setFixedWidth(40)
        rm.setToolTip("Убрать эту частоту")
        rm.clicked.connect(lambda *_a, rw=row: self._remove_freq_row(rw))
        h.addWidget(rm)
        self._freq_rows_layout.addWidget(row)
        self._freq_row_widgets.append(row)
        self._freq_combos.append(combo)
        self._update_freq_slot_label()
        if log:
            sel = float(combo.currentData())
            self._log_line(f"Добавлена лампа {len(self._freq_combos)}: {sel:g} Hz → Apply frequencies.")

    def _on_freq_add_row(self) -> None:
        if len(self._freq_combos) >= self.MAX_LAMPS:
            return
        extras = (12.0, 15.0, 20.0, 8.0, 18.0)
        k = len(self._freq_combos)
        initial = float(extras[(k - len(self.DEFAULT_LAMP_FREQS)) % len(extras)]) if k >= len(
            self.DEFAULT_LAMP_FREQS
        ) else float(self.DEFAULT_LAMP_FREQS[k])
        self._add_freq_row(initial_hz=initial)

    def _session_time_base(self) -> Optional[float]:
        if self._buf_t.size > 0:
            return float(self._buf_t[-1])
        return None

    def _on_session_start(self) -> None:
        if self._inlet is None:
            QMessageBox.warning(self, "Сессия", "Сначала подключите LSL EEG (Connect).")
            return
        if not self._freqs_hz:
            QMessageBox.warning(self, "Сессия", "Сначала нажмите Apply frequencies.")
            return
        t0 = self._session_time_base()
        if t0 is None:
            QMessageBox.warning(self, "Сессия", "Нет данных EEG — дождитесь потока.")
            return
        self._session_active = True
        self._session_t0 = t0
        self._session_points.clear()
        self._refresh_session_plot()
        self._lbl_session.setText("Сессия: запись…")
        self._lbl_session.setStyleSheet("color: #5cb85c;")
        self._log_line(f"Сессия: старт (t0 LSL = {t0:.3f})")

    def _on_session_stop(self) -> None:
        if not self._session_active:
            self._log_line("Сессия: не была запущена.")
            return
        self._session_active = False
        self._lbl_session.setText(f"Сессия: стоп, точек {len(self._session_points)}")
        self._lbl_session.setStyleSheet("color: #f0ad4e;")
        self._refresh_session_plot()
        self._log_line("Сессия: стоп — можно сохранить PNG.")
        self._on_session_save_png(auto_path=True)

    def _on_session_reset(self) -> None:
        self._session_active = False
        self._session_t0 = None
        self._session_points.clear()
        self._msi_events.clear()
        self._clear_gate_history()
        self._last_winner = None
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=self.WINDOW_SEC))
        if self._freqs_hz:
            self._burst_gate.set_active_lamps(len(self._freqs_hz))
        if self._curve_session is not None:
            self._curve_session.setData([], [])
        if self._gantt_chart is not None:
            self._gantt_chart.clear()
        self._lbl_winner.setText("CURRENT TARGET:\n—")
        self._lbl_scores.setText("Scores / Coef:\n—")
        self._lbl_session.setText("Сессия: сброшена")
        self._lbl_session.setStyleSheet("color: #888;")
        self._log_line("Сессия: сброс (график, Gantt, winner).")

    def _on_session_save_png(self, *, auto_path: bool = False) -> None:
        out_dir = _REPO / "screenshots"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default = out_dir / f"ssvep_session_{stamp}.png"
        path: Optional[Path]
        if auto_path:
            path = default
        else:
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить график сессии",
                str(default),
                "PNG (*.png)",
            )
            path = Path(chosen) if chosen else None
        if path is None:
            return
        try:
            from pyqtgraph.exporters import ImageExporter

            if self._plot_session is not None:
                exporter = ImageExporter(self._plot_session.plotItem)
                exporter.parameters()["width"] = 1200
                exporter.export(str(path))
            if self._plot_gantt is not None:
                gpath = path.with_name(path.stem + "_gantt" + path.suffix)
                exporter_g = ImageExporter(self._plot_gantt.plotItem)
                exporter_g.parameters()["width"] = 1200
                exporter_g.export(str(gpath))
            self._log_line(f"Сохранено: {path}")
            if auto_path:
                QMessageBox.information(self, "Сохранено", f"График:\n{path}\n\nGantt:\n{path.with_name(path.stem + '_gantt' + path.suffix)}")
        except Exception as e:
            QMessageBox.critical(self, "PNG", str(e))

    def _append_session_point(self, winner: int, hz: float, gate_ok: bool) -> None:
        if not self._session_active or self._session_t0 is None:
            return
        t_base = self._session_time_base()
        if t_base is None:
            return
        t_rel = float(t_base) - float(self._session_t0)
        self._session_points.append((t_rel, int(winner), float(hz), bool(gate_ok)))
        self._refresh_session_plot()

    def _refresh_session_plot(self) -> None:
        if self._curve_session is None:
            return
        if not self._session_points:
            self._curve_session.setData([], [])
            return
        xs: List[float] = []
        ys: List[float] = []
        for t_rel, winner, _hz, gate_ok in self._session_points:
            if not gate_ok or winner < 1:
                continue
            xs.append(t_rel)
            ys.append(float(winner))
        self._curve_session.setData(xs, ys)
        if xs:
            n_lamps = max(len(self._freqs_hz), 1)
            self._plot_session.setXRange(0, max(xs[-1], 1.0), padding=0.05)
            self._plot_session.setYRange(0.5, float(n_lamps) + 0.5, padding=0)

    def _collect_frequencies_from_ui(self) -> Optional[List[float]]:
        if not self._freq_combos:
            QMessageBox.warning(
                self,
                "Частоты",
                f"Нет ни одной частоты. Нажмите «+», чтобы добавить лампу (до {self.MAX_LAMPS}).",
            )
            return None
        out: List[float] = []
        for combo in self._freq_combos:
            v = combo.currentData()
            if v is None:
                QMessageBox.warning(self, "Частоты", "Не удалось прочитать выбранную частоту.")
                return None
            fv = float(v)
            if fv <= 0:
                QMessageBox.warning(self, "Частоты", "Частота должна быть больше нуля.")
                return None
            out.append(fv)
        return out

    def _ensure_msi(self) -> bool:
        if self._msi is not None:
            return True
        try:
            self._msi, msi_res, dotnet_root = tme.load_msi_runtime()
        except Exception as e:
            self._log_line(f"MSI load failed: {e}")
            QMessageBox.critical(self, "MSI", f"{e}\n\nSee scripts/test_msi_import.py / test_msi_exec.py.")
            return False
        self._log_line(f"MSI runtime loaded (msi-res={msi_res}, DOTNET_ROOT={dotnet_root}).")
        return True

    def _apply_model_signals_to_msi(self) -> None:
        assert self._msi is not None
        fs = self._nominal_fs
        np_models = tme.generate_model_signals(self._freqs_hz, fs, self.WINDOW_SEC)
        self._n_template = int(np_models[0].shape[1]) if np_models else int(round(fs * self.WINDOW_SEC))
        model_list = tme.build_model_signal_list(self._msi, np_models, verbose=False)
        self._msi.ModelSignal = model_list
        self._log_line(
            f"Frequencies updated: {', '.join(f'{f:g}' for f in self._freqs_hz)} Hz; "
            f"template length = {self._n_template} samples @ {fs:g} Hz."
        )

    def _on_apply_frequencies(self) -> None:
        freqs = self._collect_frequencies_from_ui()
        if freqs is None:
            return
        self._freqs_hz = freqs
        self._burst_gate.set_active_lamps(len(freqs))
        if not self._ensure_msi():
            return
        try:
            self._apply_model_signals_to_msi()
        except Exception as e:
            self._log_line(f"Apply frequencies error: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "MSI", str(e))
            return
        self._last_winner = None

    def _on_pull(self) -> None:
        if self._inlet is None:
            return
        try:
            try:
                chunk, ts = self._inlet.pull_chunk(timeout=0.0, max_samples=256)
            except TypeError:
                chunk, ts = self._inlet.pull_chunk(timeout=0.0)
        except Exception as e:
            self._log_line(f"LSL pull error: {e}")
            return
        if not chunk:
            if self._is_burst_mode():
                self._pull_markers()
            return
        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] != self._n_channels:
            # динамическое число каналов (редко)
            self._n_channels = int(arr.shape[1])
            self._buf = np.zeros((0, self._n_channels), dtype=np.float64)
            self._buf_t = np.zeros(0, dtype=np.float64)
            self._rebuild_channel_controls()

        n_new = int(arr.shape[0])
        self._buf_t = append_chunk_timestamps(
            self._buf_t, ts if ts is not None else [], n_new, self._nominal_fs
        )
        self._buf = np.vstack([self._buf, arr])
        if self._buf.shape[0] > self._max_buf:
            self._buf = self._buf[-self._max_buf :]
            self._buf_t = self._buf_t[-self._max_buf :]

        if self._is_burst_mode():
            self._pull_markers()

        self._update_gantt()

        n_show = min(self._buf.shape[0], int(self._nominal_fs * 3))
        tail = self._buf[-n_show:]
        x = np.arange(tail.shape[0], dtype=np.float64) / max(self._nominal_fs, 1.0)
        n_ch = tail.shape[1]
        for i, curve in enumerate(self._plot_curves):
            if i >= n_ch:
                curve.clear()
                continue
            if i < len(self._ch_checkboxes) and not self._ch_checkboxes[i].isChecked():
                curve.clear()
                continue
            curve.setData(x, tail[:, i] + float(i) * self.CHANNEL_PLOT_SEP)

    def _on_classify(self) -> None:
        if self._msi is None or not self._freqs_hz:
            return
        if self._buf.shape[0] < self._n_template:
            return
        ch_idx = self._selected_channel_indices()
        if not ch_idx:
            return

        gate_ok = True
        if self._is_burst_mode():
            if self._marker_inlet is None and not self._try_connect_markers():
                self._lbl_winner.setText("CURRENT TARGET:\n—\n\n(нет LSL маркеров)")
                self._lbl_burst_status.setText("Пауза / нет MigalkaStimMarkers")
                self._update_gantt(gate_allowed=False)
                self._append_session_point(0, 0.0, False)
                return
            allowed, reason = self._burst_gate.classify_allowed(self._buf_t)
            gate_ok = allowed
            self._lbl_burst_status.setText(f"Пакетный гейт: {reason}")
            if not allowed:
                self._lbl_burst_status.setStyleSheet("color: #f0ad4e; font-size: 12px;")
                self._lbl_winner.setText(f"CURRENT TARGET:\n—\n\nПАУЗА\n({reason})")
                self._update_gantt(gate_allowed=False)
                self._append_session_point(0, 0.0, False)
                return
            self._lbl_burst_status.setStyleSheet("color: #5cb85c; font-size: 12px;")
        else:
            self._last_gate_allowed = True

        win_full = self._buf[-self._n_template :].T.copy()  # (channels, samples)
        win = np.ascontiguousarray(win_full[ch_idx, :], dtype=np.float64)
        try:
            managed = tme.numpy_to_double_matrix2d(win, verbose=False)
            winner = int(self._msi.MSIExec(managed))
        except Exception as e:
            self._log_line(f"MSIExec error: {e}")
            return

        if winner != self._last_winner:
            self._log_line(f"Winner changed → {winner}")
            self._last_winner = winner

        # 1-based индекс шаблона (как в smoke test test_msi_exec)
        idx0 = winner - 1
        if 0 <= idx0 < len(self._freqs_hz):
            hz = self._freqs_hz[idx0]
            text = f"CURRENT TARGET:\n{hz:g} Hz"
            sub = f"SELECTED:\n{winner}"
        else:
            text = f"CURRENT TARGET:\n(winner index {winner})"
            sub = "SELECTED:\n—"
        self._lbl_winner.setText(f"{text}\n\n{sub}")

        score_lines = _coef_to_strings(self._msi, self._freqs_hz)
        self._lbl_scores.setText("Scores / Coef:\n" + "\n".join(score_lines))
        self._update_gantt(gate_allowed=gate_ok, record_msi=(winner, gate_ok))
        hz_out = self._freqs_hz[winner - 1] if 0 <= winner - 1 < len(self._freqs_hz) else 0.0
        self._append_session_point(winner, hz_out, gate_ok)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = SSVEPAnalyzerWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
