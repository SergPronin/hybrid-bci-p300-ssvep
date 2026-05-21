#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone SSVEP realtime analyzer: LSL EEG → MSI (MSIController.dll) → GUI.

Использует MSI-хелперы из scripts/test_msi_exec.py без их дублирования.
Стиль интерфейса близок к P300 Analyzer (тёмная тема, pyqtgraph).
"""

from __future__ import annotations

import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

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
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QDoubleSpinBox,
    QGroupBox,
    QSpinBox,
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
    parse_lsl_marker,
)
from ssvep_analysis.burst_debug import (  # noqa: E402
    BurstDebugSession,
    expected_msi_lamp_from_diag,
)
from ssvep_analysis.experiment_logger import (  # noqa: E402
    SSVEPExperimentLogger,
    coef_values,
)
from ssvep_analysis.gantt_timeline import GanttExtraRow, StimGanttChart  # noqa: E402
from ssvep_analysis.migalka_lsl import STREAM_NAME as MIGALKA_MARKER_STREAM  # noqa: E402

EXPERIMENT_LOG_DIR = _REPO / "ssvep_experiment_logs"
BURST_DEBUG_DIR = _REPO / "ssvep_burst_debug"

from ssvep_analysis.lamp_frequencies import (  # noqa: E402
    CHANNEL_CB_COLUMNS as _CHANNEL_CB_COLUMNS,
    MSI_DEFAULT_FS as _MSI_DEFAULT_FS,
    MSI_DEFAULT_WINDOW_SEC as _MSI_DEFAULT_WINDOW_SEC,
    lamp_frequency_choices,
    lamp_frequency_closest_index,
)


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
    DEFAULT_FS = _MSI_DEFAULT_FS
    WINDOW_SEC = _MSI_DEFAULT_WINDOW_SEC
    BUFFER_MARGIN = 1.15
    MAX_ROLLING_BUFFER_SEC = 8.0
    CLASSIFY_MS = 200
    PULL_MS = 40
    PLOT_REFRESH_MS = 120
    GANTT_REFRESH_MS = 450
    LOG_MAX_LINES = 250
    # В поток ламп — не чаще одного MSI-решения на длину окна (см. _append_winner_trace).
    WINNER_TRACE_PER_LINE = 24
    CHANNEL_PLOT_SEP = 100.0
    CHANNEL_CB_COLUMNS = _CHANNEL_CB_COLUMNS
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
        self._msi_window_sec: float = self.WINDOW_SEC
        self._burst_gate = BurstGate(BurstGateConfig(window_sec=self._msi_window_sec))
        self._buf_t: np.ndarray = np.zeros(0, dtype=np.float64)
        self._nominal_fs: float = self.DEFAULT_FS
        self._n_channels: int = 1
        self._last_msi_exec_ms: Optional[float] = None
        self._last_msi_run_winner: Optional[int] = None
        self._last_msi_run_n: int = 0

        self._msi = None
        self._freqs_hz: List[float] = []
        self._n_template: int = int(round(self.DEFAULT_FS * self._msi_window_sec))
        self._freq_row_widgets: List[QWidget] = []
        self._freq_combos: List[QComboBox] = []

        # rolling EEG (samples, channels) — кольцевой буфер фикс. размера
        self._max_buf: int = 0
        self._ring_eeg: Optional[np.ndarray] = None
        self._ring_t: Optional[np.ndarray] = None
        self._ring_fill: int = 0
        self._buf: np.ndarray = np.zeros((0, 1), dtype=np.float64)

        self._ch_labels: List[str] = []
        self._ch_checkboxes: List[QCheckBox] = []
        self._plot_curves: List[Any] = []
        self._ch_grid_host: Optional[QWidget] = None
        self._ch_grid_layout: Optional[QGridLayout] = None

        self._last_winner: Optional[int] = None
        self._winner_trace: List[int] = []
        self.WINNER_TRACE_MAX = 800
        self._last_trace_lsl_t: Optional[float] = None
        self._last_burst_msi_lsl_t: Optional[float] = None
        self._burst_debug: Optional[BurstDebugSession] = None
        self._last_plot_mono: float = 0.0
        self._last_gantt_mono: float = 0.0
        self._last_gate_debug_allowed: Optional[bool] = None
        self._last_gate_allowed: bool = True
        self._msi_events: Deque[Tuple[float, int]] = deque(maxlen=self.GANTT_MSI_HISTORY)
        self._gate_open_t: Optional[float] = None
        self._gate_intervals: List[Tuple[float, float]] = []
        self._gantt_chart: Optional[StimGanttChart] = None
        self._plot_gantt: Optional[pg.PlotWidget] = None
        self._plot_session: Optional[pg.PlotWidget] = None
        self._curve_session: Optional[Any] = None

        self._analysis_active = False
        self._session_t0: Optional[float] = None
        self._session_points: List[Tuple[float, int, float, bool]] = []
        self._exp_logger: Optional[SSVEPExperimentLogger] = None
        self._last_log_dir: Optional[Path] = None

        self._timer_pull = QTimer(self)
        self._timer_pull.setInterval(self.PULL_MS)
        self._timer_pull.timeout.connect(self._on_pull)

        self._timer_class = QTimer(self)
        self._timer_class.setInterval(self.CLASSIFY_MS)
        self._timer_class.timeout.connect(self._on_classify)

        self._recompute_msi_buffer()
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

        msi_box = QGroupBox("Параметры MSI")
        msi_grid = QGridLayout(msi_box)
        msi_grid.addWidget(QLabel("Дискретизация, Гц:"), 0, 0)
        self._spin_msi_fs = QSpinBox()
        self._spin_msi_fs.setRange(1, 20000)
        self._spin_msi_fs.setValue(int(round(self.DEFAULT_FS)))
        self._spin_msi_fs.setToolTip(
            "Частота дискретизации для шаблонов sin/cos и длины окна MSIExec.\n"
            "После Connect подставляется из LSL (можно изменить вручную)."
        )
        self._spin_msi_fs.valueChanged.connect(self._on_msi_fs_changed)
        msi_grid.addWidget(self._spin_msi_fs, 0, 1)

        msi_grid.addWidget(QLabel("Окно MSI, отсч.:"), 1, 0)
        self._spin_msi_samples = QSpinBox()
        self._spin_msi_samples.setRange(8, 200000)
        self._spin_msi_samples.setValue(self._n_template)
        self._spin_msi_samples.setToolTip(
            "Сколько последних сэмплов EEG подаётся в MSIExec за один прогон (не накапливается).\n"
            "Связано с полем «Окно, с»: отсчёты = Гц × секунды."
        )
        self._spin_msi_samples.valueChanged.connect(self._on_msi_samples_changed)
        msi_grid.addWidget(self._spin_msi_samples, 1, 1)

        msi_grid.addWidget(QLabel("Окно, с:"), 2, 0)
        self._spin_msi_window_sec = QDoubleSpinBox()
        self._spin_msi_window_sec.setRange(0.1, 120.0)
        self._spin_msi_window_sec.setDecimals(2)
        self._spin_msi_window_sec.setSingleStep(0.1)
        self._spin_msi_window_sec.setValue(self._msi_window_sec)
        self._spin_msi_window_sec.setToolTip(
            "Длительность окна = число отсчётов / дискретизация. Меняет оба поля согласованно."
        )
        self._spin_msi_window_sec.valueChanged.connect(self._on_msi_window_sec_changed)
        msi_grid.addWidget(self._spin_msi_window_sec, 2, 1)

        msi_grid.addWidget(QLabel("Точек:"), 3, 0)
        self._lbl_msi_points = QLabel("0")
        self._lbl_msi_points.setToolTip("Число частот / шаблонов ModelSignal (лампы).")
        msi_grid.addWidget(self._lbl_msi_points, 3, 1)

        msi_grid.addWidget(QLabel("Готов к MSI:"), 4, 0)
        self._lbl_msi_ready = QLabel("нет EEG")
        self._lbl_msi_ready.setToolTip(
            "В rolling-буфере должно быть ≥ «Окно MSI, отсч.» сэмплов.\n"
            "Число в буфере растёт при приёме LSL и не сбрасывается после MSI — это нормально."
        )
        msi_grid.addWidget(self._lbl_msi_ready, 4, 1)

        msi_grid.addWidget(QLabel("Последний MSI:"), 5, 0)
        self._lbl_msi_last_run = QLabel("—")
        self._lbl_msi_last_run.setToolTip(
            "Обновляется после каждого MSIExec: сколько отсчётов взято и номер лампы (L1…)."
        )
        msi_grid.addWidget(self._lbl_msi_last_run, 5, 1)

        msi_grid.addWidget(QLabel("Время MSI:"), 6, 0)
        self._lbl_msi_exec = QLabel("—")
        self._lbl_msi_exec.setToolTip("Длительность последнего вызова MSIExec.")
        msi_grid.addWidget(self._lbl_msi_exec, 6, 1)

        msi_grid.addWidget(QLabel("Целевая лампа №:"), 7, 0)
        self._spin_target_lamp = QSpinBox()
        self._spin_target_lamp.setRange(0, self.MAX_LAMPS)
        self._spin_target_lamp.setValue(0)
        self._spin_target_lamp.setSpecialValueText("—")
        self._spin_target_lamp.setToolTip(
            "Номер лампы (1…N), на которую смотрит человек (эталон для лога).\n"
            "Обязательно перед «Начать анализ»."
        )
        self._spin_target_lamp.valueChanged.connect(self._on_target_lamp_changed)
        msi_grid.addWidget(self._spin_target_lamp, 7, 1)

        msi_grid.addWidget(QLabel("Цель vs MSI:"), 8, 0)
        self._lbl_target_match = QLabel("—")
        self._lbl_target_match.setWordWrap(True)
        self._lbl_target_match.setToolTip("Совпадение последнего решения MSI с целевой лампой.")
        msi_grid.addWidget(self._lbl_target_match, 8, 1)
        left.addWidget(msi_box)

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

        anal_row = QHBoxLayout()
        self._btn_analysis_start = QPushButton("Начать анализ")
        self._btn_analysis_start.setToolTip(
            "Запись полного лога (EEG, маркеры, MSI, параметры) + график сессии."
        )
        self._btn_analysis_start.clicked.connect(self._on_analysis_start)
        self._btn_analysis_stop = QPushButton("Остановить")
        self._btn_analysis_stop.setToolTip("Остановить анализ, сохранить лог и PNG в ssvep_experiment_logs/.")
        self._btn_analysis_stop.clicked.connect(self._on_analysis_stop)
        self._btn_analysis_reset = QPushButton("Сбросить")
        self._btn_analysis_reset.setToolTip("Сброс графиков, Gantt, winner; при записи — сохранить лог и остановить.")
        self._btn_analysis_reset.clicked.connect(self._on_analysis_reset)
        self._btn_save_log = QPushButton("Сохранить лог")
        self._btn_save_log.setToolTip(
            f"Принудительно завершить запись в {EXPERIMENT_LOG_DIR.name}/ (если идёт анализ)."
        )
        self._btn_save_log.clicked.connect(self._on_save_experiment_log)
        for w in (
            self._btn_analysis_start,
            self._btn_analysis_stop,
            self._btn_analysis_reset,
            self._btn_save_log,
        ):
            anal_row.addWidget(w)
        left.addLayout(anal_row)
        self._lbl_analysis = QLabel("Анализ: не запущен")
        self._lbl_analysis.setWordWrap(True)
        self._lbl_analysis.setStyleSheet("color: #888; font-size: 11px;")
        left.addWidget(self._lbl_analysis)

        left.addWidget(QLabel("Status / log"))
        left.addWidget(QLabel("MSI лампы (≈раз в окно):"))
        self._msi_trace = QPlainTextEdit()
        self._msi_trace.setReadOnly(True)
        self._msi_trace.setMaximumBlockCount(200)
        self._msi_trace.setMaximumHeight(88)
        self._msi_trace.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._msi_trace.setStyleSheet(
            "font-family: monospace; font-size: 12px; color: #9cf; background: #111;"
        )
        self._msi_trace.setToolTip(
            "Одна цифра после каждого MSIExec, когда прошло ≥ длины окна MSI (поле «Окно, с»).\n"
            "Внутри MSI считается чаще (таймер 200 ms), но сюда попадает шаг окна.\n"
            f"Перенос строки каждые {self.WINNER_TRACE_PER_LINE} числа."
        )
        self._msi_trace.setPlainText("—")
        left.addWidget(self._msi_trace)
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
            "Заданы 4 лампы — Apply, Connect EEG, «Начать анализ» для полного лога."
        )
        self._update_freq_slot_label()
        self._refresh_msi_status_labels()

    def _sync_msi_spins_from_state(self) -> None:
        widgets = (self._spin_msi_fs, self._spin_msi_samples, self._spin_msi_window_sec)
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            self._spin_msi_fs.setValue(max(1, int(round(self._nominal_fs))))
            self._spin_msi_samples.setValue(max(8, int(self._n_template)))
            if self._nominal_fs > 0:
                self._spin_msi_window_sec.setValue(self._n_template / self._nominal_fs)
            else:
                self._spin_msi_window_sec.setValue(self._msi_window_sec)
        finally:
            for w, was in zip(widgets, blocked):
                w.blockSignals(was)

    def _recompute_msi_buffer(self) -> None:
        need = max(8, int(self._n_template))
        margin = int(self._nominal_fs * self._msi_window_sec * self.BUFFER_MARGIN) + 64
        cap = int(self._nominal_fs * self.MAX_ROLLING_BUFFER_SEC) + 64
        self._max_buf = max(need, min(margin, cap))
        self._ring_eeg = None
        self._ring_t = None
        self._ring_fill = 0
        self._buf = np.zeros((0, max(1, self._n_channels)), dtype=np.float64)
        self._buf_t = np.zeros(0, dtype=np.float64)
        cfg = self._burst_gate.config
        self._burst_gate = BurstGate(
            BurstGateConfig(
                window_sec=self._msi_window_sec,
                min_on_fraction=cfg.min_on_fraction,
                min_on_sec=cfg.min_on_sec,
            )
        )
        if self._freqs_hz:
            self._burst_gate.set_active_lamps(len(self._freqs_hz))

    def _on_msi_fs_changed(self, hz: int) -> None:
        self._nominal_fs = float(max(1, hz))
        self._n_template = max(8, int(round(self._nominal_fs * self._msi_window_sec)))
        self._sync_msi_spins_from_state()
        self._recompute_msi_buffer()
        self._refresh_msi_status_labels()

    def _on_msi_samples_changed(self, n: int) -> None:
        self._n_template = max(8, int(n))
        if self._nominal_fs > 0:
            self._msi_window_sec = self._n_template / self._nominal_fs
        self._sync_msi_spins_from_state()
        self._recompute_msi_buffer()
        self._refresh_msi_status_labels()

    def _on_msi_window_sec_changed(self, sec: float) -> None:
        self._msi_window_sec = max(0.1, float(sec))
        self._last_trace_lsl_t = None
        self._n_template = max(8, int(round(self._nominal_fs * self._msi_window_sec)))
        self._sync_msi_spins_from_state()
        self._recompute_msi_buffer()
        self._refresh_msi_status_labels()

    def _refresh_msi_status_labels(self) -> None:
        n_lamps = len(self._freq_combos) if self._freq_combos else len(self._freqs_hz)
        self._lbl_msi_points.setText(str(n_lamps))
        buf_n = int(self._buf.shape[0]) if self._buf is not None else 0
        need = int(self._n_template)
        cap = int(self._max_buf)
        if buf_n >= need:
            self._lbl_msi_ready.setText(
                f"да · {buf_n}/{cap} отсч. (окно MSI {need})"
            )
            self._lbl_msi_ready.setStyleSheet("color: #5cb85c;")
        elif buf_n > 0:
            self._lbl_msi_ready.setText(f"нет · {buf_n} / {need} отсч.")
            self._lbl_msi_ready.setStyleSheet("color: #f0ad4e;")
        else:
            self._lbl_msi_ready.setText("нет · нет EEG")
            self._lbl_msi_ready.setStyleSheet("color: #888;")
        if self._last_msi_run_winner is not None and self._last_msi_run_n > 0:
            self._lbl_msi_last_run.setText(
                f"{self._last_msi_run_n} отсч. → L{self._last_msi_run_winner}"
            )
            self._lbl_msi_last_run.setStyleSheet("color: #9cf; font-weight: bold;")
        else:
            self._lbl_msi_last_run.setText("—")
            self._lbl_msi_last_run.setStyleSheet("color: #888;")
        if self._last_msi_exec_ms is not None:
            self._lbl_msi_exec.setText(f"{self._last_msi_exec_ms:.1f} мс")
        else:
            self._lbl_msi_exec.setText("—")

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
            msi_window_sec=self._msi_window_sec,
            gate_allowed=self._last_gate_allowed,
            msi_events=list(self._msi_events),
        )

    def _log_line(self, msg: str) -> None:
        self._log.append(msg)
        doc = self._log.document()
        if doc.blockCount() > self.LOG_MAX_LINES:
            cursor = self._log.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            cursor.movePosition(
                cursor.MoveOperation.Down,
                cursor.MoveMode.KeepAnchor,
                doc.blockCount() - self.LOG_MAX_LINES,
            )
            cursor.removeSelectedText()
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _ensure_ring_storage(self, n_ch: int) -> None:
        n_ch = max(1, int(n_ch))
        cap = max(8, int(self._max_buf))
        if (
            self._ring_eeg is None
            or self._ring_eeg.shape != (cap, n_ch)
        ):
            self._ring_eeg = np.zeros((cap, n_ch), dtype=np.float64)
            self._ring_t = np.zeros(cap, dtype=np.float64)
            self._ring_fill = 0

    def _rebuild_buf_views(self) -> None:
        """Собрать линейный вид для MSI/графика из кольцевого буфера."""
        if self._ring_eeg is None or self._ring_fill <= 0:
            self._buf = np.zeros((0, max(1, self._n_channels)), dtype=np.float64)
            self._buf_t = np.zeros(0, dtype=np.float64)
            return
        cap = self._ring_eeg.shape[0]
        n = min(self._ring_fill, cap)
        n_ch = self._ring_eeg.shape[1]
        if self._ring_fill <= cap:
            self._buf = np.ascontiguousarray(self._ring_eeg[:n, :n_ch], dtype=np.float64)
            self._buf_t = np.ascontiguousarray(self._ring_t[:n], dtype=np.float64)
        else:
            start = self._ring_fill % cap
            idx = (start + np.arange(n, dtype=np.int64)) % cap
            self._buf = np.ascontiguousarray(self._ring_eeg[idx, :n_ch], dtype=np.float64)
            self._buf_t = np.ascontiguousarray(self._ring_t[idx], dtype=np.float64)

    def _push_eeg_chunk(self, arr: np.ndarray, ts: Any) -> int:
        """Добавить LSL-chunk; при переполнении старые отсчёты замещаются (без роста RAM)."""
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_new = int(arr.shape[0])
        if n_new <= 0:
            return 0
        n_ch = int(arr.shape[1])
        self._ensure_ring_storage(n_ch)
        assert self._ring_eeg is not None and self._ring_t is not None
        cap = int(self._ring_eeg.shape[0])

        if n_new >= cap:
            arr = arr[-cap:]
            n_new = cap

        t_prev = (
            np.asarray([float(self._ring_t[(self._ring_fill - 1) % cap])], dtype=np.float64)
            if self._ring_fill > 0
            else np.zeros(0, dtype=np.float64)
        )
        new_t = append_chunk_timestamps(
            t_prev,
            ts if ts is not None else [],
            n_new,
            self._nominal_fs,
        )
        if new_t.size != n_new:
            step = 1.0 / max(self._nominal_fs, 1.0)
            if self._ring_fill > 0:
                t0 = float(self._ring_t[(self._ring_fill - 1) % cap]) + step
            else:
                t0 = 0.0
            new_t = t0 + step * np.arange(n_new, dtype=np.float64)

        for i in range(n_new):
            w = self._ring_fill % cap
            self._ring_eeg[w, :n_ch] = arr[i, :n_ch]
            self._ring_t[w] = float(new_t[i])
            self._ring_fill += 1

        self._rebuild_buf_views()
        return n_new

    def _ui_due(self, last_mono: float, interval_ms: int) -> bool:
        now = time.monotonic()
        if (now - last_mono) * 1000.0 >= float(interval_ms):
            return True
        return False

    def _clear_winner_trace(self) -> None:
        self._winner_trace.clear()
        self._last_trace_lsl_t = None
        self._last_burst_msi_lsl_t = None
        self._msi_trace.setPlainText("—")

    def _burst_classify_period_due(self, t_lsl: float) -> bool:
        """Пакетный режим: не чаще одного MSIExec на длину окна EEG."""
        period = max(0.1, float(self._msi_window_sec))
        if self._last_burst_msi_lsl_t is None:
            return True
        return (float(t_lsl) - float(self._last_burst_msi_lsl_t)) >= period * 0.98

    def _mark_burst_classify(self, t_lsl: float) -> None:
        self._last_burst_msi_lsl_t = float(t_lsl)

    def _start_burst_debug(self) -> None:
        if self._burst_debug is not None:
            return
        try:
            self._burst_debug = BurstDebugSession.open_new(
                BURST_DEBUG_DIR,
                start_payload={
                    "stim_mode": "burst",
                    "params": self._snapshot_experiment_params(),
                },
            )
            self._log_line(f"Пакетный автолог: {self._burst_debug.session_dir}")
        except Exception as e:
            self._burst_debug = None
            self._log_line(f"Пакетный автолог: ошибка {e}")

    def _stop_burst_debug(self, *, reason: str) -> None:
        if self._burst_debug is None:
            return
        try:
            log_dir = self._burst_debug.finalize(
                stop_payload={"reason": reason, "last_winner": self._last_winner},
                channel_labels=self._ch_labels,
                target_lamp=self._get_target_lamp() or None,
            )
            self._log_line(f"Пакетный автолог сохранён: {log_dir}")
        except Exception as e:
            self._log_line(f"Пакетный автолог: ошибка сохранения {e}")
        finally:
            self._burst_debug = None
            self._last_gate_debug_allowed = None

    def _burst_debug_write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self._burst_debug is not None:
            self._burst_debug.write(event, data)

    def _winner_trace_display_text(self) -> str:
        if not self._winner_trace:
            return "—"
        nums = [str(w) for w in self._winner_trace]
        n = self.WINNER_TRACE_PER_LINE
        lines = [" ".join(nums[i : i + n]) for i in range(0, len(nums), n)]
        return "\n".join(lines)

    def _msi_trace_period_due(self, t_lsl: float) -> bool:
        """Новая цифра в потоке не чаще одного раза на длину окна MSI."""
        period = max(0.1, float(self._msi_window_sec))
        if self._last_trace_lsl_t is None:
            return True
        return (float(t_lsl) - float(self._last_trace_lsl_t)) >= period * 0.98

    def _append_winner_trace(self, winner: int, *, t_lsl: float) -> None:
        """Цифра в поток после MSIExec; в пакетном режиме — вместе с отфильтрованным MSI."""
        if not self._is_burst_mode() and not self._msi_trace_period_due(t_lsl):
            return
        self._last_trace_lsl_t = float(t_lsl)
        self._winner_trace.append(int(winner))
        if len(self._winner_trace) > self.WINNER_TRACE_MAX:
            self._winner_trace = self._winner_trace[-self.WINNER_TRACE_MAX :]
        self._msi_trace.setPlainText(self._winner_trace_display_text())
        sb = self._msi_trace.verticalScrollBar()
        sb.setValue(sb.maximum())

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_burst_debug(reason="window_close")
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
                f"Нужен LSL «{MIGALKA_MARKER_STREAM}». Автолог → {BURST_DEBUG_DIR.name}/"
            )
            self._try_connect_markers()
            if self._inlet is not None:
                self._start_burst_debug()
        else:
            self._stop_burst_debug(reason="mode_continuous")
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
            parsed = parse_lsl_marker(val)
            payload = {
                "lsl_time": float(ts),
                "raw": str(val),
                "lamp_index_0idx": parsed[0] if parsed else None,
                "lamp_msi_1idx": (int(parsed[0]) + 1) if parsed else None,
                "is_on": parsed[1] if parsed else None,
            }
            self._exp_write("lsl_marker", payload)
            self._burst_debug_write("lsl_marker", payload)

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
        self._n_template = max(8, int(round(self._nominal_fs * self._msi_window_sec)))
        self._n_channels = int(info.channel_count())
        if self._n_channels < 1:
            self._n_channels = 1

        self._sync_msi_spins_from_state()
        self._recompute_msi_buffer()
        self._ring_fill = 0
        self._rebuild_buf_views()
        self._last_msi_run_winner = None
        self._last_msi_run_n = 0
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
        self._refresh_msi_status_labels()

        self._timer_pull.start()
        self._timer_class.start()
        if self._is_burst_mode():
            self._stop_burst_debug(reason="reconnect")
            self._start_burst_debug()

        if self._is_burst_mode():
            self._try_connect_markers()

        # пересобрать шаблоны под фактический fs, если MSI уже инициализирован
        if self._msi is not None and self._freqs_hz:
            self._apply_model_signals_to_msi()

    def _update_freq_slot_label(self) -> None:
        n = len(self._freq_combos)
        self._lbl_freq_slots.setText(f"Ламп: {n}/{self.MAX_LAMPS}")
        self._btn_freq_add.setEnabled(n < self.MAX_LAMPS)
        self._spin_target_lamp.setMaximum(max(1, n) if n else self.MAX_LAMPS)
        self._refresh_msi_status_labels()

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

    def _stream_info_dict(self) -> Dict[str, Any]:
        info = self._stream_info
        if info is None:
            return {}
        try:
            return {
                "name": info.name(),
                "type": info.type(),
                "channel_count": int(info.channel_count()),
                "nominal_srate": float(info.nominal_srate() or 0.0),
                "session_id": info.session_id(),
                "source_id": info.source_id(),
                "channel_labels": list(self._ch_labels),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_target_lamp(self) -> int:
        return int(self._spin_target_lamp.value())

    def _target_ground_truth_payload(self) -> Dict[str, Any]:
        lamp = self._get_target_lamp()
        if lamp < 1:
            return {"target_lamp": None, "target_hz": None}
        hz: Optional[float] = None
        if lamp <= len(self._freqs_hz):
            hz = float(self._freqs_hz[lamp - 1])
        return {"target_lamp": lamp, "target_hz": hz}

    def _classification_match(self, predicted_lamp: int) -> Optional[bool]:
        target = self._get_target_lamp()
        if target < 1:
            return None
        if predicted_lamp < 1:
            return False
        return int(predicted_lamp) == target

    def _refresh_target_match_label(self, predicted_lamp: Optional[int]) -> None:
        target = self._get_target_lamp()
        if target < 1:
            self._lbl_target_match.setText("—")
            self._lbl_target_match.setStyleSheet("color: #888;")
            return
        if predicted_lamp is None or predicted_lamp < 1:
            self._lbl_target_match.setText(f"цель L{target}")
            self._lbl_target_match.setStyleSheet("color: #aaa;")
            return
        if predicted_lamp == target:
            self._lbl_target_match.setText(f"✓ L{target}")
            self._lbl_target_match.setStyleSheet("color: #5cb85c; font-weight: bold;")
        else:
            self._lbl_target_match.setText(f"✗ цель L{target}, MSI L{predicted_lamp}")
            self._lbl_target_match.setStyleSheet("color: #f0ad4e; font-weight: bold;")

    def _on_target_lamp_changed(self, _value: int) -> None:
        n = len(self._freq_combos) or len(self._freqs_hz) or self.MAX_LAMPS
        self._spin_target_lamp.setMaximum(max(1, n))
        self._refresh_target_match_label(self._last_winner)
        payload = self._target_ground_truth_payload()
        if self._analysis_active:
            self._exp_write("target_lamp_changed", payload)
        if payload["target_lamp"]:
            hz = payload.get("target_hz")
            extra = f" ({hz:g} Hz)" if hz else ""
            self._log_line(f"Целевая лампа: L{payload['target_lamp']}{extra}")

    def _snapshot_experiment_params(self) -> Dict[str, Any]:
        cfg = self._burst_gate.config
        return {
            **self._target_ground_truth_payload(),
            "stim_mode": str(self._cb_stim_mode.currentData()),
            "nominal_fs_hz": float(self._nominal_fs),
            "n_template_samples": int(self._n_template),
            "msi_window_sec": float(self._msi_window_sec),
            "classify_interval_ms": int(self.CLASSIFY_MS),
            "pull_interval_ms": int(self.PULL_MS),
            "gantt_span_sec": float(self.GANTT_SPAN_SEC),
            "freqs_hz": [float(f) for f in self._freqs_hz],
            "n_lamps_ui": len(self._freq_combos),
            "channel_labels": list(self._ch_labels),
            "selected_channel_indices": self._selected_channel_indices(),
            "n_channels_stream": int(self._n_channels),
            "burst_gate": {
                "window_sec": float(cfg.window_sec),
                "min_on_fraction": float(cfg.min_on_fraction),
                "min_on_sec": float(cfg.min_on_sec),
                "active_lamps": list(self._burst_gate.active_lamps),
            },
            "stream": self._stream_info_dict(),
            "marker_stream": MIGALKA_MARKER_STREAM,
            "buffer_samples": int(self._buf.shape[0]) if self._buf.size else 0,
            "max_buf_samples": int(self._max_buf),
        }

    def _exp_write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self._exp_logger is not None and self._analysis_active:
            self._exp_logger.write(event, data)

    def _finalize_experiment_log(self, *, reason: str) -> Optional[Path]:
        if self._exp_logger is None:
            return self._last_log_dir
        stop = {
            "reason": reason,
            "session_points_count": len(self._session_points),
            "target_lamp": self._get_target_lamp() or None,
            "session_points": [
                {
                    "t_rel_s": float(t),
                    "target_lamp": self._get_target_lamp() or None,
                    "predicted_lamp": int(w),
                    "hz": float(hz),
                    "gate_ok": bool(g),
                    "match": self._classification_match(int(w)) if g and w >= 1 else None,
                }
                for t, w, hz, g in self._session_points
            ],
            "msi_events_gantt": len(self._msi_events),
            "last_winner": self._last_winner,
            "params_end": self._snapshot_experiment_params(),
        }
        log_dir = self._exp_logger.finalize(
            stop_payload=stop,
            channel_labels=self._ch_labels,
        )
        self._last_log_dir = log_dir
        self._exp_logger = None
        return log_dir

    def _save_analysis_pngs(self, plots_dir: Path) -> List[Path]:
        saved: List[Path] = []
        try:
            from pyqtgraph.exporters import ImageExporter

            plots_dir.mkdir(parents=True, exist_ok=True)
            if self._plot_session is not None:
                p = plots_dir / "session_plot.png"
                ex = ImageExporter(self._plot_session.plotItem)
                ex.parameters()["width"] = 1200
                ex.export(str(p))
                saved.append(p)
            if self._plot_gantt is not None:
                p = plots_dir / "gantt.png"
                ex = ImageExporter(self._plot_gantt.plotItem)
                ex.parameters()["width"] = 1200
                ex.export(str(p))
                saved.append(p)
            if self._plot is not None:
                p = plots_dir / "eeg_rolling.png"
                ex = ImageExporter(self._plot.plotItem)
                ex.parameters()["width"] = 1200
                ex.export(str(p))
                saved.append(p)
        except Exception as e:
            self._log_line(f"PNG export error: {e}")
        return saved

    def _on_analysis_start(self) -> None:
        if self._inlet is None:
            QMessageBox.warning(self, "Анализ", "Сначала подключите LSL EEG (Connect).")
            return
        if not self._freqs_hz:
            QMessageBox.warning(self, "Анализ", "Сначала нажмите Apply frequencies.")
            return
        if self._msi is None:
            QMessageBox.warning(self, "Анализ", "MSI не загружен — нажмите Apply frequencies.")
            return
        t0 = self._session_time_base()
        if t0 is None:
            QMessageBox.warning(self, "Анализ", "Нет данных EEG — дождитесь потока.")
            return
        if self._analysis_active:
            self._log_line("Анализ уже запущен.")
            return
        target = self._get_target_lamp()
        if target < 1:
            QMessageBox.warning(
                self,
                "Анализ",
                "Укажите номер целевой лампы (1…N) — на какую лампу смотрит человек.\n"
                "Поле «Целевая лампа №» в блоке «Параметры MSI».",
            )
            return
        n_lamps = len(self._freqs_hz)
        if target > n_lamps:
            QMessageBox.warning(
                self,
                "Анализ",
                f"Целевая лампа L{target}, но задано только {n_lamps} частот(ы).",
            )
            return
        try:
            self._exp_logger = SSVEPExperimentLogger.open_new(
                output_root=EXPERIMENT_LOG_DIR,
                start_payload=self._snapshot_experiment_params(),
            )
        except Exception as e:
            QMessageBox.critical(self, "Лог", f"Не удалось создать лог:\n{e}")
            return
        self._analysis_active = True
        self._session_t0 = t0
        self._session_points.clear()
        self._msi_events.clear()
        self._clear_gate_history()
        self._last_winner = None
        self._refresh_session_plot()
        log_dir = self._exp_logger.session_dir
        self._lbl_analysis.setText(f"Анализ: запись →\n{log_dir.name}")
        self._lbl_analysis.setStyleSheet("color: #5cb85c; font-size: 11px;")
        self._log_line(f"Анализ: старт, лог {log_dir}")
        self._exp_write("analysis_ui_start", {"t0_lsl": t0})

    def _on_analysis_stop(self) -> None:
        if not self._analysis_active:
            self._log_line("Анализ не был запущен.")
            return
        self._analysis_active = False
        log_dir = self._finalize_experiment_log(reason="user_stop")
        pngs: List[Path] = []
        if log_dir is not None:
            pngs = self._save_analysis_pngs(log_dir / "plots")
        self._lbl_analysis.setText(
            f"Анализ: остановлен, точек {len(self._session_points)}\n{log_dir}"
            if log_dir
            else "Анализ: остановлен"
        )
        self._lbl_analysis.setStyleSheet("color: #f0ad4e; font-size: 11px;")
        self._refresh_session_plot()
        self._log_line(f"Анализ: стоп. Лог: {log_dir}")
        if log_dir:
            QMessageBox.information(
                self,
                "Лог сохранён",
                f"Каталог:\n{log_dir}\n\nevents.ndjson, manifest.json, eeg.npz\nPNG: {len(pngs)}",
            )

    def _on_analysis_reset(self) -> None:
        if self._analysis_active:
            self._analysis_active = False
            log_dir = self._finalize_experiment_log(reason="user_reset")
            if log_dir:
                self._log_line(f"Анализ: лог сохранён при сбросе → {log_dir}")
        self._session_t0 = None
        self._session_points.clear()
        self._msi_events.clear()
        self._clear_gate_history()
        self._last_winner = None
        self._clear_winner_trace()
        self._recompute_msi_buffer()
        if self._curve_session is not None:
            self._curve_session.setData([], [])
        if self._gantt_chart is not None:
            self._gantt_chart.clear()
        self._lbl_winner.setText("CURRENT TARGET:\n—")
        self._lbl_scores.setText("Scores / Coef:\n—")
        self._lbl_analysis.setText("Анализ: сброшен")
        self._lbl_analysis.setStyleSheet("color: #888; font-size: 11px;")
        self._log_line("Анализ: сброс (графики, Gantt, winner).")

    def _on_save_experiment_log(self) -> None:
        if self._analysis_active and self._exp_logger is not None:
            self._analysis_active = False
            log_dir = self._finalize_experiment_log(reason="manual_save")
            if log_dir:
                self._save_analysis_pngs(log_dir / "plots")
                self._lbl_analysis.setText(f"Лог сохранён:\n{log_dir.name}")
                QMessageBox.information(self, "Лог", f"Сохранено:\n{log_dir}")
                self._log_line(f"Лог сохранён вручную: {log_dir}")
            return
        if self._last_log_dir is not None and self._last_log_dir.is_dir():
            QMessageBox.information(
                self,
                "Лог",
                f"Последний каталог:\n{self._last_log_dir}",
            )
            return
        QMessageBox.warning(self, "Лог", "Нет активной записи. Сначала «Начать анализ».")

    def _append_session_point(self, winner: int, hz: float, gate_ok: bool) -> None:
        if not self._analysis_active or self._session_t0 is None:
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
        np_models = tme.generate_model_signals(self._freqs_hz, fs, self._msi_window_sec)
        self._n_template = (
            int(np_models[0].shape[1]) if np_models else int(round(fs * self._msi_window_sec))
        )
        self._sync_msi_spins_from_state()
        model_list = tme.build_model_signal_list(self._msi, np_models, verbose=False)
        self._msi.ModelSignal = model_list
        self._log_line(
            f"Frequencies updated: {', '.join(f'{f:g}' for f in self._freqs_hz)} Hz; "
            f"template length = {self._n_template} samples @ {fs:g} Hz."
        )
        self._refresh_msi_status_labels()

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
        self._last_msi_run_winner = None
        self._last_msi_run_n = 0
        self._refresh_msi_status_labels()

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
            self._ring_eeg = None
            self._ring_t = None
            self._ring_fill = 0
            self._rebuild_channel_controls()

        n_new = self._push_eeg_chunk(arr, ts)
        n_log = min(n_new, int(self._buf.shape[0]))
        tail_t = self._buf_t[-n_log:] if n_log > 0 else None
        tail_eeg = self._buf[-n_log:] if n_log > 0 else None

        if (
            self._analysis_active
            and self._exp_logger is not None
            and tail_t is not None
            and tail_eeg is not None
        ):
            self._exp_logger.append_eeg_chunk(
                tail_t,
                tail_eeg,
                max_total_samples=int(self._nominal_fs * 120),
            )
        if self._burst_debug is not None and tail_t is not None and tail_eeg is not None:
            self._burst_debug.append_eeg_chunk(tail_t, tail_eeg)

        if self._is_burst_mode():
            self._pull_markers()

        now_mono = time.monotonic()
        if self._ui_due(self._last_gantt_mono, self.GANTT_REFRESH_MS):
            self._last_gantt_mono = now_mono
            self._update_gantt()
        self._refresh_msi_status_labels()

        if not self._ui_due(self._last_plot_mono, self.PLOT_REFRESH_MS):
            return
        self._last_plot_mono = now_mono
        n_show = min(self._buf.shape[0], int(self._nominal_fs * 3))
        if n_show < 2:
            return
        tail = self._buf[-n_show:]
        t_end = float(self._buf_t[-1]) if self._buf_t.size else 0.0
        t_start = t_end - (n_show - 1) / max(self._nominal_fs, 1.0)
        x = t_start + np.arange(n_show, dtype=np.float64) / max(self._nominal_fs, 1.0)
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
        t_buf_end = float(self._buf_t[-1]) if self._buf_t.size else None
        win_diag: Dict[str, Any] = {}
        if self._is_burst_mode():
            if self._marker_inlet is None and not self._try_connect_markers():
                self._lbl_winner.setText("CURRENT TARGET:\n—\n\n(нет LSL маркеров)")
                self._lbl_burst_status.setText("Пауза / нет MigalkaStimMarkers")
                self._update_gantt(gate_allowed=False)
                self._append_session_point(0, 0.0, False)
                self._exp_write("burst_gate", {"allowed": False, "reason": "no_marker_stream"})
                self._burst_debug_write(
                    "burst_gate", {"allowed": False, "reason": "no_marker_stream"}
                )
                return
            win_diag = self._burst_gate.window_diagnostics(self._buf_t)
            allowed, reason = self._burst_gate.classify_allowed(self._buf_t)
            gate_ok = allowed
            gate_payload = {
                "allowed": allowed,
                "reason": reason,
                "lsl_time": t_buf_end,
                **win_diag,
            }
            if allowed != self._last_gate_debug_allowed:
                self._last_gate_debug_allowed = allowed
                self._burst_debug_write("burst_gate", gate_payload)
            self._lbl_burst_status.setText(f"Пакетный гейт: {reason}")
            if not allowed:
                self._lbl_burst_status.setStyleSheet("color: #f0ad4e; font-size: 12px;")
                self._lbl_winner.setText(f"CURRENT TARGET:\n—\n\nПАУЗА\n({reason})")
                self._update_gantt(gate_allowed=False)
                self._append_session_point(0, 0.0, False)
                self._exp_write("burst_gate", {"allowed": False, "reason": reason, **win_diag})
                return
            self._lbl_burst_status.setStyleSheet("color: #5cb85c; font-size: 12px;")
            if t_buf_end is not None and not self._burst_classify_period_due(t_buf_end):
                self._burst_debug_write(
                    "burst_skip_throttle",
                    {"lsl_time": t_buf_end, "reason": "wait_next_window", **win_diag},
                )
                return
        else:
            self._last_gate_allowed = True

        win_full = self._buf[-self._n_template :].T.copy()  # (channels, samples)
        win = np.ascontiguousarray(win_full[ch_idx, :], dtype=np.float64)
        try:
            managed = tme.numpy_to_double_matrix2d(win, verbose=False)
            t0 = time.perf_counter()
            winner = int(self._msi.MSIExec(managed))
            self._last_msi_exec_ms = (time.perf_counter() - t0) * 1000.0
            self._last_msi_run_n = int(self._n_template)
            self._last_msi_run_winner = int(winner)
            self._refresh_msi_status_labels()
        except Exception as e:
            self._log_line(f"MSIExec error: {e}")
            self._exp_write("msi_exec_error", {"error": str(e)})
            return

        t_lsl = self._session_time_base()
        if self._is_burst_mode() and t_lsl is not None:
            self._mark_burst_classify(float(t_lsl))
        coefs = coef_values(self._msi, len(self._freqs_hz))
        prev_winner = self._last_winner
        tgt = self._target_ground_truth_payload()
        match = self._classification_match(int(winner))
        self._refresh_target_match_label(int(winner))
        expected_marker = (
            expected_msi_lamp_from_diag(win_diag) if win_diag else None
        )
        match_marker = (
            int(winner) == int(expected_marker)
            if expected_marker is not None
            else None
        )
        classify_payload = {
            "lsl_time": t_lsl,
            "target_lamp": tgt["target_lamp"],
            "target_hz": tgt["target_hz"],
            "predicted_lamp": int(winner),
            "predicted_hz": (
                float(self._freqs_hz[winner - 1])
                if 1 <= winner <= len(self._freqs_hz)
                else None
            ),
            "expected_from_markers": expected_marker,
            "match_target": match,
            "match_markers": match_marker,
            "match": match,
            "winner": int(winner),
            "winner_prev": prev_winner,
            "gate_ok": bool(gate_ok),
            "coef_by_freq_hz": {
                f"{self._freqs_hz[i]:g}": coefs[i]
                for i in range(len(self._freqs_hz))
            },
            "coefs": coefs,
            "msi_exec_ms": self._last_msi_exec_ms,
            "channels_used": ch_idx,
            "n_template": int(self._n_template),
            "buf_samples": int(self._buf.shape[0]),
            "buf_t_first": float(self._buf_t[-self._n_template])
            if self._buf_t.size >= self._n_template
            else None,
            "buf_t_last": float(self._buf_t[-1]) if self._buf_t.size else None,
            **(win_diag if self._is_burst_mode() else {}),
        }
        self._exp_write("msi_classify", classify_payload)
        if self._is_burst_mode():
            self._burst_debug_write("burst_msi", classify_payload)

        if t_lsl is not None:
            self._append_winner_trace(int(winner), t_lsl=float(t_lsl))

        if winner != self._last_winner:
            self._exp_write(
                "winner_changed",
                {
                    "winner": int(winner),
                    "prev": prev_winner,
                    "lsl_time": t_lsl,
                    "target_lamp": tgt["target_lamp"],
                    "predicted_lamp": int(winner),
                    "match": match,
                },
            )
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
        self._last_gantt_mono = time.monotonic()
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
