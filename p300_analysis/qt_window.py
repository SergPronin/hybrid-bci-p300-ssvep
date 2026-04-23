#!/usr/bin/env python3 f
# -*- coding: utf-8 -*-
"""Главное окно Qt онлайн P300-анализатора."""

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QPoint, QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from pylsl import StreamInlet, StreamInfo, local_clock as lsl_local_clock
import pyqtgraph as pg

from p300_analysis.constants import (
    EEG_KEEP_SECONDS,
    EPOCH_DURATION_MS,
    EPOCH_RESERVE_MS,
    EEG_PULL_MAX_SAMPLES,
    MARKERS_PULL_MAX_SAMPLES,
    MIN_EPOCHS_TO_DECIDE,
    MONITOR_EEG_PLOT_MAX,
    MONITOR_LOG_INTERVAL_S,
    MONITOR_MARKER_EVENTS_MAX,
    WINNER_LABEL_STYLE_COLLECTING,
    WINNER_LABEL_STYLE_IDLE,
    WINNER_LABEL_STYLE_MATCH,
    WINNER_LABEL_STYLE_MISMATCH,
)
from p300_analysis.debug_ndjson import debug_ndjson
from p300_analysis.epoch_geometry import EpochGeometry
from p300_analysis.epoch_indexing import resolve_epoch_indices_for_marker
from p300_analysis.exam_session_detail_logger import (
    ExamSessionDetailLogger,
    epoch_roi_summary,
    pending_snapshot_for_log,
    summarize_eeg_chunk,
)
from p300_analysis.erp_compute import (
    build_averaged_erp,
    check_can_decide,
    compute_corrected_and_integrated,
    compute_winner_metrics,
    winner_display_lines,
)
from p300_analysis.lsl_streams import (
    find_allowed_eeg_streams,
    resolve_marker_streams,
    stream_inlet_with_buffer,
    unwrap_combo_userdata,
)
from p300_analysis.marker_parsing import (
    marker_value_to_stim_key,
    parse_trial_target_tile_id,
    stim_key_to_tile_digit,
)
from p300_analysis.run_export import (
    export_run_continuous_csv,
    export_run_data,
    export_run_data_all_stims,
    stim_indices_in_run,
)
from p300_analysis.session_recorder import SessionRecorder
from p300_analysis.winner_selection import (
    WINNER_MODE_AUC,
    mode_to_short_label,
)

LOG = logging.getLogger("p300_analyzer")

pg.setConfigOptions(useOpenGL=False, antialias=False, useCupy=False)
pg.setConfigOption("background", "#0a0a0a")
pg.setConfigOption("foreground", "#E0E0E0")


class P300AnalyzerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("P300 Analyzer (Online BCI)")
        self.setMinimumWidth(1100)

        # Real-time state required by the task
        self.eeg_buffer: List[np.ndarray] = []  # each element: (n_channels,) per timepoint
        self.eeg_times: List[float] = []
        self.pending_markers: List[Tuple[float, str]] = []
        self.epochs_data: Dict[str, List[np.ndarray]] = {}

        self._inlet_eeg: Optional[StreamInlet] = None
        self._inlet_markers: Optional[StreamInlet] = None

        self._epoch_geom = EpochGeometry()

        self._need_redraw_params: bool = False

        self._eeg_monitor_buf: Deque[float] = deque(maxlen=MONITOR_EEG_PLOT_MAX)
        self._marker_mono_buf: Deque[float] = deque(maxlen=MONITOR_MARKER_EVENTS_MAX)
        self._last_monitor_log_t: float = 0.0
        self._curve_eeg_monitor: Optional[Any] = None
        self._curve_marker_monitor: Optional[Any] = None

        self._monitor_eeg_win: Optional[QWidget] = None
        self._monitor_markers_win: Optional[QWidget] = None
        self._epoch_summary_win: Optional[QWidget] = None
        self._epoch_summary_text: Optional[QTextEdit] = None
        # Метки времени последних принятых маркеров вспышек (для оценки ISI). Ось произвольная.
        self._stimulus_marker_ts_history: Deque[float] = deque(maxlen=256)
        self._last_stim_marker_ts: Optional[float] = None
        self._monitor_eeg_status: Optional[QLabel] = None
        self._monitor_markers_status: Optional[QLabel] = None
        self._plot_eeg_monitor: Optional[pg.PlotWidget] = None
        self._plot_marker_monitor: Optional[pg.PlotWidget] = None

        self._recording_epochs: bool = False
        self._dbg_epoch_lag_n: int = 0
        self._dbg_winner_n: int = 0
        self._dbg_cue_n: int = 0
        # Offset переводит marker_ts в ось pylsl.local_clock() приёмника (ту же, что и
        # _lsl_clock_at_buffer_end). Для индексации: seconds_back = ref - (marker_ts + offset).
        self._marker_eeg_ts_offset: Optional[float] = None
        # Первый маркер вспышки сессии (для offset), пока не сопоставили с ЭЭГ в том же тике.
        self._calib_first_marker_ts: Optional[float] = None
        # pylsl.local_clock() на приёмнике в момент прихода первой вспышки (совместимая ось с ref).
        self._calib_first_marker_lsl_clock: Optional[float] = None
        # LSL-clock time when the latest EEG chunk was received.
        # Used for direct sample-index calculation (bypasses coarse EEG timestamps).
        self._lsl_clock_at_buffer_end: Optional[float] = None
        # LSL-clock time когда пришла первая пачка ЭЭГ в записи — нужен для оценки
        # реальной частоты дискретизации (если nominal_srate() = 0, что бывает у NeuroSpectrum).
        self._lsl_clock_at_buffer_start: Optional[float] = None
        # Количество отсчётов в _run_eeg_samples_export в момент self._lsl_clock_at_buffer_end
        # (фиксируется на каждый pull, используется при финализации для привязки маркеров).
        self._lsl_clock_buffer_end_n_samples: int = 0
        # Целевая плитка из LSL: -1|trial_start|target=N (PsychoPy после cue)
        self._lsl_cue_target_id: Optional[int] = None
        # Сводка по прогонам записи (для сравнения нескольких запусков подряд)
        self._exp_run_seq: int = 0
        self._exp_trial_targets: List[int] = []
        self._exp_last_winner_digit: Optional[int] = None
        # LSL trial_start: отсечь вспышки до cue (см. chk_epochs_after_trial)
        self._marker_ts_last_trial_start: Optional[float] = None
        self._session_recorder = SessionRecorder()
        self._session_run_id: Optional[str] = None
        # True after first stimulus marker in current recording run.
        self._has_seen_stimulus_marker_in_run: bool = False
        self._run_markers_export: List[Dict[str, Any]] = []
        self._run_eeg_ts_export: List[float] = []
        self._run_eeg_samples_export: List[List[float]] = []
        self._run_winners_export: List[Dict[str, Any]] = []
        self._run_epoch_segments_export: List[Dict[str, Any]] = []
        self._last_run_export_data: Optional[Dict[str, Any]] = None
        self._exam_detail_log: Optional[ExamSessionDetailLogger] = None
        self._exam_qt_tick_seq: int = 0
        self._recording_started_mono: float = 0.0
        self._shown_no_marker_hint: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_loop)

        self._setup_ui()
        self._setup_stream_monitor_windows()

    def _epoch_counts_snapshot(self) -> Dict[str, int]:
        return {k: len(v) for k, v in self.epochs_data.items()}

    def _stim_keys_sorted_for_summary(self, keys: List[str]) -> List[str]:
        def sk(k: str) -> Tuple[int, str]:
            if k.startswith("стимул_"):
                tail = k[len("стимул_") :]
                try:
                    return int(tail), k
                except ValueError:
                    pass
            return 10_000_000, k

        return sorted(keys, key=sk)

    def _median_isi_ms(self) -> Optional[float]:
        """Медиана межстимульного интервала по последним маркерам |on (мс) или None."""
        hist = sorted(self._stimulus_marker_ts_history)
        if len(hist) < 2:
            return None
        diffs = [
            (hist[i + 1] - hist[i]) * 1000.0
            for i in range(len(hist) - 1)
            if 0.0 < (hist[i + 1] - hist[i]) < 10.0
        ]
        if not diffs:
            return None
        diffs.sort()
        mid = len(diffs) // 2
        return float(
            diffs[mid] if len(diffs) % 2 == 1 else (diffs[mid - 1] + diffs[mid]) / 2.0
        )

    def _format_epoch_summary_text(self) -> str:
        lines = [
            f"Прогон записи (run_seq): {self._exp_run_seq}",
            "",
            "Эпохи по плитке (класс, маркер N|on):",
            "",
        ]
        if not self.epochs_data:
            lines.append("  (пока нет накопленных эпох)")
        else:
            total = 0
            for k in self._stim_keys_sorted_for_summary(list(self.epochs_data.keys())):
                n = len(self.epochs_data[k])
                total += n
                if k.startswith("стимул_"):
                    tile = k[len("стимул_") :]
                    line = f"  • плитка {tile} ({k}): {n} эпох"
                else:
                    line = f"  • {k}: {n} эпох"
                lines.append(line)
            lines.append("")
            lines.append(f"Всего эпох (вспышек |on): {total}")
        lines.append("")
        pend = len(self.pending_markers)
        lines.append(f"Маркеров в очереди (ждут буфер): {pend}")

        isi_ms = self._median_isi_ms()
        if isi_ms is not None:
            wy = int(self.spin_y.value())
            epoch_ms = int(EPOCH_DURATION_MS)
            lines.append("")
            lines.append(
                f"Средний ISI между вспышками: ~{isi_ms:.0f} мс "
                f"(по {len(self._stimulus_marker_ts_history)} маркерам)"
            )
            if wy > isi_ms + 10.0:
                lines.append(
                    f"⚠  Окно анализа Y={wy} мс длиннее ISI — в окно попадают отклики "
                    f"следующих плиток, ERP соседних плиток «подмешиваются»."
                )
                lines.append(
                    f"   Рекомендация: уменьшить Y примерно до {int(isi_ms)} мс "
                    "(и/или увеличить число повторений)."
                )
            if epoch_ms > isi_ms + 10.0:
                lines.append(
                    f"ℹ  Эпоха {epoch_ms} мс > ISI: в каждой эпохе присутствуют "
                    "~{} следующих вспышек других плиток (нормально для быстрой парадигмы, "
                    "усредняется повторениями).".format(max(1, int(epoch_ms // isi_ms) - 1))
                )
        return "\n".join(lines)

    def _update_epoch_summary_panel(self) -> None:
        if self._epoch_summary_text is None:
            return
        self._epoch_summary_text.setPlainText(self._format_epoch_summary_text())

    def _on_epoch_summary_clicked(self) -> None:
        if self._epoch_summary_win is None:
            return
        self._update_epoch_summary_panel()
        self._position_epoch_summary_win()
        self._epoch_summary_win.show()
        self._epoch_summary_win.raise_()
        self._epoch_summary_win.activateWindow()

    def _setup_stream_monitor_windows(self) -> None:
        """Два отдельных окна: ЭЭГ (нейроспектр) и маркеры плиток (PsychoPy)."""
        def _make(title: str) -> Tuple[QWidget, QLabel, pg.PlotWidget]:
            w = QWidget(None, Qt.Tool)
            w.setWindowTitle(title)
            w.setFixedSize(440, 220)
            w.setAttribute(Qt.WA_QuitOnClose, False)
            w.setStyleSheet("background-color: #121212; color: #e0e0e0;")
            lay = QVBoxLayout(w)
            lay.setContentsMargins(8, 8, 8, 8)
            st = QLabel("Нет подключения к LSL.")
            st.setWordWrap(True)
            st.setStyleSheet("color: #aaa; font-size: 11px;")
            plot = pg.PlotWidget()
            plot.setBackground("#0a0a0a")
            plot.setFixedHeight(130)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", "Время, с")
            lay.addWidget(st)
            lay.addWidget(plot)
            return w, st, plot

        self._monitor_eeg_win, self._monitor_eeg_status, self._plot_eeg_monitor = _make(
            "Монитор: ЭЭГ (нейроспектр / симулятор)"
        )
        self._plot_eeg_monitor.setLabel("left", "Ампл.")
        self._curve_eeg_monitor = self._plot_eeg_monitor.plot(pen=pg.mkPen("#7cfc00", width=1))

        self._monitor_markers_win, self._monitor_markers_status, self._plot_marker_monitor = _make(
            "Монитор: маркеры плиток (LSL Markers)"
        )
        self._plot_marker_monitor.setLabel("left", "события")
        self._curve_marker_monitor = self._plot_marker_monitor.plot(
            pen=None, symbol="o", symbolBrush="#ff6b6b", symbolSize=6
        )

        self._monitor_eeg_win.show()
        self._monitor_markers_win.show()

        # Отдельное окно без Qt.Tool: иначе на macOS/полном экране часто уезжает за край
        # (позиция была g.right()+10 от главного) и не видно при нажатии кнопки.
        self._epoch_summary_win = QDialog(self)
        self._epoch_summary_win.setWindowTitle("Сводка: плитки и эпохи P300")
        self._epoch_summary_win.setModal(False)
        self._epoch_summary_win.setFixedSize(420, 320)
        self._epoch_summary_win.setAttribute(Qt.WA_QuitOnClose, False)
        self._epoch_summary_win.setStyleSheet("background-color: #121212; color: #e0e0e0;")
        es_lay = QVBoxLayout(self._epoch_summary_win)
        es_lay.setContentsMargins(8, 8, 8, 8)
        es_hint = QLabel(
            "Эпохи — это вспышки «id|on» по LSL; по каждой плитке показано, сколько таких эпох "
            "накоплено с последнего сброса. «В очереди» — маркеры, ждущие хвоста буфера ЭЭГ."
        )
        es_hint.setWordWrap(True)
        es_hint.setStyleSheet("color: #aaa; font-size: 11px;")
        self._epoch_summary_text = QTextEdit()
        self._epoch_summary_text.setReadOnly(True)
        self._epoch_summary_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a1a; color: #e8e8e8; "
            "font-family: monospace; font-size: 12px; border: 1px solid #444; }"
        )
        es_lay.addWidget(es_hint)
        es_lay.addWidget(self._epoch_summary_text, stretch=1)
        self._epoch_summary_win.hide()

    def _position_epoch_summary_win(self) -> None:
        """Ставит окно сводки рядом с главным, но целиком в пределах availableGeometry()."""
        if self._epoch_summary_win is None:
            return
        w = self._epoch_summary_win
        margin = 12
        top_left = self.mapToGlobal(QPoint(0, 0))
        bot_right = self.mapToGlobal(QPoint(self.width(), self.height()))
        pref_x = bot_right.x() + margin
        pref_y = top_left.y() + margin

        if self._monitor_eeg_win is not None and self._monitor_markers_win is not None:
            mr_bottom = self._monitor_markers_win.frameGeometry().bottom()
            pref_y = max(pref_y, int(mr_bottom) + margin)

        scr = QApplication.screenAt(QPoint(pref_x, pref_y))
        if scr is None:
            scr = QApplication.screenAt(top_left)
        if scr is None:
            scr = QApplication.primaryScreen()
        ag = scr.availableGeometry()
        w_w, w_h = w.width(), w.height()
        x, y = int(pref_x), int(pref_y)
        if x + w_w > ag.right():
            x = int(top_left.x()) - w_w - margin
        if x + w_w > ag.right():
            x = int(ag.right()) - w_w - margin
        if x < ag.left():
            x = int(ag.left()) + margin
        if y + w_h > ag.bottom():
            y = int(ag.bottom()) - w_h - margin
        if y < ag.top():
            y = int(ag.top()) + margin
        w.move(x, y)

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if self._monitor_eeg_win and self._monitor_markers_win:
            g = self.geometry()
            self._monitor_eeg_win.move(g.right() + 10, g.top())
            self._monitor_markers_win.move(
                g.right() + 10, g.top() + self._monitor_eeg_win.height() + 12
            )
        if self._epoch_summary_win is not None and self._epoch_summary_win.isVisible():
            self._position_epoch_summary_win()

    def _setup_ui(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0a0a0a; color: white; }
            QLabel { color: white; }
            QPushButton {
                background-color: #333;
                color: white;
                border-radius: 3px;
                padding: 8px 10px;
            }
            QPushButton:hover { background-color: #444; }
            QSpinBox {
                background-color: #2d2d2d;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                min-width: 72px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #3d3d3d;
                border: none;
            }
            """
        )

        central = QWidget()
        self.setCentralWidget(central)
        self.setMinimumSize(960, 560)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # Left sidebar
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar.setFixedWidth(270)

        # Выбор потока
        stream_layout = QHBoxLayout()
        self.combo_streams = QComboBox()
        self.combo_streams.setStyleSheet(
            "background-color: #2d2d2d; color: white; padding: 5px; border: 1px solid #555; border-radius: 3px;"
        )
        self.btn_refresh_streams = QPushButton("🔄")
        self.btn_refresh_streams.setFixedWidth(40)
        self.btn_refresh_streams.clicked.connect(self._on_refresh_streams_clicked)
        stream_layout.addWidget(self.combo_streams)
        stream_layout.addWidget(self.btn_refresh_streams)

        sidebar_layout.addWidget(QLabel("Поток ЭЭГ:"))
        sidebar_layout.addLayout(stream_layout)

        self._markers_presence_label = QLabel("Поток плиток (Markers): не проверен — нажмите 🔄")
        self._markers_presence_label.setWordWrap(True)
        self._markers_presence_label.setStyleSheet("color: #888; font-size: 11px;")
        sidebar_layout.addWidget(self._markers_presence_label)

        self.btn_connect = QPushButton("Подключиться к LSL")
        self.btn_connect.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #218838; }"
        )
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        sidebar_layout.addWidget(self.btn_connect)

        self.btn_start_analysis = QPushButton("Начать анализ")
        self.btn_start_analysis.setStyleSheet(
            "QPushButton { background-color: #007bff; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #0069d9; } "
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.btn_start_analysis.setEnabled(False)
        self.btn_start_analysis.clicked.connect(self._on_start_analysis_clicked)
        sidebar_layout.addWidget(self.btn_start_analysis)

        self.btn_stop_analysis = QPushButton("Остановить анализ")
        self.btn_stop_analysis.setStyleSheet(
            "QPushButton { background-color: #fd7e14; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #e96b02; } "
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.btn_stop_analysis.setEnabled(False)
        self.btn_stop_analysis.clicked.connect(self._on_stop_analysis_clicked)
        sidebar_layout.addWidget(self.btn_stop_analysis)

        self.btn_reset_analysis = QPushButton("Сброс анализа")
        self.btn_reset_analysis.setStyleSheet(
            "QPushButton { background-color: #6c757d; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #5a6268; } "
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.btn_reset_analysis.setEnabled(False)
        self.btn_reset_analysis.clicked.connect(self._on_reset_analysis_clicked)
        sidebar_layout.addWidget(self.btn_reset_analysis)

        self.btn_save_exam = QPushButton("Сохранить обследование")
        self.btn_save_exam.setStyleSheet(
            "QPushButton { background-color: #20c997; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #1aa97f; } "
            "QPushButton:disabled { background-color: #444; color: #888; }"
        )
        self.btn_save_exam.setEnabled(False)
        self.btn_save_exam.clicked.connect(self._on_save_exam_clicked)
        sidebar_layout.addWidget(self.btn_save_exam)

        self.btn_disconnect = QPushButton("Отключить LSL")
        self.btn_disconnect.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #c82333; }"
        )
        self.btn_disconnect.setEnabled(False)
        self.btn_disconnect.clicked.connect(self._on_disconnect_clicked)
        sidebar_layout.addWidget(self.btn_disconnect)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(QLabel("Параметры анализа:"))

        self.spin_baseline = QSpinBox()
        self.spin_baseline.setRange(1, 800)
        self.spin_baseline.setValue(100)
        self.spin_baseline.setSuffix(" мс")
        self.spin_baseline.setKeyboardTracking(False)
        self.spin_baseline.valueChanged.connect(self._on_params_changed)

        # Блок выбора каналов (ROI)
        btn_ch_layout = QHBoxLayout()
        btn_all_ch = QPushButton("Все")
        btn_clear_ch = QPushButton("Сброс")
        btn_all_ch.clicked.connect(lambda: self._set_all_channels(True))
        btn_clear_ch.clicked.connect(lambda: self._set_all_channels(False))
        btn_ch_layout.addWidget(btn_all_ch)
        btn_ch_layout.addWidget(btn_clear_ch)

        self.scroll_channels = QScrollArea()
        self.scroll_channels.setWidgetResizable(True)
        self.scroll_channels.setMaximumHeight(180)
        self.scroll_channels.setStyleSheet(
            "QScrollArea { border: 1px solid #333; background-color: #1a1a1a; }"
        )

        self.channels_container = QWidget()
        self.channels_container.setStyleSheet("background-color: transparent;")
        self.channels_cb_layout = QVBoxLayout(self.channels_container)
        self.channels_cb_layout.setSpacing(2)
        self.scroll_channels.setWidget(self.channels_container)

        self.channel_checkboxes: List[QCheckBox] = []

        self.spin_x = QSpinBox()
        self.spin_x.setRange(0, 799)
        self.spin_x.setValue(200)
        self.spin_x.setSuffix(" мс")
        self.spin_x.setKeyboardTracking(False)
        self.spin_x.valueChanged.connect(self._on_params_changed)

        self.spin_y = QSpinBox()
        self.spin_y.setRange(1, 800)
        self.spin_y.setValue(600)
        self.spin_y.setKeyboardTracking(False)
        self.spin_y.valueChanged.connect(self._on_params_changed)

        self._status_label = QLabel(
            "Отключено. Запустите ЭЭГ, нажмите 🔄, выберите поток, «Подключиться к LSL», затем «Начать анализ»."
        )
        self._status_label.setWordWrap(True)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(QLabel("Baseline (мс):"))
        sidebar_layout.addWidget(self.spin_baseline)
        sidebar_layout.addWidget(QLabel("Каналы ROI:"))
        sidebar_layout.addLayout(btn_ch_layout)
        sidebar_layout.addWidget(self.scroll_channels)
        sidebar_layout.addWidget(QLabel("Начало окна X (мс):"))
        sidebar_layout.addWidget(self.spin_x)
        sidebar_layout.addWidget(QLabel("Конец окна Y (мс):"))
        sidebar_layout.addWidget(self.spin_y)

        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(QLabel("Как выбрать победителя:"))
        self.combo_winner_mode = QComboBox()
        self.combo_winner_mode.setStyleSheet(
            "background-color: #2d2d2d; color: white; padding: 5px; border: 1px solid #555; border-radius: 3px;"
        )
        self.combo_winner_mode.addItem(
            "Интегрирование по модулю |corrected| в окне [X–Y]", WINNER_MODE_AUC
        )
        self.combo_winner_mode.setCurrentIndex(0)
        self.combo_winner_mode.currentIndexChanged.connect(self._on_params_changed)
        sidebar_layout.addWidget(self.combo_winner_mode)

        self.chk_epochs_after_trial = QCheckBox(
            "Накапливать эпохи только после trial_start (cue из LSL)"
        )
        self.chk_epochs_after_trial.setChecked(False)
        self.chk_epochs_after_trial.setStyleSheet("color: #aaa; font-size: 11px;")
        self.chk_epochs_after_trial.setToolTip(
            "Отсекает вспышки до первого маркера trial_start в этой записи. "
            "Включите до «Начать анализ» или сделайте сброс после смены режима."
        )
        self.chk_epochs_after_trial.stateChanged.connect(self._on_params_changed)
        sidebar_layout.addWidget(self.chk_epochs_after_trial)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self._status_label)

        self.winner_label = QLabel("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self.winner_label.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.winner_label.setWordWrap(True)

        self.winner_scroll = QScrollArea()
        self.winner_scroll.setWidgetResizable(True)
        self.winner_scroll.setFrameShape(QFrame.StyledPanel)
        self.winner_scroll.setStyleSheet(
            "QScrollArea { border: 1px solid #444; background-color: #1a1a1a; }"
        )
        self.winner_scroll.setMinimumHeight(96)
        self.winner_scroll.setMaximumHeight(200)
        self.winner_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.winner_scroll.setWidget(self.winner_label)

        self.btn_epoch_summary = QPushButton("Сводка по плиткам / эпохам")
        self.btn_epoch_summary.setToolTip("Окно: в какие классы (плитки) писались эпохи и сколько вспышек |on")
        self.btn_epoch_summary.clicked.connect(self._on_epoch_summary_clicked)

        sidebar_layout.addSpacing(12)
        sidebar_layout.addWidget(self.winner_scroll)
        sidebar_layout.addWidget(self.btn_epoch_summary)
        sidebar_layout.addStretch(1)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFrameShape(QFrame.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sidebar_scroll.setStyleSheet("QScrollArea { background: #0a0a0a; border: none; }")
        sidebar_scroll.setWidget(sidebar)

        # Right plots: вертикальная прокрутка — иначе 4 графика сжимают подписи
        plots_scroll = QScrollArea()
        plots_scroll.setWidgetResizable(True)
        plots_scroll.setFrameShape(QFrame.NoFrame)
        plots_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        plots_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plots_scroll.setStyleSheet("QScrollArea { background: #0a0a0a; border: none; }")

        plots_inner = QWidget()
        plots_layout = QVBoxLayout(plots_inner)
        plots_layout.setContentsMargins(0, 0, 4, 0)
        plots_layout.setSpacing(10)

        self.plot_raw = pg.PlotWidget()
        self.plot_raw.setBackground("#0a0a0a")
        self.plot_corrected = pg.PlotWidget()
        self.plot_corrected.setBackground("#0a0a0a")
        self.plot_integrated = pg.PlotWidget()
        self.plot_integrated.setBackground("#0a0a0a")
        self.plot_raw_last = pg.PlotWidget()
        self.plot_raw_last.setBackground("#0a0a0a")

        _ph = 220
        for _pw in (
            self.plot_raw,
            self.plot_raw_last,
            self.plot_corrected,
            self.plot_integrated,
        ):
            _pw.setMinimumHeight(_ph)

        self._setup_plot(self.plot_raw, title="① Mean ERP (среднее по эпохам)")
        self._setup_plot(self.plot_raw_last, title="② Последняя эпоха (без mean)")
        self._setup_plot(self.plot_corrected, title="③ Baseline correction")
        self._setup_plot(self.plot_integrated, title="④ ∫|corr| (AUC)")

        plots_layout.addWidget(self.plot_raw)
        plots_layout.addWidget(self.plot_raw_last)
        plots_layout.addWidget(self.plot_corrected)
        plots_layout.addWidget(self.plot_integrated)

        plots_scroll.setWidget(plots_inner)

        root_layout.addWidget(sidebar_scroll)
        root_layout.addWidget(plots_scroll, stretch=1)

    @staticmethod
    def _setup_plot(plot_widget: pg.PlotWidget, *, title: str) -> None:
        plot_widget.setTitle(title, color="#5bc0be", size="10pt")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setLabel("bottom", "Время, мс")
        if "AUC" in title or "модулю" in title:
            left_lbl = "Накопл. AUC (условн. ед.)"
        elif "Baseline" in title or "выравнивания" in title:
            left_lbl = "Амплитуда (условн. ед.)"
        else:
            left_lbl = "Амплитуда (условн. ед.)"
        plot_widget.setLabel("left", left_lbl)
        plot_widget.addLegend(offset=(4, 4))

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _update_markers_presence_label(self) -> None:
        streams = resolve_marker_streams(timeout=0.8, attempts=1)
        if streams:
            info0 = streams[0]
            name = info0.name() or "Markers"
            nch = info0.channel_count()
            self._markers_presence_label.setText(
                f"Поток плиток (Markers): есть — «{name}», {nch} ch"
            )
            self._markers_presence_label.setStyleSheet("color: #5cb85c; font-size: 11px;")
            LOG.info("Обнаружен поток Markers: name=%r channels=%s", name, nch)
        else:
            self._markers_presence_label.setText(
                "Поток плиток (Markers): не найден — запустите PsychoPy; "
                "на другом ПК проверьте Wi‑Fi (multicast) и брандмауэр"
            )
            self._markers_presence_label.setStyleSheet("color: #d9534f; font-size: 11px;")
            LOG.info("Поток Markers не найден (resolve timeout)")

    def _reset_monitor_windows_disconnected(self) -> None:
        self._eeg_monitor_buf.clear()
        self._marker_mono_buf.clear()
        if self._monitor_eeg_status:
            self._monitor_eeg_status.setText("Нет подключения к LSL. Подключитесь для потока ЭЭГ.")
        if self._monitor_markers_status:
            self._monitor_markers_status.setText(
                "Нет подключения к LSL. Подключитесь для потока маркеров плиток."
            )
        if self._curve_eeg_monitor:
            self._curve_eeg_monitor.setData([], [])
        if self._curve_marker_monitor:
            self._curve_marker_monitor.setData([], [])

    def _append_eeg_monitor_samples(self, ch0: np.ndarray) -> None:
        if ch0.size == 0:
            return
        flat = np.asarray(ch0, dtype=np.float64).ravel()
        self._eeg_monitor_buf.extend(flat.tolist())

    def _append_marker_monitor_events(self, n_markers: int) -> None:
        now = time.monotonic()
        for _ in range(n_markers):
            self._marker_mono_buf.append(now)

    def _refresh_monitor_ui(
        self,
        *,
        eeg_samples_this_tick: int,
        marker_samples_this_tick: int,
        connected: bool,
    ) -> None:
        if not connected:
            return
        if self._monitor_eeg_status:
            buf_len = len(self._eeg_monitor_buf)
            rate_txt = f"+{eeg_samples_this_tick} сэмплов за тик" if eeg_samples_this_tick else "нет новых сэмплов"
            self._monitor_eeg_status.setText(
                f"ЭЭГ: данные идут — в буфере графика {buf_len} отсчётов. {rate_txt}"
            )
            self._monitor_eeg_status.setStyleSheet("color: #7cfc00; font-size: 11px;")
        if self._curve_eeg_monitor and self._eeg_monitor_buf:
            y = np.asarray(self._eeg_monitor_buf, dtype=np.float64)
            self._curve_eeg_monitor.setData(np.arange(len(y), dtype=np.float64), y)
            if self._plot_eeg_monitor:
                self._plot_eeg_monitor.setLabel("bottom", "Сэмпл (последние в окне)")
                self._plot_eeg_monitor.enableAutoRange("y", True)

        if self._monitor_markers_status:
            nbuf = len(self._marker_mono_buf)
            self._monitor_markers_status.setText(
                f"Маркеры: +{marker_samples_this_tick} за тик, всего в окне {nbuf} событий"
            )
            mk_style = "#ff6b6b;" if marker_samples_this_tick else "#aaaaaa;"
            self._monitor_markers_status.setStyleSheet(f"color: {mk_style} font-size: 11px;")
        if self._curve_marker_monitor and self._marker_mono_buf:
            arr = np.asarray(self._marker_mono_buf, dtype=np.float64)
            xs = arr - arr[0]
            ys = np.ones_like(arr)
            self._curve_marker_monitor.setData(xs, ys)
            if self._plot_marker_monitor:
                span = float(xs[-1]) if xs.size else 1.0
                self._plot_marker_monitor.setXRange(0, max(span, 0.5))

        now = time.monotonic()
        if now - self._last_monitor_log_t >= MONITOR_LOG_INTERVAL_S:
            self._last_monitor_log_t = now
            LOG.info(
                "LSL тик: ЭЭГ +%s сэмплов, маркеры +%s; буфер ЭЭГ_plot=%s, маркеров=%s",
                eeg_samples_this_tick,
                marker_samples_this_tick,
                len(self._eeg_monitor_buf),
                len(self._marker_mono_buf),
            )

    def _on_params_changed(self) -> None:
        self._need_redraw_params = True

    def _on_refresh_streams_clicked(self) -> None:
        self.combo_streams.clear()
        self._set_status("Поиск потоков ЭЭГ...")
        QApplication.processEvents()

        eeg_candidates = find_allowed_eeg_streams(timeout=1.0)
        if not eeg_candidates:
            self._set_status("Потоки ЭЭГ не найдены.")
            return

        for info in eeg_candidates:
            name = info.name() or "Unknown"
            ch_count = info.channel_count()
            display_text = f"{name} ({ch_count} ch)"
            self.combo_streams.addItem(display_text, userData=info)

        self._set_status(f"Найдено потоков: {len(eeg_candidates)}")
        self._update_markers_presence_label()

    def _build_channel_checkboxes(self, count: int) -> None:
        while self.channels_cb_layout.count():
            item = self.channels_cb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.channel_checkboxes.clear()

        for i in range(count):
            cb = QCheckBox(f"Канал {i + 1}")
            cb.setChecked(True)
            cb.setStyleSheet("color: white;")
            cb.stateChanged.connect(self._on_params_changed)
            self.channel_checkboxes.append(cb)
            self.channels_cb_layout.addWidget(cb)
        self.channels_cb_layout.addStretch()

    def _set_all_channels(self, state: bool) -> None:
        for cb in self.channel_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        self._on_params_changed()

    def _begin_connection_session(self) -> None:
        """LSL открыт, прогрев: читаем потоки, но эпохи для анализа не накапливаем."""
        if self._inlet_eeg is None or self._inlet_markers is None:
            return
        self._recording_epochs = False
        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._epoch_geom.reset()
        self._need_redraw_params = False
        self._marker_eeg_ts_offset = None
        self._calib_first_marker_ts = None
        self._lsl_clock_at_buffer_end = None
        self._lsl_clock_at_buffer_start = None
        self._lsl_clock_buffer_end_n_samples = 0
        self._lsl_cue_target_id = None
        self._marker_ts_last_trial_start = None
        self._eeg_monitor_buf.clear()
        self._marker_mono_buf.clear()
        self._last_monitor_log_t = 0.0
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._update_epoch_summary_panel()
        self._ensure_epoch_template()
        if not self._timer.isActive():
            self._timer.start()
        self.btn_connect.setEnabled(False)
        self.btn_start_analysis.setEnabled(True)
        self.btn_stop_analysis.setEnabled(False)
        self.btn_reset_analysis.setEnabled(True)
        self.btn_disconnect.setEnabled(True)
        self._set_status(
            "Подключено (прогрев). Можно сначала запустить плитки, затем «Начать анализ» — "
            "или наоборот; выравнивание времени — по первому маркеру вспышки."
        )
        try:
            nch = self._inlet_eeg.info().channel_count()
            ename = self._inlet_eeg.info().name() or "EEG"
        except Exception:
            nch, ename = -1, "EEG"
        LOG.info("Начата сессия LSL (прогрев): ЭЭГ «%s», каналов=%s", ename, nch)

    def _begin_recording_session(self) -> None:
        """Сброс буферов и накопление эпох только с этого момента."""
        if self._inlet_eeg is None or self._inlet_markers is None:
            return
        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._epoch_geom.reset()
        self._need_redraw_params = False
        self._recording_epochs = True
        self._recording_started_mono = time.monotonic()
        self._shown_no_marker_hint = False
        self._marker_eeg_ts_offset = None
        self._calib_first_marker_ts = None
        self._lsl_clock_at_buffer_end = None
        self._lsl_clock_at_buffer_start = None
        self._lsl_clock_buffer_end_n_samples = 0
        self._lsl_cue_target_id = None
        self._marker_ts_last_trial_start = None
        self._dbg_epoch_lag_n = 0
        self._dbg_winner_n = 0
        self._dbg_cue_n = 0
        self._has_seen_stimulus_marker_in_run = False
        self._run_markers_export = []
        self._run_eeg_ts_export = []
        self._run_eeg_samples_export = []
        self._run_winners_export = []
        self._run_epoch_segments_export = []
        self._last_run_export_data = None
        self._exp_run_seq += 1
        self._exp_trial_targets = []
        self._exp_last_winner_digit = None
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._update_epoch_summary_panel()
        self._ensure_epoch_template()
        self.btn_start_analysis.setEnabled(False)
        self.btn_stop_analysis.setEnabled(True)
        self.btn_save_exam.setEnabled(False)
        self._set_status(
            f"Запись эпох. Для выбора победителя нужно ≥{MIN_EPOCHS_TO_DECIDE} эпох по каждому классу с данными. "
            "Если нажали «Начать» до запуска плиток — дождитесь первой вспышки (калибровка времени по ней)."
        )
        LOG.info("Начата запись эпох для анализа")
        run_meta: Dict[str, Any] = {
            "run_seq": self._exp_run_seq,
            "winner_mode": self.combo_winner_mode.currentData(),
            "baseline_ms": int(self.spin_baseline.value()),
            "window_x_ms": int(self.spin_x.value()),
            "window_y_ms": int(self.spin_y.value()),
            "epochs_after_trial_only": bool(self.chk_epochs_after_trial.isChecked()),
            "selected_roi_channels_0idx": [
                i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()
            ],
            "eeg_stream_name": (self._inlet_eeg.info().name() if self._inlet_eeg else None),
            "eeg_stream_srate": (
                float(self._inlet_eeg.info().nominal_srate()) if self._inlet_eeg else None
            ),
            "eeg_stream_channels": (
                int(self._inlet_eeg.info().channel_count()) if self._inlet_eeg else None
            ),
            "markers_stream_name": (
                self._inlet_markers.info().name() if self._inlet_markers else None
            ),
            "recorder_file": str(self._session_recorder.output_path),
        }
        self._session_run_id = self._session_recorder.start_run(run_meta)
        LOG.info(
            "Запущена запись сырых данных для офлайн-отладки: run_id=%s file=%s",
            self._session_run_id,
            self._session_recorder.output_path,
        )
        try:
            self._exam_detail_log = ExamSessionDetailLogger.open_new(
                run_seq=self._exp_run_seq,
                exam_start_data={
                    **run_meta,
                    "session_ndjson_run_id": self._session_run_id,
                    "qt_timer_interval_ms": int(self._timer.interval()),
                    "eeg_keep_seconds": float(EEG_KEEP_SECONDS),
                    "epoch_duration_ms": int(EPOCH_DURATION_MS),
                    "epoch_reserve_ms": int(EPOCH_RESERVE_MS),
                    "min_epochs_to_decide": int(MIN_EPOCHS_TO_DECIDE),
                    "eeg_pull_max_samples": int(EEG_PULL_MAX_SAMPLES),
                    "markers_pull_max_samples": int(MARKERS_PULL_MAX_SAMPLES),
                },
            )
            self._exam_qt_tick_seq = 0
            LOG.info("Подробный NDJSON-лог обследования: %s", self._exam_detail_log.path)
        except Exception:
            self._exam_detail_log = None
            LOG.exception("Не удалось открыть подробный лог обследования (NDJSON)")
        # region agent log
        debug_ndjson(
            {
                "hypothesisId": "H5_experiment",
                "message": "run_start",
                "data": {
                    "run_seq": self._exp_run_seq,
                    "window_ms": [int(self.spin_x.value()), int(self.spin_y.value())],
                    "baseline_ms": int(self.spin_baseline.value()),
                    "selected_roi_channels_0idx": [
                        i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()
                    ],
                },
            }
        )
        # endregion

    def _on_start_analysis_clicked(self) -> None:
        if self._inlet_eeg is None or self._inlet_markers is None:
            return
        self._begin_recording_session()

    def _on_stop_analysis_clicked(self) -> None:
        if not self._recording_epochs:
            return
        self._finalize_pending_epochs_for_stop()
        self._recording_epochs = False
        self.pending_markers = []
        self.btn_start_analysis.setEnabled(True)
        self.btn_stop_analysis.setEnabled(False)
        self._set_status(
            "Анализ остановлен (LSL активен). Нажмите «Начать анализ» снова или «Отключить LSL»."
        )
        LOG.info("Запись эпох остановлена пользователем")
        # region agent log
        self._log_experiment_run_end("stop_clicked")
        # endregion

    def _finalize_pending_epochs_for_stop(self) -> None:
        """Best-effort finalize pending markers before stopping recording.

        Useful for very short runs (e.g. one flash per tile), where the user
        clicks Stop immediately and epochs may still be queued in pending_markers.
        """
        if (
            not self._recording_epochs
            or self._epoch_geom.epoch_len is None
            or self._epoch_geom.dt_ms is None
            or self._epoch_geom.time_ms_template is None
            or not self.eeg_buffer
            or self._lsl_clock_at_buffer_end is None
            or not self.pending_markers
        ):
            return

        dt_s = float(self._epoch_geom.dt_ms) / 1000.0
        if dt_s <= 0:
            return
        srate = 1.0 / dt_s
        el = int(self._epoch_geom.epoch_len)
        buf_len = len(self.eeg_buffer)
        buffer_end_ts = float(self._lsl_clock_at_buffer_end)
        time_arr = np.asarray(self.eeg_times, dtype=np.float64) if self.eeg_times else np.empty(0)
        buf_2d = np.stack(self.eeg_buffer)
        _roi = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]
        _valid = [c for c in _roi if 0 <= c < buf_2d.shape[1]] if buf_2d.ndim == 2 else []
        if buf_2d.ndim == 2 and _valid:
            buf_arr = np.mean(buf_2d[:, _valid], axis=1)
        elif buf_2d.ndim == 2:
            buf_arr = np.mean(buf_2d, axis=1)
        else:
            buf_arr = buf_2d.ravel()

        extracted_now = 0
        for marker_ts, stim_key in list(self.pending_markers):
            start_idx, end_idx, wait_more = self._resolve_epoch_indices_for_marker(
                marker_ts=float(marker_ts),
                buf_len=buf_len,
                srate=srate,
                epoch_len=el,
                lsl_ref=buffer_end_ts,
                time_arr=time_arr,
            )
            if wait_more:
                continue
            if start_idx is None or end_idx is None:
                continue
            epoch = buf_arr[start_idx:end_idx]
            if epoch.shape[0] != el:
                continue
            self.epochs_data.setdefault(stim_key, []).append(epoch.copy())
            raw_segment = buf_2d[start_idx:end_idx]
            if raw_segment.ndim == 2:
                raw_epoch_samples = raw_segment.astype(np.float64).tolist()
            else:
                raw_epoch_samples = np.asarray(raw_segment, dtype=np.float64).reshape(-1, 1).tolist()
            if time_arr.shape[0] == buf_len and end_idx <= time_arr.shape[0]:
                raw_epoch_ts = [float(x) for x in time_arr[start_idx:end_idx]]
            else:
                raw_epoch_ts = [None for _ in range(int(el))]
            self._run_epoch_segments_export.append(
                {
                    "stim_key": stim_key,
                    "marker_ts": float(marker_ts),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "eeg_ts": raw_epoch_ts,
                    "eeg_samples": raw_epoch_samples,
                }
            )
            extracted_now += 1

        if extracted_now:
            LOG.info(
                "Перед остановкой добрали эпох из pending_markers: %d (для коротких прогонов).",
                extracted_now,
            )
            if self._exam_detail_log is not None:
                self._exam_detail_log.write(
                    "stop_finalize_epochs",
                    {"extracted_epochs": int(extracted_now)},
                )
            self._redraw_from_epochs()

    def _log_experiment_run_end(self, reason: str) -> None:
        """Одна строка на прогон записи: цели LSL по порядку vs победитель UI (для сравнения запусков)."""
        targets = list(self._exp_trial_targets)
        last_cue = targets[-1] if targets else None
        win = self._exp_last_winner_digit
        match_last: Optional[bool] = None
        if last_cue is not None and win is not None:
            match_last = last_cue == win
        counts = {k: len(v) for k, v in self.epochs_data.items()}
        try:
            n_lag = self._dbg_epoch_lag_n
        except Exception:
            n_lag = 0
        summary = {
            "run_seq": self._exp_run_seq,
            "lsl_cue_targets_in_order": targets,
            "n_cues": len(targets),
            "unique_cues": sorted(set(targets)),
            "last_lsl_cue": last_cue,
            "ui_winner_tile_id": win,
            "match_last_cue_vs_winner": match_last,
            "epoch_counts_by_stim": counts,
            "n_epoch_align_logs": n_lag,
            "marker_eeg_offset": self._marker_eeg_ts_offset,
            "lsl_clock_at_buffer_end": (
                float(self._lsl_clock_at_buffer_end)
                if self._lsl_clock_at_buffer_end is not None
                else None
            ),
            "lsl_clock_at_buffer_start": (
                float(self._lsl_clock_at_buffer_start)
                if self._lsl_clock_at_buffer_start is not None
                else None
            ),
            "lsl_clock_buffer_end_n_samples": int(self._lsl_clock_buffer_end_n_samples),
            "pending_markers_count": len(self.pending_markers),
            "eeg_buffer_len": len(self.eeg_buffer),
            "eeg_times_len": len(self.eeg_times),
            "analysis_params": {
                "baseline_ms": int(self.spin_baseline.value()),
                "window_x_ms": int(self.spin_x.value()),
                "window_y_ms": int(self.spin_y.value()),
                "epochs_after_trial_only": bool(self.chk_epochs_after_trial.isChecked()),
                "sampling_rate_hz": (
                    float(self._inlet_eeg.info().nominal_srate())
                    if self._inlet_eeg is not None
                    else None
                ),
            },
        }
        if self._exam_detail_log is not None:
            summary["detail_exam_log_path"] = str(self._exam_detail_log.path)
            self._exam_detail_log.write("exam_end", {"reason": reason, "summary": summary})
            self._exam_detail_log.close()
            self._exam_detail_log = None

        # Оценка реальной частоты дискретизации (NeuroSpectrum частенько отдаёт nominal_srate()=0).
        nominal_srate = summary["analysis_params"].get("sampling_rate_hz")
        try:
            nominal_srate = float(nominal_srate) if nominal_srate is not None else 0.0
        except (TypeError, ValueError):
            nominal_srate = 0.0
        est_srate: Optional[float] = None
        lc_end = self._lsl_clock_at_buffer_end
        lc_start = self._lsl_clock_at_buffer_start
        n_total = len(self._run_eeg_samples_export)
        if (
            lc_end is not None
            and lc_start is not None
            and n_total > 1
            and float(lc_end) > float(lc_start)
        ):
            est_srate = float(n_total) / float(float(lc_end) - float(lc_start))
        effective_srate = nominal_srate if nominal_srate and nominal_srate > 0 else (est_srate or 0.0)
        summary["analysis_params"]["sampling_rate_hz_estimated"] = est_srate
        summary["analysis_params"]["sampling_rate_hz_effective"] = (
            effective_srate if effective_srate > 0 else None
        )

        # Предвычисленная привязка каждого маркера к индексу отсчёта в _run_eeg_samples_export.
        # Логика совпадает с resolve_epoch_indices_for_marker: ось — pylsl.local_clock(),
        # ref = lsl_clock_at_buffer_end, обратный ход по sampling_rate.
        markers_with_sample_idx: List[Dict[str, Any]] = []
        mk_offset = self._marker_eeg_ts_offset
        ref_lc = lc_end
        ref_n = int(self._lsl_clock_buffer_end_n_samples)
        can_map = effective_srate and effective_srate > 0 and ref_lc is not None and ref_n > 0
        for item in self._run_markers_export:
            new_item = dict(item)
            sample_idx: Optional[int] = None
            if can_map:
                try:
                    ts_m = float(item.get("ts"))
                    t_mark_lc = ts_m + float(mk_offset) if mk_offset is not None else ts_m
                    seconds_back = float(ref_lc) - t_mark_lc
                    j = ref_n - 1 - int(round(seconds_back * float(effective_srate)))
                    if 0 <= j < n_total:
                        sample_idx = j
                except (TypeError, ValueError):
                    sample_idx = None
            new_item["sample_idx"] = sample_idx
            markers_with_sample_idx.append(new_item)

        self._last_run_export_data = {
            "run_seq": self._exp_run_seq,
            "saved_at_ms": int(time.time() * 1000),
            "summary": summary,
            "markers": markers_with_sample_idx,
            "eeg_ts": list(self._run_eeg_ts_export),
            "eeg_samples": [list(x) for x in self._run_eeg_samples_export],
            "winner_updates": list(self._run_winners_export),
            "epoch_segments": list(self._run_epoch_segments_export),
            "epochs_data": {
                k: [np.asarray(ep, dtype=np.float64).tolist() for ep in v]
                for k, v in self.epochs_data.items()
            },
            "epoch_time_ms": (
                self._epoch_geom.time_ms_template.tolist()
                if self._epoch_geom.time_ms_template is not None
                else []
            ),
        }
        self.btn_save_exam.setEnabled(True)
        debug_ndjson(
            {
                "hypothesisId": "H5_experiment",
                "message": "run_end",
                "data": {
                    "reason": reason,
                    **summary,
                },
            }
        )
        self._session_recorder.stop_run(reason=reason, summary=summary)
        self._session_run_id = None

    def _on_reset_analysis_clicked(self) -> None:
        if self._inlet_eeg is None or self._inlet_markers is None:
            return
        was_recording = self._recording_epochs
        # region agent log
        if was_recording:
            self._log_experiment_run_end("reset_clicked")
        # endregion
        self._recording_epochs = False
        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._epoch_geom.reset()
        self._need_redraw_params = False
        self._marker_eeg_ts_offset = None
        self._calib_first_marker_ts = None
        self._lsl_clock_at_buffer_end = None
        self._lsl_clock_at_buffer_start = None
        self._lsl_clock_buffer_end_n_samples = 0
        self._lsl_cue_target_id = None
        self._marker_ts_last_trial_start = None
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._update_epoch_summary_panel()
        self._ensure_epoch_template()
        self.btn_start_analysis.setEnabled(True)
        self.btn_stop_analysis.setEnabled(False)
        self._set_status(
            "Анализ сброшен. Нажмите «Начать анализ», чтобы начать новую запись эпох."
        )
        LOG.info("Сброс анализа (буферы эпох очищены)")
        self._exp_trial_targets = []
        self._exp_last_winner_digit = None

    def _on_connect_clicked(self) -> None:
        if self._timer.isActive():
            return

        idx = self.combo_streams.currentIndex()
        if idx < 0:
            QMessageBox.warning(
                self,
                "Ошибка",
                "Сначала выберите поток ЭЭГ из списка (нажмите 🔄).",
            )
            return

        eeg_raw = unwrap_combo_userdata(self.combo_streams.itemData(idx))
        if not isinstance(eeg_raw, StreamInfo):
            QMessageBox.warning(
                self,
                "Ошибка",
                "Некорректные данные потока в списке. Нажмите 🔄 и выберите поток снова.",
            )
            return
        eeg_info = eeg_raw

        self._set_status("Подключение к маркерам...")
        QApplication.processEvents()

        marker_streams = resolve_marker_streams(timeout=6.0, attempts=3)

        if not marker_streams:
            QMessageBox.warning(
                self,
                "LSL",
                "Не найден поток маркеров (type='Markers').\n\n"
                "Проверьте: стимуляция PsychoPy запущена на том же ПК или в сети; на втором ноутбуке "
                "LSL использует multicast — оба в одной Wi‑Fi, без «клиентской изоляции» AP, "
                "разрешите входящий UDP в брандмауэре Windows/macOS.\n"
                "Если ЭЭГ виден, а маркеры нет: запустите LabRecorder на машине со стимуляцией "
                "или используйте провод/Ethernet.",
            )
            self._set_status("Не найден поток маркеров.")
            return

        # Закрываем старые
        try:
            if self._inlet_eeg is not None:
                self._inlet_eeg.close_stream()
        except Exception:
            pass
        try:
            if self._inlet_markers is not None:
                self._inlet_markers.close_stream()
        except Exception:
            pass

        # Открываем новые (int для буфера; сначала max_buffered)
        try:
            eeg_buf_s = int(round(float(EEG_KEEP_SECONDS)))
            self._inlet_eeg = stream_inlet_with_buffer(eeg_info, eeg_buf_s)
            self._inlet_eeg.open_stream(timeout=1.0)

            self._inlet_markers = stream_inlet_with_buffer(marker_streams[0], 60)
            self._inlet_markers.open_stream(timeout=1.0)
        except Exception as e:
            LOG.exception("Ошибка открытия LSL: %s", e)
            QMessageBox.critical(self, "Ошибка LSL", f"Не удалось открыть потоки:\n{e}")
            return

        # Генерируем чекбоксы каналов по количеству каналов в ЭЭГ
        self._build_channel_checkboxes(eeg_info.channel_count())
        LOG.info("Inlet открыты: ЭЭГ + Markers")
        self._begin_connection_session()

    def _on_disconnect_clicked(self) -> None:
        if self._recording_epochs:
            self._log_experiment_run_end("disconnect_clicked")
        self._recording_epochs = False
        if self._timer.isActive():
            self._timer.stop()

        # Best-effort close
        try:
            if self._inlet_eeg is not None:
                self._inlet_eeg.close_stream()
        except Exception:
            pass
        try:
            if self._inlet_markers is not None:
                self._inlet_markers.close_stream()
        except Exception:
            pass

        self._inlet_eeg = None
        self._inlet_markers = None

        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._epoch_geom.reset()
        self._need_redraw_params = False
        self._marker_eeg_ts_offset = None
        self._calib_first_marker_ts = None
        self._lsl_clock_at_buffer_end = None
        self._lsl_clock_at_buffer_start = None
        self._lsl_clock_buffer_end_n_samples = 0
        self._lsl_cue_target_id = None
        self._marker_ts_last_trial_start = None

        self.btn_connect.setEnabled(True)
        self.btn_start_analysis.setEnabled(False)
        self.btn_stop_analysis.setEnabled(False)
        self.btn_reset_analysis.setEnabled(False)
        self.btn_save_exam.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self._set_status("Остановлено. Обновите список 🔄 и снова «Подключиться к LSL».")
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._update_epoch_summary_panel()
        self._reset_monitor_windows_disconnected()
        LOG.info("Сессия LSL остановлена пользователем")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._recording_epochs:
            self._log_experiment_run_end("window_closed")
        if self._timer.isActive():
            self._timer.stop()
        try:
            if self._inlet_eeg is not None:
                self._inlet_eeg.close_stream()
        except Exception:
            pass
        try:
            if self._inlet_markers is not None:
                self._inlet_markers.close_stream()
        except Exception:
            pass
        if self._monitor_eeg_win is not None:
            self._monitor_eeg_win.close()
        if self._monitor_markers_win is not None:
            self._monitor_markers_win.close()
        if self._epoch_summary_win is not None:
            self._epoch_summary_win.close()
        LOG.info("Окно анализатора закрыто")
        super().closeEvent(event)

    def _clear_plots(self) -> None:
        self.plot_raw.clear()
        self.plot_raw_last.clear()
        self.plot_corrected.clear()
        self.plot_integrated.clear()

        self._setup_plot(self.plot_raw, title="① Mean ERP (среднее по эпохам)")
        self._setup_plot(self.plot_raw_last, title="② Последняя эпоха (без mean)")
        self._setup_plot(self.plot_corrected, title="③ Baseline correction")
        self._setup_plot(self.plot_integrated, title="④ ∫|corr| (AUC)")

    def _ensure_epoch_template(self) -> None:
        self._epoch_geom.ensure_template(self._inlet_eeg, self.eeg_times)

    def _compute_epoch_start_index(
        self, time_arr: np.ndarray, t_eff: float
    ) -> Optional[int]:
        return self._epoch_geom.compute_start_index(time_arr, t_eff)

    def _resolve_epoch_indices_for_marker(
        self,
        *,
        marker_ts: float,
        buf_len: int,
        srate: float,
        epoch_len: int,
        lsl_ref: float,
        time_arr: np.ndarray,
    ) -> Tuple[Optional[int], Optional[int], bool]:
        """Return (start_idx, end_idx, wait_more_data). См. p300_analysis.epoch_indexing."""
        return resolve_epoch_indices_for_marker(
            marker_ts=marker_ts,
            buf_len=buf_len,
            srate=srate,
            epoch_len=epoch_len,
            lsl_ref=lsl_ref,
            time_arr=time_arr,
            marker_eeg_offset=self._marker_eeg_ts_offset,
            compute_start_index=self._compute_epoch_start_index,
        )

    def _redraw_from_epochs(self) -> None:
        el = self._epoch_geom.epoch_len
        time_ms = self._epoch_geom.time_ms_template
        if el is None or time_ms is None:
            return

        stim_keys, raw_averaged = build_averaged_erp(self.epochs_data, el)
        if not stim_keys:
            self._clear_plots()
            self.winner_label.setText("РЕЗУЛЬТАТ: ?")
            self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
            self._update_epoch_summary_panel()
            return

        baseline_ms = int(self.spin_baseline.value())
        window_x_ms = int(self.spin_x.value())
        window_y_ms = int(self.spin_y.value())

        corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
            raw_averaged, time_ms, baseline_ms, window_x_ms, window_y_ms
        )

        can_decide, min_n = check_can_decide(stim_keys, self.epochs_data)

        if not can_decide or integrated.size == 0:
            self.winner_label.setText(
                f"Сбор данных...\n(мин. по классам: {min_n}/{MIN_EPOCHS_TO_DECIDE})"
            )
            self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_COLLECTING)
        else:
            winner_idx, mode_used, dbg = compute_winner_metrics(
                stim_keys,
                raw_averaged,
                corrected,
                time_ms,
                wx,
                wy,
                winner_mode=str(self.combo_winner_mode.currentData() or WINNER_MODE_AUC),
            )
            winner_key = stim_keys[winner_idx]
            dbg["lsl_cue_target_id"] = self._lsl_cue_target_id
            dbg["run_seq"] = self._exp_run_seq
            self._dbg_winner_n += 1
            dbg["winner_event_seq"] = self._dbg_winner_n
            dbg["marker_eeg_offset"] = self._marker_eeg_ts_offset
            dbg["pending_markers_count"] = len(self.pending_markers)
            dbg["epoch_counts_by_stim"] = {k: len(self.epochs_data.get(k, [])) for k in stim_keys}
            debug_ndjson({"hypothesisId": "H1_metric", "message": "winner_compare", "data": dbg})
            lines, win_digit, match_lsl = winner_display_lines(
                winner_key, mode_to_short_label(mode_used), self._lsl_cue_target_id
            )
            self._exp_last_winner_digit = win_digit
            self._run_winners_export.append(
                {
                    "event_seq": self._dbg_winner_n,
                    "winner_key": winner_key,
                    "winner_digit": win_digit,
                    "match_lsl_cue": match_lsl,
                }
            )
            self._session_recorder.log_winner(
                {
                    "run_seq": self._exp_run_seq,
                    "winner_event_seq": self._dbg_winner_n,
                    "winner_key": winner_key,
                    "winner_digit": win_digit,
                    "mode": mode_used,
                    "match_lsl_cue": match_lsl,
                    "lsl_cue_target_id": self._lsl_cue_target_id,
                    "winner_debug": dbg,
                    "window_ms": [wx, wy],
                    "time_axis_ms": [float(x) for x in time_ms],
                    "time_crop_ms": [float(x) for x in time_crop],
                    "stim_keys": stim_keys,
                    "epoch_counts_by_stim": {k: len(self.epochs_data.get(k, [])) for k in stim_keys},
                    "raw_averaged": raw_averaged.tolist(),
                    "corrected": corrected.tolist(),
                    "integrated": integrated.tolist(),
                    "pending_markers_count": len(self.pending_markers),
                    "marker_eeg_offset": self._marker_eeg_ts_offset,
                    "analysis_params": {
                        "baseline_ms": baseline_ms,
                        "window_x_ms": wx,
                        "window_y_ms": wy,
                        "min_epochs_to_decide": MIN_EPOCHS_TO_DECIDE,
                        "epochs_after_trial_only": bool(self.chk_epochs_after_trial.isChecked()),
                        "roi_channels_0idx": [
                            i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()
                        ],
                    },
                }
            )
            if self._exam_detail_log is not None:
                self._exam_detail_log.write(
                    "winner_update",
                    {
                        "winner_event_seq": int(self._dbg_winner_n),
                        "winner_key": winner_key,
                        "winner_digit": win_digit,
                        "mode": str(mode_used),
                        "match_lsl_cue": bool(match_lsl),
                        "lsl_cue_target_id": self._lsl_cue_target_id,
                        "stim_keys": list(stim_keys),
                        "epoch_counts_by_stim": {
                            k: len(self.epochs_data.get(k, [])) for k in stim_keys
                        },
                        "winner_debug": dbg,
                        "window_ms": [int(wx), int(wy)],
                        "pending_markers_count": len(self.pending_markers),
                        "marker_eeg_offset": self._marker_eeg_ts_offset,
                    },
                )
            self.winner_label.setText("\n".join(lines))
            self.winner_label.setStyleSheet(
                WINNER_LABEL_STYLE_MATCH if match_lsl else WINNER_LABEL_STYLE_MISMATCH
            )

        self._plot_all(
            raw_averaged,
            corrected,
            integrated,
            labels=stim_keys,
            time_ms=time_ms,
            time_crop=time_crop,
        )
        self._plot_last_single_epochs(stim_keys, time_ms)

        counts = ", ".join([f"{k}:{len(self.epochs_data[k])}" for k in stim_keys])
        need_hint = (
            f" Нужно ≥{MIN_EPOCHS_TO_DECIDE} эпох на каждый класс с данными."
            if not can_decide
            else ""
        )
        filter_hint = ""
        if getattr(self, "chk_epochs_after_trial", None) and self.chk_epochs_after_trial.isChecked():
            filter_hint = " Фильтр: только после trial_start."
        self._set_status(
            f"Обновлено: baseline={baseline_ms} мс, окно=[{wx}, {wy}] мс. "
            f"Эпохи: {counts}.{need_hint}{filter_hint}"
        )
        self._update_epoch_summary_panel()

    def _plot_last_single_epochs(
        self,
        stim_keys: List[str],
        time_ms: np.ndarray,
    ) -> None:
        """Одна последняя эпоха на каждый класс — без mean по повторениям."""
        self.plot_raw_last.clear()
        self._setup_plot(self.plot_raw_last, title="② Последняя эпоха (без mean)")
        n_colors = max(1, len(stim_keys))
        t = np.asarray(time_ms, dtype=np.float64)
        for i, key in enumerate(stim_keys):
            epochs = self.epochs_data.get(key, [])
            if not epochs:
                continue
            last = np.asarray(epochs[-1], dtype=np.float64).ravel()
            n = int(min(last.size, t.size))
            if n <= 0:
                continue
            label = f"{key} (последняя эпоха)"
            self.plot_raw_last.plot(
                t[:n],
                last[:n],
                pen=pg.mkPen(pg.intColor(i, hues=n_colors, values=1).name(), width=2),
                name=label,
            )
        self.plot_raw_last.setXRange(0, 800)

    def _plot_all(
        self,
        raw: np.ndarray,
        corrected: np.ndarray,
        integrated: np.ndarray,
        *,
        labels: List[str],
        time_ms: np.ndarray,
        time_crop: np.ndarray,
    ) -> None:
        n_stim = raw.shape[0]
        n_colors = max(1, n_stim)

        # Graph 1: raw averaged ERP
        self.plot_raw.clear()
        self._setup_plot(self.plot_raw, title="① Mean ERP (среднее по эпохам)")
        for i in range(n_stim):
            label = labels[i] if i < len(labels) else f"Стимул {i + 1}"
            self.plot_raw.plot(
                time_ms,
                raw[i],
                pen=pg.mkPen(pg.intColor(i, hues=n_colors, values=1).name(), width=2),
                name=label,
            )
        self.plot_raw.setXRange(0, 800)

        # Graph 2: baseline corrected ERP
        self.plot_corrected.clear()
        self._setup_plot(self.plot_corrected, title="③ Baseline correction")
        for i in range(n_stim):
            label = labels[i] if i < len(labels) else f"Стимул {i + 1}"
            self.plot_corrected.plot(
                time_ms,
                corrected[i],
                pen=pg.mkPen(pg.intColor(i, hues=n_colors, values=1).name(), width=2),
                name=label,
            )
        self.plot_corrected.setXRange(0, 800)

        # Graph 3: |corrected| integrated in the decision window
        self.plot_integrated.clear()
        self._setup_plot(self.plot_integrated, title="④ ∫|corr| (AUC)")
        for i in range(n_stim):
            label = labels[i] if i < len(labels) else f"Стимул {i + 1}"
            self.plot_integrated.plot(
                time_crop,
                integrated[i],
                pen=pg.mkPen(pg.intColor(i, hues=n_colors, values=1).name(), width=2),
                name=label,
            )
        if time_crop.size:
            self.plot_integrated.setXRange(float(time_crop[0]), float(time_crop[-1]))

    def _update_loop(self) -> None:
        need_redraw = False

        if self._inlet_eeg is None or self._inlet_markers is None:
            return

        eeg_samples_this_tick = 0
        marker_samples_this_tick = 0

        # 1) Pull markers and enqueue them
        try:
            marker_chunk, marker_ts = self._inlet_markers.pull_chunk(
                timeout=0.0, max_samples=MARKERS_PULL_MAX_SAMPLES
            )
        except TypeError:
            marker_chunk, marker_ts = self._inlet_markers.pull_chunk(timeout=0.0)

        if marker_ts:
            marker_samples_this_tick = len(marker_ts)
            self._append_marker_monitor_events(marker_samples_this_tick)
            if self._recording_epochs:
                self._session_recorder.log_markers(marker_chunk=marker_chunk, marker_ts=marker_ts)
                for sample, ts in zip(marker_chunk, marker_ts):
                    if isinstance(sample, (list, tuple)):
                        marker_value = sample[0] if sample else ""
                    else:
                        marker_value = sample
                    self._run_markers_export.append({"ts": float(ts), "value": str(marker_value)})
                for sample, ts in zip(marker_chunk, marker_ts):
                    if isinstance(sample, (list, tuple)):
                        marker_value = sample[0] if sample else ""
                    else:
                        marker_value = sample
                    cue_tid = parse_trial_target_tile_id(sample)
                    if cue_tid is not None:
                        self._lsl_cue_target_id = cue_tid
                        self._marker_ts_last_trial_start = float(ts)
                        self._exp_trial_targets.append(cue_tid)
                        # region agent log
                        debug_ndjson(
                            {
                                "hypothesisId": "H5_experiment",
                                "message": "trial_cue",
                                "data": {
                                    "run_seq": self._exp_run_seq,
                                    "trial_index": len(self._exp_trial_targets) - 1,
                                    "target_tile_id": cue_tid,
                                },
                            }
                        )
                        self._dbg_cue_n += 1
                        debug_ndjson(
                            {
                                "hypothesisId": "H3_cue",
                                "message": "trial_target_lsl",
                                "data": {
                                    "cue_event_seq": self._dbg_cue_n,
                                    "target_tile_id": cue_tid,
                                    "marker_ts": float(ts),
                                    "pending_markers_count": len(self.pending_markers),
                                    "marker_eeg_offset": self._marker_eeg_ts_offset,
                                    "epoch_counts_by_stim": self._epoch_counts_snapshot(),
                                },
                            }
                        )
                        if self._exam_detail_log is not None:
                            self._exam_detail_log.write(
                                "trial_cue_lsl",
                                {
                                    "cue_event_seq": int(self._dbg_cue_n),
                                    "trial_index": len(self._exp_trial_targets) - 1,
                                    "target_tile_id": int(cue_tid),
                                    "marker_ts": float(ts),
                                    "pending_markers_count": len(self.pending_markers),
                                    "epoch_counts_by_stim": self._epoch_counts_snapshot(),
                                },
                            )
                        # endregion
                    stim_key = marker_value_to_stim_key(sample)
                    if stim_key is None:
                        if self._exam_detail_log is not None:
                            self._exam_detail_log.write(
                                "lsl_marker_sample",
                                {
                                    "lsl_ts": float(ts),
                                    "raw_value": str(marker_value),
                                    "parsed_stim_key": None,
                                    "parsed_trial_target_tile_id": cue_tid,
                                    "enqueued_as_stimulus": False,
                                    "note": "not_stimulus_on_marker",
                                },
                            )
                        continue
                    if not self._has_seen_stimulus_marker_in_run:
                        self._has_seen_stimulus_marker_in_run = True
                        self._session_recorder.log_event(
                            "stimulus_stream_started",
                            {
                                "run_seq": self._exp_run_seq,
                                "first_stimulus_marker_ts": float(ts),
                                "stim_key": stim_key,
                            },
                        )
                        if self._exam_detail_log is not None:
                            self._exam_detail_log.write(
                                "stimulus_stream_started",
                                {
                                    "first_stimulus_marker_ts": float(ts),
                                    "stim_key": stim_key,
                                },
                            )
                    tsf = float(ts)
                    if (
                        self.chk_epochs_after_trial.isChecked()
                        and self._marker_ts_last_trial_start is not None
                        and tsf < self._marker_ts_last_trial_start
                    ):
                        if self._exam_detail_log is not None:
                            self._exam_detail_log.write(
                                "lsl_marker_sample",
                                {
                                    "lsl_ts": tsf,
                                    "raw_value": str(marker_value),
                                    "parsed_stim_key": stim_key,
                                    "parsed_trial_target_tile_id": cue_tid,
                                    "enqueued_as_stimulus": False,
                                    "skipped_before_trial_start": True,
                                    "trial_start_ts": float(self._marker_ts_last_trial_start),
                                },
                            )
                        continue
                    self.pending_markers.append((tsf, stim_key))
                    self._stimulus_marker_ts_history.append(tsf)
                    self._last_stim_marker_ts = tsf
                    if self._exam_detail_log is not None:
                        self._exam_detail_log.write(
                            "lsl_marker_sample",
                            {
                                "lsl_ts": tsf,
                                "raw_value": str(marker_value),
                                "parsed_stim_key": stim_key,
                                "parsed_trial_target_tile_id": cue_tid,
                                "enqueued_as_stimulus": True,
                                "pending_len_after": len(self.pending_markers),
                            },
                        )
                    if (
                        self._marker_eeg_ts_offset is None
                        and self._calib_first_marker_ts is None
                    ):
                        self._calib_first_marker_ts = tsf
                        # Фиксируем момент прихода первой вспышки в оси lsl_local_clock()
                        # приёмника — ту же, что используется для ref=_lsl_clock_at_buffer_end.
                        self._calib_first_marker_lsl_clock = float(lsl_local_clock())
                # prevent unbounded growth if marker producer is faster than extraction
                if len(self.pending_markers) > 5000:
                    n_pm = len(self.pending_markers)
                    self.pending_markers = self.pending_markers[-5000:]
                    if self._exam_detail_log is not None:
                        self._exam_detail_log.write(
                            "pending_markers_capped",
                            {"dropped_oldest": int(n_pm - 5000), "kept": 5000},
                        )

        # 2) Pull EEG and extend buffers (take only channel 0)
        try:
            eeg_chunk, eeg_ts = self._inlet_eeg.pull_chunk(
                timeout=0.0, max_samples=EEG_PULL_MAX_SAMPLES
            )
        except TypeError:
            eeg_chunk, eeg_ts = self._inlet_eeg.pull_chunk(timeout=0.0)

        if eeg_ts:
            eeg_samples_this_tick = len(eeg_ts)
            arr = np.asarray(eeg_chunk, dtype=np.float64)
            if arr.size:
                if self._recording_epochs:
                    # Skip idle EEG logging before first real stimulus marker.
                    if self._has_seen_stimulus_marker_in_run:
                        self._session_recorder.log_eeg_chunk(eeg_chunk=arr, eeg_ts=eeg_ts)
                # Normalize to 2D (n_samples, n_channels)
                if arr.ndim == 1:
                    arr_2d = arr.reshape(-1, 1)
                elif arr.ndim == 2:
                    arr_2d = arr
                else:
                    arr_2d = arr.reshape(arr.shape[0], -1)

                # ch0 for monitor: average only selected ROI channels
                roi_channels = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]
                valid_channels = [c for c in roi_channels if 0 <= c < arr_2d.shape[1]]
                if valid_channels:
                    ch0 = np.mean(arr_2d[:, valid_channels], axis=1)
                else:
                    ch0 = np.mean(arr_2d, axis=1)

                self._append_eeg_monitor_samples(ch0)
                if self._recording_epochs:
                    # Store ALL channels per timepoint (channel averaging deferred to epoch extraction)
                    self.eeg_buffer.extend(arr_2d)
                    self.eeg_times.extend([float(t) for t in eeg_ts])
                    self._run_eeg_ts_export.extend([float(t) for t in eeg_ts])
                    self._run_eeg_samples_export.extend(arr_2d.tolist())
                    self._lsl_clock_at_buffer_end = lsl_local_clock()
                    self._lsl_clock_buffer_end_n_samples = len(self._run_eeg_samples_export)
                    if self._lsl_clock_at_buffer_start is None:
                        self._lsl_clock_at_buffer_start = float(self._lsl_clock_at_buffer_end)
                    self._ensure_epoch_template()
                    if (
                        self._exam_detail_log is not None
                        and self._has_seen_stimulus_marker_in_run
                    ):
                        self._exam_detail_log.write(
                            "eeg_chunk",
                            {
                                **summarize_eeg_chunk(arr_2d, [float(t) for t in eeg_ts]),
                                "buf_len_after": len(self.eeg_buffer),
                                "lsl_clock_at_buffer_end": float(self._lsl_clock_at_buffer_end),
                            },
                        )
                    if (
                        self._marker_eeg_ts_offset is None
                        and self._calib_first_marker_ts is not None
                        and self._calib_first_marker_lsl_clock is not None
                    ):
                        fm = float(self._calib_first_marker_ts)
                        fm_lc = float(self._calib_first_marker_lsl_clock)
                        eeg_first = float(self.eeg_times[0]) if self.eeg_times else 0.0
                        eeg_last = float(self.eeg_times[-1]) if self.eeg_times else 0.0
                        # Ось времени привязки — lsl_local_clock() на приёмнике.
                        # ref для индексации эпох тоже берётся там (self._lsl_clock_at_buffer_end),
                        # поэтому offset переводит marker_ts в ту же ось и даёт микросекундную
                        # точность даже когда штампы ЭЭГ (Neurospectrum) округлены до секунды.
                        self._marker_eeg_ts_offset = fm_lc - fm
                        calib_method = "first_flash_lsl_local_clock"

                        self._calib_first_marker_ts = None
                        self._calib_first_marker_lsl_clock = None
                        LOG.info(
                            "LSL выравнивание маркер/ЭЭГ: offset=%.6f method=%s "
                            "(eeg_first=%.6f, eeg_last=%.6f, first_flash_marker=%.6f)",
                            self._marker_eeg_ts_offset,
                            calib_method,
                            eeg_first,
                            eeg_last,
                            fm,
                        )
                        self._session_recorder.log_event(
                            "time_alignment_calibrated",
                            {
                                "run_seq": self._exp_run_seq,
                                "offset_diagnostic": self._marker_eeg_ts_offset,
                                "calib_method": calib_method,
                                "epoch_index_method": "lsl_clock_direct",
                                "lsl_clock_at_buffer_end": self._lsl_clock_at_buffer_end,
                                "eeg_first_ts": eeg_first,
                                "eeg_last_ts": eeg_last,
                                "first_flash_marker_ts": fm,
                                "eeg_buffer_len": len(self.eeg_buffer),
                                "pending_markers_count": len(self.pending_markers),
                            },
                        )
                        if self._exam_detail_log is not None:
                            self._exam_detail_log.write(
                                "time_alignment_calibrated",
                                {
                                    "run_seq": self._exp_run_seq,
                                    "offset_diagnostic": self._marker_eeg_ts_offset,
                                    "calib_method": calib_method,
                                    "lsl_clock_at_buffer_end": self._lsl_clock_at_buffer_end,
                                    "eeg_first_ts": eeg_first,
                                    "eeg_last_ts": eeg_last,
                                    "first_flash_marker_ts": fm,
                                    "eeg_buffer_len": len(self.eeg_buffer),
                                    "pending_markers_count": len(self.pending_markers),
                                },
                            )

        # 3) Extract epochs for pending markers (up to what current buffer allows)
        #
        # Index calculation: bypass coarse EEG timestamps entirely.
        # Use pylsl.local_clock() recorded at the last EEG pull + nominal srate
        # to compute the buffer index for each marker_ts (both in LSL-clock domain).
        if (
            self._recording_epochs
            and self._epoch_geom.epoch_len is not None
            and self._epoch_geom.time_ms_template is not None
            and self.eeg_buffer
            and self._lsl_clock_at_buffer_end is not None
        ):
            dt_s = float(self._epoch_geom.dt_ms) / 1000.0
            srate = 1.0 / dt_s
            el = int(self._epoch_geom.epoch_len)
            buf_len = len(self.eeg_buffer)
            # Опора — lsl_local_clock() на приёмнике (микросекундная точность).
            # Штампы NeuronSpectrum в eeg_times округлены до секунды, поэтому ими как
            # ref пользоваться нельзя — эпохи будут смещаться на ±1 сек.
            time_ref = float(self._lsl_clock_at_buffer_end)
            reserve_samples = max(1, int(EPOCH_RESERVE_MS / 1000.0 / dt_s))

            time_arr = np.asarray(self.eeg_times, dtype=np.float64) if self.eeg_times else np.empty(0)
            # eeg_buffer stores (n_channels,) per timepoint; average ROI channels here
            buf_2d = np.stack(self.eeg_buffer)  # (n_timepoints, n_channels)
            _roi = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]
            _valid = [c for c in _roi if 0 <= c < buf_2d.shape[1]] if buf_2d.ndim == 2 else []
            if buf_2d.ndim == 2 and _valid:
                buf_arr = np.mean(buf_2d[:, _valid], axis=1)
            elif buf_2d.ndim == 2:
                buf_arr = np.mean(buf_2d, axis=1)
            else:
                buf_arr = buf_2d.ravel()
            new_pending: List[Tuple[float, str]] = []

            for marker_ts, stim_key in self.pending_markers:
                start_idx, end_idx, wait_more = self._resolve_epoch_indices_for_marker(
                    marker_ts=float(marker_ts),
                    buf_len=buf_len,
                    srate=srate,
                    epoch_len=el,
                    lsl_ref=time_ref,
                    time_arr=time_arr,
                )
                if wait_more:
                    new_pending.append((marker_ts, stim_key))
                    if self._exam_detail_log is not None:
                        self._exam_detail_log.write(
                            "marker_epoch_wait_more",
                            {
                                "marker_ts": float(marker_ts),
                                "stim_key": stim_key,
                                "buf_len": int(buf_len),
                                "eeg_time_ref": float(time_ref),
                                "lsl_local_clock": (
                                    float(self._lsl_clock_at_buffer_end)
                                    if self._lsl_clock_at_buffer_end is not None
                                    else None
                                ),
                                "epoch_len": int(el),
                                "srate": float(srate),
                            },
                        )
                    continue
                if start_idx is None or end_idx is None:
                    if self._exam_detail_log is not None:
                        self._exam_detail_log.write(
                            "marker_epoch_dropped",
                            {
                                "marker_ts": float(marker_ts),
                                "stim_key": stim_key,
                                "buf_len": int(buf_len),
                                "eeg_time_ref": float(time_ref),
                            },
                        )
                    continue
                if end_idx + reserve_samples > buf_len:
                    new_pending.append((marker_ts, stim_key))
                    if self._exam_detail_log is not None:
                        self._exam_detail_log.write(
                            "marker_epoch_wait_reserve",
                            {
                                "marker_ts": float(marker_ts),
                                "stim_key": stim_key,
                                "buf_len": int(buf_len),
                                "end_idx": int(end_idx),
                                "reserve_samples": int(reserve_samples),
                                "eeg_time_ref": float(time_ref),
                            },
                        )
                    continue

                epoch = buf_arr[start_idx:end_idx]
                if epoch.shape[0] == el:
                    n_epochs_before = sum(len(v) for v in self.epochs_data.values())
                    self.epochs_data.setdefault(stim_key, []).append(epoch.copy())
                    raw_segment = buf_2d[start_idx:end_idx]
                    if raw_segment.ndim == 2:
                        raw_epoch_samples = raw_segment.astype(np.float64).tolist()
                    else:
                        raw_epoch_samples = (
                            np.asarray(raw_segment, dtype=np.float64).reshape(-1, 1).tolist()
                        )
                    if time_arr.shape[0] == buf_len and end_idx <= time_arr.shape[0]:
                        raw_epoch_ts = [float(x) for x in time_arr[start_idx:end_idx]]
                    else:
                        raw_epoch_ts = [None for _ in range(int(el))]
                    self._run_epoch_segments_export.append(
                        {
                            "stim_key": stim_key,
                            "marker_ts": float(marker_ts),
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "eeg_ts": raw_epoch_ts,
                            "eeg_samples": raw_epoch_samples,
                        }
                    )
                    # region agent log
                    self._dbg_epoch_lag_n += 1
                    # lag_ms: how far start_idx is from the ideal marker position.
                    # Ideal index = buf_len - 1 - seconds_back * srate (float, before rounding)
                    _off = self._marker_eeg_ts_offset
                    _t_m = float(marker_ts) + float(_off) if _off is not None else float(marker_ts)
                    seconds_back = time_ref - _t_m
                    ideal_idx = buf_len - 1 - seconds_back * srate
                    lag_ms = (start_idx - ideal_idx) * dt_s * 1000.0
                    span_nominal_s = (float(el) - 1.0) * dt_s
                    # Timestamp resolution diagnostic
                    if time_arr.shape[0] == buf_len and end_idx <= time_arr.shape[0]:
                        epoch_ts = time_arr[start_idx:end_idx]
                        ts_unique_count = int(np.unique(epoch_ts).shape[0])
                    else:
                        ts_unique_count = -1
                    ts_resolution_hint = (
                        "sub-sample" if ts_unique_count > el // 2
                        else f"{ts_unique_count}_unique_in_{el}"
                    )
                    debug_ndjson(
                        {
                            "hypothesisId": "H2_lag",
                            "message": "epoch_align",
                            "data": {
                                "epoch_align_event_seq": self._dbg_epoch_lag_n,
                                "run_seq": self._exp_run_seq,
                                "stim_key": stim_key,
                                "marker_ts": float(marker_ts),
                                "eeg_time_ref": time_ref,
                                "seconds_back": seconds_back,
                                "lag_ms": lag_ms,
                                "span_nominal_s": span_nominal_s,
                                "index_rule": "eeg_time_end_direct",
                                "start_idx": int(start_idx),
                                "end_idx": int(end_idx),
                                "buf_len": buf_len,
                                "epoch_len": el,
                                "srate": srate,
                                "ts_unique_in_epoch": ts_unique_count,
                                "ts_resolution_hint": ts_resolution_hint,
                                "roi_channels_used": _valid,
                            },
                        }
                    )
                    self._session_recorder.log_event(
                        "epoch_extracted",
                        {
                            "run_seq": self._exp_run_seq,
                            "event_seq": self._dbg_epoch_lag_n,
                            "stim_key": stim_key,
                            "marker_ts": float(marker_ts),
                            "eeg_time_ref": time_ref,
                            "seconds_back": seconds_back,
                            "lag_ms": lag_ms,
                            "start_idx": int(start_idx),
                            "end_idx": int(end_idx),
                            "buf_len": buf_len,
                            "epoch_samples": epoch.tolist(),
                            "epoch_counts_by_stim": self._epoch_counts_snapshot(),
                        },
                    )
                    if self._exam_detail_log is not None:
                        self._exam_detail_log.write(
                            "epoch_extracted",
                            {
                                "event_seq": int(self._dbg_epoch_lag_n),
                                "stim_key": stim_key,
                                "marker_ts": float(marker_ts),
                                "eeg_time_ref": float(time_ref),
                                "seconds_back": float(seconds_back),
                                "lag_ms": float(lag_ms),
                                "start_idx": int(start_idx),
                                "end_idx": int(end_idx),
                                "buf_len": int(buf_len),
                                "epoch_len": int(el),
                                "srate": float(srate),
                                "ts_unique_in_epoch": int(ts_unique_count),
                                "ts_resolution_hint": ts_resolution_hint,
                                "roi_channels_used": list(_valid),
                                "epoch_roi": epoch_roi_summary(epoch),
                                "epoch_counts_by_stim": self._epoch_counts_snapshot(),
                            },
                        )
                    # endregion
                    if n_epochs_before == 0:
                        LOG.info(
                            "Первая эпоха ERP: %s, index=[%d:%d] в буфере из %d "
                            "(%.1f с назад от буфера, %d отсч. @ %.1f Гц)",
                            stim_key,
                            start_idx,
                            end_idx,
                            buf_len,
                            seconds_back,
                            el,
                            srate,
                        )
                    # Basic cap: keep most recent epochs per stimulus
                    if len(self.epochs_data[stim_key]) > 300:
                        self.epochs_data[stim_key] = self.epochs_data[stim_key][-300:]
                    need_redraw = True

            self.pending_markers = new_pending

        # 4) Trim old EEG samples to keep memory bounded (sample-count based)
        if self._recording_epochs and self.eeg_buffer and self._epoch_geom.dt_ms:
            _srate_trim = 1000.0 / float(self._epoch_geom.dt_ms)
            max_keep = int(EEG_KEEP_SECONDS * _srate_trim)
            if len(self.eeg_buffer) > max_keep:
                cut = len(self.eeg_buffer) - max_keep
                self.eeg_buffer = self.eeg_buffer[cut:]
                self.eeg_times = self.eeg_times[cut:]
                if self._exam_detail_log is not None:
                    self._exam_detail_log.write(
                        "eeg_buffer_trim",
                        {
                            "removed_samples": int(cut),
                            "buf_len_after": len(self.eeg_buffer),
                            "max_keep_samples": int(max_keep),
                            "eeg_keep_seconds": float(EEG_KEEP_SECONDS),
                        },
                    )

            # Drop pending markers that are too old (их эпоха уже вышла за EEG_KEEP_SECONDS буфера).
            # Сравниваем в оси lsl_local_clock(): маркер со смещением offset и хвост буфера в той же оси.
            if self._lsl_clock_at_buffer_end is not None:
                buf_end = float(self._lsl_clock_at_buffer_end)
                buf_cut_t = buf_end - float(EEG_KEEP_SECONDS)
                off = self._marker_eeg_ts_offset
                n_pending_before = len(self.pending_markers)
                if off is None:
                    self.pending_markers = [
                        (mts, sk) for mts, sk in self.pending_markers if float(mts) >= buf_cut_t
                    ]
                else:
                    of = float(off)
                    self.pending_markers = [
                        (mts, sk) for mts, sk in self.pending_markers
                        if float(mts) + of >= buf_cut_t
                    ]
                if self._exam_detail_log is not None and n_pending_before != len(
                    self.pending_markers
                ):
                    self._exam_detail_log.write(
                        "pending_markers_time_trim",
                        {
                            "before": int(n_pending_before),
                            "after": int(len(self.pending_markers)),
                            "lsl_cutoff_time": float(buf_cut_t),
                            "lsl_clock_buffer_end": float(buf_end),
                            "marker_eeg_offset_applied": off is not None,
                        },
                    )

        # 5) Redraw if new epochs arrived or GUI parameters changed
        if self._need_redraw_params:
            need_redraw = True
            self._need_redraw_params = False

        if need_redraw:
            self._redraw_from_epochs()

        if (
            self._recording_epochs
            and not self._shown_no_marker_hint
            and not self._has_seen_stimulus_marker_in_run
            and len(self.eeg_times) > 200
            and (time.monotonic() - self._recording_started_mono) > 12.0
        ):
            self._shown_no_marker_hint = True
            self._set_status(
                "ЭЭГ идёт, маркеров плиток (LSL) нет — графики ERP не появятся. "
                "Другой ноутбук: та же LAN, без блокировки multicast, брандмауэр; "
                "или LabRecorder на ПК со стимуляцией как мост."
            )

        if (
            self._recording_epochs
            and self._exam_detail_log is not None
            and self._inlet_eeg is not None
            and self._inlet_markers is not None
        ):
            self._exam_qt_tick_seq += 1
            self._exam_detail_log.write(
                "qt_tick",
                {
                    "tick_seq": int(self._exam_qt_tick_seq),
                    "eeg_samples_this_tick": int(eeg_samples_this_tick),
                    "marker_samples_this_tick": int(marker_samples_this_tick),
                    "buf_len": len(self.eeg_buffer),
                    "lsl_local_clock": (
                        float(self._lsl_clock_at_buffer_end)
                        if self._lsl_clock_at_buffer_end is not None
                        else None
                    ),
                    "eeg_time_end": float(self.eeg_times[-1]) if self.eeg_times else None,
                    "marker_eeg_offset": self._marker_eeg_ts_offset,
                    "epoch_len": self._epoch_geom.epoch_len,
                    "dt_ms": self._epoch_geom.dt_ms,
                    "pending": pending_snapshot_for_log(self.pending_markers),
                },
            )

        self._refresh_monitor_ui(
            eeg_samples_this_tick=eeg_samples_this_tick,
            marker_samples_this_tick=marker_samples_this_tick,
            connected=True,
        )

    def _show_export_options_dialog(self) -> Optional[Dict[str, Any]]:
        dlg = QDialog(self)
        dlg.setWindowTitle("Сохранение обследования")
        layout = QVBoxLayout(dlg)

        form = QFormLayout()
        combo_format = QComboBox()
        combo_format.addItem("CSV", "csv")
        combo_format.addItem("TXT", "txt")
        combo_format.addItem("XLSX", "xlsx")
        form.addRow("Формат:", combo_format)
        combo_mode = QComboBox()
        combo_mode.addItem("Одна плитка (по индексу)", "one_stim")
        combo_mode.addItem("Все плитки в один файл (эпохи по порядку вспышек)", "all_stims")
        combo_mode.addItem(
            "Вся ЭЭГ за прогон, включая паузы (непрерывный CSV/XLSX)", "continuous"
        )
        form.addRow("Что сохранять:", combo_mode)
        spin_stim_index = QSpinBox()
        spin_stim_index.setRange(0, 999)
        spin_stim_index.setValue(0)
        form.addRow("Индекс плитки (stim):", spin_stim_index)

        def _mode_changed(_: int = 0) -> None:
            spin_stim_index.setEnabled(combo_mode.currentData() == "one_stim")
            # В непрерывном режиме доступны только CSV/XLSX (TXT не имеет смысла).
            is_cont = combo_mode.currentData() == "continuous"
            if is_cont:
                # Отключаем TXT, если оно выбрано — сдвигаем на CSV.
                i_txt = combo_format.findData("txt")
                if i_txt >= 0 and combo_format.currentIndex() == i_txt:
                    i_csv = combo_format.findData("csv")
                    if i_csv >= 0:
                        combo_format.setCurrentIndex(i_csv)

        combo_mode.currentIndexChanged.connect(_mode_changed)
        _mode_changed()
        combo_channels = QComboBox()
        combo_channels.addItem("Все каналы", "all")
        combo_channels.addItem("Только ROI-каналы (выбранные в UI)", "roi")
        combo_channels.addItem("Вручную (индексы через запятую, 1-based)", "manual")
        form.addRow("Каналы:", combo_channels)
        edit_manual_channels = QLineEdit()
        edit_manual_channels.setPlaceholderText("Напр.: 1,2,5,10")
        edit_manual_channels.setEnabled(False)
        form.addRow("Индексы каналов:", edit_manual_channels)
        combo_channels.currentIndexChanged.connect(
            lambda _: edit_manual_channels.setEnabled(combo_channels.currentData() == "manual")
        )
        layout.addLayout(form)

        info = QLabel(
            "• «Одна плитка» — эпохи только выбранного индекса.\n"
            "• «Все плитки» — эпохи всех плиток в порядке вспышек; колонки segment_id, "
            "stim_key, marker_ts, sample_idx, ts, каналы.\n"
            "• «Вся ЭЭГ за прогон» — один CSV/XLSX, одна строка = один отсчёт ЭЭГ (в т.ч. "
            "когда никакая плитка не моргает); колонки sample_idx, t_rel_s, ts, каналы, "
            "marker (0 = пауза; иначе номер плитки стоит на каждом отсчёте на всём "
            "протяжении её горения, от |on до |off), in_epoch (попал ли в окно эпохи).\n"
            "CSV пишется в «русском» формате Excel: разделитель «;», дробная часть через «,»."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(info)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(dlg.accept)
        box.rejected.connect(dlg.reject)
        layout.addWidget(box)

        if dlg.exec_() != QDialog.Accepted:
            return None

        channels_mode = str(combo_channels.currentData())
        selected_channels: Optional[List[int]] = None
        if channels_mode == "roi":
            selected_channels = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]
        elif channels_mode == "manual":
            raw = edit_manual_channels.text().strip()
            if not raw:
                QMessageBox.warning(self, "Каналы", "Укажите индексы каналов для ручного режима.")
                return None
            parts = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
            selected_channels = []
            for p in parts:
                try:
                    idx_1based = int(p)
                except ValueError:
                    QMessageBox.warning(self, "Каналы", f"Некорректный индекс канала: {p}")
                    return None
                if idx_1based <= 0:
                    QMessageBox.warning(self, "Каналы", f"Индекс должен быть >= 1: {p}")
                    return None
                selected_channels.append(idx_1based - 1)
            selected_channels = sorted(set(selected_channels))

        return {
            "format": str(combo_format.currentData()),
            "stim_index": int(spin_stim_index.value()),
            "mode": str(combo_mode.currentData()),
            "selected_channels": selected_channels,
        }

    def _on_save_exam_clicked(self) -> None:
        if self._recording_epochs:
            QMessageBox.information(
                self,
                "Сохранение недоступно",
                "Сначала остановите анализ, затем сохраните обследование.",
            )
            return
        if not self._last_run_export_data:
            QMessageBox.information(
                self,
                "Нет данных",
                "Нет завершённого обследования для сохранения.",
            )
            return

        opts = self._show_export_options_dialog()
        if opts is None:
            return
        run_seq = self._last_run_export_data.get("run_seq") or 0
        mode = str(opts.get("mode") or "one_stim")
        if mode == "all_stims":
            default_name = f"exam_run_{run_seq}_all_stims"
        elif mode == "continuous":
            default_name = f"exam_run_{run_seq}_continuous"
        else:
            default_name = f"exam_run_{run_seq}_stim_{opts['stim_index']}"
        suffix_hint = {"csv": ".csv", "txt": ".txt", "xlsx": ".xlsx"}.get(opts["format"], ".csv")
        default_dir = str((Path(__file__).resolve().parent.parent / "data" / "exports"))
        selected_path, _ = QFileDialog.getSaveFileName(
            self,
            "Куда сохранить обследование",
            str(Path(default_dir) / f"{default_name}{suffix_hint}"),
            "CSV (*.csv);;TXT (*.txt);;Excel (*.xlsx);;All files (*.*)",
        )
        if not selected_path:
            return

        try:
            if mode == "all_stims":
                if not stim_indices_in_run(self._last_run_export_data):
                    QMessageBox.warning(
                        self,
                        "Нет эпох",
                        "В данных обследования нет сегментов epoch_segments — нечего экспортировать.",
                    )
                    return
                created_files = export_run_data_all_stims(
                    run_data=self._last_run_export_data,
                    output_path=Path(selected_path),
                    file_format=str(opts["format"]),
                    selected_channels=opts["selected_channels"],
                )
            elif mode == "continuous":
                cont_format = str(opts["format"]) if str(opts["format"]) in {"csv", "xlsx"} else "csv"
                cpath = export_run_continuous_csv(
                    run_data=self._last_run_export_data,
                    output_path=Path(selected_path),
                    selected_channels=opts["selected_channels"],
                    file_format=cont_format,
                )
                created_files = [cpath]
                self._show_continuous_diagnostics(cpath)
                return
            else:
                created_files = export_run_data(
                    run_data=self._last_run_export_data,
                    output_path=Path(selected_path),
                    file_format=str(opts["format"]),
                    stim_index=int(opts["stim_index"]),
                    selected_channels=opts["selected_channels"],
                )
        except Exception as e:
            LOG.exception("Ошибка сохранения обследования: %s", e)
            QMessageBox.critical(self, "Ошибка сохранения", str(e))
            return

        files_txt = "\n".join(str(p) for p in created_files)
        QMessageBox.information(
            self,
            "Сохранено",
            f"Обследование сохранено.\n\nФайлы:\n{files_txt}",
        )

    def _show_continuous_diagnostics(self, saved_path: Path) -> None:
        """После сохранения непрерывного экспорта показываем сводку: сколько вспышек
        из LSL-потока плиток попало в файл, с какой частотой дискретизации.

        Помогает отличить ситуации «маркеров вообще не было» и «маркеры есть, но не
        привязались к отсчётам ЭЭГ».
        """
        run = self._last_run_export_data or {}
        markers = list(run.get("markers") or [])
        summary = dict(run.get("summary") or {})
        ap = dict(summary.get("analysis_params") or {})

        n_flashes = 0
        n_mapped = 0
        per_tile: Dict[int, int] = {}
        for item in markers:
            sk = marker_value_to_stim_key(item.get("value"))
            if sk is None:
                continue
            try:
                d = int(stim_key_to_tile_digit(sk))
            except Exception:
                continue
            if d < 0:
                continue
            n_flashes += 1
            per_tile[d] = per_tile.get(d, 0) + 1
            si = item.get("sample_idx")
            if isinstance(si, int) and si >= 0:
                n_mapped += 1

        srate_eff = ap.get("sampling_rate_hz_effective")
        srate_nom = ap.get("sampling_rate_hz")
        srate_est = ap.get("sampling_rate_hz_estimated")
        offset = summary.get("marker_eeg_offset")
        lc_end = summary.get("lsl_clock_at_buffer_end")

        tiles_txt = (
            ", ".join(f"плитка {k}: {v}" for k, v in sorted(per_tile.items())) or "—"
        )

        def _fmt(x: Any) -> str:
            if x is None:
                return "—"
            try:
                return f"{float(x):.4f}"
            except (TypeError, ValueError):
                return str(x)

        text = (
            f"Файл: {saved_path}\n\n"
            f"Маркеры плиток (LSL-стрим моргающих плиток, не NeuroSpectrum):\n"
            f"  всего вспышек в прогоне: {n_flashes}\n"
            f"  привязано к отсчётам ЭЭГ (marker != 0 в файле): {n_mapped}\n"
            f"  по плиткам: {tiles_txt}\n\n"
            f"Частота дискретизации ЭЭГ:\n"
            f"  nominal_srate (от NeuroSpectrum): {_fmt(srate_nom)} Гц\n"
            f"  оценка по чанкам (N/Δlsl_clock): {_fmt(srate_est)} Гц\n"
            f"  использовано при привязке: {_fmt(srate_eff)} Гц\n\n"
            f"Калибровка оси маркер↔ЭЭГ:\n"
            f"  marker_eeg_offset: {_fmt(offset)}\n"
            f"  lsl_clock_at_buffer_end: {_fmt(lc_end)}\n"
        )

        if n_flashes == 0:
            text += (
                "\nВ LSL-потоке плиток не было ни одной вспышки за прогон.\n"
                "Проверь, что стим-стрим (PsychoPy/GUI плиток) запущен и виден в LSL."
            )
        elif n_mapped == 0:
            text += (
                "\nВспышки есть, но ни одна не привязалась к ЭЭГ.\n"
                "Скорее всего, не получилось оценить частоту дискретизации "
                "(nominal_srate() от NeuroSpectrum = 0, а чанков пришло мало, "
                "чтобы оценить по времени). Попробуй запись подлиннее."
            )
        elif n_mapped < n_flashes:
            text += (
                f"\nЧасть вспышек ({n_flashes - n_mapped}) не попала в окно записи ЭЭГ "
                f"(мало отсчётов до/после стимула). Это нормально для краёв прогона."
            )

        QMessageBox.information(self, "Сохранено: сводка по маркерам", text)
