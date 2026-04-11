#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pylsl import StreamInlet, StreamInfo, resolve_byprop
import pyqtgraph as pg


pg.setConfigOptions(useOpenGL=False, antialias=False, useCupy=False)
pg.setConfigOption("background", "#0a0a0a")
pg.setConfigOption("foreground", "#E0E0E0")


EPOCH_DURATION_MS = 800
EPOCH_RESERVE_MS = 50
EEG_KEEP_SECONDS = 10.0
MIN_EPOCHS_TO_DECIDE = 5
MARKERS_PULL_MAX_SAMPLES = 256
EEG_PULL_MAX_SAMPLES = 2048

WINNER_LABEL_STYLE_IDLE = (
    "QLabel { background-color: #1a1a1a; color: #4dff88; font-size: 18px; "
    "font-weight: bold; padding: 15px; border: 2px solid #333; border-radius: 5px; }"
)

# Фильтр «разрешённых» потоков ЭЭГ (симулятор / NeuroSpectr)
SIMULATOR_NAME = "EEG_Simulator"
SIMULATOR_SOURCE_ID = "eeg-simulator-neurospectr"
NEUROSPECTR_MARKER = "neuro"
EEG_STREAM_TYPES = ("EEG", "Signal")

LOG = logging.getLogger("p300_analyzer")
MONITOR_EEG_PLOT_MAX = 2500
MONITOR_MARKER_EVENTS_MAX = 120
MONITOR_LOG_INTERVAL_S = 2.0


def configure_logging() -> Path:
    """Файл рядом со скриптом + stderr; только логгер p300_analyzer."""
    log_path = Path(__file__).resolve().parent / "p300_analyzer.log"
    lg = logging.getLogger("p300_analyzer")
    if lg.handlers:
        return log_path
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    lg.addHandler(fh)
    lg.addHandler(sh)
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    return log_path


def _is_allowed_stream(info: StreamInfo) -> bool:
    try:
        name = (info.name() or "").strip().lower()
        sid = (info.source_id() or "").strip().lower()
    except Exception:
        return False
    if name == SIMULATOR_NAME.lower() or SIMULATOR_SOURCE_ID in sid:
        return True
    if NEUROSPECTR_MARKER in name or NEUROSPECTR_MARKER in sid:
        return True
    return False


def find_allowed_eeg_streams(timeout: float = 3.0) -> List[StreamInfo]:
    all_streams: List[StreamInfo] = []
    for stream_type in EEG_STREAM_TYPES:
        try:
            streams = resolve_byprop("type", stream_type, timeout=timeout)
            all_streams.extend(streams)
        except Exception:
            pass
    return [s for s in all_streams if _is_allowed_stream(s)]


def resolve_marker_streams(timeout: float = 0.5) -> List[StreamInfo]:
    try:
        return list(resolve_byprop("type", "Markers", timeout=timeout))
    except Exception:
        return []


def _unwrap_combo_userdata(data: Any) -> Any:
    """QComboBox.itemData иногда отдаёт QVariant; pylsl ждёт «сырой» StreamInfo."""
    if data is None:
        return None
    try:
        from PyQt5.QtCore import QVariant

        if isinstance(data, QVariant):
            return data.value()
    except Exception:
        pass
    return data


def _stream_inlet_with_buffer(info: StreamInfo, buffer_seconds: int) -> StreamInlet:
    """
    pylsl: второй параметр должен быть int; часть сборок знает только max_buffered,
    часть — max_buflen. Float (например 10.0) даёт «Don't know how to convert parameter 2».
    """
    try:
        return StreamInlet(info, max_buffered=buffer_seconds)
    except TypeError:
        pass
    try:
        return StreamInlet(info, max_buflen=buffer_seconds)
    except TypeError:
        pass
    return StreamInlet(info)


def marker_value_to_stim_key(marker_value: Any) -> str:
    """
    Convert marker value to dict key like "стимул_1".
    Accepts numeric values or strings containing digits.
    """
    mv = marker_value

    if isinstance(mv, (list, tuple, np.ndarray)) and len(mv) == 1:
        mv = mv[0]

    if isinstance(mv, (bytes, bytearray)):
        mv = mv.decode("utf-8", errors="ignore")

    if isinstance(mv, str):
        s = mv.strip()
        m = re.search(r"(\d+)", s)
        if m:
            return f"стимул_{int(m.group(1))}"
        return s

    if isinstance(mv, (int, np.integer)):
        return f"стимул_{int(mv)}"

    if isinstance(mv, (float, np.floating)):
        return f"стимул_{int(round(float(mv)))}"

    return str(mv)


def stim_key_sort_key(stim_key: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", stim_key)
    if m:
        return int(m.group(1)), stim_key
    return 10**9, stim_key


def baseline_correction(raw: np.ndarray, time_ms: np.ndarray, baseline_ms: int) -> np.ndarray:
    """
    Baseline correction: corrected = raw - mean(raw[:baseline_idx]).

    raw: shape (..., n_time)
    returns: shape (..., n_time)
    """
    if raw.ndim < 1:
        raise ValueError("raw must have at least 1 dimension")
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be a 1D array")
    if raw.shape[-1] != time_ms.shape[0]:
        raise ValueError("raw and time_ms length mismatch")

    dt_ms = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    baseline_idx = int(round(float(baseline_ms) / dt_ms))
    baseline_idx = max(1, min(baseline_idx, time_ms.shape[0]))  # avoid empty slice

    # corrected = raw - np.mean(raw[:baseline_idx])
    baseline_mean = np.mean(raw[..., :baseline_idx], axis=-1, keepdims=True)
    corrected = raw - baseline_mean
    return corrected


def integrated_cumsum(
    corrected: np.ndarray,
    time_ms: np.ndarray,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integration via monotonic CumSum:
        integrated = np.cumsum(np.abs(corrected[x_idx:y_idx]))
    """
    if corrected.ndim < 1:
        raise ValueError("corrected must have at least 1 dimension")
    if time_ms.ndim != 1:
        raise ValueError("time_ms must be 1D array")
    if corrected.shape[-1] != time_ms.shape[0]:
        raise ValueError("corrected and time_ms length mismatch")

    dt_ms = float(time_ms[1] - time_ms[0]) if time_ms.shape[0] > 1 else 1.0
    x_idx = int(round(float(window_x_ms) / dt_ms))
    y_idx = int(round(float(window_y_ms) / dt_ms)) + 1  # include endpoint sample

    x_idx = max(0, min(x_idx, time_ms.shape[0] - 1))
    y_idx = max(x_idx + 1, min(y_idx, time_ms.shape[0]))

    segment = corrected[..., x_idx:y_idx]
    integrated = np.cumsum(np.abs(segment), axis=-1)
    time_crop = time_ms[x_idx:y_idx]
    return integrated, time_crop


class P300AnalyzerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("P300 Analyzer (Online BCI)")
        self.setMinimumWidth(1100)

        # Real-time state required by the task
        self.eeg_buffer: List[float] = []
        self.eeg_times: List[float] = []
        self.pending_markers: List[Tuple[float, str]] = []
        self.epochs_data: Dict[str, List[np.ndarray]] = {}

        self._inlet_eeg: Optional[StreamInlet] = None
        self._inlet_markers: Optional[StreamInlet] = None

        self._time_ms_template: Optional[np.ndarray] = None
        self._epoch_len: Optional[int] = None
        self._dt_ms: Optional[float] = None

        self._need_redraw_params: bool = False

        self._eeg_monitor_buf: Deque[float] = deque(maxlen=MONITOR_EEG_PLOT_MAX)
        self._marker_mono_buf: Deque[float] = deque(maxlen=MONITOR_MARKER_EVENTS_MAX)
        self._last_monitor_log_t: float = 0.0
        self._curve_eeg_monitor: Optional[Any] = None
        self._curve_marker_monitor: Optional[Any] = None

        self._monitor_eeg_win: Optional[QWidget] = None
        self._monitor_markers_win: Optional[QWidget] = None
        self._monitor_eeg_status: Optional[QLabel] = None
        self._monitor_markers_status: Optional[QLabel] = None
        self._plot_eeg_monitor: Optional[pg.PlotWidget] = None
        self._plot_marker_monitor: Optional[pg.PlotWidget] = None

        self._recording_epochs: bool = False

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_loop)

        self._setup_ui()
        self._setup_stream_monitor_windows()

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

    def showEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().showEvent(event)
        if self._monitor_eeg_win and self._monitor_markers_win:
            g = self.geometry()
            self._monitor_eeg_win.move(g.right() + 10, g.top())
            self._monitor_markers_win.move(
                g.right() + 10, g.top() + self._monitor_eeg_win.height() + 12
            )

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
        self.spin_x.setValue(250)
        self.spin_x.setSuffix(" мс")
        self.spin_x.setKeyboardTracking(False)
        self.spin_x.valueChanged.connect(self._on_params_changed)

        self.spin_y = QSpinBox()
        self.spin_y.setRange(1, 800)
        self.spin_y.setValue(500)
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

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self._status_label)

        self.winner_label = QLabel("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self.winner_label.setAlignment(Qt.AlignCenter)

        sidebar_layout.addSpacing(20)
        sidebar_layout.addWidget(self.winner_label)
        sidebar_layout.addStretch(1)

        # Right plots
        plots_container = QWidget()
        plots_layout = QVBoxLayout(plots_container)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(6)

        self.plot_raw = pg.PlotWidget()
        self.plot_raw.setBackground("#0a0a0a")
        self.plot_corrected = pg.PlotWidget()
        self.plot_corrected.setBackground("#0a0a0a")
        self.plot_integrated = pg.PlotWidget()
        self.plot_integrated.setBackground("#0a0a0a")

        self._setup_plot(self.plot_raw, title="Сырые усредненные потенциалы")
        self._setup_plot(self.plot_corrected, title="После выравнивания (Baseline Correction)")
        self._setup_plot(self.plot_integrated, title="Интегрирование (AUC / CumSum)")

        plots_layout.addWidget(self.plot_raw, stretch=1)
        plots_layout.addWidget(self.plot_corrected, stretch=1)
        plots_layout.addWidget(self.plot_integrated, stretch=1)

        root_layout.addWidget(sidebar)
        root_layout.addWidget(plots_container, stretch=1)

    @staticmethod
    def _setup_plot(plot_widget: pg.PlotWidget, *, title: str) -> None:
        plot_widget.setTitle(title, color="#5bc0be", size="14pt")
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setLabel("bottom", "Время, мс")
        plot_widget.setLabel(
            "left",
            "Амплитуда (условн. ед.)"
            if title == "Сырые усредненные потенциалы"
            else ("Амплитуда (условн. ед.)" if "Baseline" in title else "Накопл. AUC (условн. ед.)"),
        )
        plot_widget.addLegend(offset=(10, 10))

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    def _update_markers_presence_label(self) -> None:
        streams = resolve_marker_streams(timeout=0.6)
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
                "Поток плиток (Markers): не найден (запустите стимуляцию PsychoPy)"
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
        self._time_ms_template = None
        self._epoch_len = None
        self._dt_ms = None
        self._need_redraw_params = False
        self._eeg_monitor_buf.clear()
        self._marker_mono_buf.clear()
        self._last_monitor_log_t = 0.0
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._ensure_epoch_template()
        if not self._timer.isActive():
            self._timer.start()
        self.btn_connect.setEnabled(False)
        self.btn_start_analysis.setEnabled(True)
        self.btn_stop_analysis.setEnabled(False)
        self.btn_reset_analysis.setEnabled(True)
        self.btn_disconnect.setEnabled(True)
        self._set_status(
            "Подключено (прогрев). Когда стимуляция готова — нажмите «Начать анализ»."
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
        self._time_ms_template = None
        self._epoch_len = None
        self._dt_ms = None
        self._need_redraw_params = False
        self._recording_epochs = True
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._ensure_epoch_template()
        self.btn_start_analysis.setEnabled(False)
        self.btn_stop_analysis.setEnabled(True)
        self._set_status(
            f"Запись эпох. Для выбора победителя нужно ≥{MIN_EPOCHS_TO_DECIDE} эпох по каждому классу с данными."
        )
        LOG.info("Начата запись эпох для анализа")

    def _on_start_analysis_clicked(self) -> None:
        if self._inlet_eeg is None or self._inlet_markers is None:
            return
        self._begin_recording_session()

    def _on_stop_analysis_clicked(self) -> None:
        if not self._recording_epochs:
            return
        self._recording_epochs = False
        self.pending_markers = []
        self.btn_start_analysis.setEnabled(True)
        self.btn_stop_analysis.setEnabled(False)
        self._set_status(
            "Анализ остановлен (LSL активен). Нажмите «Начать анализ» снова или «Отключить LSL»."
        )
        LOG.info("Запись эпох остановлена пользователем")

    def _on_reset_analysis_clicked(self) -> None:
        if self._inlet_eeg is None or self._inlet_markers is None:
            return
        self._recording_epochs = False
        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._time_ms_template = None
        self._epoch_len = None
        self._dt_ms = None
        self._need_redraw_params = False
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._ensure_epoch_template()
        self.btn_start_analysis.setEnabled(True)
        self.btn_stop_analysis.setEnabled(False)
        self._set_status(
            "Анализ сброшен. Нажмите «Начать анализ», чтобы начать новую запись эпох."
        )
        LOG.info("Сброс анализа (буферы эпох очищены)")

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

        eeg_raw = _unwrap_combo_userdata(self.combo_streams.itemData(idx))
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

        marker_streams = resolve_marker_streams(timeout=2.0)

        if not marker_streams:
            QMessageBox.warning(
                self,
                "LSL",
                "Не найден поток маркеров (type='Markers'). Убедитесь, что стимуляция запущена.",
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
            self._inlet_eeg = _stream_inlet_with_buffer(eeg_info, eeg_buf_s)
            self._inlet_eeg.open_stream(timeout=1.0)

            self._inlet_markers = _stream_inlet_with_buffer(marker_streams[0], 20)
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
        self._time_ms_template = None
        self._epoch_len = None
        self._dt_ms = None
        self._need_redraw_params = False

        self.btn_connect.setEnabled(True)
        self.btn_start_analysis.setEnabled(False)
        self.btn_stop_analysis.setEnabled(False)
        self.btn_reset_analysis.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self._set_status("Остановлено. Обновите список 🔄 и снова «Подключиться к LSL».")
        self._clear_plots()
        self.winner_label.setText("РЕЗУЛЬТАТ: ?")
        self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
        self._reset_monitor_windows_disconnected()
        LOG.info("Сессия LSL остановлена пользователем")

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
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
        LOG.info("Окно анализатора закрыто")
        super().closeEvent(event)

    def _clear_plots(self) -> None:
        self.plot_raw.clear()
        self.plot_corrected.clear()
        self.plot_integrated.clear()

        self._setup_plot(self.plot_raw, title="Сырые усредненные потенциалы")
        self._setup_plot(
            self.plot_corrected, title="После выравнивания (Baseline Correction)"
        )
        self._setup_plot(self.plot_integrated, title="Интегрирование (AUC / CumSum)")

    def _ensure_epoch_template(self) -> None:
        if self._time_ms_template is not None and self._epoch_len is not None and self._dt_ms is not None:
            return

        # Prefer nominal sampling rate from stream info
        dt_ms: Optional[float] = None
        try:
            if self._inlet_eeg is not None:
                srate = float(self._inlet_eeg.info().nominal_srate())
                if srate > 0:
                    dt_ms = 1000.0 / srate
        except Exception:
            dt_ms = None

        # Fallback: estimate dt from timestamps
        if dt_ms is None and len(self.eeg_times) >= 100:
            times = np.asarray(self.eeg_times[-200:], dtype=np.float64)
            diffs = np.diff(times)
            diffs = diffs[diffs > 0]
            if diffs.size:
                dt_ms = float(np.median(diffs) * 1000.0)

        if dt_ms is None or dt_ms <= 0:
            return

        self._dt_ms = dt_ms
        self._epoch_len = int(round(EPOCH_DURATION_MS / dt_ms)) + 1
        self._time_ms_template = np.arange(self._epoch_len, dtype=np.float64) * dt_ms

    def _redraw_from_epochs(self) -> None:
        if self._epoch_len is None or self._time_ms_template is None:
            return

        stim_keys = [k for k, v in self.epochs_data.items() if v]
        if not stim_keys:
            self._clear_plots()
            self.winner_label.setText("РЕЗУЛЬТАТ: ?")
            self.winner_label.setStyleSheet(WINNER_LABEL_STYLE_IDLE)
            return

        stim_keys.sort(key=stim_key_sort_key)
        n_stim = len(stim_keys)

        raw_averaged = np.zeros((n_stim, self._epoch_len), dtype=np.float64)
        for i, key in enumerate(stim_keys):
            epochs = self.epochs_data.get(key, [])
            if not epochs:
                continue
            # Ensure consistent length (should already be fixed by extraction)
            stack = np.stack([e[: self._epoch_len] for e in epochs], axis=0)
            raw_averaged[i, :] = np.mean(stack, axis=0)

        time_ms = self._time_ms_template
        baseline_ms = int(self.spin_baseline.value())
        window_x_ms = int(self.spin_x.value())
        window_y_ms = int(self.spin_y.value())

        if window_y_ms <= window_x_ms:
            window_y_ms = window_x_ms + 1

        corrected = baseline_correction(raw_averaged, time_ms, baseline_ms=baseline_ms)
        integrated, time_crop = integrated_cumsum(
            corrected,
            time_ms,
            window_x_ms=window_x_ms,
            window_y_ms=window_y_ms,
        )

        # === Логика выбора победителя ===
        can_decide = True
        for key in stim_keys:
            if len(self.epochs_data.get(key, [])) < MIN_EPOCHS_TO_DECIDE:
                can_decide = False
                break

        if not can_decide or integrated.size == 0:
            min_n = min(len(self.epochs_data.get(k, [])) for k in stim_keys) if stim_keys else 0
            self.winner_label.setText(
                f"Сбор данных...\n(мин. по классам: {min_n}/{MIN_EPOCHS_TO_DECIDE})"
            )
            self.winner_label.setStyleSheet(
                "QLabel { background-color: #1a1a1a; color: #ffcc33; font-size: 16px; font-weight: bold; padding: 15px; border: 2px solid #333; border-radius: 5px; }"
            )
        else:
            final_auc_values = integrated[:, -1]
            winner_idx = int(np.argmax(final_auc_values))
            winner_key = stim_keys[winner_idx]

            m = re.search(r"(\d+)", winner_key)
            display_name = f"ПЛИТКА {m.group(1)}" if m else winner_key.upper()

            self.winner_label.setText(f"РЕЗУЛЬТАТ:\n{display_name}")
            self.winner_label.setStyleSheet(
                "QLabel { background-color: #0d2614; color: #4dff88; font-size: 20px; font-weight: bold; padding: 15px; border: 2px solid #28a745; border-radius: 5px; }"
            )

        self._plot_all(
            raw_averaged,
            corrected,
            integrated,
            labels=stim_keys,
            time_ms=time_ms,
            time_crop=time_crop,
        )

        counts = ", ".join([f"{k}:{len(self.epochs_data[k])}" for k in stim_keys])
        need_hint = (
            f" Нужно ≥{MIN_EPOCHS_TO_DECIDE} эпох на каждый класс с данными."
            if not can_decide
            else ""
        )
        self._set_status(
            f"Обновлено: baseline={baseline_ms} мс, окно=[{window_x_ms}, {window_y_ms}] мс. "
            f"Эпохи: {counts}.{need_hint}"
        )

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
        colors = ["#ff4d4d", "#4d79ff", "#4dff88", "#ffcc33", "#b366ff", "#33ffd8"]
        n_stim = raw.shape[0]

        # Graph 1: raw averaged ERP
        self.plot_raw.clear()
        self._setup_plot(self.plot_raw, title="Сырые усредненные потенциалы")
        for i in range(n_stim):
            label = labels[i] if i < len(labels) else f"Стимул {i + 1}"
            self.plot_raw.plot(
                time_ms,
                raw[i],
                pen=pg.mkPen(colors[i % len(colors)], width=2),
                name=label,
            )
        self.plot_raw.setXRange(0, 800)

        # Graph 2: baseline corrected ERP
        self.plot_corrected.clear()
        self._setup_plot(
            self.plot_corrected, title="После выравнивания (Baseline Correction)"
        )
        for i in range(n_stim):
            label = labels[i] if i < len(labels) else f"Стимул {i + 1}"
            self.plot_corrected.plot(
                time_ms,
                corrected[i],
                pen=pg.mkPen(colors[i % len(colors)], width=2),
                name=label,
            )
        self.plot_corrected.setXRange(0, 800)

        # Graph 3: integration (monotonic cumsum of abs)
        self.plot_integrated.clear()
        self._setup_plot(self.plot_integrated, title="Интегрирование (AUC / CumSum)")
        for i in range(n_stim):
            label = labels[i] if i < len(labels) else f"Стимул {i + 1}"
            self.plot_integrated.plot(
                time_crop,
                integrated[i],
                pen=pg.mkPen(colors[i % len(colors)], width=2),
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
                for sample, ts in zip(marker_chunk, marker_ts):
                    stim_key = marker_value_to_stim_key(sample)
                    self.pending_markers.append((float(ts), stim_key))
                # prevent unbounded growth if marker producer is faster than extraction
                if len(self.pending_markers) > 5000:
                    self.pending_markers = self.pending_markers[-5000:]

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
                if arr.ndim == 1:
                    ch0 = arr
                elif arr.ndim == 2:
                    # Собираем индексы включенных каналов (0-indexed)
                    roi_channels = [i for i, cb in enumerate(self.channel_checkboxes) if cb.isChecked()]

                    n_avail_channels = arr.shape[1]
                    valid_channels = [c for c in roi_channels if 0 <= c < n_avail_channels]

                    if valid_channels:
                        # Average only selected valid channels
                        ch0 = np.mean(arr[:, valid_channels], axis=1)
                    else:
                        # Fallback: average all available channels
                        ch0 = np.mean(arr, axis=1)
                else:
                    ch0 = arr.reshape(-1)

                self._append_eeg_monitor_samples(ch0)
                if self._recording_epochs:
                    self.eeg_buffer.extend(ch0.tolist())
                    self.eeg_times.extend([float(t) for t in eeg_ts])
                    self._ensure_epoch_template()

        # 3) Extract epochs for pending markers (up to what current buffer allows)
        if (
            self._recording_epochs
            and self._epoch_len is not None
            and self._time_ms_template is not None
            and self.eeg_times
        ):
            reserve_s = EPOCH_RESERVE_MS / 1000.0
            target_end_s = EPOCH_DURATION_MS / 1000.0

            time_arr = np.asarray(self.eeg_times, dtype=np.float64)
            buf_arr = np.asarray(self.eeg_buffer, dtype=np.float64)
            new_pending: List[Tuple[float, str]] = []

            for marker_ts, stim_key in self.pending_markers:
                # If we already have data reaching marker_ts + 800 ms (plus reserve), try to cut epoch.
                if time_arr[-1] < marker_ts + target_end_s + reserve_s:
                    new_pending.append((marker_ts, stim_key))
                    continue

                start_idx = int(np.searchsorted(time_arr, marker_ts, side="left"))
                end_idx = start_idx + self._epoch_len
                if end_idx > time_arr.shape[0]:
                    new_pending.append((marker_ts, stim_key))
                    continue

                start_t = float(time_arr[start_idx])
                end_t = float(time_arr[end_idx - 1])
                span_s = end_t - start_t
                # Раньше здесь сравнивали end_t с marker+800ms в узком окне (~60ms) — на практике эпохи
                # никогда не проходили. Достаточно: есть epoch_len сэмплов после маркера и буфер по времени.

                epoch = buf_arr[start_idx:end_idx]
                if epoch.shape[0] == self._epoch_len:
                    n_epochs_before = sum(len(v) for v in self.epochs_data.values())
                    self.epochs_data.setdefault(stim_key, []).append(epoch.copy())
                    if n_epochs_before == 0:
                        LOG.info(
                            "Первая эпоха ERP: %s, span по LSL=%.4f с (ось графика — nominal srate)",
                            stim_key,
                            span_s,
                        )
                    # Basic cap: keep most recent epochs per stimulus
                    if len(self.epochs_data[stim_key]) > 300:
                        self.epochs_data[stim_key] = self.epochs_data[stim_key][-300:]
                    need_redraw = True
                else:
                    new_pending.append((marker_ts, stim_key))

            self.pending_markers = new_pending

        # 4) Trim old EEG samples to keep memory bounded
        if self._recording_epochs and self.eeg_times:
            latest = float(self.eeg_times[-1])
            cutoff = latest - EEG_KEEP_SECONDS

            time_arr = np.asarray(self.eeg_times, dtype=np.float64)
            cut_idx = int(np.searchsorted(time_arr, cutoff, side="left"))
            if cut_idx > 0:
                self.eeg_buffer = self.eeg_buffer[cut_idx:]
                self.eeg_times = self.eeg_times[cut_idx:]

            # Pending markers older than the buffer start can't be extracted anymore
            new_pending = []
            for marker_ts, stim_key in self.pending_markers:
                if marker_ts >= cutoff:
                    new_pending.append((marker_ts, stim_key))
            self.pending_markers = new_pending

        # 5) Redraw if new epochs arrived or GUI parameters changed
        if self._need_redraw_params:
            need_redraw = True
            self._need_redraw_params = False

        if need_redraw:
            self._redraw_from_epochs()

        self._refresh_monitor_ui(
            eeg_samples_this_tick=eeg_samples_this_tick,
            marker_samples_this_tick=marker_samples_this_tick,
            connected=True,
        )


def main() -> None:
    log_path = configure_logging()
    LOG.info("Старт P300 Analyzer, лог: %s", log_path)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = P300AnalyzerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

