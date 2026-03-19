#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pylsl import StreamInlet, resolve_byprop
import pyqtgraph as pg


pg.setConfigOptions(useOpenGL=False, antialias=False, useCupy=False)
pg.setConfigOption("background", "#0a0a0a")
pg.setConfigOption("foreground", "#E0E0E0")


EPOCH_DURATION_MS = 800
EPOCH_RESERVE_MS = 50
EEG_KEEP_SECONDS = 10.0
MARKERS_PULL_MAX_SAMPLES = 256
EEG_PULL_MAX_SAMPLES = 2048


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

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_loop)

        self._setup_ui()

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

        self.btn_connect = QPushButton("Подключиться к LSL")
        self.btn_connect.setStyleSheet(
            "QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #218838; }"
        )
        self.btn_connect.clicked.connect(self._on_connect_clicked)
        sidebar_layout.addWidget(self.btn_connect)

        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #dc3545; color: white; font-weight: bold; padding: 10px 12px; border-radius: 6px; } "
            "QPushButton:hover { background-color: #c82333; }"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        sidebar_layout.addWidget(self.btn_stop)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(QLabel("Параметры анализа:"))

        self.spin_baseline = QSpinBox()
        self.spin_baseline.setRange(1, 800)
        self.spin_baseline.setValue(100)
        self.spin_baseline.setSuffix(" мс")
        self.spin_baseline.setKeyboardTracking(False)
        self.spin_baseline.valueChanged.connect(self._on_params_changed)

        self.line_channels = QLineEdit()
        self.line_channels.setText("19, 20, 21")  # Pz, Cz, Oz по умолчанию для 21-канальной
        self.line_channels.setStyleSheet(
            "background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; border-radius: 3px; padding: 4px;"
        )
        self.line_channels.textChanged.connect(self._on_params_changed)

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

        self._status_label = QLabel("Отключено. Нажмите «Подключиться к LSL».")
        self._status_label.setWordWrap(True)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(QLabel("Baseline (мс):"))
        sidebar_layout.addWidget(self.spin_baseline)
        sidebar_layout.addWidget(QLabel("Каналы ROI (через запятую, с 1):"))
        sidebar_layout.addWidget(self.line_channels)
        sidebar_layout.addWidget(QLabel("Начало окна X (мс):"))
        sidebar_layout.addWidget(self.spin_x)
        sidebar_layout.addWidget(QLabel("Конец окна Y (мс):"))
        sidebar_layout.addWidget(self.spin_y)

        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self._status_label)
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

    def _on_params_changed(self) -> None:
        self._need_redraw_params = True

    def _on_connect_clicked(self) -> None:
        if self._timer.isActive():
            return

        self._set_status("Подключение к LSL...")
        try:
            eeg_streams = resolve_byprop("type", "EEG", timeout=2)
        except Exception:
            eeg_streams = []

        if not eeg_streams:
            QMessageBox.warning(self, "LSL", "Не найден поток ЭЭГ (type='EEG').")
            self._set_status("Не найден поток ЭЭГ (type='EEG').")
            return

        try:
            marker_streams = resolve_byprop("type", "Markers", timeout=2)
        except Exception:
            marker_streams = []

        if not marker_streams:
            QMessageBox.warning(self, "LSL", "Не найден поток маркеров (type='Markers').")
            self._set_status("Не найден поток маркеров (type='Markers').")
            return

        try:
            self._inlet_eeg = StreamInlet(eeg_streams[0], max_buflen=EEG_KEEP_SECONDS)
        except TypeError:
            self._inlet_eeg = StreamInlet(eeg_streams[0])

        try:
            self._inlet_markers = StreamInlet(marker_streams[0], max_buflen=20)
        except TypeError:
            self._inlet_markers = StreamInlet(marker_streams[0])

        # Reset buffers and epochs
        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._time_ms_template = None
        self._epoch_len = None
        self._dt_ms = None
        self._need_redraw_params = False

        self._timer.start()
        self.btn_connect.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("Подключено. Ожидаю данные...")

    def _on_stop_clicked(self) -> None:
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
        self.btn_stop.setEnabled(False)
        self._set_status("Остановлено. Нажмите «Подключиться к LSL».")
        self._clear_plots()

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

        self._plot_all(
            raw_averaged,
            corrected,
            integrated,
            labels=stim_keys,
            time_ms=time_ms,
            time_crop=time_crop,
        )

        counts = ", ".join([f"{k}:{len(self.epochs_data[k])}" for k in stim_keys])
        self._set_status(
            f"Обновлено: baseline={baseline_ms} мс, окно=[{window_x_ms}, {window_y_ms}] мс. Epochs: {counts}"
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

        # 1) Pull markers and enqueue them
        try:
            marker_chunk, marker_ts = self._inlet_markers.pull_chunk(
                timeout=0.0, max_samples=MARKERS_PULL_MAX_SAMPLES
            )
        except TypeError:
            marker_chunk, marker_ts = self._inlet_markers.pull_chunk(timeout=0.0)

        if marker_ts:
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
            arr = np.asarray(eeg_chunk, dtype=np.float64)
            if arr.size:
                if arr.ndim == 1:
                    ch0 = arr
                elif arr.ndim == 2:
                    # Parse ROI channels from UI (1-indexed in UI -> 0-indexed in numpy)
                    try:
                        raw_text = str(self.line_channels.text())
                    except Exception:
                        raw_text = ""

                    roi_channels: List[int] = []
                    try:
                        for x in raw_text.split(","):
                            x = x.strip()
                            if x.isdigit():
                                idx = int(x) - 1
                                if idx >= 0:
                                    roi_channels.append(idx)
                    except Exception:
                        roi_channels = []

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

                self.eeg_buffer.extend(ch0.tolist())
                self.eeg_times.extend([float(t) for t in eeg_ts])

                self._ensure_epoch_template()

        # 3) Extract epochs for pending markers (up to what current buffer allows)
        if self._epoch_len is not None and self._time_ms_template is not None and self.eeg_times:
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

                end_time = time_arr[end_idx - 1]
                target_end_time = marker_ts + target_end_s
                if end_time < target_end_time - 0.01 or end_time > target_end_time + reserve_s:
                    new_pending.append((marker_ts, stim_key))
                    continue

                epoch = buf_arr[start_idx:end_idx]
                if epoch.shape[0] == self._epoch_len:
                    self.epochs_data.setdefault(stim_key, []).append(epoch.copy())
                    # Basic cap: keep most recent epochs per stimulus
                    if len(self.epochs_data[stim_key]) > 300:
                        self.epochs_data[stim_key] = self.epochs_data[stim_key][-300:]
                    need_redraw = True
                else:
                    new_pending.append((marker_ts, stim_key))

            self.pending_markers = new_pending

        # 4) Trim old EEG samples to keep memory bounded
        if self.eeg_times:
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


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = P300AnalyzerWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

