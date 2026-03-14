#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аппаратная валидация ЭЭГ: EXTREME PERFORMANCE + DYNAMIC UI + MANUAL SCALING + DATA SAVING
Оптимизации:
- Сохранение данных из нейроспектра в CSV для сравнения
- Интерактивный сайдбар для включения/выключения каналов.
- Глобальный тумблер "Автомасштаб Y" для возможности ручного зума (колесико мыши).
- Жесткая защита от мусорных данных (NaN, Inf), предотвращающая краш C++ ядра Qt.
- Оптимизированный рендеринг (120+ FPS) скрытых и активных каналов.
"""

import sys
import time
import logging
import queue
import numpy as np
from typing import Optional, List
from datetime import datetime
import os

from pylsl import StreamInlet, StreamInfo, resolve_byprop, ContinuousResolver

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QGroupBox, QInputDialog, QCheckBox,
    QPushButton, QScrollArea, QFrame, QFileDialog, QMessageBox,
    QDoubleSpinBox,
)
from PyQt5.QtCore import QTimer, Qt, QDateTime, QThread, pyqtSignal
from PyQt5.QtGui import QFont
import pyqtgraph as pg

# ==============================================================================
# КОНФИГУРАЦИЯ
# ==============================================================================
EEG_STREAM_TYPES = ("EEG", "Signal")
WINDOW_SEC = 0.2
COV_UPDATE_INTERVAL = 10.0
UPDATE_INTERVAL_MS = 50   # чаще обрабатываем очередь — меньше задержка и потерь в начале
STATS_INTERVAL_MS = 500
SAVE_INTERVAL_MS = 5000  # Автосохранение каждые 5 секунд (можно отключить)
# Поток чтения LSL: быстрый drain буфера, чтобы не терять начальные сэмплы при старте потока в NeuroSpectrum
LSL_PULL_TIMEOUT_S = 0.01  # 10 ms — быстрая реакция на появление данных
LSL_PULL_POLL_S = 0.002   # при отсутствии данных опрашивать каждые 2 ms
LSL_MAX_SAMPLES_PER_CHUNK = 8192  # за один pull забираем больше сэмплов
LSL_MAX_BUFFERED_SEC = 600  # буфер inlet в секундах (для записи — с запасом)

SIMULATOR_NAME = "EEG_Simulator"
SIMULATOR_SOURCE_ID = "eeg-simulator-neurospectr"
NEUROSPECTR_MARKER = "neuro"

# Стандартные полные названия 21 канала (10-20 система), порядок фиксирован
DEFAULT_CHANNEL_NAMES_21 = [
    "Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
]

FIXED_RECORDING_CHANNELS = 21

pg.setConfigOptions(useOpenGL=False, antialias=False, useCupy=False)


# ==============================================================================
# ЛОГИРОВАНИЕ
# ==============================================================================
def setup_logging():
    logger = logging.getLogger("EEG_Validation")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("hardware_validation.log", mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


log = setup_logging()


# ==============================================================================
# ПОИСК ПОТОКОВ И МЕТАДАННЫЕ
# ==============================================================================
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


def find_eeg_streams(timeout: float = 3.0) -> List[StreamInfo]:
    log.info("Поиск LSL потоков ЭЭГ...")
    all_streams = []
    for stream_type in EEG_STREAM_TYPES:
        streams = resolve_byprop("type", stream_type, timeout=timeout)
        all_streams.extend(streams)

    allowed = [s for s in all_streams if _is_allowed_stream(s)]
    log.info(f"Найдено подходящих потоков: {len(allowed)}")
    return allowed


# ==============================================================================
# ПОИСК ПОТОКА В ФОНЕ (пока не нажата "Остановить поиск")
# ==============================================================================
class StreamSearchThread(QThread):
    """Поиск LSL через ContinuousResolver: реакция на появление потока за миллисекунды, без блокирующего resolve."""
    streams_found = pyqtSignal(list)  # list of StreamInfo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        resolvers = []
        try:
            resolvers.append(ContinuousResolver())
        except TypeError:
            for stream_type in EEG_STREAM_TYPES:
                try:
                    resolvers.append(ContinuousResolver("type", stream_type))
                except Exception:
                    pass
        except Exception:
            pass
        if not resolvers:
            return
        while not self._stop_requested:
            streams = []
            for resolver in resolvers:
                try:
                    streams.extend(resolver.results())
                except Exception:
                    pass
            allowed = [s for s in streams if _is_allowed_stream(s)]
            if allowed:
                self.streams_found.emit(allowed)
                break
            time.sleep(0.05)


class LSLPullThread(QThread):
    """
    Фоновый поток, непрерывно читающий из LSL и складывающий чанки в очередь.
    При появлении данных тянем сразу (timeout=0), иначе короткий sleep — минимизируем потерю начала.
    """
    def __init__(self, inlet: StreamInlet, chunk_queue: queue.Queue, stop_flag: List[bool], parent=None):
        super().__init__(parent)
        self.inlet = inlet
        self.chunk_queue = chunk_queue
        self._stop_flag = stop_flag  # [False] -> ставим [0]=True для остановки

    def run(self):
        while not self._stop_flag[0]:
            try:
                # Если данные уже есть — забираем без ожидания; иначе коротко ждём
                try:
                    n = self.inlet.samples_available()
                except Exception:
                    n = 0
                timeout = 0.0 if n > 0 else LSL_PULL_TIMEOUT_S
                chunk, timestamps = self.inlet.pull_chunk(
                    timeout=timeout,
                    max_samples=LSL_MAX_SAMPLES_PER_CHUNK,
                )
                if chunk:
                    self.chunk_queue.put((chunk, timestamps))
                elif n == 0:
                    time.sleep(LSL_PULL_POLL_S)
            except Exception as e:
                log.debug("LSL pull thread: %s", e)
                break


def _create_inlet(info: StreamInfo) -> StreamInlet:
    """Создать inlet и сразу открыть поток — без блокирующего time_correction, чтобы не терять первые сэмплы."""
    try:
        inlet = StreamInlet(info, max_buffered=LSL_MAX_BUFFERED_SEC)
    except TypeError:
        inlet = StreamInlet(info)
    try:
        inlet.open_stream(timeout=1.0)
    except Exception:
        pass
    return inlet


def get_channel_names(info: StreamInfo, n_channels: int) -> List[str]:
    channels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for _ in range(n_channels):
            name = ch.child_value("label") or ch.child_value("name") or ch.child_value("type")
            channels.append(name.strip() if name else f"Ch {len(channels) + 1}")
            ch = ch.next_sibling()
    except Exception:
        pass
    while len(channels) < n_channels:
        channels.append(f"Ch {len(channels) + 1}")
    return channels[:n_channels]


# ==============================================================================
# GUI КОМПОНЕНТЫ
# ==============================================================================
class ChannelWidget(QFrame):

    def __init__(self, channel_id: int, channel_name: str, sampling_rate: float):
        super().__init__()
        self.channel_id = channel_id
        self.channel_name = channel_name
        self.sampling_rate = sampling_rate
        self.buffer_size = int(WINDOW_SEC * sampling_rate)

        self.y_data = np.zeros(self.buffer_size, dtype=np.float32)
        self.x_data = np.linspace(-WINDOW_SEC, 0, self.buffer_size, dtype=np.float32)
        self.filled = 0

        # Для сохранения полных данных (не только буфер)
        self.full_data = []  # будет хранить все полученные данные

        self.current_y_min = -10.0
        self.current_y_max = 10.0

        self.is_active = True

        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame { background-color: #1a1a1a; border: 1px solid #333; border-radius: 4px; }")

        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)

        header = QLabel(f"[{self.channel_id + 1}] {self.channel_name}")
        header.setFont(QFont("Arial", 8, QFont.Bold))
        header.setStyleSheet("color: white; background-color: transparent; border: none; padding: 2px;")
        layout.addWidget(header)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setDownsampling(mode=None)
        self.plot_widget.setClipToView(True)

        # Разрешаем зум мышью только по оси Y (чтобы шкала времени не ломалась)
        self.plot_widget.setMouseEnabled(x=False, y=True)
        self.plot_widget.setMenuEnabled(True)  # Включаем меню по правому клику

        self.plot_widget.getViewBox().setLimits(yMin=-1000000, yMax=1000000)

        self.plot_widget.setLabel('left', 'мкВ', color='white', fontsize=6)
        self.plot_widget.setBackground('#000000')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setMinimumHeight(80)
        self.plot_widget.setXRange(-WINDOW_SEC, 0, padding=0)

        colors = pg.intColor(self.channel_id, hues=15, values=1, maxValue=255, minValue=150)
        self.signal_line = self.plot_widget.plot(self.x_data, self.y_data, pen=pg.mkPen(colors, width=1.0),
                                                 autoDownsample=False)

        layout.addWidget(self.plot_widget)

        self.stats_label = QLabel("Ожидание...")
        self.stats_label.setFont(QFont("Courier", 7))
        self.stats_label.setStyleSheet("color: #a8ff9e; background-color: transparent; border: none;")
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    def push_chunk(self, new_data: np.ndarray):
        if not self.is_active:
            return

        n = len(new_data)
        if n == 0: return

        try:
            # Защита от переполнения и NaN
            clean_data = np.nan_to_num(new_data, nan=0.0, posinf=0.0, neginf=0.0)
            clean_data = np.clip(clean_data, -250000.0, 250000.0)

            # Сохраняем полные данные (для экспорта)
            self.full_data.extend(clean_data.tolist())

        except Exception:
            clean_data = np.zeros(n, dtype=np.float32)

        self.filled = min(self.buffer_size, self.filled + n)

        if n >= self.buffer_size:
            self.y_data[:] = clean_data[-self.buffer_size:]
        else:
            self.y_data[:-n] = self.y_data[n:]
            self.y_data[-n:] = clean_data

    def clear_saved_data(self):
        """Очистить сохраненные данные (для нового сеанса записи)"""
        self.full_data = []
        log.info(f"Канал {self.channel_name}: данные очищены")

    def update_plot(self, auto_scale: bool):
        if not self.is_active or self.filled == 0 or not self.isVisible():
            return

        try:
            self.signal_line.setData(self.x_data, self.y_data, skipFiniteCheck=True)

            # Применяем умное масштабирование, ТОЛЬКО если включена галочка
            if auto_scale:
                y_min, y_max = float(np.min(self.y_data)), float(np.max(self.y_data))

                span = y_max - y_min
                if span < 20.0:
                    center = (y_max + y_min) / 2.0
                    target_min, target_max = center - 10.0, center + 10.0
                else:
                    margin = span * 0.15
                    target_min, target_max = y_min - margin, y_max + margin

                if abs(target_min - self.current_y_min) > 2.0 or abs(target_max - self.current_y_max) > 2.0:
                    self.current_y_min, self.current_y_max = target_min, target_max
                    self.plot_widget.setYRange(self.current_y_min, self.current_y_max, padding=0, update=False)
        except Exception:
            pass

    def set_y_range(self, y_min: float, y_max: float):
        """Задать масштаб оси Y по значениям (мкВ)."""
        if y_min >= y_max:
            return
        self.current_y_min = float(y_min)
        self.current_y_max = float(y_max)
        self.plot_widget.setYRange(self.current_y_min, self.current_y_max, padding=0, update=False)

    def set_time_window(self, sec: float):
        """Изменить длительность окна по оси X (секунды)."""
        if sec <= 0:
            return
        new_size = max(1, int(sec * self.sampling_rate))
        old_size = len(self.y_data)
        new_y = np.zeros(new_size, dtype=np.float32)
        if new_size <= old_size:
            new_y[:] = self.y_data[-new_size:]
        else:
            new_y[-old_size:] = self.y_data[:]
        self.y_data = new_y
        self.x_data = np.linspace(-sec, 0, new_size, dtype=np.float32)
        self.buffer_size = new_size
        self.filled = min(self.filled, new_size)
        self.plot_widget.setXRange(-sec, 0, padding=0)

    def update_stats(self):
        if not self.is_active or self.filled < 10 or not self.isVisible():
            return

        try:
            data = self.y_data[-self.filled:]
            mean_val, std_val = float(np.mean(data)), float(np.std(data))
            rms_val = float(np.sqrt(np.mean(data ** 2)))
            p2p = float(np.max(data) - np.min(data))

            if p2p > 200000:
                self.stats_label.setStyleSheet("color: #ff4d4d; font-weight: bold; border: none;")
                self.stats_label.setText("⚠ ОШИБКА КОНТАКТА ⚠")
            else:
                self.stats_label.setStyleSheet("color: #a8ff9e; border: none;")
                self.stats_label.setText(
                    f"Mean:{mean_val:6.1f} | STD:{std_val:6.1f} | RMS:{rms_val:6.1f} | P2P:{p2p:6.1f}")
        except Exception:
            pass


# ==============================================================================
# ГЛАВНОЕ ОКНО С САЙДБАРОМ
# ==============================================================================
class HardwareValidationWindow(QMainWindow):

    def __init__(
        self,
        inlet: Optional[StreamInlet] = None,
        full_info: Optional[StreamInfo] = None,
        channel_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.inlet = inlet
        self.stream_name = (full_info.name() or "EEG") if full_info else "—"
        self.channel_names = channel_names or []
        self.n_channels = full_info.channel_count() if full_info else 0
        self.sampling_rate = full_info.nominal_srate() if full_info else 0
        self._has_stream = inlet is not None and full_info is not None

        self.channel_widgets: List[ChannelWidget] = []
        self.checkboxes: List[QCheckBox] = []
        self.info_panel: Optional[QWidget] = None
        self.grid_widget: Optional[QWidget] = None
        self.placeholder_widget: Optional[QWidget] = None

        self.start_time = time.time()
        self.last_cov_time = self.start_time
        self.sample_count = 0

        self.recording = False
        self.save_folder = "saved_data"
        # Каналы для записи: 21 чекбокс, можно выбрать до подключения потока
        self.record_channel_checked = [True] * FIXED_RECORDING_CHANNELS  # какие из 21 записывать
        self.record_checkboxes: List[QCheckBox] = []  # чекбоксы "каналы для записи"
        # Буфер записи: один список на канал (по порядку 0..20), заполняется при recording и при подключённом потоке
        self.recording_buffer: List[List[float]] = [[] for _ in range(FIXED_RECORDING_CHANNELS)]
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

        self._search_thread: Optional[StreamSearchThread] = None
        self._searching = False
        # Очередь чанков от фонового потока LSL (чтобы не терять начальные сэмплы)
        self._pull_queue: queue.Queue = queue.Queue()
        self._pull_stop: List[bool] = [False]
        self._pull_thread: Optional[LSLPullThread] = None

        self._setup_ui()
        if self._has_stream:
            self._setup_timers()
            self._pull_stop[0] = False
            self._pull_thread = LSLPullThread(self.inlet, self._pull_queue, self._pull_stop, self)
            self._pull_thread.start()
            log.info("GUI готово. Режим DYNAMIC GRID активирован. Фоновое чтение LSL запущено.")
        else:
            log.info("GUI готово. Поток не подключён — нажмите «Поиск потока».")

    def _setup_ui(self):
        self.setWindowTitle(f"LSL Validation — {self.stream_name}")
        self.setStyleSheet("""
            QMainWindow { background-color: #0a0a0a; color: white; }
            QLabel { color: white; }
            QCheckBox { color: white; spacing: 5px; font-size: 13px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QPushButton { background-color: #333; color: white; border-radius: 3px; padding: 5px; }
            QPushButton:hover { background-color: #444; }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 1. ЛЕВАЯ ПАНЕЛЬ (САЙДБАР)
        sidebar = QWidget()
        sidebar.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        # Кнопка поиска / остановки поиска потока (поиск идёт, пока не нажата "Остановить поиск")
        self.btn_search_stream = QPushButton("🔍 Поиск потока")
        self.btn_search_stream.setStyleSheet("QPushButton { background-color: #007bff; font-weight: bold; }")
        self.btn_search_stream.clicked.connect(self._on_search_or_stop_stream)
        sidebar_layout.addWidget(self.btn_search_stream)

        # Кнопка автомасштаба (актуальна только при подключённом потоке)
        self.cb_autoscale = QCheckBox("Автомасштаб Y")
        self.cb_autoscale.setChecked(True)
        self.cb_autoscale.setStyleSheet("QCheckBox { color: #5bc0be; font-weight: bold; margin-bottom: 10px; }")
        sidebar_layout.addWidget(self.cb_autoscale)

        # Ручной масштаб Y (мкВ) — ввод по цифрам
        scale_group = QGroupBox("Масштаб Y (мкВ)")
        scale_group.setStyleSheet(
            "QGroupBox { color: white; margin-top: 6px; } "
            "QDoubleSpinBox { background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px; min-width: 72px; selection-background-color: #007bff; } "
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { background-color: #3d3d3d; border: none; } "
            "QLabel { color: #ccc; }"
        )
        scale_layout = QVBoxLayout()
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Мин:"))
        self.spin_y_min = QDoubleSpinBox()
        self.spin_y_min.setRange(-500000, 500000)
        self.spin_y_min.setValue(-100)
        self.spin_y_min.setDecimals(1)
        self.spin_y_min.setSingleStep(10)
        self.spin_y_min.valueChanged.connect(self._apply_manual_scale)
        scale_row.addWidget(self.spin_y_min)
        scale_layout.addLayout(scale_row)
        scale_row2 = QHBoxLayout()
        scale_row2.addWidget(QLabel("Макс:"))
        self.spin_y_max = QDoubleSpinBox()
        self.spin_y_max.setRange(-500000, 500000)
        self.spin_y_max.setValue(100)
        self.spin_y_max.setDecimals(1)
        self.spin_y_max.setSingleStep(10)
        self.spin_y_max.valueChanged.connect(self._apply_manual_scale)
        scale_row2.addWidget(self.spin_y_max)
        scale_layout.addLayout(scale_row2)
        scale_group.setLayout(scale_layout)
        sidebar_layout.addWidget(scale_group)

        # Масштаб по X (время, секунды)
        scale_x_group = QGroupBox("Масштаб X (с)")
        scale_x_group.setStyleSheet(
            "QGroupBox { color: white; margin-top: 6px; } "
            "QDoubleSpinBox { background-color: #2d2d2d; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 3px; padding: 4px; min-width: 72px; selection-background-color: #007bff; } "
            "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button { background-color: #3d3d3d; border: none; } "
            "QLabel { color: #ccc; }"
        )
        scale_x_layout = QVBoxLayout()
        scale_x_row = QHBoxLayout()
        scale_x_row.addWidget(QLabel("Окно:"))
        self.spin_x_window = QDoubleSpinBox()
        self.spin_x_window.setRange(0.05, 60.0)
        self.spin_x_window.setValue(WINDOW_SEC)
        self.spin_x_window.setDecimals(2)
        self.spin_x_window.setSingleStep(0.1)
        self.spin_x_window.setSuffix(" с")
        self.spin_x_window.valueChanged.connect(self._apply_x_window)
        scale_x_row.addWidget(self.spin_x_window)
        scale_x_layout.addLayout(scale_x_row)
        scale_x_group.setLayout(scale_x_layout)
        sidebar_layout.addWidget(scale_x_group)

        # СЕКЦИЯ КАНАЛЫ ДЛЯ ЗАПИСИ (доступна до подключения потока, 21 канал по порядку)
        record_channels_group = QGroupBox("Каналы для записи")
        record_channels_group.setStyleSheet("QGroupBox { color: #5bc0be; margin-top: 10px; font-weight: bold; }")
        record_channels_layout = QVBoxLayout()
        record_btn_row = QHBoxLayout()
        btn_record_all = QPushButton("Все")
        btn_record_none = QPushButton("Сброс")
        btn_record_all.clicked.connect(self._set_all_record_channels)
        btn_record_none.clicked.connect(lambda: self._set_all_record_channels(False))
        record_btn_row.addWidget(btn_record_all)
        record_btn_row.addWidget(btn_record_none)
        record_channels_layout.addLayout(record_btn_row)
        scroll_record = QScrollArea()
        scroll_record.setWidgetResizable(True)
        scroll_record.setMaximumHeight(180)
        scroll_record.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        record_cb_container = QWidget()
        record_cb_container.setStyleSheet("background-color: transparent;")
        self.record_cb_layout = QVBoxLayout(record_cb_container)
        self.record_cb_layout.setSpacing(2)
        for i in range(FIXED_RECORDING_CHANNELS):
            name = DEFAULT_CHANNEL_NAMES_21[i] if i < len(DEFAULT_CHANNEL_NAMES_21) else f"Ch{i+1}"
            cb = QCheckBox(f"[{i+1}] {name}")
            cb.setChecked(True)
            cb.setProperty("channel_index", i)
            cb.stateChanged.connect(self._on_record_channel_toggled)
            self.record_checkboxes.append(cb)
            self.record_cb_layout.addWidget(cb)
        self.record_cb_layout.addStretch()
        scroll_record.setWidget(record_cb_container)
        record_channels_layout.addWidget(scroll_record)
        record_channels_group.setLayout(record_channels_layout)
        sidebar_layout.addWidget(record_channels_group)

        # СЕКЦИЯ СОХРАНЕНИЯ ДАННЫХ
        save_group = QGroupBox("Сохранение данных")
        save_group.setStyleSheet("QGroupBox { color: white; margin-top: 10px; }")
        save_layout = QVBoxLayout()

        # Кнопка начала/остановки записи (можно нажать до подключения потока)
        self.btn_record = QPushButton("⏺ Начать запись")
        self.btn_record.setStyleSheet("QPushButton { background-color: #28a745; font-weight: bold; }")
        self.btn_record.clicked.connect(self.toggle_recording)
        save_layout.addWidget(self.btn_record)

        # Кнопка сохранения текущих данных
        self.btn_save = QPushButton("💾 Сохранить в файл")
        self.btn_save.setStyleSheet("QPushButton { background-color: #007bff; }")
        self.btn_save.clicked.connect(self.save_data_to_file)
        save_layout.addWidget(self.btn_save)

        # Кнопка очистки буфера записи
        self.btn_clear = QPushButton("🗑 Очистить буфер записи")
        self.btn_clear.setStyleSheet("QPushButton { background-color: #dc3545; }")
        self.btn_clear.clicked.connect(self.clear_saved_data)
        save_layout.addWidget(self.btn_clear)

        # Индикатор записи
        self.recording_label = QLabel("⚫ Запись остановлена")
        self.recording_label.setStyleSheet("color: #dc3545; font-weight: bold;")
        save_layout.addWidget(self.recording_label)

        save_group.setLayout(save_layout)
        sidebar_layout.addWidget(save_group)

        lbl = QLabel("ОТОБРАЖЕНИЕ:")
        lbl.setFont(QFont("Arial", 10, QFont.Bold))
        sidebar_layout.addWidget(lbl)

        btn_layout = QHBoxLayout()
        btn_all = QPushButton("Все")
        btn_all.clicked.connect(lambda: self._set_all_channels(True))
        btn_none = QPushButton("Сброс")
        btn_none.clicked.connect(lambda: self._set_all_channels(False))
        btn_layout.addWidget(btn_all)
        btn_layout.addWidget(btn_none)
        sidebar_layout.addLayout(btn_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        cb_container = QWidget()
        cb_container.setStyleSheet("background-color: transparent;")
        self.cb_layout = QVBoxLayout(cb_container)
        self.cb_layout.setSpacing(6)
        for i, name in enumerate(self.channel_names):
            cb = QCheckBox(f"[{i + 1}] {name}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._rebuild_grid)
            self.checkboxes.append(cb)
            self.cb_layout.addWidget(cb)
        self.cb_layout.addStretch()
        scroll.setWidget(cb_container)
        sidebar_layout.addWidget(scroll)

        # 2. ПРАВАЯ ПАНЕЛЬ (графики или заглушка)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)

        if self._has_stream:
            self._build_stream_content()
        else:
            self._build_placeholder()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.right_panel, stretch=1)
        self.resize(1600, 1000)

    def _build_placeholder(self):
        """Правая панель при отсутствии потока."""
        self.placeholder_widget = QWidget()
        pl_layout = QVBoxLayout(self.placeholder_widget)
        pl_layout.setAlignment(Qt.AlignCenter)
        pl_layout.setSpacing(20)
        lbl = QLabel("Поток ЭЭГ не подключён.\nЗапустите нейроспектр или симулятор и нажмите «Поиск потока» в боковой панели.")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 14px;")
        lbl.setWordWrap(True)
        pl_layout.addWidget(lbl)
        self.right_layout.addWidget(self.placeholder_widget, stretch=1)

    def _build_stream_content(self):
        """Правая панель с информацией и графиками каналов."""
        if self.placeholder_widget:
            self.right_layout.removeWidget(self.placeholder_widget)
            self.placeholder_widget.setParent(None)
            self.placeholder_widget = None
        self.info_panel = QHBoxLayout()
        self.info_panel.addWidget(QLabel(f"<b>Поток:</b> {self.stream_name}"))
        self.info_panel.addWidget(QLabel(f"<b>Частота:</b> {self.sampling_rate} Гц"))
        self.samples_label = QLabel(f"<b>Сэмплов:</b> {self.sample_count}")
        self.info_panel.addWidget(self.samples_label)
        self.info_panel.addStretch()
        self.right_layout.addLayout(self.info_panel)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        for ch in range(self.n_channels):
            cw = ChannelWidget(ch, self.channel_names[ch], self.sampling_rate)
            self.channel_widgets.append(cw)
        self.right_layout.addWidget(self.grid_widget, stretch=1)
        self._rebuild_grid()

    def toggle_recording(self):
        """Включить/выключить непрерывную запись данных"""
        self.recording = not self.recording
        if self.recording:
            self.btn_record.setText("⏹ Остановить запись")
            self.btn_record.setStyleSheet("QPushButton { background-color: #dc3545; font-weight: bold; }")
            self.recording_label.setText("🔴 ИДЕТ ЗАПИСЬ...")
            self.recording_label.setStyleSheet("color: #28a745; font-weight: bold;")
            log.info("Начало непрерывной записи данных")
        else:
            self.btn_record.setText("⏺ Начать запись")
            self.btn_record.setStyleSheet("QPushButton { background-color: #28a745; font-weight: bold; }")
            self.recording_label.setText("⚫ Запись остановлена")
            self.recording_label.setStyleSheet("color: #dc3545; font-weight: bold;")
            log.info("Остановка записи данных")

    def save_data_to_file(self):
        """Сохранить буфер записи в файл: один столбец — один канал (по порядку), 3 знака после запятой, в имени — дата/время, в файле — частота дискретизации."""
        try:
            # Используем буфер записи (выбранные каналы в порядке 0..20)
            channels_to_save = [i for i in range(FIXED_RECORDING_CHANNELS) if self.record_channel_checked[i]]
            if not channels_to_save:
                QMessageBox.warning(self, "Нет каналов", "Выберите хотя бы один канал для записи в блоке «Каналы для записи».")
                return

            max_len = max(len(self.recording_buffer[i]) for i in channels_to_save)
            if max_len == 0:
                QMessageBox.warning(self, "Нет данных", "Буфер записи пуст. Запустите запись и дождитесь поступления данных.")
                return

            # Имя файла: дата и время с точностью до секунды (три знака после запятой в данных, не в имени)
            dt = datetime.now()
            timestamp = dt.strftime("%Y-%m-%d_%H-%M-%S")
            default_name = os.path.join(self.save_folder, f"eeg_{timestamp}.csv")

            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить запись ЭЭГ",
                default_name,
                "CSV (*.csv);;Текстовый файл (*.txt);;Все файлы (*.*)",
                options=options,
            )

            if not file_path:
                return

            self._write_recording_buffer_to_file(file_path, channels_to_save, max_len)
            srate = self.sampling_rate if self._has_stream else 0
            log.info(f"Запись сохранена: {file_path}, каналы по порядку, sampling_rate={srate}")
            QMessageBox.information(
                self,
                "Успех",
                f"Данные сохранены:\n{file_path}\n\nКаналы по порядку, точность 3 знака. Частота дискретизации: {srate} Гц.",
            )

        except Exception as e:
            log.error(f"Ошибка при сохранении данных: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить данные:\n{str(e)}")


    def clear_saved_data(self):
        """Очистить буфер записи и сохранённые данные по каналам."""
        reply = QMessageBox.question(
            self,
            "Очистка буфера записи",
            "Очистить буфер записи (и данные по каналам отображения)?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            for i in range(FIXED_RECORDING_CHANNELS):
                self.recording_buffer[i] = []
            for cw in self.channel_widgets:
                cw.clear_saved_data()
            log.info("Буфер записи и данные каналов очищены")
            QMessageBox.information(self, "Готово", "Буфер записи очищен")

    def _apply_manual_scale(self):
        """Применить масштаб Y из полей ввода ко всем каналам."""
        if not self._has_stream or not self.channel_widgets:
            return
        y_min = self.spin_y_min.value()
        y_max = self.spin_y_max.value()
        if y_min >= y_max:
            return
        self.cb_autoscale.setChecked(False)
        for cw in self.channel_widgets:
            cw.set_y_range(y_min, y_max)

    def _apply_x_window(self):
        """Применить длительность окна по X ко всем каналам."""
        if not self._has_stream or not self.channel_widgets:
            return
        sec = self.spin_x_window.value()
        if sec <= 0:
            return
        for cw in self.channel_widgets:
            cw.set_time_window(sec)

    def _set_all_record_channels(self, checked: bool = True):
        """Включить или выключить все каналы для записи."""
        for i, cb in enumerate(self.record_checkboxes):
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            self.record_channel_checked[i] = checked

    def _on_record_channel_toggled(self):
        """Синхронизировать состояние чекбоксов записи с record_channel_checked."""
        for i, cb in enumerate(self.record_checkboxes):
            if i < FIXED_RECORDING_CHANNELS:
                self.record_channel_checked[i] = cb.isChecked()

    def _set_all_channels(self, state: bool):
        for cb in self.checkboxes:
            cb.blockSignals(True)
            cb.setChecked(state)
            cb.blockSignals(False)
        self._rebuild_grid()

    def _rebuild_grid(self):
        if not self._has_stream or not self.channel_widgets:
            return
        for cw in self.channel_widgets:
            self.grid_layout.removeWidget(cw)
            cw.setVisible(False)
            cw.is_active = False

        active_indices = [i for i, cb in enumerate(self.checkboxes) if cb.isChecked()]
        n_active = len(active_indices)

        if n_active == 0:
            return

        n_cols = 1 if n_active == 1 else (2 if n_active <= 4 else 3)

        for grid_idx, ch_idx in enumerate(active_indices):
            cw = self.channel_widgets[ch_idx]
            cw.is_active = True
            cw.setVisible(True)
            row, col = grid_idx // n_cols, grid_idx % n_cols
            self.grid_layout.addWidget(cw, row, col)

    def _setup_timers(self):
        self.plot_timer = QTimer()
        self.plot_timer.setTimerType(Qt.PreciseTimer)
        self.plot_timer.timeout.connect(self._pull_and_plot)
        self.plot_timer.start(UPDATE_INTERVAL_MS)

        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_text_stats)
        self.stats_timer.start(STATS_INTERVAL_MS)

        # Таймер для автосохранения (если включена запись)
        self.save_timer = QTimer()
        self.save_timer.timeout.connect(self._auto_save)
        self.save_timer.start(SAVE_INTERVAL_MS)

    def _auto_save(self):
        """Автоматическое сохранение буфера записи при включенной записи (тот же формат: столбцы по порядку, 3 знака)."""
        if not self.recording:
            return
        channels_to_save = [i for i in range(FIXED_RECORDING_CHANNELS) if self.record_channel_checked[i]]
        if not channels_to_save:
            return
        max_len = max(len(self.recording_buffer[i]) for i in channels_to_save)
        if max_len == 0:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filepath = os.path.join(self.save_folder, f"eeg_{timestamp}.csv")
        self._write_recording_buffer_to_file(filepath, channels_to_save, max_len)
        log.info(f"Автосохранение: {filepath}")

    def _write_recording_buffer_to_file(self, filepath: str, channels_to_save: List[int], max_len: int):
        """Записать буфер записи в файл: один столбец — один канал по порядку, 3 знака, в первой строке sampling_rate."""
        try:
            srate = self.sampling_rate if self._has_stream else 0
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# sampling_rate={srate}\n")
                header_names = [DEFAULT_CHANNEL_NAMES_21[i] if i < len(DEFAULT_CHANNEL_NAMES_21) else f"Ch{i+1}" for i in channels_to_save]
                f.write(",".join(header_names) + "\n")
                for row_idx in range(max_len):
                    row_vals = []
                    for ch in channels_to_save:
                        buf = self.recording_buffer[ch]
                        val = buf[row_idx] if row_idx < len(buf) else 0.0
                        row_vals.append(f"{val:.3f}".replace(".", ","))
                    f.write(",".join(row_vals) + "\n")
        except Exception as e:
            log.error(f"Ошибка записи в файл: {e}")

    def _on_search_or_stop_stream(self):
        """Запуск поиска потока (идет до нажатия «Остановить поиск») или остановка поиска."""
        if self._searching:
            self._stop_stream_search()
            return
        self._start_stream_search()

    def _start_stream_search(self):
        """Запустить фоновый поиск LSL-потока (пока не нажата «Остановить поиск»)."""
        if self._search_thread is not None and self._search_thread.isRunning():
            return
        self._searching = True
        self.btn_search_stream.setText("⏹ Остановить поиск")
        self.btn_search_stream.setStyleSheet("QPushButton { background-color: #dc3545; font-weight: bold; }")
        self._search_thread = StreamSearchThread(self)
        self._search_thread.streams_found.connect(self._on_streams_found)
        self._search_thread.finished.connect(self._on_search_thread_finished)
        self._search_thread.start()
        log.info("Поиск потока запущен (остановить — кнопкой «Остановить поиск»).")

    def _stop_stream_search(self):
        """Остановить поиск потока."""
        self._searching = False
        if self._search_thread is not None and self._search_thread.isRunning():
            self._search_thread.request_stop()
            self._search_thread.wait(5000)
        self._search_thread = None
        self.btn_search_stream.setText("🔍 Поиск потока")
        self.btn_search_stream.setStyleSheet("QPushButton { background-color: #007bff; font-weight: bold; }")
        log.info("Поиск потока остановлен.")

    def _on_search_thread_finished(self):
        """Поиск завершён (найден поток или остановлен)."""
        if not self._searching:
            return
        self._searching = False
        self.btn_search_stream.setText("🔍 Поиск потока")
        self.btn_search_stream.setStyleSheet("QPushButton { background-color: #007bff; font-weight: bold; }")
        self._search_thread = None

    def _on_streams_found(self, streams: list):
        """Подключиться к найденному потоку и остановить поиск."""
        self._search_thread.request_stop()
        self._on_search_thread_finished()
        if not streams:
            return
        eeg_info = streams[0] if len(streams) == 1 else select_stream_gui(streams, self)
        if not eeg_info:
            return
        inlet = _create_inlet(eeg_info)
        full_info = inlet.info()
        channel_names = get_channel_names(full_info, full_info.channel_count())
        self._connect_stream(inlet, full_info, channel_names)
        QMessageBox.information(self, "Поток найден", f"Подключено к потоку: {full_info.name() or 'EEG'}.")

    def _connect_stream(self, inlet: StreamInlet, full_info: StreamInfo, channel_names: List[str]):
        """Подключить найденный поток и переключить UI на отображение каналов."""
        self.inlet = inlet
        self.stream_name = full_info.name() or "EEG"
        self.channel_names = channel_names
        self.n_channels = full_info.channel_count()
        self.sampling_rate = full_info.nominal_srate()
        self._has_stream = True
        self.setWindowTitle(f"LSL Validation — {self.stream_name}")

        # 1. КРИТИЧЕСКИЙ ФИКС: ЗАПУСКАЕМ ЧТЕНИЕ ДО ОТРИСОВКИ ТЯЖЕЛОГО ИНТЕРФЕЙСА
        self._pull_stop[0] = False
        self._pull_queue = queue.Queue()
        self._pull_thread = LSLPullThread(self.inlet, self._pull_queue, self._pull_stop, self)
        self._pull_thread.start()
        log.info("Поток подключен. Фоновое чтение LSL запущено ДО отрисовки графиков.")

        # 2. ПОТОМ СТРОИМ UI
        # Очистить старые чекбоксы в сайдбаре и добавить новые
        while self.cb_layout.count():
            item = self.cb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.checkboxes.clear()
        for i, name in enumerate(self.channel_names):
            cb = QCheckBox(f"[{i + 1}] {name}")
            cb.setChecked(True)
            cb.stateChanged.connect(self._rebuild_grid)
            self.checkboxes.append(cb)
            self.cb_layout.addWidget(cb)
        self.cb_layout.addStretch()

        self._build_stream_content()
        self._apply_x_window()
        self._setup_timers()

    def _pull_and_plot(self):
        if not self.inlet:
            return
        try:
            # Обрабатываем все чанки из очереди (накопленные фоновым потоком) — без потери начальных сэмплов
            while True:
                try:
                    chunk, timestamps = self._pull_queue.get_nowait()
                except queue.Empty:
                    break
                if not chunk:
                    continue
                arr = np.array(chunk)
                if arr.shape[1] != self.n_channels and arr.shape[0] == self.n_channels:
                    arr = arr.T

                self.sample_count += len(arr)
                if hasattr(self, "samples_label") and self.samples_label:
                    self.samples_label.setText(f"<b>Сэмплов:</b> {self.sample_count}")

                for ch in range(self.n_channels):
                    self.channel_widgets[ch].push_chunk(arr[:, ch])

                # Буфер записи: только выбранные каналы, в порядке 0..20 (можно запись до подключения потока)
                if self.recording:
                    n_record = min(FIXED_RECORDING_CHANNELS, arr.shape[1])
                    for ch in range(n_record):
                        if self.record_channel_checked[ch]:
                            col = arr[:, ch].astype(np.float64)
                            col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
                            self.recording_buffer[ch].extend(col.tolist())

            auto_scale_enabled = self.cb_autoscale.isChecked()
            for cw in self.channel_widgets:
                cw.update_plot(auto_scale_enabled)
            # Показывать текущий масштаб в полях (при автомасштабе)
            if auto_scale_enabled and self.channel_widgets:
                cw = next((c for c in self.channel_widgets if c.is_active), self.channel_widgets[0])
                if hasattr(self, "spin_y_min") and hasattr(self, "spin_y_max"):
                    self.spin_y_min.blockSignals(True)
                    self.spin_y_max.blockSignals(True)
                    self.spin_y_min.setValue(cw.current_y_min)
                    self.spin_y_max.setValue(cw.current_y_max)
                    self.spin_y_min.blockSignals(False)
                    self.spin_y_max.blockSignals(False)

            now = time.time()
            if now - self.last_cov_time >= COV_UPDATE_INTERVAL:
                self.last_cov_time = now
                self._calculate_covariance()
        except Exception:
            pass

    def _update_text_stats(self):
        for cw in self.channel_widgets:
            cw.update_stats()

    def _calculate_covariance(self):
        try:
            active_cws = [cw for cw in self.channel_widgets if cw.is_active]

            if len(active_cws) < 2:
                return

            data_matrix = np.column_stack([cw.y_data for cw in active_cws])
            data_matrix = np.nan_to_num(data_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            data_matrix = data_matrix - np.mean(data_matrix, axis=0)
            cov_matrix = np.cov(data_matrix.T)

            names = [cw.channel_name for cw in active_cws]
            n_active = len(names)

            log.info(f"--- МАТРИЦА КОВАРИАЦИЙ ({n_active} активных, сэмплов: {self.sample_count}) ---")
            col_w = 10
            header = " " * col_w + "".join([f"{name[:col_w]:>{col_w}}" for name in names])
            log.info(header)

            for i in range(n_active):
                row_str = f"{names[i][:col_w]:>{col_w}}"
                for j in range(n_active):
                    val = cov_matrix[i, j]
                    if np.isnan(val) or np.isinf(val): val = 0.0
                    row_str += f"{val:{col_w}.2f}"
                log.info(row_str)

            np.fill_diagonal(cov_matrix, 0)
            max_idx = np.unravel_index(np.argmax(np.abs(cov_matrix)), cov_matrix.shape)
            max_val = cov_matrix[max_idx]

            ch1, ch2 = names[max_idx[0]], names[max_idx[1]]
            log.info(f"Макс. наводка: {ch1} ↔ {ch2} = {max_val:.2f}\n")
        except Exception:
            pass

    def closeEvent(self, event):
        """Остановить фоновый поток чтения LSL при закрытии окна."""
        self._pull_stop[0] = True
        if self._pull_thread is not None and self._pull_thread.isRunning():
            self._pull_thread.wait(2000)
        super().closeEvent(event)


def select_stream_gui(streams: List[StreamInfo], parent=None):
    items = []
    for info in streams:
        name = info.name() or "Unknown"
        stype = info.type() or "EEG"
        items.append(f"{name} (type={stype}, ch={info.channel_count()})")

    item, ok = QInputDialog.getItem(parent, "Выбор LSL‑потока", "Выберите поток:", items, 0, False)
    return streams[items.index(item)] if ok else None


def main():
    log.info("=== СТАРТ АППАРАТНОЙ ВАЛИДАЦИИ ===")
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    inlet = None
    full_info = None
    channel_names = None
    try:
        streams = find_eeg_streams()
        if streams:
            eeg_info = streams[0] if len(streams) == 1 else select_stream_gui(streams, None)
            if eeg_info:
                inlet = _create_inlet(eeg_info)
                full_info = inlet.info()
                channel_names = get_channel_names(full_info, full_info.channel_count())
    except Exception as e:
        log.warning("Поток при старте не найден или ошибка: %s. Окно откроется без потока.", e)

    window = HardwareValidationWindow(inlet=inlet, full_info=full_info, channel_names=channel_names)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()