#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Операторский GUI гибридного протокола v2 (калибровка → 45 экспериментов)."""

from __future__ import annotations

import sys
import time
from pathlib import Path
import subprocess

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_protocol.protocol_log import info as plog_info  # noqa: E402
from experiment_protocol.protocol_runner import ProtocolConfig, ProtocolRunner  # noqa: E402
from experiment_protocol.unified_logger import UnifiedExperimentLogger  # noqa: E402
from experiment_protocol.ssvep_cue_overlay import SsvepCueOverlay  # noqa: E402
from p300_analysis.analysis_profiles import (  # noqa: E402
    ANALYSIS_PROFILE_GENERAL,
    ANALYSIS_PROFILE_RECENT,
    get_analysis_profile,
)
from p300_analysis.lsl_streams import (  # noqa: E402
    BCI_STIM_MARKER_STREAM_NAME,
    discover_eeg_streams,
    select_eeg_stream,
    stream_channel_labels,
    stream_display_label,
    stream_inlet_with_buffer,
    wait_for_stimulus_marker_stream,
)
from ssvep_analyzer import lamp_frequency_choices, lamp_frequency_closest_index  # noqa: E402

# Как SSVEPAnalyzerWindow.DEFAULT_LAMP_FREQS — 4 лампы
_DEFAULT_LAMP_FREQS = (
    10.0,
    1000.0 / 99.0,
    1000.0 / 87.0,
    1000.0 / 76.0,
)
_MAX_LAMPS = 6
_CHANNEL_COLUMNS = 4


def _list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports

        return sorted({p.device for p in list_ports.comports()})
    except Exception:
        return []


def _make_lamp_freq_combo(initial_hz: float) -> QComboBox:
    combo = QComboBox()
    combo.addItem("0 (выкл)", 0.0)
    for text, val in lamp_frequency_choices():
        combo.addItem(text, float(val))
    if float(initial_hz) <= 0.0:
        combo.setCurrentIndex(0)
    else:
        combo.setCurrentIndex(int(lamp_frequency_closest_index(float(initial_hz))) + 1)
    combo.setToolTip("Дискретные частоты 1000/i Гц (как в анализаторе SSVEP и migalka.py)")
    return combo


def _combo_freq_hz(combo: QComboBox) -> float:
    v = combo.currentData()
    return float(v) if v is not None else 0.0


def _eeg_stream_key(info) -> tuple[str, str]:
    try:
        return (str(info.name() or ""), str(info.session_id() or ""))
    except Exception:
        return ("", "")


def _clear_grid_layout(layout: QGridLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.deleteLater()


def _selected_channels_0idx(checkboxes: list[QCheckBox]) -> tuple[int, ...]:
    """Выбранные каналы (0-based). Пустой кортеж = все каналы отмечены."""
    if not checkboxes:
        return ()
    idx = [i for i, cb in enumerate(checkboxes) if cb.isChecked()]
    if not idx or len(idx) >= len(checkboxes):
        return ()
    return tuple(idx)


class ProtocolRunnerWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Гибридный протокол (P300×2 + SSVEP×2)")
        self.setMinimumWidth(720)
        self.resize(760, 640)

        self._runner: ProtocolRunner | None = None
        self._stimulus_proc: subprocess.Popen[str] | None = None
        self._session_dir: Path | None = None
        self._last_stim_restart_attempt: float = 0.0
        self._last_status_printed: str = ""
        self._eeg_test_inlet = None
        self._ssvep_cue_overlay: SsvepCueOverlay | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        scroll_content = QWidget()
        scroll_content.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)

        form = QFormLayout()

        self.ed_subject = QLineEdit("испытуемый_001")
        self.ed_output = QLineEdit(str((_ROOT / "experiment_runs").resolve()))

        eeg_row = QHBoxLayout()
        self.cb_eeg = QComboBox()
        self.cb_eeg.setMinimumWidth(360)
        self.btn_refresh_eeg = QPushButton("Обновить")
        self.btn_refresh_eeg.setToolTip("Найти потоки LSL типа EEG/Signal в сети")
        self.btn_test_eeg = QPushButton("Проверить")
        self.btn_test_eeg.setToolTip("Открыть выбранный поток и прочитать тестовый фрагмент данных")
        eeg_row.addWidget(self.cb_eeg, stretch=1)
        eeg_row.addWidget(self.btn_refresh_eeg)
        eeg_row.addWidget(self.btn_test_eeg)
        self._eeg_row_widget = QWidget()
        self._eeg_row_widget.setLayout(eeg_row)
        self.lbl_eeg_status = QLabel("Поток ЭЭГ не выбран. Нажмите «Обновить».")
        self.lbl_eeg_status.setWordWrap(True)
        self.lbl_eeg_status.setStyleSheet("color: #555; font-size: 11px;")

        com_row = QHBoxLayout()
        self.cb_com = QComboBox()
        self.cb_com.setEditable(True)
        self.cb_com.setMinimumWidth(220)
        self.btn_refresh_com = QPushButton("Обновить")
        self.btn_refresh_com.setToolTip("Повторно найти COM-порты (как в окне migalka.py)")
        com_row.addWidget(self.cb_com, stretch=1)
        com_row.addWidget(self.btn_refresh_com)
        self._com_row_widget = QWidget()
        self._com_row_widget.setLayout(com_row)

        self._freq_combos: list[QComboBox] = []
        self._lamp_row_widgets: list[QWidget] = []
        migalka_box = QGroupBox("Мигалка (ССВП) — COM-порт и частоты ламп")
        migalka_outer = QVBoxLayout(migalka_box)
        migalka_form = QFormLayout()
        migalka_form.addRow("COM-порт:", self._com_row_widget)
        migalka_outer.addLayout(migalka_form)
        self._lamp_rows_host = QWidget()
        self._lamp_rows_layout = QVBoxLayout(self._lamp_rows_host)
        self._lamp_rows_layout.setContentsMargins(0, 0, 0, 0)
        migalka_outer.addWidget(self._lamp_rows_host)
        lamp_btn_row = QHBoxLayout()
        self.btn_add_lamp = QPushButton("+ лампа")
        self.btn_add_lamp.setToolTip(f"Добавить лампу (не более {_MAX_LAMPS}, как в анализаторе SSVEP)")
        self.btn_remove_lamp = QPushButton("− лампа")
        self.btn_remove_lamp.setToolTip("Убрать последнюю лампу (не меньше 4 по умолчанию)")
        lamp_btn_row.addWidget(self.btn_add_lamp)
        lamp_btn_row.addWidget(self.btn_remove_lamp)
        lamp_btn_row.addStretch(1)
        migalka_outer.addLayout(lamp_btn_row)
        for f0 in _DEFAULT_LAMP_FREQS:
            self._add_lamp_freq_row(float(f0))

        self._p300_ch_checkboxes: list[QCheckBox] = []
        self._ssvep_ch_checkboxes: list[QCheckBox] = []
        self._eeg_channel_count = 0

        ssvep_box = QGroupBox("ССВП — каналы ЭЭГ для MSI")
        ssvep_outer = QVBoxLayout(ssvep_box)
        ssvep_top = QHBoxLayout()
        self.cb_ssvep_profile = QComboBox()
        self.cb_ssvep_profile.addItem("Все каналы", "all")
        self.cb_ssvep_profile.addItem("Общий профиль (канал 4)", "general")
        self.cb_ssvep_profile.addItem("Последние 7 сессий (каналы 5, 6)", "recent")
        self.cb_ssvep_profile.addItem("Вручную", "custom")
        ssvep_top.addWidget(QLabel("Профиль:"))
        ssvep_top.addWidget(self.cb_ssvep_profile, stretch=1)
        self.btn_ssvep_ch_all = QPushButton("Все")
        self.btn_ssvep_ch_none = QPushButton("Снять")
        ssvep_top.addWidget(self.btn_ssvep_ch_all)
        ssvep_top.addWidget(self.btn_ssvep_ch_none)
        ssvep_outer.addLayout(ssvep_top)
        self._ssvep_ch_host = QWidget()
        self._ssvep_ch_layout = QGridLayout(self._ssvep_ch_host)
        self._lbl_ssvep_ch_hint = QLabel("Сначала выберите поток ЭЭГ (LSL) выше.")
        self._lbl_ssvep_ch_hint.setWordWrap(True)
        self._lbl_ssvep_ch_hint.setStyleSheet("color: #666;")
        ssvep_outer.addWidget(self._lbl_ssvep_ch_hint)
        ssvep_outer.addWidget(self._ssvep_ch_host)

        p300_box = QGroupBox("P300 — каналы ЭЭГ (область интереса)")
        p300_outer = QVBoxLayout(p300_box)
        p300_top = QHBoxLayout()
        self.cb_p300_profile = QComboBox()
        self.cb_p300_profile.addItem("Общий профиль (канал 4)", "general")
        self.cb_p300_profile.addItem("Последние 7 сессий (каналы 5, 6)", "recent")
        self.cb_p300_profile.addItem("Все каналы", "all")
        self.cb_p300_profile.addItem("Вручную", "custom")
        p300_top.addWidget(QLabel("Профиль:"))
        p300_top.addWidget(self.cb_p300_profile, stretch=1)
        self.btn_p300_ch_all = QPushButton("Все")
        self.btn_p300_ch_none = QPushButton("Снять")
        p300_top.addWidget(self.btn_p300_ch_all)
        p300_top.addWidget(self.btn_p300_ch_none)
        p300_outer.addLayout(p300_top)
        self._p300_ch_host = QWidget()
        self._p300_ch_layout = QGridLayout(self._p300_ch_host)
        self._lbl_p300_ch_hint = QLabel("Сначала выберите поток ЭЭГ (LSL) выше.")
        self._lbl_p300_ch_hint.setWordWrap(True)
        self._lbl_p300_ch_hint.setStyleSheet("color: #666;")
        p300_outer.addWidget(self._lbl_p300_ch_hint)
        p300_outer.addWidget(self._p300_ch_host)
        self._p300_ch_host.hide()
        self._ssvep_ch_host.hide()

        proto_box = QGroupBox("Протокол (калибровка → 45 экспериментов в случайном порядке)")
        proto_form = QFormLayout(proto_box)

        self.spin_calib_trials = QSpinBox()
        self.spin_calib_trials.setRange(1, 50)
        self.spin_calib_trials.setValue(3)
        self.spin_calib_trials.setToolTip(
            "Подряд в начале: прогоны P300 на одной плитке для эталона."
        )

        self.spin_calib_target = QSpinBox()
        self.spin_calib_target.setRange(0, 8)
        self.spin_calib_target.setValue(4)

        self.spin_template_epochs = QSpinBox()
        self.spin_template_epochs.setRange(4, 50)
        self.spin_template_epochs.setValue(12)

        self.spin_p300_main = QSpinBox()
        self.spin_p300_main.setRange(1, 200)
        self.spin_p300_main.setValue(15)
        self.spin_p300_main.setToolTip(
            "В основном блоке: каждый прогон P300 — AUC и сравнение с шаблоном (один trial)."
        )

        self.spin_ssvep = QSpinBox()
        self.spin_ssvep.setRange(1, 200)
        self.spin_ssvep.setValue(15)
        self.spin_ssvep.setToolTip("По 15 блоков ССВП: непрерывный и пакетный (вперемешку с P300).")

        self.spin_pause = QDoubleSpinBox()
        self.spin_pause.setRange(0.0, 60.0)
        self.spin_pause.setDecimals(1)
        self.spin_pause.setSingleStep(0.5)
        self.spin_pause.setValue(2.0)

        self.spin_shuffle_seed = QSpinBox()
        self.spin_shuffle_seed.setRange(-1, 2_000_000_000)
        self.spin_shuffle_seed.setValue(-1)
        self.spin_shuffle_seed.setToolTip("-1 = новый случайный порядок при каждом запуске.")

        self.spin_ssvep_block_sec = QDoubleSpinBox()
        self.spin_ssvep_block_sec.setRange(1.0, 120.0)
        self.spin_ssvep_block_sec.setDecimals(1)
        self.spin_ssvep_block_sec.setValue(6.0)

        proto_form.addRow("Калибровка: число прогонов P300:", self.spin_calib_trials)
        proto_form.addRow("Калибровка: целевая плитка (0…8):", self.spin_calib_target)
        proto_form.addRow("Мин. эпох на шаблон:", self.spin_template_epochs)
        proto_form.addRow("Основной блок: прогонов P300:", self.spin_p300_main)
        proto_form.addRow("ССВП: блоков на режим (cont+burst):", self.spin_ssvep)
        proto_form.addRow("Пауза между экспериментами (с):", self.spin_pause)
        proto_form.addRow("Seed перемешивания (-1=случайно):", self.spin_shuffle_seed)
        proto_form.addRow("Длительность блока ССВП (с):", self.spin_ssvep_block_sec)

        stim_box = QGroupBox("Стимулятор P300 (PsychoPy, авто-режим)")
        stim_form = QFormLayout(stim_box)

        self.spin_inter_trial = QDoubleSpinBox()
        self.spin_inter_trial.setRange(0.0, 30.0)
        self.spin_inter_trial.setDecimals(1)
        self.spin_inter_trial.setSingleStep(0.5)
        self.spin_inter_trial.setValue(1.0)

        self.spin_sequences = QSpinBox()
        self.spin_sequences.setRange(1, 50)
        self.spin_sequences.setValue(12)

        self.chk_run_stimulus = QCheckBox(
            "Запускать экранную стимуляцию (PsychoPy, без кнопки START)"
        )
        self.chk_run_stimulus.setChecked(True)
        self.chk_run_stimulus.setToolTip(
            "PsychoPy: P300 с серым оверлеем между trial (как 3a0a997), по stim_control.json.\n"
            "ССВП — отдельное окно SsvepCueOverlay, PsychoPy закрывается."
        )

        stim_form.addRow("Пауза между trial P300 (с):", self.spin_inter_trial)
        stim_form.addRow("Последовательностей в прогоне:", self.spin_sequences)
        stim_form.addRow("", self.chk_run_stimulus)

        form.addRow("ID испытуемого:", self.ed_subject)
        form.addRow("Папка результатов:", self.ed_output)
        form.addRow("Поток ЭЭГ (LSL):", self._eeg_row_widget)
        form.addRow("", self.lbl_eeg_status)

        scroll_layout.addLayout(form)
        scroll_layout.addWidget(proto_box)
        scroll_layout.addWidget(stim_box)
        scroll_layout.addWidget(migalka_box)
        scroll_layout.addWidget(ssvep_box)
        scroll_layout.addWidget(p300_box)

        self.lbl_settings_hint = QLabel(
            "Поток ЭЭГ — запись и онлайн-анализ; COM и частоты — мигалка; каналы — ROI для P300/ССВП. "
            "Перед стартом проверьте ЭЭГ кнопкой «Проверить». Консоль: [protocol]."
        )
        self.lbl_settings_hint.setWordWrap(True)
        self.lbl_settings_hint.setStyleSheet("color: #555; font-size: 11px;")
        scroll_layout.addWidget(self.lbl_settings_hint)
        scroll_layout.addStretch(1)

        scroll.setWidget(scroll_content)
        root.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("Запустить протокол")
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        root.addLayout(btn_row)

        self.lbl_status = QLabel("Ожидание. Подключите ЭЭГ/LSL и стимулятор, затем нажмите «Запустить протокол».")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("padding: 4px 0;")
        root.addWidget(self.lbl_status)

        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._on_tick)

        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_refresh_com.clicked.connect(self._refresh_com_ports)
        self.btn_refresh_eeg.clicked.connect(self._refresh_eeg_streams)
        self.btn_test_eeg.clicked.connect(self._on_test_eeg_connect)
        self.cb_eeg.currentIndexChanged.connect(self._rebuild_channel_checkboxes_from_selection)
        self.cb_p300_profile.currentIndexChanged.connect(self._on_p300_profile_changed)
        self.cb_ssvep_profile.currentIndexChanged.connect(self._on_ssvep_profile_changed)
        self.btn_p300_ch_all.clicked.connect(lambda: self._set_channel_checks(self._p300_ch_checkboxes, True))
        self.btn_p300_ch_none.clicked.connect(lambda: self._set_channel_checks(self._p300_ch_checkboxes, False))
        self.btn_ssvep_ch_all.clicked.connect(lambda: self._set_channel_checks(self._ssvep_ch_checkboxes, True))
        self.btn_ssvep_ch_none.clicked.connect(lambda: self._set_channel_checks(self._ssvep_ch_checkboxes, False))
        self.btn_add_lamp.clicked.connect(self._on_add_lamp)
        self.btn_remove_lamp.clicked.connect(self._on_remove_lamp)

        self._refresh_com_ports()
        self._refresh_eeg_streams()
        self._update_lamp_buttons()

    def _close_eeg_test_inlet(self) -> None:
        if self._eeg_test_inlet is None:
            return
        try:
            self._eeg_test_inlet.close_stream()
        except Exception:
            pass
        self._eeg_test_inlet = None

    def _refresh_eeg_streams(self) -> None:
        self._close_eeg_test_inlet()
        current_key: tuple[str, str] = ("", "")
        if self.cb_eeg.currentIndex() >= 0:
            data = self.cb_eeg.currentData()
            if isinstance(data, tuple) and len(data) == 2:
                current_key = (str(data[0]), str(data[1]))
        self.cb_eeg.blockSignals(True)
        self.cb_eeg.clear()
        try:
            streams = discover_eeg_streams(timeout=1.5)
        except Exception as e:
            self.cb_eeg.addItem(f"(ошибка поиска: {e})")
            self.cb_eeg.blockSignals(False)
            self.lbl_eeg_status.setText(f"Ошибка поиска потоков LSL: {e}")
            return
        if not streams:
            self.cb_eeg.addItem("(потоки ЭЭГ не найдены)")
            self.cb_eeg.blockSignals(False)
            self.lbl_eeg_status.setText("Потоки не найдены. Запустите усилитель/запись с LSL и нажмите «Обновить».")
            return
        for info in streams:
            self.cb_eeg.addItem(stream_display_label(info), userData=_eeg_stream_key(info))
        self.cb_eeg.blockSignals(False)
        if current_key[0]:
            for i in range(self.cb_eeg.count()):
                data = self.cb_eeg.itemData(i)
                if isinstance(data, tuple) and tuple(data) == current_key:
                    self.cb_eeg.setCurrentIndex(i)
                    break
        self._rebuild_channel_checkboxes_from_selection()

    def _set_channel_checks(self, checkboxes: list[QCheckBox], checked: bool) -> None:
        for cb in checkboxes:
            cb.setChecked(bool(checked))

    def _profile_channels_0idx(self, profile_key: str, n_channels: int) -> list[int]:
        if profile_key == "all":
            return list(range(n_channels))
        if profile_key == "general":
            prof = get_analysis_profile(ANALYSIS_PROFILE_GENERAL)
            return [c for c in prof.roi_channels_0idx if 0 <= c < n_channels]
        if profile_key == "recent":
            prof = get_analysis_profile(ANALYSIS_PROFILE_RECENT)
            return [c for c in prof.roi_channels_0idx if 0 <= c < n_channels]
        return []

    def _apply_channel_profile(self, checkboxes: list[QCheckBox], profile_key: str) -> None:
        if not checkboxes or profile_key == "custom":
            return
        n = len(checkboxes)
        if profile_key == "all":
            self._set_channel_checks(checkboxes, True)
            return
        want = set(self._profile_channels_0idx(profile_key, n))
        for i, cb in enumerate(checkboxes):
            cb.setChecked(i in want)

    def _build_channel_checkbox_grid(
        self,
        layout: QGridLayout,
        storage: list[QCheckBox],
        *,
        count: int,
        labels: list[str],
    ) -> None:
        storage.clear()
        _clear_grid_layout(layout)
        for i in range(count):
            text = labels[i] if i < len(labels) else f"Канал {i + 1}"
            if not text.startswith("Канал"):
                text = f"{i + 1}: {text}"
            cb = QCheckBox(text)
            storage.append(cb)
            row, col = divmod(i, _CHANNEL_COLUMNS)
            layout.addWidget(cb, row, col)

    def _rebuild_channel_checkboxes_from_selection(self) -> None:
        info = self._resolve_eeg_stream_info(timeout=0.8)
        if info is None:
            self._eeg_channel_count = 0
            self._p300_ch_checkboxes.clear()
            self._ssvep_ch_checkboxes.clear()
            _clear_grid_layout(self._p300_ch_layout)
            _clear_grid_layout(self._ssvep_ch_layout)
            self._p300_ch_host.hide()
            self._ssvep_ch_host.hide()
            self._lbl_p300_ch_hint.show()
            self._lbl_ssvep_ch_hint.show()
            self._on_eeg_selection_changed()
            return
        try:
            n = max(1, int(info.channel_count()))
        except Exception:
            n = 1
        self._eeg_channel_count = n
        labels = stream_channel_labels(info, n)
        self._build_channel_checkbox_grid(self._p300_ch_layout, self._p300_ch_checkboxes, count=n, labels=labels)
        self._build_channel_checkbox_grid(self._ssvep_ch_layout, self._ssvep_ch_checkboxes, count=n, labels=labels)
        self._lbl_p300_ch_hint.hide()
        self._lbl_ssvep_ch_hint.hide()
        self._p300_ch_host.show()
        self._ssvep_ch_host.show()
        self._apply_channel_profile(self._p300_ch_checkboxes, str(self.cb_p300_profile.currentData()))
        self._apply_channel_profile(self._ssvep_ch_checkboxes, str(self.cb_ssvep_profile.currentData()))
        self._on_eeg_selection_changed()

    def _selected_eeg_stream_key(self) -> tuple[str, str] | None:
        idx = self.cb_eeg.currentIndex()
        if idx < 0:
            return None
        data = self.cb_eeg.itemData(idx)
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        name, sid = str(data[0]), str(data[1])
        if not name or name.startswith("("):
            return None
        return (name, sid)

    def _resolve_eeg_stream_info(self, *, timeout: float = 1.5):
        key = self._selected_eeg_stream_key()
        if key is None:
            return None
        streams = discover_eeg_streams(timeout=timeout)
        return select_eeg_stream(streams, name=key[0], session_id=key[1])

    def _on_eeg_selection_changed(self) -> None:
        key = self._selected_eeg_stream_key()
        if key is None:
            self.lbl_eeg_status.setText("Выберите поток из списка.")
            return
        self.lbl_eeg_status.setText(
            f"Выбран: {key[0]!r}. Нажмите «Проверить» для тестового подключения."
        )

    def _on_test_eeg_connect(self) -> None:
        key = self._selected_eeg_stream_key()
        if key is None:
            QMessageBox.warning(self, "ЭЭГ", "Сначала выберите поток в списке (кнопка «Обновить»).")
            return
        info = self._resolve_eeg_stream_info()
        if info is None:
            QMessageBox.warning(
                self,
                "ЭЭГ",
                f"Поток «{key[0]}» не найден в сети LSL.\nЗапустите запись ЭЭГ и нажмите «Обновить».",
            )
            return
        self._close_eeg_test_inlet()
        try:
            inlet = stream_inlet_with_buffer(info, buffer_seconds=5)
            try:
                chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=32)
            except TypeError:
                chunk, ts = inlet.pull_chunk(timeout=1.0)
            n_samp = len(ts) if ts else 0
            inlet.close_stream()
            name = info.name() or "?"
            ch = int(info.channel_count())
            fs = float(info.nominal_srate() or 0.0)
            self.lbl_eeg_status.setText(
                f"Подключение OK: {name}, {ch} кан., Fs={fs:g} Гц, получено сэмплов: {n_samp}."
            )
            plog_info(f"тест ЭЭГ: {stream_display_label(info)}, samples={n_samp}")
            self._rebuild_channel_checkboxes_from_selection()
        except Exception as e:
            self.lbl_eeg_status.setText(f"Ошибка подключения: {e}")
            QMessageBox.warning(self, "ЭЭГ", f"Не удалось подключиться к потоку:\n{e}")

    def _add_lamp_freq_row(self, initial_hz: float) -> None:
        if len(self._freq_combos) >= _MAX_LAMPS:
            return
        n = len(self._freq_combos) + 1
        combo = _make_lamp_freq_combo(initial_hz)
        row_w = QWidget()
        row = QHBoxLayout(row_w)
        row.addWidget(QLabel(f"Лампа {n}:"))
        row.addWidget(combo, stretch=1)
        self._freq_combos.append(combo)
        self._lamp_row_widgets.append(row_w)
        self._lamp_rows_layout.addWidget(row_w)

    def _on_add_lamp(self) -> None:
        if len(self._freq_combos) >= _MAX_LAMPS:
            return
        extras = (12.0, 15.0, 20.0, 8.57, 7.5)
        k = len(self._freq_combos)
        hz = float(extras[(k - len(_DEFAULT_LAMP_FREQS)) % len(extras)]) if k >= len(_DEFAULT_LAMP_FREQS) else 0.0
        self._add_lamp_freq_row(hz)
        self._update_lamp_buttons()

    def _on_remove_lamp(self) -> None:
        if len(self._freq_combos) <= len(_DEFAULT_LAMP_FREQS):
            return
        row_w = self._lamp_row_widgets.pop()
        self._freq_combos.pop()
        self._lamp_rows_layout.removeWidget(row_w)
        row_w.deleteLater()
        self._update_lamp_buttons()

    def _update_lamp_buttons(self) -> None:
        n = len(self._freq_combos)
        self.btn_add_lamp.setEnabled(n < _MAX_LAMPS)
        self.btn_remove_lamp.setEnabled(n > len(_DEFAULT_LAMP_FREQS))
        self.btn_add_lamp.setText(f"+ лампа ({n}/{_MAX_LAMPS})")

    def _refresh_com_ports(self) -> None:
        ports = _list_serial_ports()
        current = self.cb_com.currentText().strip()
        self.cb_com.blockSignals(True)
        self.cb_com.clear()
        if not ports:
            self.cb_com.addItem("(порты не найдены — введите вручную)")
        else:
            for p in ports:
                self.cb_com.addItem(p)
        self.cb_com.blockSignals(False)
        if current and self.cb_com.findText(current) >= 0:
            self.cb_com.setCurrentText(current)
        elif ports:
            self.cb_com.setCurrentIndex(0)

    def _on_p300_profile_changed(self, _index: int) -> None:
        self._apply_channel_profile(self._p300_ch_checkboxes, str(self.cb_p300_profile.currentData()))

    def _on_ssvep_profile_changed(self, _index: int) -> None:
        self._apply_channel_profile(self._ssvep_ch_checkboxes, str(self.cb_ssvep_profile.currentData()))

    def _ssvep_freqs_hz(self) -> tuple[float, ...]:
        return tuple(_combo_freq_hz(c) for c in self._freq_combos)

    def _cleanup_before_start(self) -> None:
        """Перед новым прогоном: закрыть старый runner/COM/LSL и убить зомби run_app."""
        self._close_eeg_test_inlet()
        self._dismiss_ssvep_overlay()
        if self._stimulus_proc is not None:
            try:
                if self._stimulus_proc.poll() is None:
                    self._stimulus_proc.terminate()
                    try:
                        self._stimulus_proc.wait(timeout=3)
                    except Exception:
                        pass
            except Exception:
                pass
            self._stimulus_proc = None
        if self._runner is not None:
            try:
                if str(self._runner.state) not in ("idle", "stopped"):
                    self._runner.stop(reason="restart")
            except Exception:
                pass
            self._runner = None
        self._session_dir = None
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _spawn_stimulus(self, session_dir: Path) -> bool:
        """Запустить run_app.py с stim_control (без ожидания LSL)."""
        run_script = _ROOT / "run_app.py"
        if not run_script.exists():
            return False
        stim_args = [
            sys.executable,
            str(run_script),
            "--auto-random-protocol",
            "--no-analyzer",
            "--inter-trial-s",
            str(float(self.spin_inter_trial.value())),
            "--sequences",
            str(int(self.spin_sequences.value())),
            "--auto-plan-trials",
            "0",
            "--auto-plan-target-tile-id",
            str(int(self.spin_calib_target.value())),
            "--stim-control-dir",
            str(session_dir),
        ]
        self._stimulus_proc = subprocess.Popen(stim_args, cwd=str(_ROOT))
        return True

    def _stimulus_needed_now(self) -> bool:
        """Нужен живой PsychoPy (калибровка или P300 в main, не во время ССВП)."""
        r = self._runner
        if r is None or not self.chk_run_stimulus.isChecked():
            return False
        if r.state in ("ssvep_continuous", "ssvep_burst", "finalize", "stopped", "idle", "preflight"):
            return False
        if bool(r.ssvep_cue_visible) or bool(r.ssvep_blackout_visible):
            return False
        return r.state in ("p300_calib", "main")

    def _ensure_stimulus_for_p300(self) -> None:
        """После ССВП стимулятор убивается — перед P300 в очереди перезапускаем."""
        if not self._stimulus_needed_now():
            return
        if self._session_dir is None:
            return
        if self._stimulus_proc is not None and self._stimulus_proc.poll() is None:
            return
        now = time.time()
        if now - float(self._last_stim_restart_attempt) < 8.0:
            return
        self._last_stim_restart_attempt = now
        if self._spawn_stimulus(self._session_dir):
            plog_info("перезапуск PsychoPy для P300 (после этапа ССВП)")

    def _wait_bci_stim_markers(self, max_wait_sec: float = 35.0) -> bool:
        """Ждём BCI_StimMarkers после запуска run_app (не MigalkaStimMarkers от прошлого ССВП)."""
        deadline = time.time() + float(max_wait_sec)
        while time.time() < deadline:
            info_mk, _ = wait_for_stimulus_marker_stream(max_wait_sec=0.8, poll_interval_sec=0.2)
            if info_mk is not None:
                try:
                    plog_info(f"стимулятор LSL OK: {info_mk.name()!r}")
                except Exception:
                    plog_info("стимулятор LSL OK")
                return True
            remain = max(0.0, deadline - time.time())
            self.lbl_status.setText(
                f"Ожидание {BCI_STIM_MARKER_STREAM_NAME}… ({remain:.0f} с)"
            )
            QApplication.processEvents()
            time.sleep(0.25)
        return False

    def _on_start(self) -> None:
        self._cleanup_before_start()
        subject = self.ed_subject.text().strip()
        out_root = self.ed_output.text().strip()
        com = self.cb_com.currentText().strip()
        if not subject:
            QMessageBox.warning(self, "Испытуемый", "Введите ID испытуемого.")
            return
        if not out_root:
            QMessageBox.warning(self, "Результаты", "Укажите папку для сохранения результатов.")
            return
        if not com or com.startswith("("):
            QMessageBox.warning(
                self,
                "COM-порт",
                "Выберите COM-порт мигалки из списка или введите вручную (COM3, /dev/tty.usbmodem…).\n"
                "Нажмите «Обновить», если устройство подключено позже.",
            )
            return

        eeg_key = self._selected_eeg_stream_key()
        if eeg_key is None:
            QMessageBox.warning(
                self,
                "ЭЭГ",
                "Выберите поток ЭЭГ (LSL): «Обновить» → выбор в списке → «Проверить».",
            )
            return
        eeg_name, eeg_sid = eeg_key
        if self._resolve_eeg_stream_info() is None:
            QMessageBox.warning(
                self,
                "ЭЭГ",
                f"Поток «{eeg_name}» сейчас не найден в LSL. Нажмите «Обновить» и проверьте запись ЭЭГ.",
            )
            return

        if not self._p300_ch_checkboxes:
            QMessageBox.warning(self, "P300", "Выберите поток ЭЭГ и дождитесь списка каналов.")
            return
        if not self._ssvep_ch_checkboxes:
            QMessageBox.warning(self, "ССВП", "Выберите поток ЭЭГ и дождитесь списка каналов.")
            return
        roi = _selected_channels_0idx(self._p300_ch_checkboxes)
        ssvep_roi = _selected_channels_0idx(self._ssvep_ch_checkboxes)
        if not any(cb.isChecked() for cb in self._p300_ch_checkboxes):
            QMessageBox.warning(self, "P300", "Отметьте хотя бы один канал ЭЭГ (или «Все»).")
            return
        if not any(cb.isChecked() for cb in self._ssvep_ch_checkboxes):
            QMessageBox.warning(self, "ССВП", "Отметьте хотя бы один канал ЭЭГ (или «Все»).")
            return
        freqs = self._ssvep_freqs_hz()
        if not any(float(f) > 0 for f in freqs):
            QMessageBox.warning(self, "Мигалка", "Задайте хотя бы одну лампу с частотой > 0.")
            return

        session_dir = UnifiedExperimentLogger.allocate_session_dir(
            output_root=Path(out_root),
            subject_id=subject,
        )
        self._session_dir = session_dir

        if self.chk_run_stimulus.isChecked():
            if self._spawn_stimulus(session_dir):
                plog_info("запуск стимулятора (P300 по stim_control)")
                plog_info(f"ожидание {BCI_STIM_MARKER_STREAM_NAME} (до 35 с)…")
                if not self._wait_bci_stim_markers(max_wait_sec=35.0):
                    QMessageBox.warning(
                        self,
                        "Стимулятор",
                        f"Поток «{BCI_STIM_MARKER_STREAM_NAME}» не появился за 35 с.\n"
                        "Дождитесь окна PsychoPy или перезапустите «Запустить протокол».",
                    )
                    try:
                        if self._stimulus_proc.poll() is None:
                            self._stimulus_proc.terminate()
                    except Exception:
                        pass
                    self._stimulus_proc = None
                    return

        cfg = ProtocolConfig(
            output_root=Path(out_root),
            subject_id=subject,
            session_dir=session_dir,
            com_port=com,
            eeg_stream_name=eeg_name,
            eeg_stream_session_id=eeg_sid,
            p300_calib_trials=int(self.spin_calib_trials.value()),
            calib_target_tile_id=int(self.spin_calib_target.value()),
            template_warmup_target_epochs=int(self.spin_template_epochs.value()),
            p300_main_trials=int(self.spin_p300_main.value()),
            ssvep_blocks_per_mode=int(self.spin_ssvep.value()),
            pause_between_experiments_s=float(self.spin_pause.value()),
            shuffle_seed=int(self.spin_shuffle_seed.value()),
            ssvep_block_sec=float(self.spin_ssvep_block_sec.value()),
            ssvep_freqs_hz=freqs,
            roi_channels_0idx=roi,
            ssvep_roi_channels_0idx=ssvep_roi,
        )
        self._runner = ProtocolRunner(cfg)
        self._runner.set_ssvep_display_clear_callback(self._restore_operator_window)
        ch_log = ",".join(str(c + 1) for c in roi) if roi else "ALL"
        ssvep_ch_log = ",".join(str(c + 1) for c in ssvep_roi) if ssvep_roi else "ALL"
        plog_info(
            f"GUI Start: subject={subject!r} EEG={eeg_name!r} "
            f"calib={self.spin_calib_trials.value()} main_P300={self.spin_p300_main.value()} "
            f"SSVEP={self.spin_ssvep.value()}×2 seed={self.spin_shuffle_seed.value()} "
            f"COM={com!r} P300_ch={ch_log} SSVEP_ch={ssvep_ch_log} "
            f"migalka_Hz={list(freqs)} inter_trial={self.spin_inter_trial.value()} "
            f"seq={self.spin_sequences.value()}"
        )
        # Preflight ждёт BCI_StimMarkers до ~20 с — стимулятор должен успеть стартовать
        self._runner.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._timer.start()
        self.lbl_status.setText("Запущено. Проверка перед стартом…")

    def _dismiss_ssvep_overlay(self) -> None:
        if self._ssvep_cue_overlay is not None:
            ov = self._ssvep_cue_overlay
            self._ssvep_cue_overlay = None
            ov.setVisible(False)
            ov.hide()
            ov.close()
            ov.deleteLater()
        QApplication.processEvents()

    def _restore_operator_window(self) -> None:
        """После ССВП — снова окно настроек поверх чёрного оверлея."""
        if self._runner is not None:
            self._runner.ssvep_blackout_visible = False
            self._runner.ssvep_cue_visible = False
        if self._ssvep_cue_overlay is not None:
            ov = self._ssvep_cue_overlay
            self._ssvep_cue_overlay = None
            ov.setVisible(False)
            ov.hide()
            ov.close()
            ov.deleteLater()
        self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()
        QApplication.processEvents()

    def _sync_ssvep_cue_overlay(self) -> None:
        if self._runner is None:
            self._dismiss_ssvep_overlay()
            return
        if self._runner.state in ("finalize", "stopped"):
            self._restore_operator_window()
            return
        if bool(self._runner.ssvep_blackout_visible):
            if self._ssvep_cue_overlay is None:
                self._ssvep_cue_overlay = SsvepCueOverlay()
            self._ssvep_cue_overlay.show_blackout()
            return
        if not bool(self._runner.ssvep_cue_visible):
            if self._ssvep_cue_overlay is not None and not bool(self._runner.ssvep_blackout_visible):
                self._dismiss_ssvep_overlay()
            return
        if self._ssvep_cue_overlay is None:
            self._ssvep_cue_overlay = SsvepCueOverlay()
        self._ssvep_cue_overlay.show_cue(
            experiment_index=int(self._runner.ssvep_cue_exp_index),
            experiment_total=int(self._runner.ssvep_cue_exp_total),
            lamp_1based=int(self._runner.ssvep_cue_lamp_1based),
            freq_hz=float(self._runner.ssvep_cue_freq_hz),
            mode_label=str(self._runner.ssvep_cue_mode_label),
        )

    def _on_stop(self) -> None:
        self._close_eeg_test_inlet()
        self._dismiss_ssvep_overlay()
        if self._stimulus_proc is not None:
            try:
                if self._stimulus_proc.poll() is None:
                    self._stimulus_proc.terminate()
            except Exception:
                pass
            self._stimulus_proc = None
        if self._runner is None:
            return
        self._runner.stop(reason="user_stop")
        self._runner = None
        self._session_dir = None
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Остановлено.")

    def _on_tick(self) -> None:
        if self._runner is None:
            return
        prev = getattr(self, "_last_status_printed", "")
        self._runner.tick()
        self._ensure_stimulus_for_p300()
        self._sync_ssvep_cue_overlay()
        QApplication.processEvents()
        st = self._runner.status_text
        self.lbl_status.setText(st)
        if st != prev:
            plog_info(f"status [{self._runner.state}]: {st}")
            self._last_status_printed = st
        # Закрываем PsychoPy до ССВП: иначе fullscreen перекрывает оверлей и мигалку не видно
        if self._stimulus_proc is not None and self._stimulus_proc.poll() is None:
            stop_stim = self._runner.state in ("ssvep_continuous", "ssvep_burst", "finalize", "stopped")
            if bool(self._runner.ssvep_cue_visible) or bool(self._runner.ssvep_blackout_visible):
                stop_stim = True
            if stop_stim:
                try:
                    self._stimulus_proc.terminate()
                except Exception:
                    pass
                self._stimulus_proc = None
                if prev != st and (
                    self._runner.state in ("ssvep_continuous", "ssvep_burst")
                    or bool(self._runner.ssvep_cue_visible)
                ):
                    plog_info("стимулятор плиток остановлен — этап ССВП (оверлей / мигалка)")
        if self._runner.state in ("finalize", "stopped"):
            self._restore_operator_window()
        if self._runner.state in ("stopped",):
            self._timer.stop()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self._restore_operator_window()
            self._runner = None
        elif str(st).startswith("Готово."):
            self._restore_operator_window()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = ProtocolRunnerWidget()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

