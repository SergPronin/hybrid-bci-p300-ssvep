from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pylsl import StreamInlet, StreamInfo, local_clock as lsl_local_clock
except Exception:  # pragma: no cover
    StreamInlet = object  # type: ignore
    StreamInfo = object  # type: ignore

    def lsl_local_clock() -> float:  # type: ignore
        return time.time()

from experiment_protocol import protocol_log as plog
from experiment_protocol.experiment_queue import QueueItem, build_main_queue, queue_summary
from experiment_protocol import stim_control as stim_ctl
from experiment_protocol.unified_logger import UnifiedExperimentLogger
from p300_analysis.marker_parsing import stim_key_to_tile_digit
from p300_analysis.constants import EEG_PULL_MAX_SAMPLES, MARKERS_PULL_MAX_SAMPLES
from p300_analysis.lsl_streams import (
    BCI_STIM_MARKER_STREAM_NAME,
    discover_eeg_streams,
    select_eeg_stream,
    stream_channel_labels,
    stream_inlet_with_buffer,
    wait_for_stimulus_marker_stream,
)
from p300_analysis.marker_parsing import parse_trial_end, parse_trial_target_tile_id
from p300_analysis.online_engine import P300Decision, P300EngineParams, P300OnlineEngine
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR
from ssvep_analysis.migalka_serial_controller import MigalkaConfig, MigalkaSerialController
from ssvep_analysis.online_engine import SSVEPDecision, SSVEPOnlineEngine, SSVEPParams


@dataclass(frozen=True)
class ProtocolConfig:
    output_root: Path
    subject_id: str
    com_port: str
    # Пустые строки — первый найденный поток ЭЭГ при предстарте
    eeg_stream_name: str = ""
    eeg_stream_session_id: str = ""

    # Основной блок: 15 P300 (AUC+шаблон за один trial) + 15 SSVEP×2 (перемешано)
    p300_main_trials: int = 15
    ssvep_blocks_per_mode: int = 15
    pause_between_experiments_s: float = 2.0
    shuffle_seed: int = -1  # -1 = случайный seed при старте main

    # Калибровка P300 в начале (подряд, до main)
    p300_calib_trials: int = 5
    calib_target_tile_id: int = 4
    template_warmup_target_epochs: int = 12

    # SSVEP timing
    ssvep_block_sec: float = 6.0
    ssvep_window_sec: float = 2.0
    ssvep_fs_hz: float = 250.0
    # 4 лампы по умолчанию (как SSVEPAnalyzerWindow.DEFAULT_LAMP_FREQS); до 6 на мигалке
    ssvep_freqs_hz: Tuple[float, ...] = (
        10.0,
        1000.0 / 99.0,
        1000.0 / 87.0,
        1000.0 / 76.0,
    )
    # P300 ROI: индексы каналов с 0; пустой кортеж = все каналы EEG
    roi_channels_0idx: Tuple[int, ...] = (3,)
    # SSVEP MSI ROI; пустой кортеж = все каналы EEG
    ssvep_roi_channels_0idx: Tuple[int, ...] = ()

    # Папка сессии (если создана до старта стимулятора — общий stim_control.json)
    session_dir: Optional[Path] = None


def _ru_stop_reason(reason: str) -> str:
    return {"user_stop": "остановка оператором"}.get(str(reason), str(reason))


def _ru_p300_winner_mode(winner_mode: str) -> str:
    if winner_mode == WINNER_MODE_AUC:
        return "AUC"
    if winner_mode == WINNER_MODE_TEMPLATE_CORR:
        return "сравнение с шаблоном"
    return str(winner_mode)


def _ru_ssvep_stim_mode(mode: str) -> str:
    if mode == "continuous":
        return "непрерывный"
    if mode == "burst":
        return "пакетный"
    return str(mode)


class ProtocolState:
    Idle = "idle"
    Preflight = "preflight"
    P300Calib = "p300_calib"
    Main = "main"
    # Под-состояния main для GUI / оверлея (совместимость)
    SSVEP_CONT = "ssvep_continuous"
    SSVEP_BURST = "ssvep_burst"
    Finalize = "finalize"
    Stopped = "stopped"


class ProtocolRunner:
    """Non-UI state machine: pulls LSL, controls Migalka, logs, produces trial/block results."""

    def __init__(self, cfg: ProtocolConfig) -> None:
        self.cfg = cfg
        self.state = ProtocolState.Idle
        self.status_text = "Ожидание"

        self._inlet_eeg: Optional[StreamInlet] = None
        self._inlet_markers: Optional[StreamInlet] = None

        self._logger: Optional[UnifiedExperimentLogger] = None
        self._p300 = P300OnlineEngine()
        self._ssvep = SSVEPOnlineEngine()
        self._migalka = MigalkaSerialController(
            mirror_lsl=True,
            on_event=lambda ev: self._on_migalka_event(ev),
        )

        self._p300_calib_trials_done = 0
        self._p300_main_trials_done = 0
        self._ssvep_block_count_cont = 0
        self._ssvep_block_count_burst = 0
        self._main_queue: List[QueueItem] = []
        self._main_queue_index = 0
        self._shuffle_seed_used: Optional[int] = None
        self._ssvep_any_continuous_done = False

        self._p300_trial_armed: bool = False
        self._p300_trial_end_seen: bool = False

        self._ssvep_block_started_at: Optional[float] = None
        self._ssvep_target_lamp: Optional[int] = None
        self._ssvep_last_msi_log_at: Optional[float] = None
        self._template_locked: bool = False

        # Неблокирующая пауза перед стартом следующей "единицы эксперимента" (trial/block).
        self._pause_until_wall: Optional[float] = None
        self._pause_status: Optional[str] = None
        self._pause_next_state: Optional[str] = None
        self._pause_next_ssvep_mode: Optional[str] = None
        self._last_state: str = ProtocolState.Idle
        self._eeg_nominal_fs_hz: float = 250.0
        self._ssvep_migalka_retry_at: float = 0.0
        self._ssvep_migalka_fail_streak: int = 0
        # Полноэкранная подсказка для оператора (рисует protocol_runner_gui)
        self.ssvep_cue_visible: bool = False
        self.ssvep_cue_exp_index: int = 0
        self.ssvep_cue_exp_total: int = 0
        self.ssvep_cue_lamp_1based: int = 1
        self.ssvep_cue_freq_hz: float = 0.0
        self.ssvep_cue_mode_label: str = ""
        self.ssvep_blackout_visible: bool = False
        self._on_ssvep_display_clear: Optional[Callable[[], None]] = None
        # Подсказки испытуемелю (читает protocol_runner_gui → ProtocolInstructionOverlay)
        self.participant_instruction: Optional[Dict[str, Any]] = None

    def set_ssvep_display_clear_callback(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_ssvep_display_clear = cb

    def _set_ssvep_blackout(self, visible: bool) -> None:
        self.ssvep_blackout_visible = bool(visible)
        if visible:
            self.ssvep_cue_visible = False
            self._update_participant_instruction(kind="blackout")

    def clear_ssvep_display(self) -> None:
        """Сброс флагов оверлея (чёрный экран / подсказка ССВП)."""
        self._set_ssvep_blackout(False)
        self._set_ssvep_cue_overlay(ssvep_mode=None)
        if self._on_ssvep_display_clear is not None:
            try:
                self._on_ssvep_display_clear()
            except Exception:
                pass

    @property
    def pause_active(self) -> bool:
        return self._pause_until_wall is not None

    @property
    def pause_before_ssvep(self) -> bool:
        return self._pause_until_wall is not None and self._pause_next_ssvep_mode is not None

    def _set_state(self, new_state: str, *, detail: str = "") -> None:
        if new_state != self._last_state:
            plog.state_change(self._last_state, new_state, detail=detail)
            self._last_state = str(new_state)
        self.state = str(new_state)

    def _total_main_experiments(self) -> int:
        return (
            int(self.cfg.p300_main_trials)
            + int(self.cfg.ssvep_blocks_per_mode) * 2
        )

    def _status_progress_label(self) -> str:
        if self.state == ProtocolState.P300Calib:
            return (
                f"Калибровка P300 {self._p300_calib_trials_done + 1}/"
                f"{int(self.cfg.p300_calib_trials)}"
            )
        if self.state == ProtocolState.Main and self._main_queue:
            return (
                f"Эксперимент {self._main_queue_index + 1}/{len(self._main_queue)}"
            )
        return "Протокол"

    def _current_queue_item(self) -> Optional[QueueItem]:
        if self.state != ProtocolState.Main:
            return None
        if self._main_queue_index < 0 or self._main_queue_index >= len(self._main_queue):
            return None
        return self._main_queue[self._main_queue_index]

    def _active_ssvep_freqs_hz(self) -> Tuple[float, ...]:
        active = tuple(float(x) for x in self.cfg.ssvep_freqs_hz if float(x) > 0.0)
        return active if active else tuple(float(x) for x in self.cfg.ssvep_freqs_hz)

    def _ssvep_target_lamp_0idx_for_item(self, item: QueueItem) -> int:
        if item.target_lamp_0idx is not None:
            return int(item.target_lamp_0idx)
        freqs = self._active_ssvep_freqs_hz()
        n_lamps = max(1, len(freqs))
        return int(self._ssvep_block_count_cont + self._ssvep_block_count_burst) % n_lamps

    def _ssvep_lamp_display(self, lamp_0idx: int) -> Tuple[int, float]:
        """Номер лампы для оператора (1-based) и частота."""
        freqs = self._active_ssvep_freqs_hz()
        lamp_1 = int(lamp_0idx) + 1
        hz = float(freqs[lamp_0idx]) if 0 <= int(lamp_0idx) < len(freqs) else 0.0
        return lamp_1, hz

    def _set_ssvep_cue_overlay(self, *, ssvep_mode: Optional[str]) -> None:
        if ssvep_mode is None:
            self.ssvep_cue_visible = False
            return
        self.ssvep_blackout_visible = False
        item = self._current_queue_item()
        lamp0 = self._ssvep_target_lamp_0idx_for_item(item) if item is not None else 0
        lamp1, hz = self._ssvep_lamp_display(lamp0)
        self.ssvep_cue_visible = True
        self.ssvep_cue_exp_index = int(self._main_queue_index + 1) if self._main_queue else 1
        self.ssvep_cue_exp_total = int(len(self._main_queue)) if self._main_queue else int(self._total_main_experiments())
        self.ssvep_cue_lamp_1based = int(lamp1)
        self.ssvep_cue_freq_hz = float(hz)
        self.ssvep_cue_mode_label = f"ССВП — {_ru_ssvep_stim_mode(str(ssvep_mode))}"
        plog.info(
            f"оверлей ССВП: эксп. {self.ssvep_cue_exp_index}/{self.ssvep_cue_exp_total}, "
            f"лампа L{self.ssvep_cue_lamp_1based} ({self.ssvep_cue_freq_hz:g} Гц)"
        )

    def _start_pause(self, *, seconds: float, status: str, next_state: str, next_ssvep_mode: Optional[str] = None) -> None:
        sec = max(0.0, float(seconds))
        self._pause_until_wall = float(time.time()) + sec
        self._pause_status = str(status)
        self._pause_next_state = str(next_state)
        self._pause_next_ssvep_mode = str(next_ssvep_mode) if next_ssvep_mode is not None else None
        if next_ssvep_mode is not None:
            self._set_ssvep_cue_overlay(ssvep_mode=str(next_ssvep_mode))
        else:
            self._set_ssvep_cue_overlay(ssvep_mode=None)
        plog.info(
            f"пауза {sec:.1f}s -> next_state={next_state}"
            + (f", ssvep_mode={next_ssvep_mode}" if next_ssvep_mode else "")
        )

    def _tick_pause(self) -> bool:
        """Returns True if we are in pause (and handled it)."""
        if self._pause_until_wall is None:
            return False
        now = float(time.time())
        rem = float(self._pause_until_wall) - now
        if rem > 0:
            base = self._pause_status or "Пауза"
            self.status_text = f"{base} ({rem:.1f} с)"
            self._update_participant_instruction(kind="pause", pause_message=base, seconds_left=rem)
            return True

        # Pause finished: apply scheduled transition once.
        next_state = self._pause_next_state
        next_mode = self._pause_next_ssvep_mode
        self._pause_until_wall = None
        self._pause_status = None
        self._pause_next_state = None
        self._pause_next_ssvep_mode = None
        self._set_ssvep_cue_overlay(ssvep_mode=None)
        if next_state:
            self._set_state(next_state, detail="pause finished")
            if next_state in (ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST) and next_mode is not None:
                plog.info(f"пауза закончилась — запуск SSVEP migalka, mode={next_mode}")
                item = self._current_queue_item()
                lamp = (
                    int(item.target_lamp_0idx)
                    if item is not None and item.target_lamp_0idx is not None
                    else self._ssvep_target_lamp
                )
                try:
                    self._start_ssvep_block(mode=str(next_mode), lamp_0idx=lamp)
                except Exception as e:
                    plog.exc("критическая ошибка _start_ssvep_block", e)
                    self.status_text = f"ССВП: ошибка запуска блока: {e}"
            elif next_state == ProtocolState.Main:
                item = self._current_queue_item()
                if item is not None and item.kind == "p300":
                    self._sync_stim_control(item)
                    self.status_text = f"{self._status_progress_label()}: P300 — смотрите на плитку…"
                    self._update_participant_instruction(kind="p300")
        return False

    @property
    def logger(self) -> Optional[UnifiedExperimentLogger]:
        return self._logger

    def _session_dir(self) -> Optional[Path]:
        if self._logger is not None:
            return self._logger.session_dir
        if self.cfg.session_dir is not None:
            return Path(self.cfg.session_dir)
        return None

    def _request_stim_p300_trial(
        self,
        *,
        target_tile_id: int,
        experiment_index: int,
        experiment_total: int,
        label: str = "P300",
    ) -> None:
        sd = self._session_dir()
        if sd is None:
            return
        stim_ctl.write_trial_request(
            sd,
            target_tile_id=int(target_tile_id),
            experiment_index=int(experiment_index),
            experiment_total=int(experiment_total),
            label=str(label),
        )
        plog.info(
            f"stim_control trial: tile={int(target_tile_id)} "
            f"({experiment_index}/{experiment_total}) {label!r}"
        )

    def _sync_stim_control(self, item: Optional[QueueItem], *, message: str = "") -> None:
        sd = self._session_dir()
        if sd is None:
            return
        total = int(len(self._main_queue)) if self._main_queue else int(self._total_main_experiments())
        idx = int(self._main_queue_index) + 1
        if item is None:
            stim_ctl.write_stim_done(sd)
            return
        if item.kind == "p300":
            tid = int(item.target_tile_id) if item.target_tile_id is not None else 0
            stim_ctl.write_trial_request(
                sd,
                target_tile_id=tid,
                experiment_index=idx,
                experiment_total=total,
                label="P300",
            )
        else:
            stim_ctl.write_paused(
                sd,
                reason="ssvep",
                message=message or "ССВП — смотрите на физическую лампу",
                experiment_index=idx,
                experiment_total=total,
            )

    def _update_participant_instruction(
        self,
        *,
        kind: str,
        pause_message: str = "",
        seconds_left: Optional[float] = None,
    ) -> None:
        total_main = int(len(self._main_queue)) if self._main_queue else 0
        if self.state == ProtocolState.P300Calib:
            self.participant_instruction = {
                "type": "calib",
                "index": int(self._p300_calib_trials_done) + 1,
                "total": int(self.cfg.p300_calib_trials),
                "tile": int(self.cfg.calib_target_tile_id),
            }
            return
        if self._pause_until_wall is not None:
            self.participant_instruction = {
                "type": "pause",
                "message": pause_message or self._pause_status or "Пауза",
                "seconds_left": seconds_left,
            }
            return
        item = self._current_queue_item()
        if item is None:
            self.participant_instruction = None
            return
        idx = int(self._main_queue_index) + 1
        if kind == "blackout" or self.ssvep_blackout_visible:
            self.participant_instruction = {"type": "blackout"}
            return
        if item.kind == "p300":
            tid = int(item.target_tile_id) if item.target_tile_id is not None else 0
            self.participant_instruction = {
                "type": "p300",
                "index": idx,
                "total": total_main,
                "tile": tid,
            }
            return
        if item.kind == "ssvep" and self.ssvep_cue_visible:
            lamp0 = self._ssvep_target_lamp_0idx_for_item(item)
            lamp1, hz = self._ssvep_lamp_display(lamp0)
            self.participant_instruction = {
                "type": "ssvep",
                "index": idx,
                "total": total_main,
                "lamp_1based": int(lamp1),
                "freq_hz": float(hz),
                "mode_label": f"ССВП — {_ru_ssvep_stim_mode(str(item.ssvep_mode or ''))}",
            }
            return
        self.participant_instruction = None

    def start(self) -> None:
        ch_disp = (
            ",".join(str(c + 1) for c in self.cfg.roi_channels_0idx)
            if self.cfg.roi_channels_0idx
            else "ALL"
        )
        ssvep_ch_disp = (
            ",".join(str(c + 1) for c in self.cfg.ssvep_roi_channels_0idx)
            if self.cfg.ssvep_roi_channels_0idx
            else "ALL"
        )
        plog.info(
            f"=== PROTOCOL START subject={self.cfg.subject_id!r} "
            f"calib={self.cfg.p300_calib_trials} main_P300={self.cfg.p300_main_trials} "
            f"SSVEP={self.cfg.ssvep_blocks_per_mode}x2 shuffle_seed={self.cfg.shuffle_seed} "
            f"COM={self.cfg.com_port!r} P300_ch(1-based)={ch_disp} "
            f"SSVEP_ch(1-based)={ssvep_ch_disp} migalka_freqs_Hz={list(self.cfg.ssvep_freqs_hz)} ==="
        )
        self._set_state(ProtocolState.Preflight, detail="start()")
        self.status_text = "Предстарт: поиск потоков LSL и COM-порта мигалки…"
        plan = {
            "schema_version": 2,
            "calibration": {
                "p300_trials": int(self.cfg.p300_calib_trials),
                "target_tile_id": int(self.cfg.calib_target_tile_id),
                "template_epochs": int(self.cfg.template_warmup_target_epochs),
            },
            "main": {
                "p300_trials": int(self.cfg.p300_main_trials),
                "ssvep_continuous": int(self.cfg.ssvep_blocks_per_mode),
                "ssvep_burst": int(self.cfg.ssvep_blocks_per_mode),
                "shuffle_seed_cfg": int(self.cfg.shuffle_seed),
                "pause_sec": float(self.cfg.pause_between_experiments_s),
            },
            "p300_methods_per_trial": [WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR],
            "ssvep": {
                "block_sec": float(self.cfg.ssvep_block_sec),
                "window_sec": float(self.cfg.ssvep_window_sec),
                "fs_hz": float(self.cfg.ssvep_fs_hz),
                "freqs_hz": [float(x) for x in self.cfg.ssvep_freqs_hz],
                "roi_channels_0idx": [int(c) for c in self.cfg.ssvep_roi_channels_0idx],
            },
        }
        self._logger = UnifiedExperimentLogger.open_new(
            output_root=Path(self.cfg.output_root),
            subject_id=str(self.cfg.subject_id),
            protocol_plan=plan,
            start_payload={"note": "protocol_start"},
            session_dir=Path(self.cfg.session_dir) if self.cfg.session_dir is not None else None,
        )
        sd = self._session_dir()
        if sd is not None:
            stim_ctl.write_paused(sd, reason="preflight", message="Подготовка протокола…")

    def stop(self, *, reason: str = "user_stop") -> None:
        plog.info(f"=== PROTOCOL STOP reason={reason!r} ===")
        self._sync_stim_control(None)
        self.participant_instruction = None
        self.clear_ssvep_display()
        try:
            self._migalka.stop_and_close()
        except Exception as e:
            plog.exc("migalka stop", e)
        self._set_state(ProtocolState.Stopped, detail=reason)
        self.status_text = f"Остановлено: {_ru_stop_reason(reason)}"
        if self._logger is not None:
            self._logger.finalize(stop_payload={"reason": reason})
            self._logger = None

    def tick(self) -> None:
        """One tick: pull streams, update engines, advance FSM."""
        if self.state in (ProtocolState.Idle, ProtocolState.Stopped):
            return
        if self.state == ProtocolState.Preflight:
            self._preflight()
            return

        # Always pull EEG + markers while protocol runs (shared across modes).
        self._pull_lsl()

        # Pause between experiments (do not advance state while pausing).
        if self._tick_pause():
            return

        if self.state == ProtocolState.P300Calib:
            self._run_p300_calib()
        elif self.state == ProtocolState.Main:
            self._run_main_queue_item()
        elif self.state in (ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST):
            mode = "continuous" if self.state == ProtocolState.SSVEP_CONT else "burst"
            self._run_ssvep_blocks(mode=mode)

        # Finalize может быть выставлен внутри _run_ssvep_blocks — обработать в том же tick.
        if self.state == ProtocolState.Finalize:
            self._finalize()
        elif self.state == ProtocolState.P300Calib:
            self._update_participant_instruction(kind="calib")
        elif self.state == ProtocolState.Main and self._pause_until_wall is None:
            self._update_participant_instruction(kind="main")

    def _preflight(self) -> None:
        assert self._logger is not None
        eeg_streams = discover_eeg_streams(timeout=1.5)
        info_mk, marker_streams = wait_for_stimulus_marker_stream(max_wait_sec=20.0)
        info_eeg = select_eeg_stream(
            eeg_streams,
            name=str(self.cfg.eeg_stream_name),
            session_id=str(self.cfg.eeg_stream_session_id),
        )
        if not eeg_streams:
            plog.error("Preflight: LSL EEG не найден")
            self.status_text = "Предстарт: не найден ни один поток LSL EEG (проверьте запись ЭЭГ)."
            return
        if info_eeg is None:
            want = self.cfg.eeg_stream_name or "?"
            plog.error(f"Preflight: выбранный поток ЭЭГ не найден: {want!r}")
            self.status_text = f"Предстарт: поток ЭЭГ «{want}» не найден. Нажмите «Обновить» в GUI и выберите поток."
            return
        if info_mk is None:
            names = []
            for s in marker_streams:
                try:
                    names.append(str(s.name() or "?"))
                except Exception:
                    names.append("?")
            plog.error(
                f"Preflight: нет потока {BCI_STIM_MARKER_STREAM_NAME!r} "
                f"(найдены Markers: {names}; нужен run_app.py / PsychoPy, не только мигалка)"
            )
            only_migalka = names == ["MigalkaStimMarkers"] or (
                len(names) == 1 and "migalka" in (names[0] or "").lower()
            )
            hint = (
                " Остался поток мигалки от прошлого ССВП — нажмите «Запустить» снова "
                "(GUI перезапустит стимулятор) или перезапустите protocol_runner_gui."
                if only_migalka
                else ""
            )
            self.status_text = (
                f"Предстарт: нет потока маркеров плиток «{BCI_STIM_MARKER_STREAM_NAME}». "
                "Запустите стимулятор (галочка в GUI или run_app.py) и подождите 5–10 с."
                f"{hint}"
            )
            return
        try:
            eeg_name = info_eeg.name()
            mk_name = info_mk.name()
        except Exception:
            eeg_name = "?"
            mk_name = "?"
        try:
            fs_raw = float(info_eeg.nominal_srate() or 0.0)
        except Exception:
            fs_raw = 0.0
        self._eeg_nominal_fs_hz = fs_raw if fs_raw > 1.0 else float(self.cfg.ssvep_fs_hz)
        plog.info(
            f"Preflight OK: EEG={eeg_name!r} fs={self._eeg_nominal_fs_hz:g} Hz, "
            f"Markers={mk_name!r}, COM={self.cfg.com_port!r}"
        )
        self._inlet_eeg = stream_inlet_with_buffer(info_eeg, buffer_seconds=20)
        self._inlet_markers = stream_inlet_with_buffer(info_mk, buffer_seconds=20)
        try:
            n_ch = max(1, int(info_eeg.channel_count()))
            labels = stream_channel_labels(info_eeg, n_ch)
            self._logger.set_eeg_channel_labels(labels)
        except Exception:
            pass
        self._logger.write(
            "preflight_ok",
            {
                "eeg_stream": {"name": getattr(info_eeg, "name", lambda: "")(), "type": getattr(info_eeg, "type", lambda: "")()},
                "markers_stream": {"name": getattr(info_mk, "name", lambda: "")(), "type": getattr(info_mk, "type", lambda: "")()},
                "com_port": str(self.cfg.com_port),
                "eeg_nominal_fs_hz": float(self._eeg_nominal_fs_hz),
            },
        )

        self._reset_for_new_p300_block()
        if int(self.cfg.p300_calib_trials) <= 0:
            plog.info("калибровка P300 отключена (0 прогонов) — сразу основной блок")
            self._begin_main_phase()
        else:
            self._set_state(ProtocolState.P300Calib, detail="preflight ok")
            self._request_stim_p300_trial(
                target_tile_id=int(self.cfg.calib_target_tile_id),
                experiment_index=1,
                experiment_total=int(self.cfg.p300_calib_trials),
                label="Калибровка P300",
            )
            self.status_text = (
                f"{self._status_progress_label()}: сбор шаблона P300, "
                f"плитка {int(self.cfg.calib_target_tile_id)} — запуск прогона…"
            )
            plog.info(self.status_text)

    def _pull_lsl(self) -> None:
        if self._inlet_eeg is None or self._inlet_markers is None or self._logger is None:
            return
        now_lc = float(lsl_local_clock())

        # markers
        try:
            marker_chunk, marker_ts = self._inlet_markers.pull_chunk(timeout=0.0, max_samples=MARKERS_PULL_MAX_SAMPLES)
        except TypeError:
            marker_chunk, marker_ts = self._inlet_markers.pull_chunk(timeout=0.0)
        if marker_ts:
            self._logger.write("markers_chunk", {"n": int(len(marker_ts))})
            for sample in marker_chunk:
                if parse_trial_end(sample):
                    self._p300_trial_end_seen = True
                    plog.event("LSL trial_end", sample=str(sample)[:80])
                tid = parse_trial_target_tile_id(sample)
                if tid is not None:
                    self._p300_trial_armed = True
                    self._p300_trial_end_seen = False
                    plog.event("LSL trial_start", target=int(tid), sample=str(sample)[:80])
                    self._logger.write("protocol_trial_start_arm", {"cue_target_tile_id": int(tid)})
                    # Обновляем статус сразу на старте trial.
                    if self.state == ProtocolState.P300Calib:
                        mode = "P300, калибровка"
                    elif self.state == ProtocolState.Main:
                        mode = "P300, AUC + шаблон"
                    else:
                        mode = "P300"
                    self.status_text = f"{self._status_progress_label()}: {mode}, целевая плитка={int(tid)}"
            if self.state in (ProtocolState.P300Calib, ProtocolState.Main):
                self._p300.ingest_marker_chunk(
                    marker_chunk=marker_chunk, marker_ts=marker_ts, lsl_local_clock_now=now_lc
                )

        # EEG
        try:
            eeg_chunk, eeg_ts = self._inlet_eeg.pull_chunk(timeout=0.0, max_samples=EEG_PULL_MAX_SAMPLES)
        except TypeError:
            eeg_chunk, eeg_ts = self._inlet_eeg.pull_chunk(timeout=0.0)
        if eeg_ts:
            arr = np.asarray(eeg_chunk, dtype=np.float64)
            self._logger.append_eeg_chunk(eeg_ts, arr, stream="EEG", lsl_local_clock=now_lc)
            self._p300.ingest_eeg_chunk(eeg_chunk=arr, eeg_ts=eeg_ts, lsl_local_clock_now=now_lc)
            self._ssvep.ingest_eeg_chunk(times=eeg_ts, samples=arr)

        if self.state in (ProtocolState.P300Calib, ProtocolState.Main):
            extracted = self._p300.extract_ready_epochs()
        else:
            extracted = 0
        if extracted and self._logger is not None:
            self._logger.write("p300_epochs_extracted", {"n": int(extracted)})
            if self.state == ProtocolState.P300Calib and not self._template_locked:
                if self._p300.try_build_external_template_from_epochs(
                    min_epochs=int(self.cfg.template_warmup_target_epochs)
                ):
                    self._logger.write(
                        "p300_template_ready",
                        {
                            "target_tile_id": int(self._p300.external_template_target_id)
                            if self._p300.external_template_target_id is not None
                            else None,
                            "n_epochs": int(self.cfg.template_warmup_target_epochs),
                            "phase": "calib",
                        },
                    )

    def _reset_for_new_p300_block(self) -> None:
        prof = P300EngineParams(
            baseline_ms=100,
            window_x_ms=550,
            window_y_ms=725,
            artifact_threshold_uv=60.0,
            use_car=False,
            roi_channels_0idx=tuple(int(c) for c in self.cfg.roi_channels_0idx),
        )
        self._p300.reset(params=prof)
        plog.info(
            f"P300 engine: roi_channels_0idx={prof.roi_channels_0idx or 'ALL'}, "
            f"window={prof.window_x_ms}-{prof.window_y_ms}ms"
        )
        self._p300_trial_armed = False
        self._p300_trial_end_seen = False

    def _p300_template_ready(self) -> bool:
        return self._p300.external_template_window is not None

    def _finish_p300_trial_armed(self) -> bool:
        """True если trial можно закрыть (решение или trial_end)."""
        if not self._p300_trial_armed:
            return False
        if self._p300.current_cue_target_id is None:
            return False
        dec_probe = self._p300.compute_decision(winner_mode=WINNER_MODE_AUC)
        if dec_probe.can_decide or self._p300_trial_end_seen:
            return True
        return False

    def _p300_tile_label(self, tile_id: Optional[int]) -> str:
        if tile_id is None:
            return "—"
        return str(int(tile_id))

    def _p300_correct_label(self, ok: Optional[bool]) -> str:
        if ok is True:
            return "верно"
        if ok is False:
            return "неверно"
        return "нет решения"

    def _log_p300_per_tile_metrics(self, method_label: str, dec_dict: Dict[str, Any], cue_tid: int) -> None:
        dbg = dec_dict.get("debug") if isinstance(dec_dict.get("debug"), dict) else {}
        keys = dbg.get("stim_keys") or []
        vals = dbg.get("final_metric_values")
        if not keys or vals is None:
            return
        plog.info(f"  {method_label} — метрики по плиткам:")
        for k, v in zip(keys, vals):
            tid = stim_key_to_tile_digit(str(k))
            mark = " ← цель" if tid is not None and int(tid) == int(cue_tid) else ""
            plog.info(f"    плитка {tid}: {float(v):.4f}{mark}")

    def _log_p300_experiment_result(
        self,
        *,
        phase: str,
        cue_tid: int,
        planned_tid: int,
        auc_d: Dict[str, Any],
        tpl_d: Dict[str, Any],
        agree: bool,
        queue_index: Optional[int] = None,
    ) -> None:
        phase_ru = {"calib": "калибровка", "main": "основной блок"}.get(str(phase), str(phase))
        exp_tag = self._status_progress_label() if phase == "main" else f"Калибровка P300"
        plog.info(f"════ Итог P300 ({phase_ru}, {exp_tag}) ════")
        plog.info(f"Испытуемый смотрел на плитку: {cue_tid}")
        if planned_tid >= 0 and int(planned_tid) != int(cue_tid):
            plog.info(f"Плановая плитка в очереди: {int(planned_tid)}")
        auc_w = auc_d.get("winner_tile_id")
        tpl_w = tpl_d.get("winner_tile_id")
        auc_ok = auc_d.get("winner_tile_id") == cue_tid if cue_tid >= 0 else None
        tpl_ok = tpl_d.get("winner_tile_id") == cue_tid if cue_tid >= 0 else None
        plog.info(
            f"AUC → плитка {self._p300_tile_label(auc_w)} ({self._p300_correct_label(auc_ok)})"
        )
        plog.info(
            f"Шаблон → плитка {self._p300_tile_label(tpl_w)} ({self._p300_correct_label(tpl_ok)})"
        )
        plog.info(f"Методы совпали: {'да' if agree else 'нет'}")
        if cue_tid >= 0:
            self._log_p300_per_tile_metrics("AUC", auc_d, cue_tid)
            self._log_p300_per_tile_metrics("Шаблон", tpl_d, cue_tid)
        if queue_index is not None:
            plog.info(f"Индекс в очереди: {int(queue_index)}")

    def _log_ssvep_msi_result(
        self,
        *,
        dec: SSVEPDecision,
        mode: str,
        target_lamp_0idx: Optional[int],
        elapsed_s: Optional[float] = None,
        final: bool = False,
    ) -> None:
        freqs = list(self._active_ssvep_freqs_hz())
        tag = "итог блока" if final else "MSI (окно)"
        elapsed_sfx = f", t={float(elapsed_s):.1f} с" if elapsed_s is not None else ""
        plog.info(f"──── ССВП {_ru_ssvep_stim_mode(mode)} — {tag}{elapsed_sfx} ────")
        if target_lamp_0idx is not None:
            plog.info(
                f"Целевая лампа: L{int(target_lamp_0idx) + 1} "
                f"({freqs[int(target_lamp_0idx)]:.3f} Гц)"
                if 0 <= int(target_lamp_0idx) < len(freqs)
                else f"Целевая лампа (0-based): {int(target_lamp_0idx)}"
            )
        if not dec.classify_allowed:
            plog.info(f"MSI: классификация недоступна — {dec.debug}")
            return
        w1 = dec.winner_1based
        w0 = dec.winner_0idx
        ok = (
            int(w0) == int(target_lamp_0idx)
            if w0 is not None and target_lamp_0idx is not None
            else None
        )
        plog.info(
            f"MSI победитель: L{w1} (индекс {w0}) — {self._p300_correct_label(ok)}"
            if w1 is not None
            else "MSI победитель: не определён"
        )
        if dec.coef:
            plog.info("MSI Coef по лампам:")
            for i, c in enumerate(dec.coef):
                freq = float(freqs[i]) if i < len(freqs) else 0.0
                mark = ""
                if target_lamp_0idx is not None and int(i) == int(target_lamp_0idx):
                    mark = " ← цель"
                if w0 is not None and int(i) == int(w0):
                    mark += " ★выбор MSI"
                plog.info(f"  L{i + 1} ({freq:.3f} Гц): {float(c):.6f}{mark}")
        dbg = dec.debug or {}
        if dbg:
            plog.info(f"MSI debug: {dbg}")

    def _decision_to_dict(self, decision: P300Decision, *, method: str) -> Dict[str, Any]:
        winner_tile = (
            stim_key_to_tile_digit(str(decision.winner_key))
            if decision.winner_key
            else None
        )
        return {
            "method": str(method),
            "can_decide": bool(decision.can_decide),
            "winner_key": decision.winner_key,
            "winner_tile_id": winner_tile,
            "mode_used": decision.mode_used,
            "min_epochs_per_class": int(decision.min_epochs_per_class),
            "debug": decision.debug,
        }

    def _record_dual_p300_experiment(self, *, phase: str, queue_index: Optional[int] = None) -> None:
        assert self._logger is not None
        cue_tid = int(self._p300.current_cue_target_id or -1)
        item = self._current_queue_item() if phase == "main" else None
        planned_tid = (
            int(item.target_tile_id)
            if item is not None and item.target_tile_id is not None
            else cue_tid
        )
        dec_auc = self._p300.compute_decision(winner_mode=WINNER_MODE_AUC)
        if not dec_auc.can_decide and self._p300_trial_end_seen:
            dec_auc = P300Decision(
                can_decide=False,
                min_epochs_per_class=int(dec_auc.min_epochs_per_class),
                winner_idx=None,
                winner_key=None,
                mode_used="no_decision",
                debug={**(dec_auc.debug or {}), "note": "trial_end_without_decision"},
            )
        dec_tpl = self._p300.compute_decision(winner_mode=WINNER_MODE_TEMPLATE_CORR)
        if not dec_tpl.can_decide and self._p300_trial_end_seen:
            dec_tpl = P300Decision(
                can_decide=False,
                min_epochs_per_class=int(dec_tpl.min_epochs_per_class),
                winner_idx=None,
                winner_key=None,
                mode_used="no_decision",
                debug={**(dec_tpl.debug or {}), "note": "trial_end_without_decision"},
            )

        auc_d = self._decision_to_dict(dec_auc, method=WINNER_MODE_AUC)
        tpl_d = self._decision_to_dict(dec_tpl, method=WINNER_MODE_TEMPLATE_CORR)
        agree = (
            auc_d.get("winner_tile_id") is not None
            and auc_d.get("winner_tile_id") == tpl_d.get("winner_tile_id")
        )
        exp_rec = {
            "phase": str(phase),
            "kind": "p300",
            "queue_index": queue_index,
            "cue_target_tile_id": cue_tid,
            "planned_target_tile_id": int(planned_tid) if planned_tid >= 0 else None,
            "ground_truth_tile_id": int(planned_tid) if planned_tid >= 0 else None,
            "auc": auc_d,
            "template_corr": tpl_d,
            "methods_agree": bool(agree),
            "correct_auc": auc_d.get("winner_tile_id") == cue_tid if cue_tid >= 0 else None,
            "correct_template": tpl_d.get("winner_tile_id") == cue_tid if cue_tid >= 0 else None,
            "epoch_counts_by_stim": {k: len(v) for k, v in self._p300.epochs_data.items()},
            "trial_end_seen": bool(self._p300_trial_end_seen),
            "template_target_id": int(self._p300.external_template_target_id)
            if self._p300.external_template_target_id is not None
            else None,
        }
        self._logger.append_experiment(exp_rec)
        legacy = {
            "phase": phase,
            "cue_target_tile_id": cue_tid,
            "auc": auc_d,
            "template_corr": tpl_d,
            "methods_agree": agree,
        }
        self._logger.append_p300_trial(legacy)
        self._logger.write("trial_decision_dual", legacy)
        self._log_p300_experiment_result(
            phase=str(phase),
            cue_tid=int(cue_tid),
            planned_tid=int(planned_tid),
            auc_d=auc_d,
            tpl_d=tpl_d,
            agree=bool(agree),
            queue_index=queue_index,
        )
        self._p300.reset(params=self._p300.params)
        self._p300_trial_armed = False
        self._p300_trial_end_seen = False

    def _run_p300_calib(self) -> None:
        assert self._logger is not None
        if not self._p300_trial_armed:
            self.status_text = f"{self._status_progress_label()}: ждём trial_start (калибровка)…"
            return
        cue = self._p300.current_cue_target_id
        if cue is None:
            self.status_text = f"{self._status_progress_label()}: нет target в маркере…"
            return
        key = f"стимул_{int(cue)}"
        n = len(self._p300.epochs_data.get(key, []))
        self.status_text = (
            f"{self._status_progress_label()}: плитка {cue}, эпох {n}/"
            f"{self.cfg.template_warmup_target_epochs}"
        )
        if not self._finish_p300_trial_armed():
            return

        self._record_dual_p300_experiment(phase="calib")
        self._p300_calib_trials_done += 1
        template_ok = self._p300_template_ready()
        calib_done = (
            template_ok
            or int(self._p300_calib_trials_done) >= int(self.cfg.p300_calib_trials)
        )
        if not calib_done:
            nxt = int(self._p300_calib_trials_done) + 1
            self._request_stim_p300_trial(
                target_tile_id=int(self.cfg.calib_target_tile_id),
                experiment_index=nxt,
                experiment_total=int(self.cfg.p300_calib_trials),
                label="Калибровка P300",
            )
            self.status_text = (
                f"Калибровка: прогон {self._p300_calib_trials_done}/"
                f"{self.cfg.p300_calib_trials}, шаблон={'да' if template_ok else 'нет'} — "
                f"следующий прогон…"
            )
            return

        if not template_ok:
            plog.warn("калибровка завершена по числу trial, шаблон не собран — template_corr может быть слабым")
        self._template_locked = True
        self._logger.write(
            "calibration_done",
            {
                "trials": int(self._p300_calib_trials_done),
                "template_ready": bool(template_ok),
                "target_tile_id": int(self._p300.external_template_target_id or self.cfg.calib_target_tile_id),
            },
        )
        self._begin_main_phase()

    def _begin_main_phase(self) -> None:
        assert self._logger is not None
        n_lamps = max(1, len(self._active_ssvep_freqs_hz()))
        seed_cfg = int(self.cfg.shuffle_seed)
        seed_arg: Optional[int] = None if seed_cfg < 0 else seed_cfg
        self._main_queue, self._shuffle_seed_used = build_main_queue(
            p300_trials=int(self.cfg.p300_main_trials),
            ssvep_continuous=int(self.cfg.ssvep_blocks_per_mode),
            ssvep_burst=int(self.cfg.ssvep_blocks_per_mode),
            n_active_lamps=n_lamps,
            shuffle_seed=seed_arg,
        )
        self._main_queue_index = 0
        self._reset_for_new_p300_block()
        self._set_state(ProtocolState.Main, detail="calibration done")
        plan_log = {
            "shuffle_seed": int(self._shuffle_seed_used or 0),
            "summary": queue_summary(self._main_queue),
            "queue": [x.to_dict() for x in self._main_queue],
        }
        self._logger.write("main_queue_ready", plan_log)
        plog.info(f"=== MAIN очередь ({len(self._main_queue)}): seed={self._shuffle_seed_used} ===")
        stim_ctl.write_paused(
            self._logger.session_dir,
            reason="main_start",
            message="Основной блок — ожидание…",
            experiment_index=1,
            experiment_total=len(self._main_queue),
        )
        self._advance_to_current_queue_item(start_pause=False)

    def _advance_to_current_queue_item(self, *, start_pause: bool) -> None:
        """Перейти к текущему элементу очереди (пауза или немедленный старт SSVEP)."""
        item = self._current_queue_item()
        if item is None:
            plog.info("=== очередь main исчерпана -> Finalize ===")
            self._sync_stim_control(None)
            self.clear_ssvep_display()
            self._set_state(ProtocolState.Finalize, detail="main queue done")
            return
        label = self._status_progress_label()
        if item.kind == "p300":
            if start_pause:
                stim_ctl.write_paused(
                    self._logger.session_dir,
                    reason="pause_before_p300",
                    message=f"Пауза перед P300, плитка {int(item.target_tile_id or 0)}",
                    experiment_index=int(self._main_queue_index) + 1,
                    experiment_total=len(self._main_queue),
                )
                self._start_pause(
                    seconds=float(self.cfg.pause_between_experiments_s),
                    status=f"{label}: пауза перед P300",
                    next_state=ProtocolState.Main,
                )
                self._update_participant_instruction(kind="pause", pause_message="Перед P300")
            else:
                self._set_state(ProtocolState.Main, detail="p300 item")
                self._sync_stim_control(item)
                self.status_text = f"{label}: P300 — смотрите на плитку {int(item.target_tile_id or 0)}…"
                self._update_participant_instruction(kind="p300")
            return
        mode = str(item.ssvep_mode or "continuous")
        lamp0 = self._ssvep_target_lamp_0idx_for_item(item)
        self._ssvep_target_lamp = int(lamp0)
        st = ProtocolState.SSVEP_CONT if mode == "continuous" else ProtocolState.SSVEP_BURST
        self._sync_stim_control(item, message=f"ССВП ({_ru_ssvep_stim_mode(mode)})")
        if start_pause:
            self._start_pause(
                seconds=float(self.cfg.pause_between_experiments_s),
                status=f"{label}: пауза перед ССВП ({_ru_ssvep_stim_mode(mode)})",
                next_state=st,
                next_ssvep_mode=mode,
            )
            self._set_ssvep_cue_overlay(ssvep_mode=mode)
            self._update_participant_instruction(kind="ssvep")
        else:
            self._set_state(st, detail="ssvep item")
            try:
                self._start_ssvep_block(mode=mode, lamp_0idx=int(lamp0))
            except Exception as e:
                plog.exc("_start_ssvep_block", e)

    def _complete_main_queue_item(self) -> None:
        self._main_queue_index += 1
        if self._main_queue_index >= len(self._main_queue):
            self.clear_ssvep_display()
            self._set_state(ProtocolState.Finalize, detail="main done")
            return
        self._advance_to_current_queue_item(start_pause=True)

    def _run_main_queue_item(self) -> None:
        item = self._current_queue_item()
        if item is None:
            self._set_state(ProtocolState.Finalize, detail="empty queue")
            return
        if item.kind != "p300":
            return
        assert self._logger is not None
        if not self._p300_trial_armed:
            self.status_text = f"{self._status_progress_label()}: P300 — ждём trial_start…"
            return
        if self._p300.current_cue_target_id is None:
            self.status_text = f"{self._status_progress_label()}: P300 — нет target…"
            return
        if not self._finish_p300_trial_armed():
            self.status_text = (
                f"{self._status_progress_label()}: P300 — накопление эпох…"
            )
            return
        self._record_dual_p300_experiment(
            phase="main",
            queue_index=int(self._main_queue_index),
        )
        self._p300_main_trials_done += 1
        self._complete_main_queue_item()

    def _on_migalka_event(self, ev: Tuple[int, bool, str]) -> None:
        # Feed burst gate via SSVEP engine
        lamp, is_on, raw = ev
        # Use local clock as lsl_time surrogate; LSL mirror is sent separately.
        self._ssvep.ingest_migalka_marker(lsl_time=float(lsl_local_clock()), value=f"{100+int(lamp)}|{'on' if is_on else 'off'}")
        if self._logger is not None:
            self._logger.write("migalka_led", {"lamp": int(lamp), "on": bool(is_on), "raw": str(raw)})

    def _start_ssvep_block(self, *, mode: str, lamp_0idx: Optional[int] = None) -> None:
        assert self._logger is not None
        plog.info(f"=== SSVEP block START mode={mode!r} COM={self.cfg.com_port!r} ===")
        active_freqs = self._active_ssvep_freqs_hz()
        n_lamps = max(1, len(active_freqs))
        fs_hz = float(self._eeg_nominal_fs_hz or self.cfg.ssvep_fs_hz)
        self._ssvep.reset(
            params=SSVEPParams(
                fs_hz=fs_hz,
                window_sec=float(self.cfg.ssvep_window_sec),
                freqs_hz=active_freqs,
                mode=str(mode),
                roi_channels_0idx=tuple(int(c) for c in self.cfg.ssvep_roi_channels_0idx),
            )
        )
        item = self._current_queue_item()
        if lamp_0idx is not None:
            self._ssvep_target_lamp = int(lamp_0idx)
        elif item is not None and item.kind == "ssvep":
            self._ssvep_target_lamp = int(self._ssvep_target_lamp_0idx_for_item(item))
        else:
            self._ssvep_target_lamp = 0
        freq_labels_list = [str(x).replace(",", ".") if float(x) > 0 else "0" for x in self.cfg.ssvep_freqs_hz]
        while len(freq_labels_list) < 6:
            freq_labels_list.append("0")
        freq_labels = tuple(freq_labels_list[:6])
        mig_mode = 0 if str(mode) == "continuous" else 1
        mcfg = MigalkaConfig(
            port=str(self.cfg.com_port),
            mode=int(mig_mode),
            freqs=freq_labels,
        )
        plog.info(
            f"SSVEP engine: lamps={n_lamps} freqs_Hz={list(active_freqs)} "
            f"roi={list(self.cfg.ssvep_roi_channels_0idx) or 'ALL'} "
            f"migalka_M={mig_mode} ({'постоянный' if mig_mode == 0 else 'пакетный'})"
        )
        try:
            if str(mode) == "burst" and self._migalka.is_open() and self._ssvep_any_continuous_done:
                plog.info("пакетный: COM открыт после continuous — handoff apply")
                self._migalka.open_and_start(mcfg)
            else:
                try:
                    self._migalka.stop_and_close()
                except Exception:
                    pass
                self._migalka.open_and_start(mcfg)
            if not self._migalka.is_open():
                raise RuntimeError("порт открыт, но is_open() == False")
        except Exception as e:
            plog.exc(f"Migalka не запустилась на {self.cfg.com_port!r}", e)
            self._ssvep_block_started_at = None
            self._ssvep_migalka_fail_streak += 1
            self._ssvep_migalka_retry_at = float(time.time()) + 3.0
            sm = _ru_ssvep_stim_mode(mode)
            self.status_text = (
                f"ССВП ({sm}): ошибка мигалки {self.cfg.com_port}: {e}. "
                f"Повтор через 3 с (закройте migalka.py, если порт занят)."
            )
            self._logger.write("migalka_error", {"port": str(self.cfg.com_port), "error": str(e), "mode": str(mode)})
            return

        self._ssvep_migalka_fail_streak = 0
        self._ssvep_block_started_at = float(time.time())
        self._ssvep_last_msi_log_at = None
        self._set_ssvep_blackout(True)
        plog.info(f"мигалка запущена, блок {self.cfg.ssvep_block_sec}s, лампа-цель={self._ssvep_target_lamp}")
        sm = _ru_ssvep_stim_mode(mode)
        self.status_text = (
            f"{self._status_progress_label()}: ССВП ({sm}), лампа {int(self._ssvep_target_lamp)} — "
            f"0/{self.cfg.ssvep_block_sec:.0f} с"
        )
        self._logger.write(
            "ssvep_block_start",
            {"mode": str(mode), "target_lamp": int(self._ssvep_target_lamp), "freqs_hz": [float(x) for x in self.cfg.ssvep_freqs_hz]},
        )

    def _run_ssvep_blocks(self, *, mode: str) -> None:
        assert self._logger is not None
        if self._ssvep_block_started_at is None:
            if self._pause_until_wall is not None:
                return
            if self._ssvep_migalka_fail_streak >= 15:
                sm = _ru_ssvep_stim_mode(mode)
                self.status_text = (
                    f"ССВП ({sm}): мигалка на {self.cfg.com_port!r} не отвечает. "
                    "Проверьте COM, питание и что migalka.py не занял порт."
                )
                return
            now = float(time.time())
            if now < float(self._ssvep_migalka_retry_at):
                sm = _ru_ssvep_stim_mode(mode)
                rem = float(self._ssvep_migalka_retry_at) - now
                self.status_text = f"ССВП ({sm}): повтор подключения мигалки через {rem:.1f} с…"
                return
            try:
                item = self._current_queue_item()
                lamp = (
                    int(item.target_lamp_0idx)
                    if item is not None and item.target_lamp_0idx is not None
                    else None
                )
                self._start_ssvep_block(mode=str(mode), lamp_0idx=lamp)
            except Exception as e:
                plog.exc("критическая ошибка _start_ssvep_block", e)
                self.status_text = f"ССВП: ошибка запуска блока: {e}"
            return
        elapsed = float(time.time()) - float(self._ssvep_block_started_at)
        sm = _ru_ssvep_stim_mode(mode)
        self.status_text = (
            f"{self._status_progress_label()}: ССВП ({sm}) — "
            f"{elapsed:.1f}/{self.cfg.ssvep_block_sec:.0f} с, лампа {self._ssvep_target_lamp}"
        )
        if elapsed < float(self.cfg.ssvep_block_sec):
            if self._ssvep.can_classify():
                last = self._ssvep_last_msi_log_at
                if last is None or (float(time.time()) - float(last)) >= 2.0:
                    snap = self._ssvep.classify()
                    if snap.classify_allowed:
                        self._log_ssvep_msi_result(
                            dec=snap,
                            mode=str(mode),
                            target_lamp_0idx=self._ssvep_target_lamp,
                            elapsed_s=elapsed,
                            final=False,
                        )
                        self._ssvep_last_msi_log_at = float(time.time())
            return

        # End of block: stop migalka and compute MSI decision from last window
        self._set_ssvep_blackout(False)
        item = self._current_queue_item()
        has_burst_later = any(
            x.kind == "ssvep" and x.ssvep_mode == "burst"
            for x in self._main_queue[self._main_queue_index + 1 :]
        )
        cont_handoff_burst = str(mode) == "continuous" and has_burst_later and self._migalka.is_open()
        if cont_handoff_burst:
            self._ssvep_any_continuous_done = True
            plog.info("последний continuous -> halt + handoff M1 (COM остаётся открыт)")
            try:
                self._migalka.halt_lamps()
                self._migalka.prepare_burst_handoff()
            except Exception as e:
                plog.exc("handoff continuous->burst", e)
        else:
            try:
                self._migalka.stop_and_close()
            except Exception:
                pass
        dec = self._ssvep.classify()
        rec = {
            "mode": str(mode),
            "target_lamp": int(self._ssvep_target_lamp) if self._ssvep_target_lamp is not None else None,
            "winner_1based": dec.winner_1based,
            "winner_0idx": dec.winner_0idx,
            "coef": dec.coef,
            "classify_allowed": bool(dec.classify_allowed),
            "debug": dec.debug,
        }
        self._logger.append_ssvep_block(rec)
        self._logger.write("ssvep_block_end", rec)
        cue_lamp = int(self._ssvep_target_lamp) if self._ssvep_target_lamp is not None else None
        exp_rec = {
            "phase": "main",
            "kind": "ssvep",
            "queue_index": int(self._main_queue_index),
            "ssvep_mode": str(mode),
            "cue_target_lamp_0idx": cue_lamp,
            "cue_target_lamp_1based": int(cue_lamp) + 1 if cue_lamp is not None else None,
            "ground_truth_lamp_0idx": cue_lamp,
            "winner_1based": dec.winner_1based,
            "winner_0idx": dec.winner_0idx,
            "correct": (
                int(dec.winner_0idx) == int(cue_lamp)
                if cue_lamp is not None and dec.winner_0idx is not None
                else None
            ),
            "coef": dec.coef,
            "classify_allowed": bool(dec.classify_allowed),
            "debug": dec.debug,
        }
        self._logger.append_experiment(exp_rec)
        self._log_ssvep_msi_result(
            dec=dec,
            mode=str(mode),
            target_lamp_0idx=cue_lamp,
            elapsed_s=float(self.cfg.ssvep_block_sec),
            final=True,
        )

        if mode == "continuous":
            self._ssvep_block_count_cont += 1
        else:
            self._ssvep_block_count_burst += 1

        self._ssvep_block_started_at = None
        self._ssvep_target_lamp = None
        self._set_state(ProtocolState.Main, detail="ssvep block done")
        self._complete_main_queue_item()

    def _finalize(self) -> None:
        assert self._logger is not None
        self._sync_stim_control(None)
        self.participant_instruction = None
        self.clear_ssvep_display()
        self.status_text = "Завершение: сохранение логов…"
        try:
            self._migalka.stop_and_close()
        except Exception:
            pass
        out_dir = self._logger.finalize(
            stop_payload={
                "reason": "protocol_done",
                "p300_calib_trials": int(self._p300_calib_trials_done),
                "p300_main_trials": int(self._p300_main_trials_done),
                "ssvep_cont_blocks": int(self._ssvep_block_count_cont),
                "ssvep_burst_blocks": int(self._ssvep_block_count_burst),
                "shuffle_seed": int(self._shuffle_seed_used or 0),
            }
        )
        self._logger = None
        self._set_state(ProtocolState.Stopped, detail="finalize")
        self.status_text = f"Готово. Логи: {out_dir}"
        plog.info(f"=== PROTOCOL DONE logs={out_dir} ===")

