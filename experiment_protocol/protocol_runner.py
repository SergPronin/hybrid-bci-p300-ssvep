from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from pylsl import StreamInlet, StreamInfo, local_clock as lsl_local_clock
except Exception:  # pragma: no cover
    StreamInlet = object  # type: ignore
    StreamInfo = object  # type: ignore

    def lsl_local_clock() -> float:  # type: ignore
        return time.time()

from experiment_protocol import protocol_log as plog
from experiment_protocol.unified_logger import UnifiedExperimentLogger
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
from ssvep_analysis.online_engine import SSVEPOnlineEngine, SSVEPParams


@dataclass(frozen=True)
class ProtocolConfig:
    output_root: Path
    subject_id: str
    com_port: str
    # Пустые строки — первый найденный поток ЭЭГ при предстарте
    eeg_stream_name: str = ""
    eeg_stream_session_id: str = ""

    p300_trials_per_mode: int = 15
    ssvep_blocks_per_mode: int = 15
    pause_between_experiments_s: float = 2.0

    # Warmup for template_corr: require at least N epochs for cue target
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
    WarmupTemplate = "warmup_template"
    P300_AUC = "p300_auc"
    P300_TEMPLATE = "p300_template"
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

        self._p300_trial_count_auc = 0
        self._p300_trial_count_template = 0
        self._ssvep_block_count_cont = 0
        self._ssvep_block_count_burst = 0

        # True после каждого LSL trial_start|target=N; сбрасываем после записи trial (см. _run_p300_trials).
        self._p300_trial_armed: bool = False
        # True после получения LSL -2|trial_end для текущего trial.
        self._p300_trial_end_seen: bool = False

        self._ssvep_block_started_at: Optional[float] = None
        self._ssvep_target_lamp: Optional[int] = None
        self._p300_external_template_ready: bool = False

        # Неблокирующая пауза перед стартом следующей "единицы эксперимента" (trial/block).
        self._pause_until_wall: Optional[float] = None
        self._pause_status: Optional[str] = None
        self._pause_next_state: Optional[str] = None
        self._pause_next_ssvep_mode: Optional[str] = None
        self._last_state: str = ProtocolState.Idle
        self._eeg_nominal_fs_hz: float = 250.0
        self._ssvep_migalka_retry_at: float = 0.0
        self._ssvep_migalka_fail_streak: int = 0

    def _set_state(self, new_state: str, *, detail: str = "") -> None:
        if new_state != self._last_state:
            plog.state_change(self._last_state, new_state, detail=detail)
            self._last_state = str(new_state)
        self.state = str(new_state)

    def _total_experiments(self) -> int:
        return int(self.cfg.p300_trials_per_mode) * 2 + int(self.cfg.ssvep_blocks_per_mode) * 2

    def _next_experiment_index_1based(self) -> int:
        return (
            int(self._p300_trial_count_auc)
            + int(self._p300_trial_count_template)
            + int(self._ssvep_block_count_cont)
            + int(self._ssvep_block_count_burst)
            + 1
        )

    def _start_pause(self, *, seconds: float, status: str, next_state: str, next_ssvep_mode: Optional[str] = None) -> None:
        sec = max(0.0, float(seconds))
        self._pause_until_wall = float(time.time()) + sec
        self._pause_status = str(status)
        self._pause_next_state = str(next_state)
        self._pause_next_ssvep_mode = str(next_ssvep_mode) if next_ssvep_mode is not None else None
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
            return True

        # Pause finished: apply scheduled transition once.
        next_state = self._pause_next_state
        next_mode = self._pause_next_ssvep_mode
        self._pause_until_wall = None
        self._pause_status = None
        self._pause_next_state = None
        self._pause_next_ssvep_mode = None
        if next_state:
            self._set_state(next_state, detail="pause finished")
            if next_state in (ProtocolState.SSVEP_CONT, ProtocolState.SSVEP_BURST) and next_mode is not None:
                plog.info(f"пауза закончилась — запуск SSVEP migalka, mode={next_mode}")
                self._start_ssvep_block(mode=str(next_mode))
        return False

    @property
    def logger(self) -> Optional[UnifiedExperimentLogger]:
        return self._logger

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
            f"P300={self.cfg.p300_trials_per_mode}x2 SSVEP={self.cfg.ssvep_blocks_per_mode}x2 "
            f"COM={self.cfg.com_port!r} P300_ch(1-based)={ch_disp} "
            f"SSVEP_ch(1-based)={ssvep_ch_disp} migalka_freqs_Hz={list(self.cfg.ssvep_freqs_hz)} ==="
        )
        self._set_state(ProtocolState.Preflight, detail="start()")
        self.status_text = "Предстарт: поиск потоков LSL и COM-порта мигалки…"
        plan = {
            "p300": {
                "modes": [WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR],
                "trials_per_mode": int(self.cfg.p300_trials_per_mode),
                "warmup_target_epochs": int(self.cfg.template_warmup_target_epochs),
            },
            "ssvep": {
                "modes": ["continuous", "burst"],
                "blocks_per_mode": int(self.cfg.ssvep_blocks_per_mode),
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
        )

    def stop(self, *, reason: str = "user_stop") -> None:
        plog.info(f"=== PROTOCOL STOP reason={reason!r} ===")
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

        if self.state == ProtocolState.WarmupTemplate:
            self._warmup_template()
        elif self.state == ProtocolState.P300_AUC:
            self._run_p300_trials(winner_mode=WINNER_MODE_AUC)
        elif self.state == ProtocolState.P300_TEMPLATE:
            self._run_p300_trials(winner_mode=WINNER_MODE_TEMPLATE_CORR)
        elif self.state == ProtocolState.SSVEP_CONT:
            self._run_ssvep_blocks(mode="continuous")
        elif self.state == ProtocolState.SSVEP_BURST:
            self._run_ssvep_blocks(mode="burst")
        elif self.state == ProtocolState.Finalize:
            self._finalize()

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
            self.status_text = (
                f"Предстарт: нет потока маркеров плиток «{BCI_STIM_MARKER_STREAM_NAME}». "
                "Запустите стимулятор (галочка в GUI или run_app.py) и подождите 5–10 с."
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

        # Start with P300 AUC block
        self._reset_for_new_p300_block()
        self._set_state(ProtocolState.P300_AUC, detail="preflight ok")
        self.status_text = (
            f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
            f"P300, режим AUC — ждём старт прогона (маркер trial_start)…"
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
                    if self.state == ProtocolState.P300_AUC:
                        mode = "P300, режим AUC"
                    elif self.state == ProtocolState.P300_TEMPLATE:
                        mode = "P300, сравнение с шаблоном"
                    else:
                        mode = "P300"
                    self.status_text = (
                        f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
                        f"{mode}, целевая плитка={int(tid)}"
                    )
            if self.state in (ProtocolState.P300_AUC, ProtocolState.P300_TEMPLATE, ProtocolState.WarmupTemplate):
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

        if self.state in (ProtocolState.P300_AUC, ProtocolState.P300_TEMPLATE, ProtocolState.WarmupTemplate):
            extracted = self._p300.extract_ready_epochs()
        else:
            extracted = 0
        if extracted and self._logger is not None:
            self._logger.write("p300_epochs_extracted", {"n": int(extracted)})
            # Try to learn template in the background during AUC stage to avoid extra warmup time.
            if self.state == ProtocolState.P300_AUC and not self._p300_external_template_ready:
                if self._p300.try_build_external_template_from_epochs(min_epochs=int(self.cfg.template_warmup_target_epochs)):
                    self._p300_external_template_ready = True
                    self._logger.write(
                        "p300_external_template_ready",
                        {
                            "target_tile_id": int(self._p300.external_template_target_id) if self._p300.external_template_target_id is not None else None,
                            "n_epochs": int(self.cfg.template_warmup_target_epochs),
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

    def _warmup_template(self) -> None:
        assert self._logger is not None
        cue = self._p300.current_cue_target_id
        if cue is None:
            self.status_text = "Подготовка шаблона: ждём целевую плитку в LSL (trial_start)…"
            return
        key = f"стимул_{int(cue)}"
        n = len(self._p300.epochs_data.get(key, []))
        self.status_text = (
            f"Подготовка шаблона: плитка {cue}, эпох {n}/{self.cfg.template_warmup_target_epochs}"
        )
        if n >= int(self.cfg.template_warmup_target_epochs):
            self._logger.write("template_warmup_done", {"target_tile_id": int(cue), "epochs": int(n)})
            self._reset_for_new_p300_block()
            self._set_state(ProtocolState.P300_TEMPLATE, detail="warmup done")
            self.status_text = "P300, сравнение с шаблоном: основной блок запущен."

    def _run_p300_trials(self, *, winner_mode: str) -> None:
        assert self._logger is not None
        if not self._p300_trial_armed:
            wm = _ru_p300_winner_mode(winner_mode)
            self.status_text = (
                f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
                f"P300, {wm} — ждём старт прогона…"
            )
            return
        cue_tid = self._p300.current_cue_target_id
        if cue_tid is None:
            wm = _ru_p300_winner_mode(winner_mode)
            self.status_text = (
                f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
                f"P300, {wm} — нет целевой плитки (ждём маркер trial_start)…"
            )
            return

        decision = self._p300.compute_decision(winner_mode=winner_mode)
        if not decision.can_decide and not self._p300_trial_end_seen:
            wm = _ru_p300_winner_mode(winner_mode)
            self.status_text = (
                f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
                f"P300, {wm} — накопление эпох (мин. {decision.min_epochs_per_class} на класс)…"
            )
            return

        if not decision.can_decide and self._p300_trial_end_seen:
            # trial завершён по времени, но данных/эпох не хватило — фиксируем trial без решения,
            # иначе протокол может никогда не перейти к SSVEP.
            decision = P300Decision(
                can_decide=False,
                min_epochs_per_class=int(decision.min_epochs_per_class),
                winner_idx=None,
                winner_key=None,
                mode_used="no_decision",
                debug={**(decision.debug or {}), "note": "trial_end_without_decision"},
            )

        # Record trial outcome; ждём следующий trial_start (даже если target тот же).
        rec = {
            "winner_mode": str(winner_mode),
            "cue_target_tile_id": int(cue_tid),
            "winner_key": decision.winner_key,
            "mode_used": decision.mode_used,
            "debug": decision.debug,
            "epoch_counts_by_stim": {k: len(v) for k, v in self._p300.epochs_data.items()},
            "trial_end_seen": bool(self._p300_trial_end_seen),
        }
        self._logger.append_p300_trial(rec)
        self._logger.write("trial_decision", rec)

        # Count trial and reset epoch accumulation to avoid leakage across trials
        if winner_mode == WINNER_MODE_AUC:
            self._p300_trial_count_auc += 1
            done = self._p300_trial_count_auc >= int(self.cfg.p300_trials_per_mode)
        else:
            self._p300_trial_count_template += 1
            done = self._p300_trial_count_template >= int(self.cfg.p300_trials_per_mode)

        plog.info(
            f"P300 trial #{self._p300_trial_count_auc if winner_mode == WINNER_MODE_AUC else self._p300_trial_count_template} "
            f"mode={winner_mode} auc={self._p300_trial_count_auc}/{self.cfg.p300_trials_per_mode} "
            f"template={self._p300_trial_count_template}/{self.cfg.p300_trials_per_mode} "
            f"winner={decision.winner_key!r} used={decision.mode_used!r}"
        )

        self._p300.reset(params=self._p300.params)  # preserve params but clear buffers
        self._p300_trial_armed = False
        self._p300_trial_end_seen = False

        if done:
            if winner_mode == WINNER_MODE_AUC:
                self._logger.write("block_done", {"block": "p300_auc", "n_trials": int(self._p300_trial_count_auc)})
                plog.info("=== P300 AUC блок завершён -> P300 template_corr ===")
                # Без отдельного warmup: держим протокол ровно в 60 единиц (15+15+15+15).
                self._reset_for_new_p300_block()
                self._start_pause(
                    seconds=float(self.cfg.pause_between_experiments_s),
                    status=f"Пауза перед экспериментом {self._next_experiment_index_1based()}/{self._total_experiments()} (P300, сравнение с шаблоном)",
                    next_state=ProtocolState.P300_TEMPLATE,
                )
            else:
                self._logger.write("block_done", {"block": "p300_template", "n_trials": int(self._p300_trial_count_template)})
                plog.info("=== P300 template_corr блок завершён -> SSVEP continuous (мигалка) ===")
                self._start_pause(
                    seconds=float(self.cfg.pause_between_experiments_s),
                    status=f"Пауза перед экспериментом {self._next_experiment_index_1based()}/{self._total_experiments()} (ССВП, непрерывный)",
                    next_state=ProtocolState.SSVEP_CONT,
                    next_ssvep_mode="continuous",
                )
        else:
            done_n = self._p300_trial_count_auc if winner_mode == WINNER_MODE_AUC else self._p300_trial_count_template
            wm = _ru_p300_winner_mode(winner_mode)
            self.status_text = (
                f"P300, {wm}: прогон записан ({int(done_n)}/{int(self.cfg.p300_trials_per_mode)}), "
                f"ждём следующий старт…"
            )

    def _on_migalka_event(self, ev: Tuple[int, bool, str]) -> None:
        # Feed burst gate via SSVEP engine
        lamp, is_on, raw = ev
        # Use local clock as lsl_time surrogate; LSL mirror is sent separately.
        self._ssvep.ingest_migalka_marker(lsl_time=float(lsl_local_clock()), value=f"{100+int(lamp)}|{'on' if is_on else 'off'}")
        if self._logger is not None:
            self._logger.write("migalka_led", {"lamp": int(lamp), "on": bool(is_on), "raw": str(raw)})

    def _start_ssvep_block(self, *, mode: str) -> None:
        assert self._logger is not None
        plog.info(f"=== SSVEP block START mode={mode!r} COM={self.cfg.com_port!r} ===")
        active_freqs = tuple(float(x) for x in self.cfg.ssvep_freqs_hz if float(x) > 0.0)
        if not active_freqs:
            active_freqs = tuple(float(x) for x in self.cfg.ssvep_freqs_hz)
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
        n_lamps = max(1, len(active_freqs))
        idx = (self._ssvep_block_count_cont + self._ssvep_block_count_burst) % n_lamps
        self._ssvep_target_lamp = int(idx)
        freq_labels_list = [str(x).replace(",", ".") if float(x) > 0 else "0" for x in self.cfg.ssvep_freqs_hz]
        while len(freq_labels_list) < 6:
            freq_labels_list.append("0")
        freq_labels = tuple(freq_labels_list[:6])
        mcfg = MigalkaConfig(
            port=str(self.cfg.com_port),
            mode=0 if mode == "continuous" else 1,
            freqs=freq_labels,
        )
        plog.info(
            f"SSVEP engine: lamps={n_lamps} freqs_Hz={list(active_freqs)} "
            f"roi={list(self.cfg.ssvep_roi_channels_0idx) or 'ALL'}"
        )
        try:
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
        plog.info(f"мигалка запущена, блок {self.cfg.ssvep_block_sec}s, лампа-цель={self._ssvep_target_lamp}")
        sm = _ru_ssvep_stim_mode(mode)
        self.status_text = (
            f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
            f"ССВП ({sm}) — мигалка включена, 0/{self.cfg.ssvep_block_sec:.0f} с"
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
            # Пауза только между блоками (не первый старт после P300 — его делает _tick_pause).
            n_done = int(self._ssvep_block_count_cont) + int(self._ssvep_block_count_burst)
            if n_done > 0:
                self._start_pause(
                    seconds=float(self.cfg.pause_between_experiments_s),
                    status=f"Пауза перед экспериментом {self._next_experiment_index_1based()}/{self._total_experiments()} (ССВП, {_ru_ssvep_stim_mode(mode)})",
                    next_state=ProtocolState.SSVEP_CONT if mode == "continuous" else ProtocolState.SSVEP_BURST,
                    next_ssvep_mode=str(mode),
                )
                return
            self._start_ssvep_block(mode=str(mode))
            return
        elapsed = float(time.time()) - float(self._ssvep_block_started_at)
        sm = _ru_ssvep_stim_mode(mode)
        self.status_text = (
            f"Эксперимент {self._next_experiment_index_1based()}/{self._total_experiments()}: "
            f"ССВП ({sm}) — {elapsed:.1f}/{self.cfg.ssvep_block_sec:.0f} с, целевая лампа {self._ssvep_target_lamp}"
        )
        if elapsed < float(self.cfg.ssvep_block_sec):
            return

        # End of block: stop migalka and compute MSI decision from last window
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

        if mode == "continuous":
            self._ssvep_block_count_cont += 1
            done = self._ssvep_block_count_cont >= int(self.cfg.ssvep_blocks_per_mode)
        else:
            self._ssvep_block_count_burst += 1
            done = self._ssvep_block_count_burst >= int(self.cfg.ssvep_blocks_per_mode)

        self._ssvep_block_started_at = None
        self._ssvep_target_lamp = None

        if done:
            self._logger.write("block_done", {"block": f"ssvep_{mode}", "n_blocks": int(self._ssvep_block_count_cont if mode == 'continuous' else self._ssvep_block_count_burst)})
            if mode == "continuous":
                self._start_pause(
                    seconds=float(self.cfg.pause_between_experiments_s),
                    status=f"Пауза перед экспериментом {self._next_experiment_index_1based()}/{self._total_experiments()} (ССВП, пакетный)",
                    next_state=ProtocolState.SSVEP_BURST,
                    next_ssvep_mode="burst",
                )
            else:
                plog.info("=== SSVEP burst завершён -> Finalize ===")
                self._set_state(ProtocolState.Finalize, detail="ssvep burst done")
        else:
            # Следующий блок стартуем через паузу (см. ветку self._ssvep_block_started_at is None).
            self.state = ProtocolState.SSVEP_CONT if mode == "continuous" else ProtocolState.SSVEP_BURST

    def _finalize(self) -> None:
        assert self._logger is not None
        self.status_text = "Завершение: сохранение логов…"
        try:
            self._migalka.stop_and_close()
        except Exception:
            pass
        out_dir = self._logger.finalize(
            stop_payload={
                "reason": "protocol_done",
                "p300_auc_trials": int(self._p300_trial_count_auc),
                "p300_template_trials": int(self._p300_trial_count_template),
                "ssvep_cont_blocks": int(self._ssvep_block_count_cont),
                "ssvep_burst_blocks": int(self._ssvep_block_count_burst),
            }
        )
        self._logger = None
        self._set_state(ProtocolState.Stopped, detail="finalize")
        self.status_text = f"Готово. Логи: {out_dir}"
        plog.info(f"=== PROTOCOL DONE logs={out_dir} ===")

