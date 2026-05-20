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

from experiment_protocol.unified_logger import UnifiedExperimentLogger
from p300_analysis.constants import EEG_PULL_MAX_SAMPLES, MARKERS_PULL_MAX_SAMPLES
from p300_analysis.lsl_streams import find_allowed_eeg_streams, resolve_marker_streams, stream_inlet_with_buffer
from p300_analysis.marker_parsing import parse_trial_target_tile_id
from p300_analysis.online_engine import P300EngineParams, P300OnlineEngine
from p300_analysis.winner_selection import WINNER_MODE_AUC, WINNER_MODE_TEMPLATE_CORR
from ssvep_analysis.migalka_serial_controller import MigalkaConfig, MigalkaSerialController
from ssvep_analysis.online_engine import SSVEPOnlineEngine, SSVEPParams


@dataclass(frozen=True)
class ProtocolConfig:
    output_root: Path
    subject_id: str
    com_port: str

    p300_trials_per_mode: int = 15
    ssvep_blocks_per_mode: int = 15

    # Warmup for template_corr: require at least N epochs for cue target
    template_warmup_target_epochs: int = 12

    # SSVEP timing
    ssvep_block_sec: float = 6.0
    ssvep_window_sec: float = 2.0
    ssvep_fs_hz: float = 250.0
    ssvep_freqs_hz: Tuple[float, ...] = (10.0, 12.0, 15.0, 20.0, 8.57, 7.5)


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
        self.status_text = "Idle"

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

        self._ssvep_block_started_at: Optional[float] = None
        self._ssvep_target_lamp: Optional[int] = None

    @property
    def logger(self) -> Optional[UnifiedExperimentLogger]:
        return self._logger

    def start(self) -> None:
        self.state = ProtocolState.Preflight
        self.status_text = "Preflight: поиск LSL и COM…"
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
            },
        }
        self._logger = UnifiedExperimentLogger.open_new(
            output_root=Path(self.cfg.output_root),
            subject_id=str(self.cfg.subject_id),
            protocol_plan=plan,
            start_payload={"note": "protocol_start"},
        )

    def stop(self, *, reason: str = "user_stop") -> None:
        try:
            self._migalka.stop_and_close()
        except Exception:
            pass
        self.state = ProtocolState.Stopped
        self.status_text = f"Stopped: {reason}"
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
        eeg_streams = find_allowed_eeg_streams(timeout=1.0)
        marker_streams = resolve_marker_streams(timeout=1.0, attempts=1)
        if not eeg_streams:
            self.status_text = "Preflight: не найден LSL EEG поток (проверьте EEG/LSL)."
            return
        if not marker_streams:
            self.status_text = "Preflight: не найден LSL Markers поток (проверьте стимулятор/LSL)."
            return
        info_eeg = eeg_streams[0]
        info_mk = marker_streams[0]
        self._inlet_eeg = stream_inlet_with_buffer(info_eeg, buffer_seconds=20)
        self._inlet_markers = stream_inlet_with_buffer(info_mk, buffer_seconds=20)
        # EEG labels from StreamInfo are optional; keep for later storage.
        try:
            labels = []
            # channel labels are not always available; leave empty if missing
            self._logger.set_eeg_channel_labels(labels)
        except Exception:
            pass
        self._logger.write(
            "preflight_ok",
            {
                "eeg_stream": {"name": getattr(info_eeg, "name", lambda: "")(), "type": getattr(info_eeg, "type", lambda: "")()},
                "markers_stream": {"name": getattr(info_mk, "name", lambda: "")(), "type": getattr(info_mk, "type", lambda: "")()},
                "com_port": str(self.cfg.com_port),
            },
        )

        # Start with P300 AUC block
        self._reset_for_new_p300_block()
        self.state = ProtocolState.P300_AUC
        self.status_text = "P300 AUC: ждём trial_start|target=… и накопления эпох."

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
                tid = parse_trial_target_tile_id(sample)
                if tid is not None:
                    self._p300_trial_armed = True
                    self._logger.write("protocol_trial_start_arm", {"cue_target_tile_id": int(tid)})
            self._p300.ingest_marker_chunk(marker_chunk=marker_chunk, marker_ts=marker_ts, lsl_local_clock_now=now_lc)

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

        # extract P300 epochs whenever possible
        extracted = self._p300.extract_ready_epochs()
        if extracted and self._logger is not None:
            self._logger.write("p300_epochs_extracted", {"n": int(extracted)})

    def _reset_for_new_p300_block(self) -> None:
        prof = P300EngineParams(
            baseline_ms=100,
            window_x_ms=550,
            window_y_ms=725,
            artifact_threshold_uv=60.0,
            use_car=False,
            roi_channels_0idx=(),
        )
        self._p300.reset(params=prof)
        self._p300_trial_armed = False

    def _warmup_template(self) -> None:
        assert self._logger is not None
        cue = self._p300.current_cue_target_id
        if cue is None:
            self.status_text = "Warmup: ждём cue target в LSL (trial_start|target=…)."
            return
        key = f"стимул_{int(cue)}"
        n = len(self._p300.epochs_data.get(key, []))
        self.status_text = f"Warmup template: target={cue}, epochs={n}/{self.cfg.template_warmup_target_epochs}"
        if n >= int(self.cfg.template_warmup_target_epochs):
            self._logger.write("template_warmup_done", {"target_tile_id": int(cue), "epochs": int(n)})
            self._reset_for_new_p300_block()
            self.state = ProtocolState.P300_TEMPLATE
            self.status_text = "P300 template_corr: стартовали основной блок."

    def _run_p300_trials(self, *, winner_mode: str) -> None:
        assert self._logger is not None
        if not self._p300_trial_armed:
            self.status_text = f"P300 {winner_mode}: ждём trial_start|target=…"
            return
        cue_tid = self._p300.current_cue_target_id
        if cue_tid is None:
            self.status_text = f"P300 {winner_mode}: нет cue target (ожидаем -1|trial_start|…)"
            return

        decision = self._p300.compute_decision(winner_mode=winner_mode)
        if not decision.can_decide:
            self.status_text = f"P300 {winner_mode}: сбор… min_epochs={decision.min_epochs_per_class}"
            return

        # Record trial outcome; ждём следующий trial_start (даже если target тот же).
        rec = {
            "winner_mode": str(winner_mode),
            "cue_target_tile_id": int(cue_tid),
            "winner_key": decision.winner_key,
            "mode_used": decision.mode_used,
            "debug": decision.debug,
            "epoch_counts_by_stim": {k: len(v) for k, v in self._p300.epochs_data.items()},
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

        self._p300.reset(params=self._p300.params)  # preserve params but clear buffers
        self._p300_trial_armed = False

        if done:
            if winner_mode == WINNER_MODE_AUC:
                self._logger.write("block_done", {"block": "p300_auc", "n_trials": int(self._p300_trial_count_auc)})
                # Warmup before template_corr
                self.state = ProtocolState.WarmupTemplate
                self.status_text = "Warmup template_corr: накопление эталона по cue target…"
            else:
                self._logger.write("block_done", {"block": "p300_template", "n_trials": int(self._p300_trial_count_template)})
                self.state = ProtocolState.SSVEP_CONT
                self._start_ssvep_block(mode="continuous")
        else:
            self.status_text = f"P300 {winner_mode}: trial записан ({self._p300_trial_count_auc if winner_mode == WINNER_MODE_AUC else self._p300_trial_count_template}/{self.cfg.p300_trials_per_mode}), ждём следующий trial_start…"

    def _on_migalka_event(self, ev: Tuple[int, bool, str]) -> None:
        # Feed burst gate via SSVEP engine
        lamp, is_on, raw = ev
        # Use local clock as lsl_time surrogate; LSL mirror is sent separately.
        self._ssvep.ingest_migalka_marker(lsl_time=float(lsl_local_clock()), value=f"{100+int(lamp)}|{'on' if is_on else 'off'}")
        if self._logger is not None:
            self._logger.write("migalka_led", {"lamp": int(lamp), "on": bool(is_on), "raw": str(raw)})

    def _start_ssvep_block(self, *, mode: str) -> None:
        assert self._logger is not None
        self._ssvep.reset(
            params=SSVEPParams(
                fs_hz=float(self.cfg.ssvep_fs_hz),
                window_sec=float(self.cfg.ssvep_window_sec),
                freqs_hz=tuple(float(x) for x in self.cfg.ssvep_freqs_hz),
                mode=str(mode),
            )
        )
        self._ssvep_block_started_at = float(time.time())
        # Pick target lamp index (0..5) round-robin to keep deterministic
        idx = (self._ssvep_block_count_cont + self._ssvep_block_count_burst) % 6
        self._ssvep_target_lamp = int(idx)
        # Activate migalka only for SSVEP blocks
        freq_labels = tuple(str(x).replace(",", ".") for x in self.cfg.ssvep_freqs_hz[:6])
        mcfg = MigalkaConfig(
            port=str(self.cfg.com_port),
            mode=0 if mode == "continuous" else 1,
            freqs=freq_labels if len(freq_labels) == 6 else ("0", "0", "0", "0", "0", "0"),
        )
        self._migalka.open_and_start(mcfg)
        self._logger.write(
            "ssvep_block_start",
            {"mode": str(mode), "target_lamp": int(self._ssvep_target_lamp), "freqs_hz": [float(x) for x in self.cfg.ssvep_freqs_hz]},
        )

    def _run_ssvep_blocks(self, *, mode: str) -> None:
        assert self._logger is not None
        if self._ssvep_block_started_at is None:
            self._start_ssvep_block(mode=mode)
            return
        elapsed = float(time.time()) - float(self._ssvep_block_started_at)
        self.status_text = f"SSVEP {mode}: блок {elapsed:.1f}/{self.cfg.ssvep_block_sec:.0f} c, цель лампа={self._ssvep_target_lamp}"
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
                self.state = ProtocolState.SSVEP_BURST
                self._start_ssvep_block(mode="burst")
            else:
                self.state = ProtocolState.Finalize
        else:
            # Start next block immediately
            self._start_ssvep_block(mode=mode)

    def _finalize(self) -> None:
        assert self._logger is not None
        self.status_text = "Finalize: сохранение логов…"
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
        self.state = ProtocolState.Stopped
        self.status_text = f"Готово. Логи: {out_dir}"

