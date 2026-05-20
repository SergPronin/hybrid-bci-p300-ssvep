from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from p300_analysis.constants import EEG_KEEP_SECONDS, EPOCH_RESERVE_MS, MIN_EPOCHS_TO_DECIDE
from p300_analysis.epoch_geometry import EpochGeometry
from p300_analysis.epoch_indexing import resolve_epoch_indices_for_marker
from p300_analysis.erp_compute import (
    build_averaged_erp,
    check_can_decide,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.marker_parsing import marker_value_to_stim_key, parse_trial_target_tile_id
from p300_analysis.signal_processing import bandpass_filter, common_average_reference
from p300_analysis.winner_selection import WINNER_MODE_AUC


@dataclass(frozen=True)
class P300EngineParams:
    baseline_ms: int = 100
    window_x_ms: int = 200
    window_y_ms: int = 600
    artifact_threshold_uv: float = 60.0
    use_car: bool = False
    roi_channels_0idx: Tuple[int, ...] = ()


@dataclass(frozen=True)
class P300Decision:
    can_decide: bool
    min_epochs_per_class: int
    winner_idx: Optional[int]
    winner_key: Optional[str]
    mode_used: str
    debug: Dict[str, Any]


class P300OnlineEngine:
    """Headless online P300 pipeline: ingest LSL EEG+markers, extract epochs, compute winner."""

    def __init__(self) -> None:
        self.params = P300EngineParams()
        self._epoch_geom = EpochGeometry()

        self.eeg_buffer: List[np.ndarray] = []  # each: (n_channels,)
        self.eeg_times: List[float] = []
        self.pending_markers: List[Tuple[float, str]] = []
        self.epochs_data: Dict[str, List[np.ndarray]] = {}

        self._marker_eeg_ts_offset: Optional[float] = None
        self._calib_first_marker_ts: Optional[float] = None
        self._calib_first_marker_lsl_clock: Optional[float] = None
        self._lsl_clock_at_buffer_end: Optional[float] = None

        self.current_cue_target_id: Optional[int] = None

    def reset(self, *, params: Optional[P300EngineParams] = None) -> None:
        if params is not None:
            self.params = params
        self._epoch_geom.reset()
        self.eeg_buffer = []
        self.eeg_times = []
        self.pending_markers = []
        self.epochs_data = {}
        self._marker_eeg_ts_offset = None
        self._calib_first_marker_ts = None
        self._calib_first_marker_lsl_clock = None
        self._lsl_clock_at_buffer_end = None
        self.current_cue_target_id = None

    def ingest_marker_chunk(
        self,
        *,
        marker_chunk: Sequence[Any],
        marker_ts: Sequence[float],
        lsl_local_clock_now: Optional[float] = None,
    ) -> None:
        now_lc = float(lsl_local_clock_now) if lsl_local_clock_now is not None else time.time()
        for sample, ts in zip(marker_chunk, marker_ts):
            cue_tid = parse_trial_target_tile_id(sample)
            if cue_tid is not None:
                self.current_cue_target_id = int(cue_tid)
            stim_key = marker_value_to_stim_key(sample)
            if stim_key is None:
                continue
            tsf = float(ts)
            self.pending_markers.append((tsf, stim_key))
            if self._marker_eeg_ts_offset is None and self._calib_first_marker_ts is None:
                self._calib_first_marker_ts = tsf
                self._calib_first_marker_lsl_clock = now_lc
        if len(self.pending_markers) > 5000:
            self.pending_markers = self.pending_markers[-5000:]

    def ingest_eeg_chunk(
        self,
        *,
        eeg_chunk: np.ndarray,
        eeg_ts: Sequence[float],
        lsl_local_clock_now: Optional[float] = None,
    ) -> None:
        if eeg_chunk is None:
            return
        arr = np.asarray(eeg_chunk, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr_2d = arr.reshape(-1, 1)
        elif arr.ndim == 2:
            arr_2d = arr
        else:
            arr_2d = arr.reshape(arr.shape[0], -1)
        ts_list = [float(t) for t in eeg_ts]
        if not ts_list:
            return

        self.eeg_buffer.extend(arr_2d)
        self.eeg_times.extend(ts_list)
        self._lsl_clock_at_buffer_end = float(lsl_local_clock_now) if lsl_local_clock_now is not None else None

        self._epoch_geom.ensure_template(
            None,  # nominal srate is optional; we estimate from timestamps when possible
            self.eeg_times,
            baseline_ms=int(self.params.baseline_ms),
        )

        # Calibrate marker->LSL local clock offset when first marker and first EEG exist.
        if (
            self._marker_eeg_ts_offset is None
            and self._calib_first_marker_ts is not None
            and self._calib_first_marker_lsl_clock is not None
        ):
            self._marker_eeg_ts_offset = float(self._calib_first_marker_lsl_clock) - float(self._calib_first_marker_ts)
            self._calib_first_marker_ts = None
            self._calib_first_marker_lsl_clock = None

        self._trim_buffers()

    def _trim_buffers(self) -> None:
        """Keep last EEG_KEEP_SECONDS in buffers (best-effort by timestamps)."""
        if not self.eeg_times:
            return
        t_last = float(self.eeg_times[-1])
        cut_t = t_last - float(EEG_KEEP_SECONDS) - float(EPOCH_RESERVE_MS) / 1000.0
        # Find first index >= cut_t
        idx = 0
        for i, t in enumerate(self.eeg_times):
            if float(t) >= cut_t:
                idx = i
                break
        if idx <= 0:
            return
        self.eeg_times = self.eeg_times[idx:]
        self.eeg_buffer = self.eeg_buffer[idx:]
        # Drop pending markers that are too old to be extracted from trimmed buffer
        self.pending_markers = [(mts, sk) for mts, sk in self.pending_markers if float(mts) >= cut_t]

    def extract_ready_epochs(self) -> int:
        """Try converting pending markers into extracted epochs. Returns number extracted now."""
        if (
            self._epoch_geom.epoch_len is None
            or self._epoch_geom.dt_ms is None
            or self._epoch_geom.time_ms_template is None
            or not self.eeg_buffer
        ):
            return 0

        dt_s = float(self._epoch_geom.dt_ms) / 1000.0
        if dt_s <= 0:
            return 0
        srate = 1.0 / dt_s
        el = int(self._epoch_geom.epoch_len)
        buf_len = len(self.eeg_buffer)
        # Prefer LSL local clock ref if available; otherwise use EEG timestamps axis.
        lsl_ref = float(self._lsl_clock_at_buffer_end) if self._lsl_clock_at_buffer_end is not None else float(self.eeg_times[-1])
        time_arr = np.asarray(self.eeg_times, dtype=np.float64)

        buf_2d_raw = np.stack(self.eeg_buffer)
        buf_2d = bandpass_filter(buf_2d_raw, srate)
        if self.params.use_car:
            buf_2d = common_average_reference(buf_2d)

        roi = list(self.params.roi_channels_0idx)
        valid = [c for c in roi if 0 <= int(c) < int(buf_2d.shape[1])] if buf_2d.ndim == 2 else []

        extracted = 0
        new_pending: List[Tuple[float, str]] = []
        pre_event_s = 0.0
        if self._epoch_geom.time_ms_template is not None and self._epoch_geom.time_ms_template.size:
            pre_event_s = max(0.0, -float(self._epoch_geom.time_ms_template[0]) / 1000.0)

        for marker_ts, stim_key in self.pending_markers:
            start_idx, end_idx, wait_more = resolve_epoch_indices_for_marker(
                marker_ts=float(marker_ts),
                buf_len=buf_len,
                srate=srate,
                epoch_len=el,
                lsl_ref=lsl_ref,
                time_arr=time_arr,
                marker_eeg_offset=self._marker_eeg_ts_offset,
                compute_start_index=self._epoch_geom.compute_start_index,
                pre_event_s=pre_event_s,
            )
            if wait_more:
                new_pending.append((marker_ts, stim_key))
                continue
            if start_idx is None or end_idx is None:
                continue
            if buf_2d.ndim == 2 and valid:
                epoch = buf_2d[start_idx:end_idx, :][:, valid]
            elif buf_2d.ndim == 2:
                epoch = buf_2d[start_idx:end_idx, :]
            else:
                epoch = buf_2d[start_idx:end_idx].reshape(-1, 1)
            if epoch.shape[0] != el:
                continue
            self.epochs_data.setdefault(stim_key, []).append(epoch.copy())
            extracted += 1

        self.pending_markers = new_pending
        return extracted

    def compute_decision(self, *, winner_mode: str) -> P300Decision:
        el = self._epoch_geom.epoch_len
        time_ms = self._epoch_geom.time_ms_template
        if el is None or time_ms is None:
            return P300Decision(False, 0, None, None, str(WINNER_MODE_AUC), {"note": "no_epoch_template"})

        stim_keys, raw_averaged, rejected_counts = build_averaged_erp(
            self.epochs_data,
            int(el),
            artifact_threshold_uv=float(self.params.artifact_threshold_uv) if self.params.artifact_threshold_uv > 0 else None,
        )
        if not stim_keys:
            return P300Decision(False, 0, None, None, str(WINNER_MODE_AUC), {"note": "no_stim_keys"})

        corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
            raw_averaged,
            time_ms,
            int(self.params.baseline_ms),
            int(self.params.window_x_ms),
            int(self.params.window_y_ms),
        )
        can_decide, min_n = check_can_decide(stim_keys, self.epochs_data)
        if not can_decide or corrected.size == 0:
            return P300Decision(
                False,
                int(min_n),
                None,
                None,
                str(WINNER_MODE_AUC),
                {
                    "note": "collecting",
                    "min_epochs_per_class": int(min_n),
                    "required": int(MIN_EPOCHS_TO_DECIDE),
                    "rejected_counts": rejected_counts,
                },
            )

        template_window = None
        if self.current_cue_target_id is not None:
            key = f"стимул_{int(self.current_cue_target_id)}"
            if key in stim_keys:
                idx = stim_keys.index(key)
                from p300_analysis.signal_processing import time_window_to_indices

                xi0, xi1 = time_window_to_indices(time_ms, int(wx), int(wy))
                template_window = np.asarray(corrected[idx, xi0:xi1], dtype=np.float64).ravel()

        winner_idx, mode_used, dbg = compute_winner_metrics(
            stim_keys,
            raw_averaged=raw_averaged,
            corrected=corrected,
            time_ms=time_ms,
            window_x_ms=int(wx),
            window_y_ms=int(wy),
            winner_mode=str(winner_mode or WINNER_MODE_AUC),
            template_window=template_window,
        )
        winner_key = stim_keys[winner_idx] if 0 <= int(winner_idx) < len(stim_keys) else None
        dbg["artifact_rejected"] = rejected_counts
        dbg["time_crop_ms"] = [float(x) for x in time_crop]
        dbg["integrated"] = integrated.tolist()
        return P300Decision(True, int(min_n), int(winner_idx), winner_key, str(mode_used), dbg)

