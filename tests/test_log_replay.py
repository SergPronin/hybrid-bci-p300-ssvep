"""
Replay recorded NDJSON session logs through the epoch-extraction + winner-selection
pipeline and verify that the corrected indexing (lsl_clock_direct) produces
reasonable results vs the broken offset-based method.

Run:  python -m pytest tests/test_log_replay.py -v --tb=short
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project imports (ERP pipeline)
# ---------------------------------------------------------------------------
from p300_analysis.signal_processing import baseline_correction
from p300_analysis.erp_compute import (
    build_averaged_erp,
    compute_corrected_and_integrated,
    compute_winner_metrics,
)
from p300_analysis.marker_parsing import (
    marker_value_to_stim_key,
    parse_trial_target_tile_id,
    stim_key_to_tile_digit,
)
from p300_analysis.constants import EPOCH_DURATION_MS

# ---------------------------------------------------------------------------
LOGS_DIR = Path(__file__).resolve().parent.parent / "input_logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_ndjson(path: Path) -> List[dict]:
    events: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _parse_run_blocks(events: List[dict]) -> List[List[dict]]:
    """Split event list into per-run blocks (by run_start … run_end)."""
    runs: List[List[dict]] = []
    current: List[dict] = []
    for e in events:
        if e.get("event") == "run_start":
            current = [e]
        elif current:
            current.append(e)
            if e.get("event") == "run_end":
                runs.append(current)
                current = []
    if current:
        runs.append(current)
    return runs


def _reconstruct_run(run_events: List[dict]) -> Optional[dict]:
    """
    From a single run's events, reconstruct:
      - EEG buffer (1-D float array, channel 0)
      - EEG timestamps
      - timestamp_ms for each EEG chunk (to simulate lsl_local_clock)
      - Marker list [(marker_ts, stim_key), ...]
      - Target tile id
      - Run params (srate, baseline, window)
    """
    rs = [e for e in run_events if e.get("event") == "run_start"]
    if not rs:
        return None
    d = rs[0]["data"]
    srate: float = d["eeg_stream_srate"]
    baseline_ms: int = d["baseline_ms"]
    window_x_ms: int = d["window_x_ms"]
    window_y_ms: int = d["window_y_ms"]

    # --- reconstruct EEG buffer -----------------------------------------
    eeg_samples: List[float] = []
    eeg_ts: List[float] = []
    # For lsl_clock simulation: record timestamp_ms of LAST eeg_chunk event
    last_eeg_event_unix_s: Optional[float] = None

    # --- reconstruct markers ---------------------------------------------
    markers: List[Tuple[float, str]] = []
    target_id: Optional[int] = None
    first_stim_marker_ts: Optional[float] = None

    # Calibration info (old offset)
    calib_offset: Optional[float] = None
    calib_eeg_last: Optional[float] = None
    calib_marker_ts: Optional[float] = None

    # --- old winner from the log -----------------------------------------
    old_winner_digit: Optional[int] = None
    old_match: Optional[bool] = None

    for e in run_events:
        ev = e.get("event")

        if ev == "eeg_chunk":
            chunk_data = e["data"]
            ts_list = chunk_data["ts"]
            samples = chunk_data["samples"]
            for i, row in enumerate(samples):
                val = row[0] if isinstance(row, list) else float(row)
                eeg_samples.append(float(val))
                eeg_ts.append(float(ts_list[i]))
            last_eeg_event_unix_s = e["timestamp_ms"] / 1000.0

        elif ev == "markers_chunk":
            for m in e["data"]["markers"]:
                mval = m["value"]
                mts = float(m["ts"])
                tid = parse_trial_target_tile_id(mval)
                if tid is not None:
                    target_id = tid
                sk = marker_value_to_stim_key(mval)
                if sk is not None:
                    markers.append((mts, sk))
                    if first_stim_marker_ts is None:
                        first_stim_marker_ts = mts

        elif ev == "time_alignment_calibrated":
            cd = e["data"]
            calib_offset = cd.get("offset") or cd.get("offset_diagnostic")
            calib_eeg_last = cd.get("eeg_last_ts")
            calib_marker_ts = cd.get("first_flash_marker_ts")

        elif ev == "winner_update":
            wd = e["data"]
            old_winner_digit = wd.get("winner_digit")
            old_match = wd.get("match_lsl_cue")

    if not eeg_samples or not markers:
        return None

    buf = np.asarray(eeg_samples, dtype=np.float64)
    ts_arr = np.asarray(eeg_ts, dtype=np.float64)

    return {
        "buf": buf,
        "ts_arr": ts_arr,
        "markers": markers,
        "target_id": target_id,
        "srate": srate,
        "baseline_ms": baseline_ms,
        "window_x_ms": window_x_ms,
        "window_y_ms": window_y_ms,
        "calib_offset": calib_offset,
        "calib_eeg_last": calib_eeg_last,
        "calib_marker_ts": calib_marker_ts,
        "last_eeg_event_unix_s": last_eeg_event_unix_s,
        "first_stim_marker_ts": first_stim_marker_ts,
        "old_winner_digit": old_winner_digit,
        "old_match": old_match,
    }


# ---------------------------------------------------------------------------
# Core: epoch extraction + winner computation
# ---------------------------------------------------------------------------

def extract_epochs_with_method(
    buf: np.ndarray,
    ts_arr: np.ndarray,
    markers: List[Tuple[float, str]],
    srate: float,
    epoch_len: int,
    method: str,
    *,
    calib_offset: Optional[float] = None,
    fallback_offset: Optional[float] = None,
    lsl_to_unix_offset: Optional[float] = None,
    last_eeg_event_unix_s: Optional[float] = None,
    pre_event_s: float = 0.0,
) -> Tuple[Dict[str, List[np.ndarray]], List[int]]:
    """
    Extract epochs from *buf* for each marker.

    method:
      "old_broken"    — t_eff = marker_ts + calib_offset (time_correction ≈ 0)
      "fallback"      — t_eff = marker_ts + fallback_offset (eeg_last - fm)
      "lsl_clock_sim" — simulate lsl_local_clock, compute index directly

    Returns epochs_data and list of start_idx per marker.
    """
    dt_s = 1.0 / srate
    n = len(buf)
    epochs: Dict[str, List[np.ndarray]] = {}
    indices: List[int] = []

    for marker_ts, stim_key in markers:
        if method == "old_broken":
            assert calib_offset is not None
            t_eff = marker_ts + calib_offset
            t0 = float(ts_arr[0])
            i_nom = int(round((t_eff - pre_event_s - t0) / dt_s))
            start_idx = max(0, min(i_nom, n - epoch_len))

        elif method == "fallback":
            assert fallback_offset is not None
            t_eff = marker_ts + fallback_offset
            t0 = float(ts_arr[0])
            i_nom = int(round((t_eff - pre_event_s - t0) / dt_s))
            start_idx = max(0, min(i_nom, n - epoch_len))

        elif method == "lsl_clock_sim":
            assert lsl_to_unix_offset is not None
            assert last_eeg_event_unix_s is not None
            lsl_ref = last_eeg_event_unix_s - lsl_to_unix_offset
            seconds_back = lsl_ref - (marker_ts - pre_event_s)
            start_idx = int(round(n - 1 - seconds_back * srate))
            start_idx = max(0, min(start_idx, n - epoch_len))

        else:
            raise ValueError(f"Unknown method: {method}")

        indices.append(start_idx)
        end_idx = start_idx + epoch_len
        if end_idx <= n:
            epochs.setdefault(stim_key, []).append(buf[start_idx:end_idx].copy())

    return epochs, indices


def compute_winner_from_epochs(
    epochs_data: Dict[str, List[np.ndarray]],
    srate: float,
    epoch_len: int,
    baseline_ms: int,
    window_x_ms: int,
    window_y_ms: int,
) -> Tuple[Optional[int], Optional[str], dict]:
    """Run the full ERP pipeline on extracted epochs and return winner_digit."""
    stim_keys, raw_avg = build_averaged_erp(epochs_data, epoch_len)
    if not stim_keys:
        return None, None, {}
    time_ms = np.arange(epoch_len, dtype=np.float64) * (1000.0 / srate) - baseline_ms
    corrected, integrated, time_crop, wx, wy = compute_corrected_and_integrated(
        raw_avg, time_ms, baseline_ms, window_x_ms, window_y_ms,
    )
    winner_idx, mode, dbg = compute_winner_metrics(
        stim_keys, raw_avg, corrected, time_ms, wx, wy,
    )
    winner_key = stim_keys[winner_idx]
    winner_digit = stim_key_to_tile_digit(winner_key)
    return winner_digit, winner_key, dbg


# ---------------------------------------------------------------------------
# Collect all run data from logs
# ---------------------------------------------------------------------------

def _collect_all_runs() -> List[Tuple[str, dict]]:
    """Return [(label, run_dict), ...] for every run across all log files."""
    if not LOGS_DIR.is_dir():
        return []
    results: List[Tuple[str, dict]] = []
    for p in sorted(LOGS_DIR.glob("*.ndjson")):
        events = load_ndjson(p)
        run_blocks = _parse_run_blocks(events)
        for i, rb in enumerate(run_blocks):
            rd = _reconstruct_run(rb)
            if rd is None:
                continue
            label = f"{p.stem}[run{i}]"
            results.append((label, rd))
    return results


ALL_RUNS = _collect_all_runs()


# ===========================================================================
#  Tests
# ===========================================================================


class TestOldMethodBroken:
    """Verify that the old offset method produced start_idx=0 for every epoch."""

    @pytest.mark.parametrize("label,rd", ALL_RUNS, ids=[r[0] for r in ALL_RUNS])
    def test_all_start_idx_zero(self, label: str, rd: dict):
        if rd["calib_offset"] is None:
            pytest.skip("no calibration data")
        # If the offset is large (fallback was used, not time_correction),
        # this test doesn't apply — skip it.
        if abs(rd["calib_offset"]) > 1.0:
            pytest.skip("run used fallback offset (large), not broken time_correction")
        srate = rd["srate"]
        epoch_len = int(round((rd["baseline_ms"] + EPOCH_DURATION_MS) / (1000.0 / srate))) + 1

        _, indices = extract_epochs_with_method(
            rd["buf"], rd["ts_arr"], rd["markers"],
            srate, epoch_len, method="old_broken",
            calib_offset=rd["calib_offset"],
            pre_event_s=rd["baseline_ms"] / 1000.0,
        )
        # The bug: nearly ALL indices should be 0 (because offset ≈ 0
        # but clocks are in different domains).
        zero_count = sum(1 for i in indices if i == 0)
        total = len(indices)
        zero_frac = zero_count / total if total else 0
        print(f"\n  {label}: {zero_count}/{total} start_idx==0  ({zero_frac:.0%})")
        assert zero_frac > 0.95, (
            f"Expected nearly all start_idx=0 with broken offset, "
            f"got {zero_frac:.0%} zeros"
        )


class TestFallbackOffset:
    """With the fallback offset (eeg_last - fm), indices should spread out."""

    @pytest.mark.parametrize("label,rd", ALL_RUNS, ids=[r[0] for r in ALL_RUNS])
    def test_indices_spread(self, label: str, rd: dict):
        if rd["calib_eeg_last"] is None or rd["calib_marker_ts"] is None:
            pytest.skip("no calibration data")
        srate = rd["srate"]
        epoch_len = int(round((rd["baseline_ms"] + EPOCH_DURATION_MS) / (1000.0 / srate))) + 1
        fallback_offset = rd["calib_eeg_last"] - rd["calib_marker_ts"]

        _, indices = extract_epochs_with_method(
            rd["buf"], rd["ts_arr"], rd["markers"],
            srate, epoch_len, method="fallback",
            fallback_offset=fallback_offset,
            pre_event_s=rd["baseline_ms"] / 1000.0,
        )
        unique_idx = len(set(indices))
        total = len(indices)
        print(f"\n  {label}: {unique_idx} unique indices out of {total} markers")
        print(f"  index range: [{min(indices)}, {max(indices)}]")
        assert unique_idx > 1, "All indices are identical with fallback — still broken"


class TestLslClockSimulation:
    """
    Simulate the NEW lsl_clock_direct method using timestamp_ms from the log
    to approximate pylsl.local_clock(). Compute winners and compare with targets.
    """

    @pytest.mark.parametrize("label,rd", ALL_RUNS, ids=[r[0] for r in ALL_RUNS])
    def test_indices_nonzero_and_spread(self, label: str, rd: dict):
        """Start indices must not all be 0 and must vary across markers."""
        if rd["calib_eeg_last"] is None or rd["calib_marker_ts"] is None:
            pytest.skip("no calibration data")
        if rd["last_eeg_event_unix_s"] is None:
            pytest.skip("no eeg event timestamps")

        srate = rd["srate"]
        epoch_len = int(round((rd["baseline_ms"] + EPOCH_DURATION_MS) / (1000.0 / srate))) + 1
        lsl_to_unix = rd["calib_eeg_last"] - rd["calib_marker_ts"]

        _, indices = extract_epochs_with_method(
            rd["buf"], rd["ts_arr"], rd["markers"],
            srate, epoch_len, method="lsl_clock_sim",
            lsl_to_unix_offset=lsl_to_unix,
            last_eeg_event_unix_s=rd["last_eeg_event_unix_s"],
            pre_event_s=rd["baseline_ms"] / 1000.0,
        )
        unique_idx = len(set(indices))
        zero_count = sum(1 for i in indices if i == 0)
        total = len(indices)
        print(f"\n  {label}: {unique_idx} unique / {total} total, "
              f"zeros={zero_count}, range=[{min(indices)}, {max(indices)}]")
        assert zero_count < total * 0.1, (
            f"Too many zero indices ({zero_count}/{total}); "
            f"lsl_clock simulation may be incorrect"
        )
        assert unique_idx > total * 0.5, (
            f"Not enough index diversity ({unique_idx}/{total})"
        )

    @pytest.mark.parametrize("label,rd", ALL_RUNS, ids=[r[0] for r in ALL_RUNS])
    def test_winner_comparison(self, label: str, rd: dict):
        """
        Extract epochs with lsl_clock_sim, compute the winner, and compare
        with the target. Also show the old (broken) winner for reference.
        """
        if rd["calib_eeg_last"] is None or rd["calib_marker_ts"] is None:
            pytest.skip("no calibration data")
        if rd["last_eeg_event_unix_s"] is None:
            pytest.skip("no eeg event timestamps")
        if rd["target_id"] is None:
            pytest.skip("no target in log")

        srate = rd["srate"]
        epoch_len = int(round((rd["baseline_ms"] + EPOCH_DURATION_MS) / (1000.0 / srate))) + 1
        lsl_to_unix = rd["calib_eeg_last"] - rd["calib_marker_ts"]

        epochs_data, indices = extract_epochs_with_method(
            rd["buf"], rd["ts_arr"], rd["markers"],
            srate, epoch_len, method="lsl_clock_sim",
            lsl_to_unix_offset=lsl_to_unix,
            last_eeg_event_unix_s=rd["last_eeg_event_unix_s"],
            pre_event_s=rd["baseline_ms"] / 1000.0,
        )

        winner_digit, winner_key, dbg = compute_winner_from_epochs(
            epochs_data, srate, epoch_len,
            rd["baseline_ms"], rd["window_x_ms"], rd["window_y_ms"],
        )

        target = rd["target_id"]
        old_winner = rd["old_winner_digit"]
        old_match = rd["old_match"]
        new_match = (winner_digit == target)

        # Signed-mean values for every stimulus (shows P300 signal quality)
        sm = dbg.get("signed_mean_final", [])
        sm_str = "  ".join(f"s{i}={v:+.2f}" for i, v in enumerate(sm))

        print(f"\n  {label}:")
        print(f"    target={target}  old_winner={old_winner} (match={old_match})")
        print(f"    new_winner={winner_digit} (match={new_match})")
        print(f"    signed_mean: {sm_str}")
        print(f"    epochs/stim: { {k: len(v) for k, v in sorted(epochs_data.items())} }")
        print(f"    index range: [{min(indices)}, {max(indices)}], "
              f"unique={len(set(indices))}")

        # We don't assert match because the signal itself might not contain
        # a detectable P300 (single channel, 1-second resolution timestamps).
        # But we DO assert that the pipeline ran and produced a valid winner.
        assert winner_digit is not None, "Pipeline failed to produce a winner"
        assert 0 <= winner_digit <= 8, f"Invalid winner digit: {winner_digit}"


class TestEpochsNotIdentical:
    """
    With the old method, ALL stimuli got the same epoch data (from buf[0:401]).
    With the new method, different stimuli at different times must get
    DIFFERENT epoch data.
    """

    @pytest.mark.parametrize("label,rd", ALL_RUNS, ids=[r[0] for r in ALL_RUNS])
    def test_epochs_differ_across_stimuli(self, label: str, rd: dict):
        if rd["calib_eeg_last"] is None or rd["calib_marker_ts"] is None:
            pytest.skip("no calibration data")
        if rd["last_eeg_event_unix_s"] is None:
            pytest.skip("no eeg event timestamps")

        srate = rd["srate"]
        epoch_len = int(round((rd["baseline_ms"] + EPOCH_DURATION_MS) / (1000.0 / srate))) + 1
        lsl_to_unix = rd["calib_eeg_last"] - rd["calib_marker_ts"]

        # --- old method: all epochs should be identical (when offset ≈ 0) ---
        offset_was_broken = (
            rd["calib_offset"] is not None and abs(rd["calib_offset"]) < 1.0
        )
        old_epochs, _ = extract_epochs_with_method(
            rd["buf"], rd["ts_arr"], rd["markers"],
            srate, epoch_len, method="old_broken",
            calib_offset=rd["calib_offset"] if rd["calib_offset"] is not None else 0.0,
            pre_event_s=rd["baseline_ms"] / 1000.0,
        )
        old_all = []
        for v in old_epochs.values():
            old_all.extend(v)
        if len(old_all) >= 2:
            old_identical = all(np.array_equal(old_all[0], ep) for ep in old_all[1:])
        else:
            old_identical = True

        # --- new method: epochs should vary ---
        new_epochs, _ = extract_epochs_with_method(
            rd["buf"], rd["ts_arr"], rd["markers"],
            srate, epoch_len, method="lsl_clock_sim",
            lsl_to_unix_offset=lsl_to_unix,
            last_eeg_event_unix_s=rd["last_eeg_event_unix_s"],
            pre_event_s=rd["baseline_ms"] / 1000.0,
        )
        new_all = []
        for v in new_epochs.values():
            new_all.extend(v)
        if len(new_all) >= 2:
            new_identical = all(np.array_equal(new_all[0], ep) for ep in new_all[1:])
        else:
            new_identical = True

        print(f"\n  {label}: old_all_identical={old_identical}, "
              f"new_all_identical={new_identical}, offset_was_broken={offset_was_broken}")
        if offset_was_broken:
            assert old_identical, "Expected old epochs to be identical (all start_idx=0)"
        assert not new_identical, "New epochs are still identical — indexing not fixed"
