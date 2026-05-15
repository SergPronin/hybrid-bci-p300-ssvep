"""
Подробный лог SSVEP-эксперимента: NDJSON-события + полный поток EEG (NPZ) + manifest.

Каталог по умолчанию: <repo>/ssvep_experiment_logs/run_YYYYMMDD_HHMMSS_<id>/
  - events.ndjson   — все события (построчный JSON)
  - manifest.json   — снимок параметров и итоги
  - eeg.npz         — times (n,), eeg (n, ch), channel_labels
  - plots/          — PNG скриншоты (опционально)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

import numpy as np

LOG_SCHEMA = "ssvep_experiment/v1"


def coef_values(msi: Any, n_freqs: int) -> List[Optional[float]]:
    """Коэффициенты MSI по частотам для JSON."""
    out: List[Optional[float]] = [None] * max(0, n_freqs)
    try:
        c = msi.Coef
    except Exception:
        return out
    if c is None:
        return out
    cnt = getattr(c, "Count", None)
    if cnt is not None:
        n = min(int(cnt), n_freqs)
        for i in range(n):
            try:
                out[i] = float(c[i])
            except Exception:
                out[i] = None
        return out
    try:
        if n_freqs > 0:
            out[0] = float(c)
    except Exception:
        pass
    return out


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    try:
        return float(obj)
    except (TypeError, ValueError):
        return repr(obj)


class SSVEPExperimentLogger:
    """Один каталог на запуск «Начать анализ»; events пишутся сразу, EEG — в память до finalize."""

    __slots__ = (
        "_dir",
        "_events_path",
        "_fh",
        "_run_id",
        "_seq",
        "_closed",
        "_eeg_times",
        "_eeg_data",
        "_n_events",
    )

    def __init__(self, session_dir: Path, fh: TextIO, run_id: str) -> None:
        self._dir = session_dir
        self._events_path = session_dir / "events.ndjson"
        self._fh = fh
        self._run_id = run_id
        self._seq = 0
        self._closed = False
        self._eeg_times: List[float] = []
        self._eeg_data: List[np.ndarray] = []
        self._n_events = 0

    @property
    def session_dir(self) -> Path:
        return self._dir

    @property
    def events_path(self) -> Path:
        return self._events_path

    @classmethod
    def open_new(
        cls,
        *,
        output_root: Path,
        start_payload: Dict[str, Any],
    ) -> "SSVEPExperimentLogger":
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pid = f"{int(time.time() * 1000) % 1_000_000:06d}"
        run_id = f"run_{stamp}_{pid}"
        session_dir = output_root / run_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "plots").mkdir(exist_ok=True)
        events_path = session_dir / "events.ndjson"
        fh = open(events_path, "a", encoding="utf-8", buffering=1)
        logger = cls(session_dir, fh, run_id)
        logger.write("experiment_start", start_payload)
        return logger

    def write(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if self._closed:
            return
        self._seq += 1
        rec = {
            "schema": LOG_SCHEMA,
            "seq": self._seq,
            "event": event,
            "wall_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "unix_ms": int(time.time() * 1000),
            "monotonic_ns": time.monotonic_ns(),
            "run_id": self._run_id,
            "data": _json_safe(data or {}),
        }
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._n_events += 1
        except Exception:
            pass

    def append_eeg_chunk(
        self,
        times: Sequence[float],
        samples: np.ndarray,
        *,
        lsl_local_clock: Optional[float] = None,
    ) -> None:
        """Накопить EEG (samples, channels) с LSL-временами."""
        if self._closed or samples.size == 0:
            return
        arr = np.asarray(samples, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        ts = np.asarray(times, dtype=np.float64)
        if ts.size != arr.shape[0]:
            if ts.size == 1:
                fs_guess = 250.0
                t0 = float(ts[0]) - (arr.shape[0] - 1) / fs_guess
                ts = t0 + np.arange(arr.shape[0], dtype=np.float64) / fs_guess
            elif ts.size > 1 and arr.shape[0] > 1:
                ts = np.linspace(float(ts[0]), float(ts[-1]), arr.shape[0])
            else:
                return
        self._eeg_times.extend(ts.tolist())
        self._eeg_data.append(arr)
        self.write(
            "eeg_chunk",
            {
                "n_samples": int(arr.shape[0]),
                "n_channels": int(arr.shape[1]),
                "t_first": float(ts[0]),
                "t_last": float(ts[-1]),
                "lsl_local_clock": lsl_local_clock,
                "total_eeg_samples": len(self._eeg_times),
            },
        )

    def finalize(
        self,
        *,
        stop_payload: Dict[str, Any],
        channel_labels: Optional[Sequence[str]] = None,
    ) -> Path:
        """Записать manifest.json и eeg.npz, закрыть events."""
        if not self._closed:
            self.write("experiment_stop", stop_payload)
        manifest: Dict[str, Any] = {
            "schema": LOG_SCHEMA,
            "run_id": self._run_id,
            "session_dir": str(self._dir),
            "events_file": str(self._events_path),
            "n_events": self._n_events,
            "stop": _json_safe(stop_payload),
        }
        if self._eeg_times and self._eeg_data:
            eeg = np.vstack(self._eeg_data)
            times = np.asarray(self._eeg_times, dtype=np.float64)
            n = min(len(times), eeg.shape[0])
            times = times[:n]
            eeg = eeg[:n]
            labels = list(channel_labels or [])
            if len(labels) < eeg.shape[1]:
                labels.extend(f"Ch{i + 1}" for i in range(len(labels), eeg.shape[1]))
            npz_path = self._dir / "eeg.npz"
            np.savez_compressed(
                npz_path,
                times=times,
                eeg=eeg,
                channel_labels=np.array(labels, dtype=object),
            )
            manifest["eeg_npz"] = str(npz_path)
            manifest["eeg_samples"] = int(eeg.shape[0])
            manifest["eeg_channels"] = int(eeg.shape[1])
            manifest["eeg_t_span_sec"] = float(times[-1] - times[0]) if n > 1 else 0.0
        else:
            manifest["eeg_npz"] = None
        manifest_path = self._dir / "manifest.json"
        try:
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        self.close()
        return self._dir

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._fh.close()
        except Exception:
            pass

    def plots_dir(self) -> Path:
        return self._dir / "plots"
