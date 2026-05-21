from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO

import numpy as np

LOG_SCHEMA = "hybrid_protocol/v2"


def _normalize_marker_value(sample: Any) -> str:
    if isinstance(sample, (list, tuple)) and sample:
        sample = sample[0]
    if isinstance(sample, (bytes, bytearray)):
        return sample.decode("utf-8", errors="ignore")
    return str(sample)


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


@dataclass(frozen=True)
class SessionPaths:
    session_dir: Path
    events_path: Path
    manifest_path: Path
    eeg_npz_path: Path
    p300_trials_path: Path
    ssvep_blocks_path: Path
    experiments_path: Path


class UnifiedExperimentLogger:
    """One session directory containing the full protocol run (all 60 trials/blocks)."""

    __slots__ = (
        "_dir",
        "_paths",
        "_fh",
        "_session_id",
        "_seq",
        "_closed",
        "_eeg_times",
        "_eeg_data",
        "_eeg_labels",
        "_n_events",
        "_manifest",
        "_p300_trials_fh",
        "_ssvep_blocks_fh",
        "_experiments_fh",
        "_n_experiments",
        "_exp_start_seq",
        "_exp_start_eeg_idx",
        "_exp_start_meta",
        "_exp_markers",
    )

    def __init__(
        self,
        session_dir: Path,
        fh: TextIO,
        *,
        session_id: str,
        paths: SessionPaths,
        manifest: Dict[str, Any],
    ) -> None:
        self._dir = session_dir
        self._paths = paths
        self._fh = fh
        self._session_id = session_id
        self._seq = 0
        self._closed = False
        self._eeg_times: List[float] = []
        self._eeg_data: List[np.ndarray] = []
        self._eeg_labels: List[str] = []
        self._n_events = 0
        self._manifest = manifest
        self._p300_trials_fh = open(paths.p300_trials_path, "a", encoding="utf-8", buffering=1)
        self._ssvep_blocks_fh = open(paths.ssvep_blocks_path, "a", encoding="utf-8", buffering=1)
        self._experiments_fh = open(paths.experiments_path, "a", encoding="utf-8", buffering=1)
        self._n_experiments = 0
        self._exp_start_seq = 0
        self._exp_start_eeg_idx = 0
        self._exp_start_meta: Dict[str, Any] = {}
        self._exp_markers: List[Dict[str, Any]] = []
        (session_dir / "experiments").mkdir(exist_ok=True)

    @property
    def session_dir(self) -> Path:
        return self._dir

    @property
    def paths(self) -> SessionPaths:
        return self._paths

    @staticmethod
    def allocate_session_dir(*, output_root: Path, subject_id: str) -> Path:
        """Создать папку сессии до старта стимулятора (общий stim_control)."""
        output_root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        pid = f"{int(time.time() * 1000) % 1_000_000:06d}"
        session_id = f"session_{stamp}_{pid}"
        session_dir = output_root / f"{session_id}_{subject_id}".strip("_")
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "plots").mkdir(exist_ok=True)
        return session_dir

    @classmethod
    def open_new(
        cls,
        *,
        output_root: Path,
        subject_id: str,
        protocol_plan: Dict[str, Any],
        start_payload: Dict[str, Any],
        session_dir: Optional[Path] = None,
    ) -> "UnifiedExperimentLogger":
        output_root.mkdir(parents=True, exist_ok=True)
        if session_dir is None:
            session_dir = cls.allocate_session_dir(output_root=output_root, subject_id=subject_id)
        else:
            session_dir = Path(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)
            (session_dir / "plots").mkdir(exist_ok=True)
        subj = str(subject_id).strip("_")
        suffix = f"_{subj}" if subj else ""
        if suffix and session_dir.name.endswith(suffix):
            session_id = session_dir.name[: -len(suffix)]
        else:
            session_id = session_dir.name

        paths = SessionPaths(
            session_dir=session_dir,
            events_path=session_dir / "events.ndjson",
            manifest_path=session_dir / "manifest.json",
            eeg_npz_path=session_dir / "eeg.npz",
            p300_trials_path=session_dir / "p300_trials.ndjson",
            ssvep_blocks_path=session_dir / "ssvep_blocks.ndjson",
            experiments_path=session_dir / "experiments.ndjson",
        )
        fh = open(paths.events_path, "a", encoding="utf-8", buffering=1)

        manifest = {
            "schema": LOG_SCHEMA,
            "session_id": session_id,
            "subject_id": subject_id,
            "protocol_plan": _json_safe(protocol_plan),
            "start": _json_safe(start_payload),
            "paths": {
                "events": str(paths.events_path),
                "manifest": str(paths.manifest_path),
                "eeg_npz": str(paths.eeg_npz_path),
                "p300_trials": str(paths.p300_trials_path),
                "ssvep_blocks": str(paths.ssvep_blocks_path),
                "experiments": str(paths.experiments_path),
            },
        }

        logger = cls(session_dir, fh, session_id=session_id, paths=paths, manifest=manifest)
        logger.write("session_start", start_payload)
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
            "session_id": self._session_id,
            "data": _json_safe(data or {}),
        }
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
            self._n_events += 1
        except Exception:
            pass

    def set_eeg_channel_labels(self, labels: Sequence[str]) -> None:
        if self._closed:
            return
        self._eeg_labels = [str(x) for x in labels]

    def append_eeg_chunk(
        self,
        times: Sequence[float],
        samples: np.ndarray,
        *,
        stream: str,
        lsl_local_clock: Optional[float] = None,
    ) -> None:
        """Accumulate full EEG stream for the whole session (for later research analysis)."""
        if self._closed or samples is None:
            return
        arr = np.asarray(samples, dtype=np.float64)
        if arr.size == 0:
            return
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)

        ts = np.asarray(times, dtype=np.float64)
        if ts.size != arr.shape[0]:
            # tolerate imperfect LSL timestamps (rare) using linear interpolation
            if ts.size > 1 and arr.shape[0] > 1:
                ts = np.linspace(float(ts[0]), float(ts[-1]), arr.shape[0])
            else:
                return
        self._eeg_times.extend(ts.tolist())
        self._eeg_data.append(arr)
        self.write(
            "eeg_chunk",
            {
                "stream": str(stream),
                "n_samples": int(arr.shape[0]),
                "n_channels": int(arr.shape[1]),
                "t_first": float(ts[0]),
                "t_last": float(ts[-1]),
                "lsl_local_clock": lsl_local_clock,
                "total_eeg_samples": int(len(self._eeg_times)),
            },
        )

    def append_p300_trial(self, rec: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._p300_trials_fh.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")
            self._p300_trials_fh.flush()
        except Exception:
            pass

    def append_ssvep_block(self, rec: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            self._ssvep_blocks_fh.write(json.dumps(_json_safe(rec), ensure_ascii=False) + "\n")
            self._ssvep_blocks_fh.flush()
        except Exception:
            pass

    def mark_experiment_start(self, **meta: Any) -> None:
        """Начало единицы протокола (trial P300 или блок SSVEP) — для среза EEG и маркеров."""
        if self._closed:
            return
        self._exp_start_seq = int(self._seq)
        self._exp_start_eeg_idx = int(len(self._eeg_times))
        self._exp_start_meta = _json_safe(meta)
        self._exp_markers = []

    def record_marker(self, *, lsl_time: float, sample: Any) -> None:
        if self._closed:
            return
        value = _normalize_marker_value(sample)
        row = {"lsl_time": float(lsl_time), "value": value}
        self._exp_markers.append(row)
        self.write("marker", row)

    def _slice_session_eeg(self) -> Optional[Dict[str, Any]]:
        if not self._eeg_times or not self._eeg_data:
            return None
        start = int(self._exp_start_eeg_idx)
        if start >= len(self._eeg_times):
            return None
        eeg = np.vstack(self._eeg_data)
        times = np.asarray(self._eeg_times, dtype=np.float64)
        n = min(int(times.size), int(eeg.shape[0]))
        times = times[:n]
        eeg = eeg[:n]
        if start >= n:
            return None
        times_s = times[start:]
        eeg_s = eeg[start:]
        labels = list(self._eeg_labels)
        if len(labels) < eeg_s.shape[1]:
            labels.extend(f"Ch{i + 1}" for i in range(len(labels), eeg_s.shape[1]))
        return {"times": times_s, "eeg": eeg_s, "channel_labels": labels}

    def _events_since_experiment_start(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        path = self._paths.events_path
        if not path.is_file():
            return out
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if int(rec.get("seq", 0)) > int(self._exp_start_seq):
                    out.append(rec)
        except Exception:
            pass
        return out

    def _write_experiment_bundle(self, exp_num: int, row: Dict[str, Any]) -> Dict[str, str]:
        exp_dir = self._dir / "experiments" / f"exp_{exp_num:05d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}

        markers_path = exp_dir / "markers.ndjson"
        try:
            with markers_path.open("w", encoding="utf-8") as fh:
                for m in self._exp_markers:
                    fh.write(json.dumps(_json_safe(m), ensure_ascii=False) + "\n")
            paths["markers"] = str(markers_path)
        except Exception:
            pass

        events_path = exp_dir / "events.ndjson"
        evs = self._events_since_experiment_start()
        try:
            with events_path.open("w", encoding="utf-8") as fh:
                for ev in evs:
                    fh.write(json.dumps(_json_safe(ev), ensure_ascii=False) + "\n")
            paths["events"] = str(events_path)
        except Exception:
            pass

        eeg_slice = self._slice_session_eeg()
        if eeg_slice is not None:
            eeg_path = exp_dir / "eeg.npz"
            try:
                np.savez_compressed(
                    eeg_path,
                    times=eeg_slice["times"],
                    eeg=eeg_slice["eeg"],
                    channel_labels=np.array(eeg_slice["channel_labels"], dtype=object),
                )
                paths["eeg_npz"] = str(eeg_path)
            except Exception:
                pass

        bundle_path = exp_dir / "experiment.json"
        bundle = {
            "schema": LOG_SCHEMA,
            "experiment_number": int(exp_num),
            "session_id": self._session_id,
            "started": self._exp_start_meta,
            "markers": list(self._exp_markers),
            "n_markers": int(len(self._exp_markers)),
            "n_events": int(len(evs)),
            **row,
            "artifacts": paths,
        }
        try:
            bundle_path.write_text(
                json.dumps(_json_safe(bundle), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            paths["experiment_json"] = str(bundle_path)
        except Exception:
            pass
        return paths

    def append_experiment(self, rec: Dict[str, Any]) -> int:
        """Одна строка = одна единица протокола (калибровка / P300 / SSVEP) для разбора медиками."""
        if self._closed:
            return self._n_experiments
        self._n_experiments += 1
        exp_num = int(self._n_experiments)
        row = dict(rec)
        row.setdefault("experiment_record_id", exp_num)
        row.setdefault("experiment_number", exp_num)
        artifacts = self._write_experiment_bundle(exp_num, row)
        row["artifacts"] = artifacts
        row["n_markers"] = int(len(self._exp_markers))
        try:
            self._experiments_fh.write(json.dumps(_json_safe(row), ensure_ascii=False) + "\n")
            self._experiments_fh.flush()
        except Exception:
            pass
        self._exp_markers = []
        return exp_num

    def finalize(self, *, stop_payload: Dict[str, Any]) -> Path:
        if self._closed:
            return self._dir
        self.write("session_stop", stop_payload)

        manifest = dict(self._manifest)
        manifest["stop"] = _json_safe(stop_payload)
        manifest["n_events"] = int(self._n_events)
        manifest["n_experiments"] = int(self._n_experiments)
        manifest["experiments_root"] = str(self._dir / "experiments")

        if self._eeg_times and self._eeg_data:
            eeg = np.vstack(self._eeg_data)
            times = np.asarray(self._eeg_times, dtype=np.float64)
            n = min(int(times.size), int(eeg.shape[0]))
            times = times[:n]
            eeg = eeg[:n]
            labels = list(self._eeg_labels)
            if len(labels) < eeg.shape[1]:
                labels.extend(f"Ch{i + 1}" for i in range(len(labels), eeg.shape[1]))
            np.savez_compressed(
                self._paths.eeg_npz_path,
                times=times,
                eeg=eeg,
                channel_labels=np.array(labels, dtype=object),
            )
            manifest["eeg_npz"] = str(self._paths.eeg_npz_path)
            manifest["eeg_samples"] = int(eeg.shape[0])
            manifest["eeg_channels"] = int(eeg.shape[1])
            manifest["eeg_t_span_sec"] = float(times[-1] - times[0]) if n > 1 else 0.0
        else:
            manifest["eeg_npz"] = None

        try:
            self._paths.manifest_path.write_text(
                json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        self._export_experiments_csv()
        self.close()
        return self._dir

    def _export_experiments_csv(self) -> None:
        src = self._paths.experiments_path
        if not src.is_file():
            return
        dst = self._dir / "experiments.csv"
        rows: List[Dict[str, Any]] = []
        try:
            for line in src.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        except Exception:
            return
        if not rows:
            return
        flat: List[Dict[str, Any]] = []
        for r in rows:
            flat.append(
                {
                    "experiment_record_id": r.get("experiment_record_id"),
                    "phase": r.get("phase"),
                    "kind": r.get("kind"),
                    "queue_index": r.get("queue_index"),
                    "cue_target_tile_id": r.get("cue_target_tile_id"),
                    "ground_truth_tile_id": r.get("ground_truth_tile_id"),
                    "auc_winner_tile": (r.get("auc") or {}).get("winner_tile_id"),
                    "template_winner_tile": (r.get("template_corr") or {}).get("winner_tile_id"),
                    "methods_agree": r.get("methods_agree"),
                    "correct_auc": r.get("correct_auc"),
                    "correct_template": r.get("correct_template"),
                    "ssvep_mode": r.get("ssvep_mode"),
                    "cue_target_lamp_1based": r.get("cue_target_lamp_1based"),
                    "winner_1based": r.get("winner_1based"),
                    "correct_ssvep": r.get("correct"),
                    "bundle_dir": (r.get("artifacts") or {}).get("experiment_json"),
                    "n_markers": r.get("n_markers"),
                }
            )
        fieldnames = list(flat[0].keys())
        try:
            with dst.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                w.writerows(flat)
        except Exception:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for h in (self._p300_trials_fh, self._ssvep_blocks_fh, self._experiments_fh, self._fh):
            try:
                h.close()
            except Exception:
                pass

