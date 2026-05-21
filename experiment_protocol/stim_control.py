"""Файловый обмен протокол ↔ PsychoPy: когда запускать следующий P300 trial."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


CONTROL_FILENAME = "stim_control.json"


def control_path(session_dir: Path) -> Path:
    return Path(session_dir) / CONTROL_FILENAME


def write_control(session_dir: Path, payload: Dict[str, Any]) -> None:
    p = control_path(session_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body["unix_ms"] = int(time.time() * 1000)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def write_paused(
    session_dir: Path,
    *,
    reason: str = "",
    message: str = "",
    experiment_index: int = 0,
    experiment_total: int = 0,
) -> None:
    write_control(
        session_dir,
        {
            "state": "paused",
            "reason": str(reason),
            "message": str(message),
            "experiment_index": int(experiment_index),
            "experiment_total": int(experiment_total),
        },
    )


def write_trial_request(
    session_dir: Path,
    *,
    target_tile_id: int,
    experiment_index: int,
    experiment_total: int,
    label: str = "P300",
) -> None:
    write_control(
        session_dir,
        {
            "state": "trial",
            "target_tile_id": int(target_tile_id),
            "experiment_index": int(experiment_index),
            "experiment_total": int(experiment_total),
            "label": str(label),
        },
    )


def write_stim_done(session_dir: Path) -> None:
    write_control(session_dir, {"state": "done"})


def read_control(session_dir: Path) -> Optional[Dict[str, Any]]:
    p = control_path(session_dir)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
