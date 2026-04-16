"""Manual export of recorded P300 runs to txt/csv/xlsx."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _round3(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        fv = float(value)
        if not math.isfinite(fv):
            return None
        return round(fv, 3)
    return value


def _rounded_row(row: Sequence[Any]) -> List[Any]:
    return [_round3(x) for x in row]


def _rows_to_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(_rounded_row(row))


def _rows_to_txt(path: Path, title: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in _rounded_row(row)) + "\n")


def _summary_rows(run_data: Dict[str, Any]) -> List[Tuple[str, Any]]:
    summary = run_data.get("summary") or {}
    params = summary.get("analysis_params") or {}
    return [
        ("run_seq", run_data.get("run_seq")),
        ("saved_at_ms", run_data.get("saved_at_ms")),
        ("n_markers", len(run_data.get("markers") or [])),
        ("n_eeg_samples", len(run_data.get("eeg_ts") or [])),
        ("n_winner_updates", len(run_data.get("winner_updates") or [])),
        ("n_epoch_classes", len(run_data.get("epochs_data") or {})),
        ("baseline_ms", params.get("baseline_ms")),
        ("window_x_ms", params.get("window_x_ms")),
        ("window_y_ms", params.get("window_y_ms")),
        ("epochs_after_trial_only", params.get("epochs_after_trial_only")),
        ("ui_winner_tile_id", summary.get("ui_winner_tile_id")),
        ("last_lsl_cue", summary.get("last_lsl_cue")),
        ("match_last_cue_vs_winner", summary.get("match_last_cue_vs_winner")),
    ]


def _marker_rows(run_data: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    rows: List[Tuple[Any, Any]] = []
    for item in run_data.get("markers") or []:
        rows.append((item.get("ts"), item.get("value")))
    return rows


def _eeg_rows(run_data: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    eeg_ts = run_data.get("eeg_ts") or []
    eeg_samples = run_data.get("eeg_samples") or []
    for i, (ts, sample) in enumerate(zip(eeg_ts, eeg_samples)):
        row: List[Any] = [i, ts]
        row.extend(sample if isinstance(sample, list) else [sample])
        rows.append(row)
    return rows


def _winner_rows(run_data: Dict[str, Any]) -> List[Tuple[Any, Any, Any, Any]]:
    rows: List[Tuple[Any, Any, Any, Any]] = []
    for item in run_data.get("winner_updates") or []:
        rows.append(
            (
                item.get("event_seq"),
                item.get("winner_digit"),
                item.get("winner_key"),
                item.get("match_lsl_cue"),
            )
        )
    return rows


def _epoch_rows(run_data: Dict[str, Any]) -> List[Tuple[Any, Any, Any, Any]]:
    rows: List[Tuple[Any, Any, Any, Any]] = []
    time_ms = run_data.get("epoch_time_ms") or []
    epochs_data = run_data.get("epochs_data") or {}
    for stim_key, epochs in epochs_data.items():
        for epoch_idx, epoch_values in enumerate(epochs):
            for sample_idx, value in enumerate(epoch_values):
                t_ms = time_ms[sample_idx] if sample_idx < len(time_ms) else None
                rows.append((stim_key, epoch_idx, t_ms, value))
    return rows


def _epoch_raw_rows(run_data: Dict[str, Any]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    segments = run_data.get("epoch_segments") or []
    for seg_idx, seg in enumerate(segments):
        stim_key = seg.get("stim_key")
        marker_ts = seg.get("marker_ts")
        eeg_ts = seg.get("eeg_ts") or []
        eeg_samples = seg.get("eeg_samples") or []
        for sample_idx, (ts, sample) in enumerate(zip(eeg_ts, eeg_samples)):
            row: List[Any] = [seg_idx, stim_key, marker_ts, sample_idx, ts]
            row.extend(sample if isinstance(sample, list) else [sample])
            rows.append(row)
    return rows


def _stim_index_from_key(stim_key: Any) -> Optional[int]:
    if not isinstance(stim_key, str):
        return None
    if "_" not in stim_key:
        return None
    tail = stim_key.rsplit("_", 1)[-1]
    try:
        return int(tail)
    except Exception:
        return None


def _epoch_raw_rows_for_stim(run_data: Dict[str, Any], stim_index: int) -> List[List[Any]]:
    rows: List[List[Any]] = []
    segments = run_data.get("epoch_segments") or []
    for seg_idx, seg in enumerate(segments):
        stim_key = seg.get("stim_key")
        if _stim_index_from_key(stim_key) != stim_index:
            continue
        marker_ts = seg.get("marker_ts")
        eeg_ts = seg.get("eeg_ts") or []
        eeg_samples = seg.get("eeg_samples") or []
        for sample_idx, (ts, sample) in enumerate(zip(eeg_ts, eeg_samples)):
            row: List[Any] = [seg_idx, stim_key, marker_ts, sample_idx, ts]
            row.extend(sample if isinstance(sample, list) else [sample])
            rows.append(row)
    return rows


def _filter_sample_channels(sample: Any, channels: List[int]) -> List[Any]:
    vals = sample if isinstance(sample, list) else [sample]
    out: List[Any] = []
    for c in channels:
        if 0 <= c < len(vals):
            out.append(vals[c])
    return out


def _filtered_run_data(run_data: Dict[str, Any], selected_channels: List[int] | None) -> Dict[str, Any]:
    if selected_channels is None:
        return run_data
    if not selected_channels:
        raise RuntimeError("Список каналов для экспорта пустой.")
    filtered = dict(run_data)
    filtered["eeg_samples"] = [
        _filter_sample_channels(sample, selected_channels) for sample in (run_data.get("eeg_samples") or [])
    ]
    segs: List[Dict[str, Any]] = []
    for seg in run_data.get("epoch_segments") or []:
        seg_copy = dict(seg)
        seg_copy["eeg_samples"] = [
            _filter_sample_channels(sample, selected_channels) for sample in (seg.get("eeg_samples") or [])
        ]
        segs.append(seg_copy)
    filtered["epoch_segments"] = segs
    filtered["selected_channels"] = list(selected_channels)
    return filtered


def export_run_data(
    *,
    run_data: Dict[str, Any],
    output_path: Path,
    file_format: str,
    stim_index: int,
    selected_channels: List[int] | None = None,
) -> List[Path]:
    """Export EEG rows only for selected stimulus index."""
    run_data = _filtered_run_data(run_data, selected_channels)
    file_format = file_format.lower().strip()
    created: List[Path] = []
    stem = output_path.with_suffix("")

    rows = _epoch_raw_rows_for_stim(run_data, stim_index=stim_index)
    if not rows:
        raise RuntimeError(f"Нет эпох для stim_index={stim_index}.")
    max_ch = max((len(x) for x in (run_data.get("eeg_samples") or [])), default=0)
    if max_ch == 0:
        max_ch = max((len(r) - 5 for r in rows), default=0)
    header = ["segment_idx", "stim_key", "marker_ts", "sample_idx", "ts"] + [
        f"ch_{i+1}" for i in range(max_ch)
    ]

    if file_format in {"csv", "txt"}:
        ext = ".csv" if file_format == "csv" else ".txt"
        p = Path(f"{stem}_stim_{stim_index}{ext}")
        if file_format == "csv":
            _rows_to_csv(p, header, rows)
        else:
            _rows_to_txt(p, f"EEG at flashes for stim {stim_index}", header, rows)
        created.append(p)
        return created

    if file_format == "xlsx":
        try:
            from openpyxl import Workbook
        except Exception as e:  # pragma: no cover - runtime dependency check
            raise RuntimeError(
                "Для экспорта в XLSX установите openpyxl (pip install openpyxl)."
            ) from e

        wb = Workbook()
        wb.remove(wb.active)

        def _add_sheet(name: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
            ws = wb.create_sheet(title=name[:31])
            ws.append(list(header))
            for row in rows:
                ws.append(_rounded_row(row))

        _add_sheet(f"stim_{stim_index}", header, rows)

        final_path = output_path if output_path.suffix.lower() == ".xlsx" else output_path.with_suffix(".xlsx")
        _ensure_parent(final_path)
        wb.save(final_path)
        created.append(final_path)
        return created

    raise ValueError(f"Unsupported file format: {file_format}")
