"""Manual export of recorded P300 runs to txt/csv/xlsx."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _rows_to_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _rows_to_txt(path: Path, title: str, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


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


def export_run_data(
    *,
    run_data: Dict[str, Any],
    output_path: Path,
    file_format: str,
    include_summary: bool,
    include_markers: bool,
    include_eeg: bool,
    include_epochs: bool,
    include_winners: bool,
) -> List[Path]:
    """Export run data and return created file paths."""
    file_format = file_format.lower().strip()
    created: List[Path] = []
    stem = output_path.with_suffix("")

    summary_rows = _summary_rows(run_data)
    marker_rows = _marker_rows(run_data)
    eeg_rows = _eeg_rows(run_data)
    winner_rows = _winner_rows(run_data)
    epoch_rows = _epoch_rows(run_data)

    if file_format in {"csv", "txt"}:
        ext = ".csv" if file_format == "csv" else ".txt"
        if include_summary:
            p = Path(f"{stem}_summary{ext}")
            if file_format == "csv":
                _rows_to_csv(p, ("field", "value"), summary_rows)
            else:
                _rows_to_txt(p, "Summary", ("field", "value"), summary_rows)
            created.append(p)
        if include_markers:
            p = Path(f"{stem}_markers{ext}")
            if file_format == "csv":
                _rows_to_csv(p, ("ts", "value"), marker_rows)
            else:
                _rows_to_txt(p, "Markers", ("ts", "value"), marker_rows)
            created.append(p)
        if include_eeg:
            max_ch = max((len(x) for x in (run_data.get("eeg_samples") or [])), default=0)
            eeg_header = ["sample_idx", "ts"] + [f"ch_{i+1}" for i in range(max_ch)]
            p = Path(f"{stem}_eeg{ext}")
            if file_format == "csv":
                _rows_to_csv(p, eeg_header, eeg_rows)
            else:
                _rows_to_txt(p, "EEG", eeg_header, eeg_rows)
            created.append(p)
        if include_winners:
            p = Path(f"{stem}_winners{ext}")
            if file_format == "csv":
                _rows_to_csv(
                    p, ("event_seq", "winner_digit", "winner_key", "match_lsl_cue"), winner_rows
                )
            else:
                _rows_to_txt(
                    p, "Winner updates", ("event_seq", "winner_digit", "winner_key", "match_lsl_cue"), winner_rows
                )
            created.append(p)
        if include_epochs:
            p = Path(f"{stem}_epochs{ext}")
            if file_format == "csv":
                _rows_to_csv(p, ("stim_key", "epoch_idx", "time_ms", "value"), epoch_rows)
            else:
                _rows_to_txt(p, "Epochs", ("stim_key", "epoch_idx", "time_ms", "value"), epoch_rows)
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
                ws.append(list(row))

        if include_summary:
            _add_sheet("summary", ("field", "value"), summary_rows)
        if include_markers:
            _add_sheet("markers", ("ts", "value"), marker_rows)
        if include_eeg:
            max_ch = max((len(x) for x in (run_data.get("eeg_samples") or [])), default=0)
            eeg_header = ["sample_idx", "ts"] + [f"ch_{i+1}" for i in range(max_ch)]
            _add_sheet("eeg", eeg_header, eeg_rows)
        if include_winners:
            _add_sheet(
                "winners", ("event_seq", "winner_digit", "winner_key", "match_lsl_cue"), winner_rows
            )
        if include_epochs:
            _add_sheet("epochs", ("stim_key", "epoch_idx", "time_ms", "value"), epoch_rows)

        final_path = output_path if output_path.suffix.lower() == ".xlsx" else output_path.with_suffix(".xlsx")
        _ensure_parent(final_path)
        wb.save(final_path)
        created.append(final_path)
        return created

    raise ValueError(f"Unsupported file format: {file_format}")
