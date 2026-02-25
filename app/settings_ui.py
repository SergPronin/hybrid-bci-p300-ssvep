from __future__ import annotations

from typing import Dict, List

from psychopy import gui

from app.config import ExperimentConfig, WindowConfig, GridConfig, ColorConfig


def _config_to_dialog_dict(cfg: ExperimentConfig) -> Dict[str, str]:
    """
    Convert current config to string values for the settings dialog.
    """
    return {
        "Window width": str(cfg.window.size[0]),
        "Window height": str(cfg.window.size[1]),
        "Fullscreen (0/1)": "1" if cfg.window.fullscreen else "0",
        "Target FPS": str(cfg.window.fps),
        "Grid rows": str(cfg.grid.rows),
        "Grid cols": str(cfg.grid.cols),
        "Tile size (norm units)": str(cfg.grid.tile_size),
        "Frequencies (Hz, comma-separated)": ", ".join(
            str(f) for f in cfg.grid.frequencies
        ),
        "Background color (r,g,b in -1..1)": ", ".join(
            str(c) for c in cfg.colors.background
        ),
        "ON color (r,g,b in -1..1)": ", ".join(str(c) for c in cfg.colors.on),
        "OFF color (r,g,b in -1..1)": ", ".join(str(c) for c in cfg.colors.off),
    }


def _parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_tuple3(value: str) -> tuple[float, float, float]:
    parts = _parse_float_list(value)
    if len(parts) != 3:
        raise ValueError("Expected exactly 3 values.")
    return parts[0], parts[1], parts[2]


def ask_user_for_config(base_config: ExperimentConfig) -> ExperimentConfig:
    """
    Show a simple GUI dialog to let the user adjust key experiment parameters.

    If the dialog is cancelled or parsing fails, the base_config is returned.
    """
    fields = _config_to_dialog_dict(base_config)

    dialog = gui.DlgFromDict(
        dictionary=fields,
        title="SSVEP Stimulus Settings",
        order=[
            "Window width",
            "Window height",
            "Fullscreen (0/1)",
            "Target FPS",
            "Grid rows",
            "Grid cols",
            "Tile size (norm units)",
            "Frequencies (Hz, comma-separated)",
            "Background color (r,g,b in -1..1)",
            "ON color (r,g,b in -1..1)",
            "OFF color (r,g,b in -1..1)",
        ],
    )

    if not dialog.OK:
        # User cancelled the dialog; keep defaults.
        return base_config

    try:
        width = int(fields["Window width"])
        height = int(fields["Window height"])
        fullscreen = fields["Fullscreen (0/1)"].strip() == "1"
        fps = int(fields["Target FPS"])

        rows = int(fields["Grid rows"])
        cols = int(fields["Grid cols"])
        tile_size = float(fields["Tile size (norm units)"])
        freqs = _parse_float_list(fields["Frequencies (Hz, comma-separated)"])

        bg = _parse_tuple3(fields["Background color (r,g,b in -1..1)"])
        on = _parse_tuple3(fields["ON color (r,g,b in -1..1)"])
        off = _parse_tuple3(fields["OFF color (r,g,b in -1..1)"])
    except Exception:
        # On any parsing error fall back to base_config to avoid crashing.
        return base_config

    window_cfg = WindowConfig(size=(width, height), fullscreen=fullscreen, fps=fps)
    grid_cfg = GridConfig(
        rows=rows,
        cols=cols,
        tile_size=tile_size,
        frequencies=freqs,
    )
    colors_cfg = ColorConfig(background=bg, on=on, off=off)

    return ExperimentConfig(window=window_cfg, grid=grid_cfg, colors=colors_cfg)

