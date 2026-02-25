import colorsys
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class WindowConfig:
    size: Tuple[int, int] = (1200, 900)
    fullscreen: bool = False
    fps: int = 60


@dataclass(frozen=True)
class GridConfig:
    rows: int = 3
    cols: int = 3
    tile_size: float = 0.2  # smaller for 9×9 in norm units
    frequencies: Tuple[float, ...] = ()   # length rows*cols, set in load_config
    tile_colors: Tuple[Tuple[float, float, float], ...] = ()  # length rows*cols


@dataclass(frozen=True)
class P300Config:
    """Minimal P300 oddball settings."""
    target_row: int = 4
    target_col: int = 4
    target_probability: float = 0.2
    interval_frames: int = 90   # one flash every N frames (e.g. 1.5 s at 60 FPS)
    flash_duration_frames: int = 2  # target highlight lasts this many frames


@dataclass(frozen=True)
class ColorConfig:
    background: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    target_flash: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # P300 target highlight
    off_brightness: float = 0.15  # 0..1, dimmer when tile is "off" (SSVEP)


@dataclass(frozen=True)
class ExperimentConfig:
    window: WindowConfig
    grid: GridConfig
    colors: ColorConfig
    p300: P300Config


def _default_frequencies_9x9() -> List[float]:
    """81 frequencies in 8–15 Hz for SSVEP (one per tile)."""
    n = 81
    return [8.0 + (i / (n - 1)) * 7.0 for i in range(n)]


def _default_tile_colors_9x9() -> List[Tuple[float, float, float]]:
    """81 distinct colors (rainbow hue spread)."""
    colors = []
    for i in range(81):
        hue = (i / 81.0) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        colors.append((r * 2 - 1, g * 2 - 1, b * 2 - 1))
    return colors


def load_config() -> ExperimentConfig:
    """
    Minimal configuration for 9×9 hybrid P300/SSVEP grid.
    All parameters can be edited directly in this file.
    """
    window_cfg = WindowConfig()
    grid_rows = 9
    grid_cols = 9
    n_tiles = grid_rows * grid_cols

    frequencies = _default_frequencies_9x9()
    tile_colors = _default_tile_colors_9x9()

    grid_cfg = GridConfig(
        rows=grid_rows,
        cols=grid_cols,
        tile_size=0.08,
        frequencies=tuple(frequencies),
        tile_colors=tuple(tile_colors),
    )

    colors_cfg = ColorConfig()
    p300_cfg = P300Config()

    assert len(grid_cfg.frequencies) == n_tiles
    assert len(grid_cfg.tile_colors) == n_tiles

    return ExperimentConfig(
        window=window_cfg,
        grid=grid_cfg,
        colors=colors_cfg,
        p300=p300_cfg,
    )
