from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .tile import Tile


Position = Tuple[float, float]
Color = Tuple[float, float, float]


@dataclass
class Grid:
    """
    Logical grid of SSVEP tiles (e.g. 9×9 for hybrid P300/SSVEP).

    Each tile has position, frequency and base color.
    """

    rows: int
    cols: int
    tile_size: float
    tiles: List[Tile]

    @classmethod
    def create(
        cls,
        rows: int,
        cols: int,
        tile_size: float,
        frequencies: Sequence[float],
        tile_colors: Sequence[Color],
    ) -> "Grid":
        n = rows * cols
        if len(frequencies) != n or len(tile_colors) != n:
            raise ValueError(
                f"Expected {n} frequencies and colors for a {rows}x{cols} grid."
            )

        positions = _generate_normalized_positions(rows, cols)
        tiles: List[Tile] = []
        for i in range(n):
            tiles.append(
                Tile(
                    position=positions[i],
                    size=tile_size,
                    frequency=frequencies[i],
                    color=tile_colors[i],
                )
            )
        return cls(rows=rows, cols=cols, tile_size=tile_size, tiles=tiles)


def _generate_normalized_positions(rows: int, cols: int) -> List[Position]:
    """
    Generate center positions for tiles in PsychoPy "norm" coordinates, but
    without depending on PsychoPy itself.

    The grid spans roughly from -0.5 to 0.5 in both axes by default. This
    keeps the implementation simple while remaining symmetrical around the
    center. More advanced layouts can be added later without changing the
    public API.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("Grid rows and cols must be positive integers.")

    # Positions are uniformly spaced in [-0.5, 0.5].
    def linspace(start: float, stop: float, num: int) -> List[float]:
        if num == 1:
            return [(start + stop) / 2.0]
        step = (stop - start) / (num - 1)
        return [start + i * step for i in range(num)]

    xs = linspace(-0.5, 0.5, cols)
    ys = linspace(0.5, -0.5, rows)  # top to bottom

    positions: List[Position] = []
    for r in range(rows):
        for c in range(cols):
            positions.append((xs[c], ys[r]))

    return positions

