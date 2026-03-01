"""Модель сетки плиток для P300/SSVEP стимуляции."""

from typing import List

from .tile import Tile


class Grid:
    """
    Сетка плиток (ячеек) для визуальной стимуляции.

    Плитки нумеруются по строкам: id = row * size + col.
    """

    def __init__(self, size: int = 3):
        """
        Args:
            size: Размер сетки (size×size плиток).
        """
        self.size = size
        self.tiles: List[Tile] = []
        self._build()

    def _build(self) -> None:
        """Создаёт плитки сетки."""
        tile_id = 0
        for row in range(self.size):
            for col in range(self.size):
                self.tiles.append(Tile(tile_id, row, col))
                tile_id += 1

    def reset(self) -> None:
        """Сбрасывает состояние всех плиток (все неактивны)."""
        for tile in self.tiles:
            tile.active = False
