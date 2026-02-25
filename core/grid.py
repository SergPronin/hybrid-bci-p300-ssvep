from typing import List
from .tile import Tile


class Grid:
    def __init__(self, size: int = 3):
        self.size = size
        self.tiles: List[Tile] = []
        self._create_grid()

    def _create_grid(self):
        tile_id = 0
        for r in range(self.size):
            for c in range(self.size):
                self.tiles.append(Tile(tile_id, r, c))
                tile_id += 1

    def reset(self):
        for tile in self.tiles:
            tile.active = False