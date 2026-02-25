class FlickerController:
    def __init__(self, tiles, fps):
        self.tiles = tiles
        self.fps = fps
        self._calculate_frames()

    def _calculate_frames(self):
        for tile in self.tiles:
            tile.frames_per_cycle = int(self.fps / tile.frequency)

    def update(self):
        for tile in self.tiles:
            tile.update()