class Tile:
    def __init__(self, position, size, frequency, color):
        self.position = position
        self.size = size
        self.frequency = frequency
        self.color = color  # (r, g, b) in [-1, 1] for SSVEP "on" state
        self.is_on = False
        self.frames_per_cycle = None
        self.current_frame = 0

    def update(self):
        self.current_frame += 1
        if self.current_frame >= self.frames_per_cycle:
            self.current_frame = 0

        half_cycle = self.frames_per_cycle // 2
        self.is_on = self.current_frame < half_cycle