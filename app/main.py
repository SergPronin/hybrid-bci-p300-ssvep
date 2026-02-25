from psychopy import visual, core, event

# =========================
# CONFIG
# =========================
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
FPS = 60

FREQUENCIES = [8, 10, 12, 15]  # Hz
TILE_SIZE = 0.3  # в нормированных координатах
BG_COLOR = [-1, -1, -1]  # черный
ON_COLOR = [1, 1, 1]     # белый
OFF_COLOR = [-1, -1, -1]  # черный


# =========================
# TILE CLASS
# =========================
class Tile:
    def __init__(self, win, position, size, frequency):
        self.win = win
        self.frequency = frequency
        self.position = position
        self.size = size

        # Считаем количество кадров на цикл
        self.frames_per_cycle = int(FPS / self.frequency)
        self.half_cycle = self.frames_per_cycle // 2

        self.frame_counter = 0
        self.is_on = False

        self.rect = visual.Rect(
            win=self.win,
            width=self.size,
            height=self.size,
            pos=self.position,
            fillColor=OFF_COLOR,
            lineColor=OFF_COLOR
        )

    def update(self):
        self.frame_counter += 1

        if self.frame_counter >= self.frames_per_cycle:
            self.frame_counter = 0

        if self.frame_counter < self.half_cycle:
            self.is_on = True
        else:
            self.is_on = False

        self.rect.fillColor = ON_COLOR if self.is_on else OFF_COLOR

    def draw(self):
        self.rect.draw()


# =========================
# CREATE WINDOW
# =========================
win = visual.Window(
    size=(SCREEN_WIDTH, SCREEN_HEIGHT),
    color=BG_COLOR,
    units="norm",
    fullscr=False  # можно поставить True
)

# =========================
# CREATE TILES (2x2 GRID)
# =========================
positions = [
    (-0.5, 0.5),
    (0.5, 0.5),
    (-0.5, -0.5),
    (0.5, -0.5)
]

tiles = []

for pos, freq in zip(positions, FREQUENCIES):
    tile = Tile(win, pos, TILE_SIZE, freq)
    tiles.append(tile)

# =========================
# MAIN LOOP
# =========================
clock = core.Clock()

while True:
    for tile in tiles:
        tile.update()
        tile.draw()

    win.flip()

    if event.getKeys(keyList=["escape"]):
        break

win.close()
core.quit()