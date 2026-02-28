import random
from psychopy import visual, core, event
from core.grid import Grid
from core.stimulus_controller import StimulusController

class StimulusApp:
    def __init__(self):
        self.win = visual.Window(size=(1200, 800), color="black", units="pix")
        # self.win = visual.Window(
        #     fullscr=True,
        #     color="black",
        #     units="pix"
        # )

        self.grid = Grid(size=3)
        self.controller = StimulusController(self.grid)

        self.tile_size = 120
        self.spacing = 20
        self.tiles_visual = []
        self.active_colors = {}
        self._create_visual_grid()

        self.start_button = visual.Rect(self.win, width=150, height=50, pos=(-100, -250), fillColor="green")
        self.stop_button = visual.Rect(self.win, width=150, height=50, pos=(100, -250), fillColor="red")
        self.start_text = visual.TextStim(self.win, text="START", pos=(-100, -250), color="black")
        self.stop_text = visual.TextStim(self.win, text="STOP", pos=(100, -250), color="black")

        self.mouse = event.Mouse(win=self.win)
        self.colors = ["yellow", "green", "red", "white"]

        self.freq_label = visual.TextStim(self.win, text="Интервал", pos=(550, 260), color="white", height=20)
        self.freq_slider = visual.Slider(self.win, ticks=(0.02, 5.0), labels=["Быстро", "Медленно"],
                                         pos=(500, 200), size=(200, 20), granularity=0.01,
                                         style="slider", color="white")
        self.freq_slider.markerPos = self.controller.isi
        self.freq_value = visual.TextStim(self.win, text=f"{self.controller.isi:.2f}", pos=(500, 180), color="white", height=20)

        # Flash duration
        self.flash_label = visual.TextStim(self.win, text="Длительность горения", pos=(550, 140), color="white", height=20)
        self.flash_slider = visual.Slider(self.win, ticks=(0.05, 2.0), labels=["Быстро", "Долго"],
                                         pos=(500, 100), size=(200, 20), granularity=0.01,
                                         style="slider", color="white")
        self.flash_slider.markerPos = self.controller.flash_duration
        self.flash_value = visual.TextStim(self.win, text=f"{self.controller.flash_duration:.2f}", pos=(500, 80), color="white", height=20)

    def _create_visual_grid(self):
        offset = (self.grid.size - 1) / 2
        for tile in self.grid.tiles:
            x = (tile.col - offset) * (self.tile_size + self.spacing)
            y = (offset - tile.row) * (self.tile_size + self.spacing)
            rect = visual.Rect(self.win, width=self.tile_size, height=self.tile_size, pos=(x, y), fillColor="gray", lineColor="white")
            self.tiles_visual.append(rect)

    def draw(self):
        for tile, rect in zip(self.grid.tiles, self.tiles_visual):
            if tile.active:
                if tile.id not in self.active_colors:
                    self.active_colors[tile.id] = random.choice(self.colors)
                rect.fillColor = self.active_colors[tile.id]
            else:
                rect.fillColor = "gray"
                if tile.id in self.active_colors:
                    del self.active_colors[tile.id]
            rect.draw()

        self.start_button.draw()
        self.stop_button.draw()
        self.start_text.draw()
        self.stop_text.draw()

        self.freq_label.draw()
        self.freq_slider.draw()
        self.freq_value.text = f"{self.controller.isi:.2f}"
        self.freq_value.draw()

        self.flash_label.draw()
        self.flash_slider.draw()
        self.flash_value.text = f"{self.controller.flash_duration:.2f}"
        self.flash_value.draw()

    def check_buttons(self):
        if self.mouse.getPressed()[0]:
            if self.mouse.isPressedIn(self.start_button):
                self.controller.start()
                core.wait(0.2)
            if self.mouse.isPressedIn(self.stop_button):
                self.controller.stop()
                core.wait(0.2)

    def run(self):
        while True:
            self.check_buttons()

            new_interval = self.freq_slider.getRating()
            if new_interval:
                self.controller.isi = new_interval

            new_flash = self.flash_slider.getRating()
            if new_flash:
                self.controller.flash_duration = new_flash

            event_data = self.controller.update()
            if event_data:
                print(event_data)

            self.draw()
            self.win.flip()

            if "escape" in event.getKeys():
                break

        self.controller.stop()
        self.win.close()
        core.quit()

