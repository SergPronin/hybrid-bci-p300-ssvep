"""Графический интерфейс приложения стимуляции (PsychoPy)."""

import random
from typing import Dict

from psychopy import visual, core, event

import config
from core.grid import Grid
from core.stimulus_controller import StimulusController


class StimulusApp:
    """
    Окно приложения: сетка плиток, кнопки START/STOP, слайдеры параметров.
    """

    def __init__(self) -> None:
        self.win = visual.Window(
            size=config.WINDOW_SIZE,
            color=config.WINDOW_COLOR,
            units="pix",
        )
        self.grid = Grid(size=config.GRID_SIZE)
        self.controller = StimulusController(self.grid)

        self._tiles_visual: list = []
        self._active_colors: Dict[int, str] = {}
        self._build_visual_grid()

        self.start_button = visual.Rect(
            self.win,
            width=config.BUTTON_WIDTH,
            height=config.BUTTON_HEIGHT,
            pos=config.START_BUTTON_POS,
            fillColor="green",
        )
        self.stop_button = visual.Rect(
            self.win,
            width=config.BUTTON_WIDTH,
            height=config.BUTTON_HEIGHT,
            pos=config.STOP_BUTTON_POS,
            fillColor="red",
        )
        self.start_text = visual.TextStim(
            self.win, text="START", pos=config.START_BUTTON_POS, color="black"
        )
        self.stop_text = visual.TextStim(
            self.win, text="STOP", pos=config.STOP_BUTTON_POS, color="black"
        )

        self.mouse = event.Mouse(win=self.win)

        self.freq_label = visual.TextStim(
            self.win,
            text="Интервал",
            pos=config.PANEL_FREQ_LABEL_POS,
            color="white",
            height=config.TEXT_HEIGHT,
        )
        self.freq_slider = visual.Slider(
            self.win,
            ticks=config.ISI_RANGE,
            labels=["Быстро", "Медленно"],
            pos=config.PANEL_FREQ_SLIDER_POS,
            size=config.SLIDER_SIZE,
            granularity=config.SLIDER_GRANULARITY,
            style="slider",
            color="white",
        )
        self.freq_slider.markerPos = self.controller.isi
        self.freq_value = visual.TextStim(
            self.win,
            text=f"{self.controller.isi:.2f}",
            pos=config.PANEL_FREQ_VALUE_POS,
            color="white",
            height=config.TEXT_HEIGHT,
        )

        self.flash_label = visual.TextStim(
            self.win,
            text="Длительность горения",
            pos=config.PANEL_FLASH_LABEL_POS,
            color="white",
            height=config.TEXT_HEIGHT,
        )
        self.flash_slider = visual.Slider(
            self.win,
            ticks=config.FLASH_DURATION_RANGE,
            labels=["Быстро", "Долго"],
            pos=config.PANEL_FLASH_SLIDER_POS,
            size=config.SLIDER_SIZE,
            granularity=config.SLIDER_GRANULARITY,
            style="slider",
            color="white",
        )
        self.flash_slider.markerPos = self.controller.flash_duration
        self.flash_value = visual.TextStim(
            self.win,
            text=f"{self.controller.flash_duration:.2f}",
            pos=config.PANEL_FLASH_VALUE_POS,
            color="white",
            height=config.TEXT_HEIGHT,
        )

    def _build_visual_grid(self) -> None:
        """Создаёт визуальные прямоугольники под каждую плитку сетки."""
        offset = (self.grid.size - 1) / 2
        for tile in self.grid.tiles:
            x = (tile.col - offset) * (config.TILE_SIZE_PX + config.TILE_SPACING_PX)
            y = (offset - tile.row) * (config.TILE_SIZE_PX + config.TILE_SPACING_PX)
            rect = visual.Rect(
                self.win,
                width=config.TILE_SIZE_PX,
                height=config.TILE_SIZE_PX,
                pos=(x, y),
                fillColor=config.TILE_DEFAULT_COLOR,
                lineColor=config.TILE_LINE_COLOR,
            )
            self._tiles_visual.append(rect)

    def _draw(self) -> None:
        """Отрисовка кадра: сетка, кнопки, слайдеры."""
        for tile, rect in zip(self.grid.tiles, self._tiles_visual):
            if tile.active:
                if tile.id not in self._active_colors:
                    self._active_colors[tile.id] = random.choice(config.FLASH_COLORS)
                rect.fillColor = self._active_colors[tile.id]
            else:
                rect.fillColor = config.TILE_DEFAULT_COLOR
                if tile.id in self._active_colors:
                    del self._active_colors[tile.id]
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

    def _handle_buttons(self) -> None:
        """Обработка кликов по кнопкам START/STOP."""
        if self.mouse.getPressed()[0]:
            if self.mouse.isPressedIn(self.start_button):
                self.controller.start()
                core.wait(0.2)
            if self.mouse.isPressedIn(self.stop_button):
                self.controller.stop()
                core.wait(0.2)

    def run(self) -> None:
        """Главный цикл приложения."""
        while True:
            self._handle_buttons()

            new_isi = self.freq_slider.getRating()
            if new_isi is not None:
                self.controller.isi = new_isi

            new_flash = self.flash_slider.getRating()
            if new_flash is not None:
                self.controller.flash_duration = new_flash

            event_data = self.controller.update()
            if event_data:
                # Отправка маркера привязана к аппаратному flip — минимизирует джиттер для SSVEP
                self.win.callOnFlip(
                    self.controller.lsl.send,
                    event_data["tile_id"],
                    event_data["event"],
                )
                print(event_data)

            self._draw()
            self.win.flip()

            if "escape" in event.getKeys():
                break

        self.controller.stop()
        self.win.close()
        core.quit()
