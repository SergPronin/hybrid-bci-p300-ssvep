"""Графический интерфейс с поддержкой четырёхфазного эксперимента."""
import random
from typing import Dict, Optional

from psychopy import visual, core, event

import config
from core.grid import Grid
from core.stimulus_controller import StimulusController


class StimulusApp:
    def __init__(self) -> None:
        self.win = visual.Window(
            size=config.WINDOW_SIZE,
            color=config.WINDOW_COLOR,
            units="pix",
            fullscr=False,
        )
        self.grid = Grid(size=config.GRID_SIZE)
        self.controller = StimulusController(
            self.grid,
            flash_duration=0.1,
            isi=0.05,
            cue_duration=2.0,
            ready_duration=1.5,
            cue_color="blue",
            stim_color="white",
        )

        # Флаг отображения элементов управления (режим оператора)
        self.show_controls = True

        # Визуальные элементы плиток
        self._tiles_visual: list = []
        self._active_colors: Dict[int, str] = {}
        self._build_visual_grid()

        # Кнопки START / STOP
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

        # Слайдеры и метки
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

        self.seq_label = visual.TextStim(
            self.win,
            text="Блоки",
            pos=config.PANEL_SEQ_LABEL_POS,
            color="white",
            height=config.TEXT_HEIGHT,
        )
        self.seq_slider = visual.Slider(
            self.win,
            ticks=config.SEQUENCES_RANGE,
            labels=["1", "20"],
            pos=config.PANEL_SEQ_SLIDER_POS,
            size=config.SLIDER_SIZE,
            granularity=1,
            style="slider",
            color="white",
        )
        self.seq_slider.markerPos = config.DEFAULT_SEQUENCES
        self.seq_value = visual.TextStim(
            self.win,
            text=str(config.DEFAULT_SEQUENCES),
            pos=config.PANEL_SEQ_VALUE_POS,
            color="white",
            height=config.TEXT_HEIGHT,
        )

        # Фиксационный крест
        self.fixation_cross = visual.ShapeStim(
            self.win,
            vertices=(
                (0, -config.FIXATION_CROSS_SIZE/2),
                (0, config.FIXATION_CROSS_SIZE/2),
                (0, 0),
                (-config.FIXATION_CROSS_SIZE/2, 0),
                (config.FIXATION_CROSS_SIZE/2, 0)
            ),
            closeShape=False,
            lineWidth=2,
            lineColor=config.FIXATION_CROSS_COLOR,
            pos=(0, 0),
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
        """Отрисовка кадра в зависимости от режима."""
        # Определяем цвета плиток
        for tile, rect in zip(self.grid.tiles, self._tiles_visual):
            if tile.active:
                # Если плитка активна, определяем её цвет
                if not self.show_controls:
                    # Режим испытуемого
                    if self.controller.get_target_id() == tile.id and self.controller.get_target_color():
                        # Фаза Cue – целевая плитка синяя
                        rect.fillColor = self.controller.get_target_color()
                    else:
                        # Фаза Stim – обычный цвет вспышки
                        rect.fillColor = self.controller.get_stim_color()
                else:
                    # Режим оператора – плитки серые
                    rect.fillColor = config.TILE_DEFAULT_COLOR
            else:
                rect.fillColor = config.TILE_DEFAULT_COLOR
            rect.draw()

        # Элементы управления только в режиме оператора
        if self.show_controls:
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

            self.seq_label.draw()
            self.seq_slider.draw()
            self.seq_value.text = str(int(self.seq_slider.getRating() or config.DEFAULT_SEQUENCES))
            self.seq_value.draw()
        else:
            # В режиме испытуемого рисуем фиксационный крест
            self.fixation_cross.draw()

    def _handle_buttons(self) -> None:
        """Обработка кликов по кнопкам START/STOP в режиме оператора."""
        if not self.show_controls:
            return

        if self.mouse.getPressed()[0]:
            if self.mouse.isPressedIn(self.start_button):
                # Считываем текущие настройки
                new_isi = self.freq_slider.getRating()
                if new_isi is not None:
                    self.controller.isi = new_isi
                new_flash = self.flash_slider.getRating()
                if new_flash is not None:
                    self.controller.flash_duration = new_flash
                new_seq = self.seq_slider.getRating()
                if new_seq is not None:
                    sequences = int(new_seq)
                else:
                    sequences = config.DEFAULT_SEQUENCES

                # Переключаемся в режим испытуемого и запускаем эксперимент
                self.show_controls = False
                self.controller.start_experiment(sequences)
                core.wait(0.2)

            if self.mouse.isPressedIn(self.stop_button):
                self.controller.stop()
                self.show_controls = True
                core.wait(0.2)

    def run(self) -> None:
        """Главный цикл."""
        while True:
            self._handle_buttons()

            if self.show_controls:
                new_isi = self.freq_slider.getRating()
                if new_isi is not None:
                    self.controller.isi = new_isi
                new_flash = self.flash_slider.getRating()
                if new_flash is not None:
                    self.controller.flash_duration = new_flash

            event_data = self.controller.update()
            if event_data:
                # Проверяем, есть ли ключ 'tile_id' (события on/off)
                if "tile_id" in event_data:
                    # Отправляем маркер двумя аргументами: tile_id и event
                    self.win.callOnFlip(
                        self.controller.lsl.send,
                        event_data["tile_id"],
                        event_data["event"],
                    )
                    print(event_data)
                # Для будущих событий trial_start/trial_end (если появятся)
                elif event_data.get("event") == "trial_start":
                    target = event_data.get("target")
                    # Используем специальный tile_id = -1 для начала раунда
                    self.win.callOnFlip(
                        self.controller.lsl.send,
                        -1,
                        f"trial_start|target={target}",
                    )
                    print(f"TRIAL START: target={target}")
                elif event_data.get("event") == "trial_end":
                    self.win.callOnFlip(
                        self.controller.lsl.send,
                        -2,  # специальный ID для конца раунда
                        "trial_end",
                    )
                    print("TRIAL END")
                    self.controller.stop()
                    self.show_controls = True

            self._draw()
            self.win.flip()

            if "escape" in event.getKeys():
                break

        self.controller.stop()
        self.win.close()
        core.quit()
