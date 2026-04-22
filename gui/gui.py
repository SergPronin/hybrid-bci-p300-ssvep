import random
from typing import Dict, Optional
from psychopy import visual, core, event
import config
from core.grid import Grid
from core.stimulus_controller import StimulusController

class StimulusApp:

    def __init__(self) -> None:
        self.win = visual.Window(size=config.WINDOW_SIZE, color=config.WINDOW_COLOR, units='pix', fullscr=True)
        self.grid = Grid(size=config.GRID_SIZE)
        self.controller = StimulusController(
            self.grid,
            flash_duration=0.1,
            isi=0.05,
            cue_duration=config.CUE_DURATION,
            ready_duration=config.READY_DURATION,
            cue_color=config.CUE_COLOR,
            stim_color=config.STIM_COLOR,
        )
        self.show_controls = True
        self._tiles_visual: list = []
        self._tile_texts: list = []
        self._active_colors: Dict[int, str] = {}
        self._build_visual_grid()
        self.start_button = visual.Rect(self.win, width=config.BUTTON_WIDTH, height=config.BUTTON_HEIGHT, pos=config.START_BUTTON_POS, fillColor='green')
        self.stop_button = visual.Rect(self.win, width=config.BUTTON_WIDTH, height=config.BUTTON_HEIGHT, pos=config.STOP_BUTTON_POS, fillColor='red')
        self.start_text = visual.TextStim(self.win, text='START', pos=config.START_BUTTON_POS, color='black')
        self.stop_text = visual.TextStim(self.win, text='STOP', pos=config.STOP_BUTTON_POS, color='black')
        self.mouse = event.Mouse(win=self.win)
        self.freq_label = visual.TextStim(self.win, text='Интервал', pos=config.PANEL_FREQ_LABEL_POS, color='white', height=config.TEXT_HEIGHT)
        self.freq_slider = visual.Slider(self.win, ticks=config.ISI_RANGE, labels=['Быстро', 'Медленно'], pos=config.PANEL_FREQ_SLIDER_POS, size=config.SLIDER_SIZE, granularity=config.SLIDER_GRANULARITY, style='slider', color='white')
        self.freq_slider.markerPos = self.controller.isi
        self.freq_value = visual.TextStim(self.win, text=f'{self.controller.isi:.2f}', pos=config.PANEL_FREQ_VALUE_POS, color='white', height=config.TEXT_HEIGHT)
        self.flash_label = visual.TextStim(self.win, text='Длительность горения', pos=config.PANEL_FLASH_LABEL_POS, color='white', height=config.TEXT_HEIGHT)
        self.flash_slider = visual.Slider(self.win, ticks=config.FLASH_DURATION_RANGE, labels=['Быстро', 'Долго'], pos=config.PANEL_FLASH_SLIDER_POS, size=config.SLIDER_SIZE, granularity=config.SLIDER_GRANULARITY, style='slider', color='white')
        self.flash_slider.markerPos = self.controller.flash_duration
        self.flash_value = visual.TextStim(self.win, text=f'{self.controller.flash_duration:.2f}', pos=config.PANEL_FLASH_VALUE_POS, color='white', height=config.TEXT_HEIGHT)
        self.seq_label = visual.TextStim(self.win, text='Блоки', pos=config.PANEL_SEQ_LABEL_POS, color='white', height=config.TEXT_HEIGHT)
        self.seq_slider = visual.Slider(self.win, ticks=config.SEQUENCES_RANGE, labels=['1', '20'], pos=config.PANEL_SEQ_SLIDER_POS, size=config.SLIDER_SIZE, granularity=1, style='slider', color='white')
        self.seq_slider.markerPos = config.DEFAULT_SEQUENCES
        self.seq_value = visual.TextStim(self.win, text=str(config.DEFAULT_SEQUENCES), pos=config.PANEL_SEQ_VALUE_POS, color='white', height=config.TEXT_HEIGHT)
        self.target_label = visual.TextStim(self.win, text='Цель (0-8)', pos=(450, -100), color='white', height=config.TEXT_HEIGHT)
        self.target_slider = visual.Slider(
            self.win,
            ticks=list(range(len(self.grid.tiles))),
            labels=['0', str(len(self.grid.tiles) - 1)],
            pos=(450, -140),
            size=config.SLIDER_SIZE,
            granularity=1,
            style='slider',
            color='white',
        )
        self.target_slider.markerPos = 0
        self.target_value = visual.TextStim(self.win, text='0', pos=(450, -160), color='white', height=config.TEXT_HEIGHT)
        self.fixation_cross = visual.ShapeStim(self.win, vertices=((0, -config.FIXATION_CROSS_SIZE / 2), (0, config.FIXATION_CROSS_SIZE / 2), (0, 0), (-config.FIXATION_CROSS_SIZE / 2, 0), (config.FIXATION_CROSS_SIZE / 2, 0)), closeShape=False, lineWidth=2, lineColor=config.FIXATION_CROSS_COLOR, pos=(0, 0))

    def _build_visual_grid(self) -> None:
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
            tile_text = visual.TextStim(
                self.win,
                text=str(tile.id),
                pos=(x, y),
                color='white',
                height=config.TILE_SIZE_PX * 0.25,
            )
            self._tiles_visual.append(rect)
            self._tile_texts.append(tile_text)

    def _draw(self) -> None:
        for (tile, rect, tile_text) in zip(self.grid.tiles, self._tiles_visual, self._tile_texts):
            if tile.active:
                if not self.show_controls:
                    if self.controller.get_target_id() == tile.id and self.controller.get_target_color():
                        rect.fillColor = self.controller.get_target_color()
                    else:
                        rect.fillColor = self.controller.get_stim_color()
                else:
                    rect.fillColor = config.TILE_DEFAULT_COLOR
            else:
                rect.fillColor = config.TILE_DEFAULT_COLOR
            rect.draw()
            tile_text.draw()
        if self.show_controls:
            self.start_button.draw()
            self.stop_button.draw()
            self.start_text.draw()
            self.stop_text.draw()
            self.freq_label.draw()
            self.freq_slider.draw()
            self.freq_value.text = f'{self.controller.isi:.2f}'
            self.freq_value.draw()
            self.flash_label.draw()
            self.flash_slider.draw()
            self.flash_value.text = f'{self.controller.flash_duration:.2f}'
            self.flash_value.draw()
            self.seq_label.draw()
            self.seq_slider.draw()
            self.seq_value.text = str(int(self.seq_slider.getRating() or config.DEFAULT_SEQUENCES))
            self.seq_value.draw()
            self.target_label.draw()
            self.target_slider.draw()
            selected_target = int(self.target_slider.getRating() or 0)
            self.target_value.text = str(selected_target)
            self.target_value.draw()
        else:
            self.fixation_cross.draw()

    def _handle_buttons(self) -> None:
        if not self.show_controls:
            return
        if self.mouse.getPressed()[0]:
            if self.mouse.isPressedIn(self.start_button):
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
                selected_target = self.target_slider.getRating()
                target_tile_id = int(selected_target) if selected_target is not None else 0
                self.show_controls = False
                self.win.mouseVisible = False
                self.controller.start_experiment(sequences, target_tile_id=target_tile_id)
                core.wait(0.2)
            if self.mouse.isPressedIn(self.stop_button):
                self.controller.stop()
                self.show_controls = True
                self.win.mouseVisible = True
                core.wait(0.2)

    def run(self) -> None:
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
                if 'tile_id' in event_data:
                    self.win.callOnFlip(self.controller.lsl.send, event_data['tile_id'], event_data['event'])
                    print(event_data)
                elif event_data.get('event') == 'trial_start':
                    target = event_data.get('target')
                    self.win.callOnFlip(self.controller.lsl.send, -1, f'trial_start|target={target}')
                    print(f'TRIAL START: target={target}')
                elif event_data.get('event') == 'trial_end':
                    self.win.callOnFlip(self.controller.lsl.send, -2, 'trial_end')
                    print('TRIAL END')
                    self.controller.stop()
                    self.show_controls = True
                    self.win.mouseVisible = True
            self._draw()
            self.win.flip()
            keys = event.getKeys()
            if 'escape' in keys:
                break
            # Экстренная остановка по Пробелу
            if 'space' in keys and not self.show_controls:
                self.controller.stop()
                self.show_controls = True
                self.win.mouseVisible = True
        self.controller.stop()
        self.win.close()
        core.quit()
