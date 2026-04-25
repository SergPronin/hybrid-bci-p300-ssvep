from typing import List, Tuple

from psychopy import core, event, visual

import config
from core.grid import Grid
from core.stimulus_controller import StimulusController

try:
    from psychopy.visual.textbox2 import TextBox2
except Exception:  # разные пути в сборках PsychoPy
    TextBox2 = getattr(visual, "TextBox2", None)  # type: ignore[assignment]
if TextBox2 is None:
    raise ImportError("Требуется psychopy.visual.textbox2.TextBox2 (установите psychopy >= 3)")


def _parse_float(
    s: str,
    default: float,
    lo: float,
    hi: float,
) -> float:
    try:
        v = float(str(s).strip().replace(",", "."))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def _parse_int(s: str, default: int, lo: int, hi: int) -> int:
    try:
        v = int(float(str(s).strip().replace(",", ".")))
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


class StimulusApp:
    def __init__(self) -> None:
        self.win = visual.Window(
            size=config.WINDOW_SIZE,
            color=config.WINDOW_COLOR,
            units="pix",
            fullscr=True,
        )
        self.grid = Grid(size=config.GRID_SIZE)
        self.controller = StimulusController(
            self.grid,
            flash_duration=config.DEFAULT_FLASH_DURATION,
            isi=config.DEFAULT_ISI,
            cue_duration=config.DEFAULT_CUE_S,
            ready_duration=config.DEFAULT_READY_S,
            inter_block_s=config.DEFAULT_INTER_BLOCK_S,
            cue_color=config.CUE_COLOR,
            stim_color=config.STIM_COLOR,
        )
        self.show_controls = True
        self._tiles_visual: list = []
        self._tile_texts: list = []
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
            self.win, text="START", pos=config.START_BUTTON_POS, color="black", height=18
        )
        self.stop_text = visual.TextStim(
            self.win, text="STOP", pos=config.STOP_BUTTON_POS, color="black", height=18
        )
        self.mouse = event.Mouse(win=self.win)

        px = self._right_panel_x()
        y = config.PANEL_FIRST_ROW_Y
        dy = config.PANEL_ROW_DY
        label_x = px + config.PANEL_LABEL_OFFSET

        def row() -> float:
            nonlocal y
            cur = y
            y -= dy
            return cur

        self._labels: List[visual.TextStim] = []
        self._tbs: List[TextBox2] = []

        def add_row(caption: str, val: str) -> TextBox2:
            ry = row()
            self._labels.append(
                visual.TextStim(
                    self.win,
                    text=caption,
                    pos=(label_x, ry),
                    color="white",
                    height=14,
                    alignText="left",
                )
            )
            tb: TextBox2 = TextBox2(
                self.win,
                text=val,
                pos=(px, ry),
                size=(config.PANEL_TB_W, config.PANEL_TB_H),
                units="pix",
                color="white",
                fillColor="#1e1e1e",
                borderColor="#666666",
                font="Arial",
                letterHeight=config.PANEL_LETTER_H,
                editable=True,
            )
            self._tbs.append(tb)
            return tb

        self.tb_cue = add_row("Показ цели (с)", f"{config.DEFAULT_CUE_S:g}")
        self.tb_ready = add_row("Пауза+крест (с)", f"{config.DEFAULT_READY_S:g}")
        self.tb_isi = add_row("ISI (с)", f"{config.DEFAULT_ISI:.2f}")
        self.tb_flash = add_row("Вспышка (с)", f"{config.DEFAULT_FLASH_DURATION:.2f}")
        self.tb_inter = add_row("Между рядами (с)", f"{config.DEFAULT_INTER_BLOCK_S:.2f}")
        self.tb_seq = add_row("Раунды", f"{config.DEFAULT_SEQUENCES}")
        self.tb_target = add_row("Цель 0–8", f"{config.DEFAULT_TARGET_ID}")

        self.hint = visual.TextStim(
            self.win,
            text="Space — остановка сессии  ·  Esc — выход из программы",
            pos=(0, config.OPERATOR_HINT_Y),
            color="#888888",
            height=config.OPERATOR_HINT_H,
        )

        self.fixation_cross = visual.ShapeStim(
            self.win,
            vertices=(
                (0, -config.FIXATION_CROSS_SIZE / 2),
                (0, config.FIXATION_CROSS_SIZE / 2),
                (0, 0),
                (-config.FIXATION_CROSS_SIZE / 2, 0),
                (config.FIXATION_CROSS_SIZE / 2, 0),
            ),
            closeShape=False,
            lineWidth=2,
            lineColor=config.FIXATION_CROSS_COLOR,
            pos=(0, 0),
        )

    def _right_panel_x(self) -> float:
        return config.PANEL_X_FRACTION * (self.win.size[0] * 0.5)

    def _build_visual_grid(self) -> None:
        scale = config.TILE_DISPLAY_SCALE
        tile_size = config.TILE_SIZE_PX * scale
        tile_spacing = config.TILE_SPACING_PX * scale
        offset = (self.grid.size - 1) / 2
        for tile in self.grid.tiles:
            x = (tile.col - offset) * (tile_size + tile_spacing)
            y = (offset - tile.row) * (tile_size + tile_spacing)
            rect = visual.Rect(
                self.win,
                width=tile_size,
                height=tile_size,
                pos=(x, y),
                fillColor=config.TILE_DEFAULT_COLOR,
                lineColor=config.TILE_LINE_COLOR,
            )
            tile_text = visual.TextStim(
                self.win,
                text=str(tile.id),
                pos=(x, y),
                color="white",
                height=tile_size * 0.25,
            )
            self._tiles_visual.append(rect)
            self._tile_texts.append(tile_text)

    def _read_settings(self) -> Tuple[int, int]:
        self.controller.cue_duration = _parse_float(
            self.tb_cue.text,
            config.DEFAULT_CUE_S,
            config.CUE_MIN,
            config.CUE_MAX,
        )
        self.tb_cue.text = f"{self.controller.cue_duration:.3g}"
        self.controller.ready_duration = _parse_float(
            self.tb_ready.text,
            config.DEFAULT_READY_S,
            config.READY_MIN,
            config.READY_MAX,
        )
        self.tb_ready.text = f"{self.controller.ready_duration:.3g}"
        self.controller.isi = _parse_float(
            self.tb_isi.text, config.DEFAULT_ISI, config.ISI_MIN, config.ISI_MAX
        )
        self.tb_isi.text = f"{self.controller.isi:.2f}"
        self.controller.flash_duration = _parse_float(
            self.tb_flash.text,
            config.DEFAULT_FLASH_DURATION,
            config.FLASH_MIN,
            config.FLASH_MAX,
        )
        self.tb_flash.text = f"{self.controller.flash_duration:.2f}"
        self.controller.inter_block_s = _parse_float(
            self.tb_inter.text,
            config.DEFAULT_INTER_BLOCK_S,
            config.INTER_BLOCK_MIN,
            config.INTER_BLOCK_MAX,
        )
        self.tb_inter.text = f"{self.controller.inter_block_s:.2f}"
        seq = _parse_int(
            self.tb_seq.text,
            config.DEFAULT_SEQUENCES,
            config.SEQUENCES_MIN,
            config.SEQUENCES_MAX,
        )
        self.tb_seq.text = str(seq)
        tgt = _parse_int(
            self.tb_target.text,
            config.DEFAULT_TARGET_ID,
            0,
            len(self.grid.tiles) - 1,
        )
        self.tb_target.text = str(tgt)
        return seq, tgt

    def _draw(self) -> None:
        for (tile, rect, tile_text) in zip(
            self.grid.tiles, self._tiles_visual, self._tile_texts
        ):
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
            for lab in self._labels:
                lab.draw()
            for tb in self._tbs:
                tb.draw()
            self.hint.draw()
        else:
            self.fixation_cross.draw()

    def _handle_buttons(self) -> None:
        if not self.show_controls:
            return
        if self.mouse.getPressed()[0]:
            if self.mouse.isPressedIn(self.start_button):
                sequences, target_tile_id = self._read_settings()
                if sequences <= 0:
                    return
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
            event_data = self.controller.update()
            if event_data:
                if "tile_id" in event_data:
                    self.win.callOnFlip(
                        self.controller.lsl.send, event_data["tile_id"], event_data["event"]
                    )
                    print(event_data)
                elif event_data.get("event") == "trial_start":
                    target = event_data.get("target")
                    self.win.callOnFlip(
                        self.controller.lsl.send, -1, f"trial_start|target={target}"
                    )
                    print(f"TRIAL START: target={target}")
                elif event_data.get("event") == "trial_end":
                    self.win.callOnFlip(self.controller.lsl.send, -2, "trial_end")
                    print("TRIAL END")
                    self.controller.stop()
                    self.show_controls = True
                    self.win.mouseVisible = True
            self._draw()
            self.win.flip()
            keys = event.getKeys()
            if "escape" in keys:
                break
            if "space" in keys and not self.show_controls:
                self.controller.stop()
                self.show_controls = True
                self.win.mouseVisible = True
        self.controller.stop()
        self.win.close()
        core.quit()
