import random
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
    def __init__(
        self,
        *,
        auto_random_trials: bool = False,
        inter_trial_s: float = 1.0,
        auto_plan_trials: int = 15,
        auto_plan_target_tile_id: int = 4,
        auto_plan_target_repeats: int = 0,
        auto_plan_target_epochs: int = 12,
    ) -> None:
        self.auto_random_trials = bool(auto_random_trials)
        self.inter_trial_s = max(0.0, float(inter_trial_s))
        self.auto_plan_trials = max(0, int(auto_plan_trials))
        self.auto_plan_target_tile_id = int(auto_plan_target_tile_id)
        self.auto_plan_target_repeats = max(0, int(auto_plan_target_repeats))
        self.auto_plan_target_epochs = max(0, int(auto_plan_target_epochs))
        self._auto_trials_started = 0
        self._auto_target_plan: list[int] = []
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
        # Автопротокол: без клика START сразу идём в полный экран и циклы trial с случайной целью.
        self.show_controls = not self.auto_random_trials
        self._auto_pending_first_trial = bool(self.auto_random_trials)
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
        if self.auto_random_trials:
            self.win.mouseVisible = False
            self._auto_target_plan = self._build_auto_target_plan()

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

    def _start_trial_with_target(self, target_tile_id: int) -> None:
        """Один trial: trial_config в LSL (для логов) + start_experiment с фиксированной целью."""
        sequences, _ = self._read_settings()
        if sequences <= 0:
            return
        tgt = int(target_tile_id)
        tgt = max(0, min(len(self.grid.tiles) - 1, tgt))
        self.tb_target.text = str(tgt)
        cfg_payload = (
            f"trial_config|target={tgt};sequences={sequences};"
            f"isi_s={self.controller.isi:.3f};flash_s={self.controller.flash_duration:.3f};"
            f"cue_s={self.controller.cue_duration:.3f};ready_s={self.controller.ready_duration:.3f};"
            f"inter_block_s={self.controller.inter_block_s:.3f};grid={self.grid.size}x{self.grid.size}"
        )
        self.win.callOnFlip(self.controller.lsl.send, -3, cfg_payload)
        self.show_controls = False
        self.win.mouseVisible = False
        self.controller.start_experiment(sequences, target_tile_id=tgt)
        core.wait(0.2)

    def _start_trial_random_target(self) -> None:
        """Авто-цель на каждый trial.

        В первые auto_plan_trials trial используем план целей, чтобы нужная плитка встретилась
        auto_plan_target_repeats раз, но не подряд (для набора шаблона без монотонности).
        После окончания плана — обычный рандом.
        """
        tgt: int
        if self._auto_trials_started < len(self._auto_target_plan):
            tgt = int(self._auto_target_plan[self._auto_trials_started])
        else:
            prev = None
            if self._auto_trials_started > 0:
                prev = int(self._auto_target_plan[-1]) if self._auto_target_plan else None
            tgt = self._rand_target_avoid(prev=prev)
        self._auto_trials_started += 1
        self._start_trial_with_target(tgt)

    def _rand_target_avoid(self, *, prev: int | None) -> int:
        n = len(self.grid.tiles)
        if n <= 1:
            return 0
        tries = 0
        while True:
            x = random.randint(0, n - 1)
            if prev is None or x != int(prev):
                return int(x)
            tries += 1
            if tries > 50:
                # fallback: just pick next different
                return int((int(prev) + 1) % n)

    def _build_auto_target_plan(self) -> list[int]:
        """Plan for first trials: target repeats spaced out, never adjacent.

        Produces a list of length <= auto_plan_trials.
        """
        n_tiles = len(self.grid.tiles)
        if n_tiles <= 0 or self.auto_plan_trials <= 0:
            return []
        trials = int(self.auto_plan_trials)
        tgt = _clamp_int(self.auto_plan_target_tile_id, 0, n_tiles - 1, default=4)
        # How many repeats do we need for a reliable template?
        # Each trial contributes approx. `sequences` target epochs (target flashes once per sequence).
        seq = _parse_int(
            self.tb_seq.text,
            config.DEFAULT_SEQUENCES,
            config.SEQUENCES_MIN,
            config.SEQUENCES_MAX,
        )
        max_non_adjacent = (trials + 1) // 2
        if self.auto_plan_target_repeats > 0:
            reps_needed = int(self.auto_plan_target_repeats)
        else:
            # auto: ensure target_epochs collected within non-adjacent constraint
            target_epochs = int(self.auto_plan_target_epochs)
            if target_epochs <= 0:
                reps_needed = 0
            else:
                # if sequences too small, raise it so we can satisfy both epochs and non-adjacent constraint
                # need: reps <= max_non_adjacent and reps*seq >= target_epochs  => seq >= ceil(target_epochs/max_non_adjacent)
                if max_non_adjacent > 0:
                    min_seq = int(np.ceil(target_epochs / float(max_non_adjacent)))
                else:
                    min_seq = target_epochs
                if seq < min_seq:
                    seq = _clamp_int(min_seq, config.SEQUENCES_MIN, config.SEQUENCES_MAX, default=config.DEFAULT_SEQUENCES)
                    self.tb_seq.text = str(seq)
                reps_needed = int(np.ceil(target_epochs / float(max(1, seq))))
        reps = max(0, min(int(reps_needed), trials, max_non_adjacent))
        if reps <= 0:
            # no special plan
            plan: list[int] = []
            prev: int | None = None
            for _ in range(trials):
                x = self._rand_target_avoid(prev=prev)
                plan.append(x)
                prev = x
            return plan

        # Choose positions for target: spread roughly evenly, avoid adjacency.
        positions: set[int] = set()
        if reps == 1:
            positions.add(trials // 2)
        else:
            step = (trials - 1) / float(reps - 1)
            for i in range(reps):
                positions.add(int(round(i * step)))
        # Fix adjacency if any
        pos_sorted = sorted(positions)
        fixed: list[int] = []
        for p in pos_sorted:
            if fixed and p == fixed[-1] + 1:
                # try shift right, else left
                pr = p + 1
                pl = p - 1
                if pr < trials and pr not in positions and (not fixed or pr != fixed[-1] + 1):
                    p = pr
                elif pl >= 0 and pl not in positions and (not fixed or p != fixed[-1] + 1):
                    p = pl
            fixed.append(p)
        positions = set(fixed)

        # Build plan
        plan: list[int] = []
        prev: int | None = None
        for i in range(trials):
            if i in positions:
                x = tgt
                if prev is not None and int(prev) == int(x):
                    # shouldn't happen; pick non-target
                    x = self._rand_target_avoid(prev=prev)
                plan.append(int(x))
                prev = int(x)
            else:
                # pick random, avoid repeating prev and avoid target if prev was target to prevent adjacency
                avoid_prev = prev
                tries = 0
                while True:
                    x = self._rand_target_avoid(prev=avoid_prev)
                    if prev is not None and int(prev) == int(tgt) and int(x) == int(tgt):
                        tries += 1
                        if tries > 50:
                            x = int((int(tgt) + 1) % n_tiles)
                        else:
                            continue
                    plan.append(int(x))
                    prev = int(x)
                    break
        return plan

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
                self._start_trial_with_target(target_tile_id)
            if self.mouse.isPressedIn(self.stop_button):
                self.controller.stop()
                self.show_controls = True
                self.win.mouseVisible = True
                core.wait(0.2)

    def run(self) -> None:
        while True:
            if self._auto_pending_first_trial:
                self._auto_pending_first_trial = False
                self._start_trial_random_target()
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
                    if self.auto_random_trials:
                        core.wait(self.inter_trial_s)
                        self.controller.stop()
                        self._start_trial_random_target()
                    else:
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
