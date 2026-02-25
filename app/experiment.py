from __future__ import annotations

import random
from typing import Dict, List, Optional

from psychopy import visual, core, event

from app.config import ExperimentConfig
from infrastructure.logger import get_logger
from stimulus.flicker_controller import FlickerController
from stimulus.grid import Grid
from stimulus.tile import Tile


def _dim_color(rgb: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    """Scale color from [-1,1] toward black by factor (0=black, 1=unchanged)."""
    return (
        rgb[0] * factor,
        rgb[1] * factor,
        rgb[2] * factor,
    )


class Experiment:
    """
    Application layer: owns the PsychoPy window and main loop.

    It coordinates:
    - configuration loading
    - grid and flicker setup
    - rendering and input handling
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = get_logger(__name__)

        self.window: visual.Window | None = None
        self.grid: Grid | None = None
        self.flicker: FlickerController | None = None

        self._tile_visuals: Dict[Tile, visual.Rect] = {}
        self._info_text: visual.TextStim | None = None
        self._running: bool = False
        # P300 oddball state
        self._p300_current_target: Optional[Tile] = None
        self._p300_fixed_target: Optional[Tile] = None
        self._p300_frames_until_next: int = 0
        self._p300_frames_visible: int = 0

    def setup(self) -> None:
        self._create_window()
        self._create_grid_and_flicker()
        self._create_visuals()
        self._log_frame_rate_estimate()

    def run(self) -> None:
        """
        Run the experiment until the user presses ESC.
        """
        self.setup()

        assert self.window is not None
        assert self.grid is not None
        assert self.flicker is not None

        clock = core.Clock()
        frame_count = 0

        self.logger.info(
            "Starting main hybrid P300/SSVEP loop. "
            "Controls: SPACE=start/stop, ESC=quit."
        )

        while True:
            frame_count += 1

            keys: List[str] = event.getKeys()
            if "escape" in keys:
                self.logger.info("ESC pressed. Exiting experiment loop.")
                break

            if "space" in keys:
                self._running = not self._running
                self.logger.info("Toggled running state to %s", self._running)

            if self._running:
                self.flicker.update()
                self._update_p300()

            # Draw current visual state
            self._draw_tiles()
            self._draw_ui_overlay()

            # Flip is the only timing primitive used for flicker
            self.window.flip()

            # Periodically log approximate FPS
            if frame_count > 0 and frame_count % self.config.window.fps == 0:
                elapsed = clock.getTime()
                if elapsed > 0:
                    fps_estimate = frame_count / elapsed
                    self.logger.debug(f"Approximate FPS: {fps_estimate:.2f}")

        self._shutdown()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _create_window(self) -> None:
        cfg = self.config.window
        colors = self.config.colors

        self.logger.info(
            f"Creating window {cfg.size}, fullscreen={cfg.fullscreen}, "
            f"target FPS={cfg.fps}"
        )

        self.window = visual.Window(
            size=cfg.size,
            fullscr=cfg.fullscreen,
            color=colors.background,
            units="norm",
        )
        self._info_text = visual.TextStim(
            win=self.window,
            text="",
            pos=(0, -0.9),
            height=0.05,
            color=(0.8, 0.8, 0.8),
            alignText="center",
        )

    def _create_grid_and_flicker(self) -> None:
        cfg = self.config.grid
        fps = self.config.window.fps

        self.logger.info(
            f"Creating {cfg.rows}x{cfg.cols} grid (P300+SSVEP)."
        )

        self.grid = Grid.create(
            rows=cfg.rows,
            cols=cfg.cols,
            tile_size=cfg.tile_size,
            frequencies=list(cfg.frequencies),
            tile_colors=list(cfg.tile_colors),
        )
        self.flicker = FlickerController(self.grid.tiles, fps=fps)
        self._p300_frames_until_next = self.config.p300.interval_frames
        self._p300_frames_visible = 0
        self._p300_current_target = None

        # Fixed P300 target tile (rare target)
        p = self.config.p300
        idx = p.target_row * cfg.cols + p.target_col
        if 0 <= idx < len(self.grid.tiles):
            self._p300_fixed_target = self.grid.tiles[idx]
        else:
            self._p300_fixed_target = None

    def _create_visuals(self) -> None:
        assert self.window is not None
        assert self.grid is not None
        for tile in self.grid.tiles:
            rect = visual.Rect(
                win=self.window,
                width=self.grid.tile_size,
                height=self.grid.tile_size,
                pos=tile.position,
                fillColor=tile.color,
                lineColor=tile.color,
            )
            self._tile_visuals[tile] = rect

    def _update_p300(self) -> None:
        """
        Simple oddball: с заданной вероятностью выбираем редкую цель,
        иначе случайную не-цель. Вспышка держится несколько кадров.
        """
        p = self.config.p300
        if self._p300_frames_visible > 0:
            self._p300_frames_visible -= 1
            if self._p300_frames_visible == 0:
                self._p300_current_target = None
            return

        self._p300_frames_until_next -= 1
        if self._p300_frames_until_next > 0:
            return

        self._p300_frames_until_next = p.interval_frames

        if self._p300_fixed_target is None:
            return

        # Выбираем цель: редкая фиксированная или одна из остальных
        if random.random() < p.target_probability:
            self._p300_current_target = self._p300_fixed_target
        else:
            others = [t for t in self.grid.tiles if t is not self._p300_fixed_target]
            if others:
                self._p300_current_target = random.choice(others)
            else:
                self._p300_current_target = self._p300_fixed_target

        self._p300_frames_visible = p.flash_duration_frames

    def _draw_tiles(self) -> None:
        colors = self.config.colors
        target_flash = colors.target_flash
        dim = colors.off_brightness

        for tile, rect in self._tile_visuals.items():
            if not self._running:
                rect.fillColor = _dim_color(tile.color, dim)
            elif self._p300_current_target is tile and self._p300_frames_visible > 0:
                rect.fillColor = target_flash
            elif tile.is_on:
                rect.fillColor = tile.color
            else:
                rect.fillColor = _dim_color(tile.color, dim)
            rect.lineColor = rect.fillColor
            rect.draw()

    def _draw_ui_overlay(self) -> None:
        if self._info_text is None:
            return
        state = "RUNNING" if self._running else "STOPPED"
        msg = (
            f"State: {state} | Grid 9×9 | P300 oddball | "
            "SPACE=start/stop  ESC=quit"
        )
        self._info_text.text = msg
        self._info_text.draw()

    def _log_frame_rate_estimate(self) -> None:
        """
        Ask PsychoPy for its measured frame rate and log the result.
        """
        if self.window is None:
            return

        # This call may take a short while; we do it once at startup
        actual_fps = self.window.getActualFrameRate(nIdentical=20, nMaxFrames=120)
        if actual_fps:
            self.logger.info(f"Measured display frame rate: {actual_fps:.2f} Hz")
        else:
            self.logger.warning(
                "Could not reliably measure display frame rate. "
                "Using target FPS from config."
            )

    def _shutdown(self) -> None:
        if self.window is not None:
            self.logger.info("Closing PsychoPy window.")
            self.window.close()
        core.quit()

