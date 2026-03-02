"""Контроллер последовательности стимулов (мигание плиток). События on/off возвращаются в GUI; отправка маркеров LSL привязана к win.flip в gui.py."""

import random
from typing import Dict, Optional

from psychopy import core

from .grid import Grid
from .lsl import LslMarkerSender
from .tile import Tile


class StimulusController:
    """
    Управляет поочерёдным миганием плиток сетки и отправкой маркеров в LSL.

    Состояния: idle → isi → on → isi → on → ...
    - isi: пауза между вспышками.
    - on: плитка подсвечена на время flash_duration.
    """

    def __init__(
        self,
        grid: Grid,
        flash_duration: float = 0.1,
        isi: float = 0.05,
    ) -> None:
        """
        Args:
            grid: Сетка плиток (модель).
            flash_duration: Длительность одной вспышки, сек.
            isi: Интервал между вспышками (Inter-Stimulus Interval), сек.
        """
        self.grid = grid
        self.flash_duration = flash_duration
        self.isi = isi
        self._clock = core.Clock()
        self._state = "idle"
        self._state_start = 0.0
        self._current_tile: Optional[Tile] = None
        self._running = False
        self._lsl = LslMarkerSender()

    @property
    def lsl(self) -> LslMarkerSender:
        """LSL-отправитель маркеров (для привязки к win.callOnFlip в GUI)."""
        return self._lsl

    def start(self) -> None:
        """Запустить стимуляцию."""
        self._running = True
        self._state = "isi"
        self._state_start = self._clock.getTime()

    def stop(self) -> None:
        """Остановить стимуляцию и сбросить сетку."""
        self._running = False
        self.grid.reset()
        self._state = "idle"

    def update(self) -> Optional[Dict[str, object]]:
        """
        Обновить состояние по таймеру. Вызывать каждый кадр.

        Returns:
            Словарь с полями tile_id, event, timestamp при смене события, иначе None.
        """
        if not self._running:
            return None

        now = self._clock.getTime()

        if self._state == "isi":
            if now - self._state_start >= self.isi:
                self._current_tile = random.choice(self.grid.tiles)
                self._current_tile.active = True
                self._state = "on"
                self._state_start = now
                event_data = {
                    "tile_id": self._current_tile.id,
                    "event": "on",
                    "timestamp": now,
                }
                return event_data

        if self._state == "on":
            if now - self._state_start >= self.flash_duration:
                self._current_tile.active = False
                event_data = {
                    "tile_id": self._current_tile.id,
                    "event": "off",
                    "timestamp": now,
                }
                self._state = "isi"
                self._state_start = now
                return event_data

        return None
