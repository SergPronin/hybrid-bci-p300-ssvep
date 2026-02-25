import random
from psychopy import core


class StimulusController:
    """
    Backend-движок стимуляции.
    UI просто вызывает update() каждый кадр.
    """

    def __init__(
        self,
        grid,
        flash_duration: float = 0.1,   # 100 ms
        isi: float = 0.05              # 50 ms inter-stimulus interval
    ):
        self.grid = grid
        self.flash_duration = flash_duration
        self.isi = isi

        self.clock = core.Clock()

        self._state = "idle"   # idle | on | isi
        self._state_start = 0.0
        self._current_tile = None
        self._running = False

    def start(self):
        self._running = True
        self._state = "isi"
        self._state_start = self.clock.getTime()

    def stop(self):
        self._running = False
        self.grid.reset()
        self._state = "idle"

    def update(self):
        """
        Вызывать каждый кадр из UI.
        Возвращает событие или None.
        """
        if not self._running:
            return None

        now = self.clock.getTime()

        # ISI → запускаем новую вспышку
        if self._state == "isi":
            if now - self._state_start >= self.isi:
                self._current_tile = random.choice(self.grid.tiles)
                self._current_tile.active = True

                self._state = "on"
                self._state_start = now

                return {
                    "tile_id": self._current_tile.id,
                    "event": "on",
                    "timestamp": now,
                }

        # Вспышка закончилась → выключаем
        elif self._state == "on":
            if now - self._state_start >= self.flash_duration:
                self._current_tile.active = False

                event = {
                    "tile_id": self._current_tile.id,
                    "event": "off",
                    "timestamp": now,
                }

                self._state = "isi"
                self._state_start = now

                return event

        return None