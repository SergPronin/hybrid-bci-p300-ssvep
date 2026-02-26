import random
from psychopy import visual, core, event
from pylsl import StreamInfo, StreamOutlet

class Tile:
    def __init__(self, row, col, id_):
        self.row = row
        self.col = col
        self.id = id_
        self.active = False

class Grid:
    def __init__(self, size=3):
        self.size = size
        self.tiles = [Tile(r, c, r*size + c) for r in range(size) for c in range(size)]

    def reset(self):
        for tile in self.tiles:
            tile.active = False

class StimulusController:
    def __init__(self, grid, flash_duration: float = 0.1, isi: float = 0.05):
        self.grid = grid
        self.flash_duration = flash_duration
        self.isi = isi
        self.clock = core.Clock()
        self._state = "idle"
        self._state_start = 0.0
        self._current_tile = None
        self._running = False
        self._lsl_outlet = None
        self._init_lsl()

    def _init_lsl(self):
        try:
            info = StreamInfo(
                name="BCI_StimMarkers",
                type="Markers",
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id="stimulus-controller-001",
            )
            self._lsl_outlet = StreamOutlet(info)
        except Exception as e:
            print(f"Failed to initialize LSL outlet: {e}")
            self._lsl_outlet = None

    def _push_lsl(self, event_data):
        if not self._lsl_outlet:
            return
        marker = f"{event_data['tile_id']}|{event_data['event']}"
        try:
            self._lsl_outlet.push_sample([marker])
        except Exception as e:
            print(f"Failed to send LSL marker: {e}")

    def start(self):
        self._running = True
        self._state = "isi"
        self._state_start = self.clock.getTime()

    def stop(self):
        self._running = False
        self.grid.reset()
        self._state = "idle"

    def update(self):
        if not self._running:
            return None

        now = self.clock.getTime()

        if self._state == "isi":
            if now - self._state_start >= self.isi:
                self._current_tile = random.choice(self.grid.tiles)
                self._current_tile.active = True
                self._state = "on"
                self._state_start = now
                event_data = {"tile_id": self._current_tile.id, "event": "on", "timestamp": now}
                self._push_lsl(event_data)
                return event_data

        elif self._state == "on":
            if now - self._state_start >= self.flash_duration:
                self._current_tile.active = False
                event_data = {"tile_id": self._current_tile.id, "event": "off", "timestamp": now}
                self._state = "isi"
                self._state_start = now
                self._push_lsl(event_data)
                return event_data

        return None