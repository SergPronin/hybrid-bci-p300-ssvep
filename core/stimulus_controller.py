"""Контроллер последовательности стимулов с поддержкой фаз Cue-Ready-Stim-End."""
import random
from typing import Dict, Optional, List

from psychopy import core

from .grid import Grid
from .lsl import LslMarkerSender
from .tile import Tile


class StimulusController:
    """
    Управляет четырёхфазным экспериментом:
      - Cue: подсветка целевой плитки синим (2 сек)
      - Ready: пауза, все плитки серые (1.5 сек)
      - Stim: блочная рандомизированная последовательность вспышек
      - End: завершение, возврат в меню

    Генерирует события и маркеры LSL:
      - trial_start|target=X
      - X|on / X|off
      - trial_end
    """

    def __init__(
        self,
        grid: Grid,
        flash_duration: float = 0.1,
        isi: float = 0.05,
        cue_duration: float = 2.0,
        ready_duration: float = 1.5,
        cue_color: str = "blue",
        stim_color: str = "white",
    ) -> None:
        self.grid = grid
        self.flash_duration = flash_duration
        self.isi = isi
        self.cue_duration = cue_duration
        self.ready_duration = ready_duration
        self.cue_color = cue_color
        self.stim_color = stim_color

        self._clock = core.Clock()
        self._state = "idle"          # idle, cue, ready, stim, end
        self._state_start = 0.0
        self._current_tile: Optional[Tile] = None
        self._running = False
        self._lsl = LslMarkerSender()

        # Параметры раунда
        self.sequences: int = 0                 # сколько блоков (полных циклов по 9 плиткам)
        self.target_tile: Optional[Tile] = None
        self._blocks_completed: int = 0
        self._current_block: List[int] = []     # перемешанный список ID плиток для текущего блока
        self._block_index: int = 0

    @property
    def lsl(self) -> LslMarkerSender:
        return self._lsl

    def get_target_id(self) -> Optional[int]:
        """Возвращает ID целевой плитки (или None, если цель не выбрана)."""
        return self.target_tile.id if self.target_tile else None

    def get_target_color(self) -> Optional[str]:
        """Возвращает цвет для целевой плитки в фазе Cue, иначе None."""
        if self._state == "cue" and self.target_tile:
            return self.cue_color
        return None

    def get_stim_color(self) -> str:
        """Цвет вспышек в фазе стимуляции."""
        return self.stim_color

    def is_running(self) -> bool:
        return self._running

    def start_experiment(self, sequences: int) -> None:
        """
        Запускает новый раунд эксперимента с заданным количеством блоков.
        """
        if sequences <= 0:
            return
        self.sequences = sequences
        self._blocks_completed = 0

        # Случайный выбор целевой плитки
        self.target_tile = random.choice(self.grid.tiles)

        self._running = True
        self._state = "cue"
        self._state_start = self._clock.getTime()
        self.grid.reset()

    def stop(self) -> None:
        """Принудительная остановка раунда."""
        self._running = False
        self.grid.reset()
        self._state = "idle"
        self.target_tile = None
        self._current_block = []
        self._block_index = 0
        # Сбрасываем подсостояние, чтобы при следующем старте оно создалось заново
        if hasattr(self, '_substate'):
            del self._substate

    def _generate_next_block(self) -> None:
        """Создаёт новый перемешанный блок из всех плиток."""
        self._current_block = list(range(len(self.grid.tiles)))
        random.shuffle(self._current_block)
        self._block_index = 0

    def update(self) -> Optional[Dict[str, object]]:
        """
        Вызывается каждый кадр.
        Возвращает словарь события при смене состояния.
        """
        if not self._running:
            return None

        now = self._clock.getTime()

        # --- Фаза CUE ---
        if self._state == "cue":
            if now - self._state_start >= self.cue_duration:
                # Переходим в READY
                self._state = "ready"
                self._state_start = now
                # Гасим целевую плитку (она была синей)
                if self.target_tile:
                    self.target_tile.active = False
            else:
                # В фазе CUE целевая плитка должна гореть синим
                if self.target_tile and not self.target_tile.active:
                    self.target_tile.active = True
                return None  # пока нет событий

        # --- Фаза READY ---
        if self._state == "ready":
            if now - self._state_start >= self.ready_duration:
                # Переходим в STIM
                self._state = "stim"
                self._state_start = now
                self._generate_next_block()
                # Первая вспышка начнётся в следующем кадре (через isi)
            return None
        # --- Фаза STIM (мигание) ---
        if self._state == "stim":
            # Если _substate ещё не создан или равен None, инициализируем его
            if not hasattr(self, '_substate') or self._substate is None:
                self._substate = "isi"
                self._substate_start = now

            # ISI (пауза между вспышками)
            if self._substate == "isi":
                if now - self._substate_start >= self.isi:
                    # Проверяем, остались ли ещё плитки в текущем блоке
                    if self._block_index >= len(self._current_block):
                        self._blocks_completed += 1
                        if self._blocks_completed >= self.sequences:
                            self._state = "end"
                            self._substate = None
                            self._running = False
                            self.grid.reset()
                            return {"event": "trial_end"}
                        else:
                            self._generate_next_block()

                    if self._block_index < len(self._current_block):
                        next_tile_id = self._current_block[self._block_index]
                        self._block_index += 1
                        self._current_tile = self.grid.tiles[next_tile_id]
                        self._current_tile.active = True
                        self._substate = "on"
                        self._substate_start = now
                        return {
                            "tile_id": self._current_tile.id,
                            "event": "on",
                            "timestamp": now,
                        }

            # ON (плитка горит)
            elif self._substate == "on":
                if now - self._substate_start >= self.flash_duration:
                    self._current_tile.active = False
                    self._substate = "isi"
                    self._substate_start = now
                    return {
                        "tile_id": self._current_tile.id,
                        "event": "off",
                        "timestamp": now,
                    }

            return None

        # --- Фаза END (уже обработана выше, но оставим заглушку) ---
        if self._state == "end":
            # После отправки trial_end контроллер должен перейти в idle
            self._state = "idle"
            self._running = False
            return None

        return None