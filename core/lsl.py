"""Отправка маркеров событий стимуляции через Lab Streaming Layer (LSL)."""

from typing import Optional

from pylsl import StreamInfo, StreamOutlet


class LslMarkerSender:
    """
    Отправитель маркеров мигания в LSL-стрим.

    Стрим создаётся при инициализации. Маркеры в формате "tile_id|event"
    (например "3|on", "3|off").
    """

    STREAM_NAME = "BCI_StimMarkers"
    STREAM_TYPE = "Markers"
    SOURCE_ID = "stimulus-controller-001"

    def __init__(self) -> None:
        self._outlet: Optional[StreamOutlet] = None
        self._init_stream()

    def _init_stream(self) -> None:
        """Создаёт и объявляет LSL-стрим маркеров."""
        try:
            info = StreamInfo(
                name=self.STREAM_NAME,
                type=self.STREAM_TYPE,
                channel_count=1,
                nominal_srate=0,
                channel_format="string",
                source_id=self.SOURCE_ID,
            )
            self._outlet = StreamOutlet(info)
        except Exception as e:
            print(f"LSL: не удалось создать стрим: {e}")
            self._outlet = None

    def send(self, tile_id: int, event: str) -> None:
        """
        Отправить маркер в LSL.

        Args:
            tile_id: ID плитки (0 .. N-1).
            event: Событие — "on" (загорание) или "off" (затухание).
        """
        if self._outlet is None:
            return
        marker = f"{tile_id}|{event}"
        try:
            self._outlet.push_sample([marker])
        except Exception as e:
            print(f"LSL: ошибка отправки маркера: {e}")
