"""
Проверка наличия LSL-стримов (для ручного запуска).

Запуск из корня проекта:
  python -m test.test_lsl_streams
"""

from pylsl import resolve_byprop

TIMEOUT = 5


def test_resolve_marker_stream() -> None:
    """Проверяет, что стрим BCI_StimMarkers можно обнаружить (при запущенном приложении)."""
    streams = resolve_byprop("name", "BCI_StimMarkers", timeout=TIMEOUT)
    assert streams, "Стрим BCI_StimMarkers не найден. Запустите приложение и нажмите START."
    info = streams[0]
    assert info.type() == "Markers"
    assert info.channel_count() == 1


def list_all_streams() -> None:
    """Выводит в консоль все доступные LSL-стримы."""
    streams = resolve_byprop("type", "Markers", timeout=2)
    streams += resolve_byprop("type", "EEG", timeout=2)
    seen = set()
    for s in streams:
        uid = s.uid()
        if uid in seen:
            continue
        seen.add(uid)
        print(f"  {s.name()} | {s.type()} | каналов={s.channel_count()}")


if __name__ == "__main__":
    print("Поиск LSL-стримов...")
    list_all_streams()
    print("\nПроверка стрима BCI_StimMarkers...")
    try:
        test_resolve_marker_stream()
        print("OK: стрим BCI_StimMarkers найден.")
    except AssertionError as e:
        print(f"  {e}")
