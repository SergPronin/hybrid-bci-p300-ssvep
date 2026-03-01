"""Точка входа: запуск приложения стимуляции."""

from gui.gui import StimulusApp


def main() -> None:
    """Запуск окна стимуляции и главного цикла."""
    app = StimulusApp()
    app.run()


if __name__ == "__main__":
    main()
