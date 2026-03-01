"""Конфигурация приложения (размеры, цвета, диапазоны параметров)."""

# Окно
WINDOW_SIZE = (1200, 800)
WINDOW_COLOR = "black"

# Сетка плиток
GRID_SIZE = 3
TILE_SIZE_PX = 120
TILE_SPACING_PX = 20
TILE_DEFAULT_COLOR = "gray"
TILE_LINE_COLOR = "white"

# Цвета мигания (случайный выбор при подсветке)
FLASH_COLORS = ["yellow", "green", "red", "white"]

# Слайдеры
ISI_RANGE = (0.02, 5.0)
FLASH_DURATION_RANGE = (0.05, 2.0)
SLIDER_GRANULARITY = 0.01

# Кнопки
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
START_BUTTON_POS = (-100, -250)
STOP_BUTTON_POS = (100, -250)

# Расположение панели настроек (позиции слайдеров и подписей)
PANEL_FREQ_LABEL_POS = (550, 260)
PANEL_FREQ_SLIDER_POS = (500, 200)
PANEL_FREQ_VALUE_POS = (500, 180)
PANEL_FLASH_LABEL_POS = (550, 140)
PANEL_FLASH_SLIDER_POS = (500, 100)
PANEL_FLASH_VALUE_POS = (500, 80)
SLIDER_SIZE = (200, 20)
TEXT_HEIGHT = 20
