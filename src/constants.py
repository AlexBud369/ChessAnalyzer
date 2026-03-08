
# раземры окна и доски
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8

# цвета
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BUTTON_COLOR = (200, 200, 200)
BUTTON_HOVER_COLOR = (170, 170, 170)
TEXT_COLOR = (50, 50, 50)

# Кнопки
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 40
BUTTON_Y = BOARD_SIZE + 20
NEW_GAME_POS = (50, BUTTON_Y)
ANALYZE_POS = (220, BUTTON_Y)

# Шрифт
FONT_SIZE = 24

# Имена файлов изображений
PIECE_TYPES = ['wP', 'wN', 'wB', 'wR', 'wQ', 'wK', 'bP', 'bN', 'bB', 'bR', 'bQ', 'bK']

# Начальная расстановка (двумерный список: строка 0 – 8-я горизонталь (чёрные фигуры))
START_BOARD = [
    ['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
    ['bP'] * 8,
    [''] * 8,
    [''] * 8,
    [''] * 8,
    [''] * 8,
    ['wP'] * 8,
    ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']
]

