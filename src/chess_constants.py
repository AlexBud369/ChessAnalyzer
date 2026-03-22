import chess

# Размеры доски
BOARD_RANKS = 8
BOARD_FILES = 8
MAX_RANK_INDEX = BOARD_RANKS - 1  # 7
MIN_SQUARE_INDEX = 0
MAX_SQUARE_INDEX = 63  # всего клеток 64, индексы 0..63

# Константы для превращения пешки
PROMOTION_RANK_WHITE = MAX_RANK_INDEX    # последняя горизонталь для белых (8-я)
PROMOTION_RANK_BLACK = MIN_SQUARE_INDEX  # последняя горизонталь для чёрных (1-я)
DEFAULT_PROMOTION = chess.QUEEN

# Начальные значения для координат и смещений (см. DragHandler)
INITIAL_DRAG_OFFSET = (0, 0)
INITIAL_MOUSE_POS = (0, 0)

# Веса фигур для материальной оценки
MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

# Отображение типа фигуры на канал тензора (для нейросети)
PIECE_TO_CHANNEL = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}