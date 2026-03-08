from src.constants import BOARD_SIZE, SQUARE_SIZE

def get_square_under_mouse(pos):
    x, y = pos
    if x < BOARD_SIZE and y < BOARD_SIZE:
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col
    return None, None
