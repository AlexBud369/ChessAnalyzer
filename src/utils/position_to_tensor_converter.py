import numpy as np
import chess
from chess_constants import BOARD_RANKS, BOARD_FILES, PIECE_TO_CHANNEL, NUM_CHANNELS, COLOR_OFFSET

def fen_to_tensor(fen: str) -> np.ndarray:
    """Преобразует FEN в тензор 8 x 8 x 12"""
    board = chess.Board(fen)
    tensor = np.zeros((BOARD_RANKS, BOARD_FILES, NUM_CHANNELS), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = BOARD_RANKS - 1 - chess.square_rank(square)
        col = chess.square_file(square)
        channel = PIECE_TO_CHANNEL[piece.piece_type]
        if piece.color == chess.WHITE:
            tensor[row, col, channel] = 1
        else:
            tensor[row, col, channel + COLOR_OFFSET] = 1
    return tensor