import numpy as np
import chess

# Константы для 12 каналов
BOARD_RANKS = 8
BOARD_FILES = 8
NUM_CHANNELS = 12
PIECE_TO_CHANNEL = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}
COLOR_OFFSET = 6

# Константы для 18 каналов
NUM_CHANNELS_18 = 18
CH_WHITE_START = 0
CH_BLACK_START = 6
CH_TURN = 12
CH_W_OO = 13
CH_W_OOO = 14
CH_B_OO = 15
CH_B_OOO = 16
CH_EN_PASSANT = 17


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


def fen_to_tensor_18ch(fen: str) -> np.ndarray:
    """
    Преобразует FEN в тензор 8x8x18 с дополнительной информацией:
    - 0..5: белые фигуры
    - 6..11: чёрные фигуры
    - 12: очередь хода (1 - белые, 0 - чёрные)
    - 13: белые короткая рокировка
    - 14: белые длинная рокировка
    - 15: чёрные короткая рокировка
    - 16: чёрные длинная рокировка
    - 17: взятие на проходе (1 на клетке цели, иначе 0)
    """
    board = chess.Board(fen)
    tensor = np.zeros((8, 8, NUM_CHANNELS_18), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        channel = PIECE_TO_CHANNEL[piece.piece_type]
        if piece.color == chess.WHITE:
            tensor[row, col, CH_WHITE_START + channel] = 1
        else:
            tensor[row, col, CH_BLACK_START + channel] = 1

    turn_value = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[:, :, CH_TURN] = turn_value

    tensor[:, :, CH_W_OO]  = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, CH_W_OOO] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[:, :, CH_B_OO]  = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[:, :, CH_B_OOO] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    ep_square = board.ep_square
    if ep_square is not None:
        row = 7 - chess.square_rank(ep_square)
        col = chess.square_file(ep_square)
        tensor[row, col, CH_EN_PASSANT] = 1.0

    return tensor