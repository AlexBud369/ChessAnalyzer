from __future__ import annotations

import numpy as np
import chess


PIECE_CHANNELS = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Преобразует позицию в массив float32 формы (18, 8, 8).

    Каналы 0–5: белые фигуры (пешка…король).
    Каналы 6–11: чёрные фигуры.
    Канал 12: поля под атакой белых.
    Канал 13: поля под атакой чёрных.
    Канал 14: ход белых (все 1.0 или 0.0).
    Канал 15: права рокировки белых — (0,7) короткая, (0,0) длинная.
    Канал 16: права рокировки чёрных — (7,7), (7,0).
    Канал 17: поле en passant.
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    for c_idx, piece_type in enumerate(PIECE_CHANNELS):
        for color, ch_offset in ((chess.WHITE, 0), (chess.BLACK, 6)):
            ch = ch_offset + c_idx
            for sq in board.pieces(piece_type, color):
                r, f = divmod(sq, 8)
                planes[ch, r, f] = 1.0

    white_attacks = _attack_mask(board, chess.WHITE)
    black_attacks = _attack_mask(board, chess.BLACK)
    for sq in range(64):
        r, f = divmod(sq, 8)
        if white_attacks & (1 << sq):
            planes[12, r, f] = 1.0
        if black_attacks & (1 << sq):
            planes[13, r, f] = 1.0

    if board.turn == chess.WHITE:
        planes[14, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        planes[15, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        planes[15, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        planes[16, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        planes[16, 7, 0] = 1.0

    if board.ep_square is not None:
        r, f = divmod(board.ep_square, 8)
        planes[17, r, f] = 1.0

    return planes


def fen_to_planes(fen: str) -> np.ndarray:
    return board_to_planes(chess.Board(fen))


def _attack_mask(board: chess.Board, color: chess.Color) -> int:
    mask = 0
    for sq in board.pieces(chess.KING, color):
        mask |= board.attacks_mask(sq)
    for sq in board.pieces(chess.QUEEN, color):
        mask |= board.attacks_mask(sq)
    for sq in board.pieces(chess.ROOK, color):
        mask |= board.attacks_mask(sq)
    for sq in board.pieces(chess.BISHOP, color):
        mask |= board.attacks_mask(sq)
    for sq in board.pieces(chess.KNIGHT, color):
        mask |= board.attacks_mask(sq)
    for sq in board.pieces(chess.PAWN, color):
        mask |= board.attacks_mask(sq)
    return mask
