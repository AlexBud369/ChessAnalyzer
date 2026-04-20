"""
Генерация синтетических позиций с чистым материальным перевесом.
Каждая позиция: начальная расстановка + случайное удаление фигур у одной стороны.
Оценка = материальная разница (с точки зрения стороны, которая ходит), обрезанная до [-30, 30].
"""

import os
import random
import csv
import chess

OUTPUT_CSV = "data/raw/synthetic_material.csv"
NUM_POSITIONS = 200_000

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

def material_score(board: chess.Board) -> float:
    """Материальный счёт с точки зрения белых."""
    score = 0.0
    for piece_type, value in PIECE_VALUES.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * value
        score -= len(board.pieces(piece_type, chess.BLACK)) * value
    return score

def random_remove_pieces(board: chess.Board, side: chess.Color, num_removals: int):
    squares = [sq for sq, piece in board.piece_map().items()
               if piece.color == side and piece.piece_type != chess.KING]
    if len(squares) < num_removals:
        num_removals = len(squares)
    to_remove = random.sample(squares, num_removals)
    for sq in to_remove:
        board.remove_piece_at(sq)
    return board

def generate_synthetic_positions(num: int, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["FEN", "Evaluation"])
        for i in range(num):
            board = chess.Board()
            side = random.choice([chess.WHITE, chess.BLACK])
            num_removals = random.randint(1, 3)
            board = random_remove_pieces(board, side, num_removals)
            if board.king(side) is None:
                continue

            raw_score = material_score(board)
            if board.turn == chess.BLACK:
                raw_score = -raw_score

            clipped_pawns = max(-30.0, min(30.0, raw_score))
            centipawns = int(round(clipped_pawns * 100))

            writer.writerow([board.fen(), f"{centipawns:+d}"])

            if (i + 1) % 50000 == 0:
                print(f"Сгенерировано {i+1} / {num} позиций")

    print(f"Синтетический датасет сохранён: {output_path} ({num} позиций)")

if __name__ == "__main__":
    generate_synthetic_positions(NUM_POSITIONS, OUTPUT_CSV)