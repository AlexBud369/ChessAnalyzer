import chess
from src.utils.chess_utils import material_score

def evaluate(board: chess.Board) -> float:
    return material_score(board)