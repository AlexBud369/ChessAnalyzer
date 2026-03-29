import chess
from src.utils.position_evalution import material_score
from src.engine.nn_evaluator import NNEvaluator

def evaluate(board: chess.Board) -> float:
    return material_score(board)

def evaluate_nn(board):
    """
    Оценка позиции через нейросеть.
    При мате/пате возвращает соответствующую большую величину.
    """
    if board.is_checkmate():
        return -1000 if board.turn == chess.WHITE else 1000
    if board.is_stalemate():
        return 0.0

    evaluator = NNEvaluator()
    return evaluator.evaluate_board(board)

