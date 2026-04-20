import chess
from src.utils.position_evalution import material_score
from src.engine.nn_evaluator import NNEvaluator

_EVALUATOR = None

def _get_evaluator():
    global _EVALUATOR
    if _EVALUATOR is None:
        _EVALUATOR = NNEvaluator()
    return _EVALUATOR

def evaluate(board: chess.Board) -> float:
    return material_score(board)

def evaluate_nn(board):
    """
    Оценка позиции через нейросеть.
    При мате/пате возвращает соответствующую величину, согласованную с CLIP_VALUE.
    """
    if board.is_checkmate():
        return -30.0 if board.turn == chess.WHITE else 30.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    evaluator = _get_evaluator()
    return evaluator.evaluate_board(board)