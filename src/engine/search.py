"""
Алгоритмы поиска для шахматного движка.
"""
import chess
from .evaluation import evaluate

def minimax(board: chess.Board, depth: int, maximizing_player: bool,
            evaluator=evaluate) -> float:
    """
    Рекурсивный минимакс
    depth – оставшаяся глубина поиска.
    maximizing_player – True для белых (максимизация), False для чёрных.
    Возвращает оценку позиции (с точки зрения белых).
    """
    if depth == 0 or board.is_game_over():
        return evaluator(board)

    if maximizing_player:
        max_eval = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, False, evaluator)
            board.pop()
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, True, evaluator)
            board.pop()
            min_eval = min(min_eval, eval)
        return min_eval

def get_best_move(board: chess.Board, depth: int,
                  evaluator=evaluate) -> tuple:
    """
    Возвращает лучший ход и его оценку.
    Если ходов нет (мат/пат), возвращает (None, None).
    """
    if board.is_game_over():
        return None, None

    best_move = None
    if board.turn == chess.WHITE:
        best_value = -float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, False, evaluator)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
    else:
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            value = minimax(board, depth - 1, True, evaluator)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move

    return best_move, best_value