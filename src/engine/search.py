"""
Алгоритм минмакса с альфа-бета отсечением
"""
import chess
from typing import List, Callable, Optional, Tuple
from .evaluation import evaluate


def order_moves(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
    """
    Сортировка ходов для улучшения отсечений.
    Сначала идут взятия и шахующие ходы
    """
    def move_score(move: chess.Move) -> int:
        score = 0
        if board.is_capture(move):
            piece = board.piece_at(move.to_square)
            if piece:
                values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}
                score += values.get(piece.piece_type, 0)
        board.push(move)
        if board.is_check():
            score += 2
        board.pop()
        return -score

    return sorted(moves, key=move_score)


def evaluate_maximizing(board: chess.Board, depth: int, alpha: float, beta: float,
                        evaluator: Callable[[chess.Board], float]) -> float:
    """Альфа-бета поиск для максимизирующего игрока (белые)"""
    if depth == 0 or board.is_game_over():
        return evaluator(board)

    max_eval = -float('inf')
    for move in order_moves(board, list(board.legal_moves)):
        board.push(move)
        eval = evaluate_minimizing(board, depth - 1, alpha, beta, evaluator)
        board.pop()
        max_eval = max(max_eval, eval)
        alpha = max(alpha, eval)
        if beta <= alpha:
            break
    return max_eval


def evaluate_minimizing(board: chess.Board, depth: int, alpha: float, beta: float,
                        evaluator: Callable[[chess.Board], float]) -> float:
    """Альфа-бета поиск для минимизирующего игрока (чёрные)"""
    if depth == 0 or board.is_game_over():
        return evaluator(board)

    min_eval = float('inf')
    for move in order_moves(board, list(board.legal_moves)):
        board.push(move)
        eval = evaluate_maximizing(board, depth - 1, alpha, beta, evaluator)
        board.pop()
        min_eval = min(min_eval, eval)
        beta = min(beta, eval)
        if beta <= alpha:
            break
    return min_eval


def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float,
              maximizing_player: bool, evaluator: Callable[[chess.Board], float] = evaluate) -> float:
    """
    Минимакс с альфа-бета отсечением.
    Возвращает оценку позиции (с точки зрения белых)
    """
    if maximizing_player:
        return evaluate_maximizing(board, depth, alpha, beta, evaluator)
    else:
        return evaluate_minimizing(board, depth, alpha, beta, evaluator)


def get_best_move(board: chess.Board, depth: int,
                  evaluator: Callable[[chess.Board], float] = evaluate) -> Tuple[Optional[chess.Move], Optional[float]]:
    """
    Возвращает лучший ход и его оценку с помощью альфа-бета поиска.
    Если ходов нет, возвращает (None, None)
    """
    if board.is_game_over():
        return None, None

    best_move = None
    if board.turn == chess.WHITE:
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move in order_moves(board, list(board.legal_moves)):
            board.push(move)
            value = evaluate_minimizing(board, depth - 1, alpha, beta, evaluator)
            board.pop()
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
    else:
        best_value = float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move in order_moves(board, list(board.legal_moves)):
            board.push(move)
            value = evaluate_maximizing(board, depth - 1, alpha, beta, evaluator)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(beta, best_value)

    return best_move, best_value