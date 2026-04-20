import chess
from typing import List, Callable, Optional, Tuple
from .transposition_table import TranspositionTable

def order_moves(board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
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
        return score
    return sorted(moves, key=move_score, reverse=True)

def evaluate_maximizing(board: chess.Board, depth: int, alpha: float, beta: float,
                        evaluator: Callable[[chess.Board], float],
                        tt: Optional[TranspositionTable] = None) -> float:
    if depth == 0 or board.is_game_over():
        return evaluator(board)
    if tt is not None:
        val, _ = tt.lookup(board, depth, alpha, beta)
        if val is not None:
            return val
    max_eval = -float('inf')
    best_move = None
    moves = list(board.legal_moves)
    if tt is not None:
        _, tt_move = tt.lookup(board, depth, alpha, beta)
        if tt_move is not None and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
    for move in order_moves(board, moves):
        board.push(move)
        eval = evaluate_minimizing(board, depth - 1, alpha, beta, evaluator, tt)
        board.pop()
        if eval > max_eval:
            max_eval = eval
            best_move = move
        alpha = max(alpha, eval)
        if beta <= alpha:
            break
    if tt is not None:
        node_type = 'exact'
        if max_eval <= alpha:
            node_type = 'upper'
        elif max_eval >= beta:
            node_type = 'lower'
        tt.store(board, depth, max_eval, best_move, node_type, alpha, beta)
    return max_eval

def evaluate_minimizing(board: chess.Board, depth: int, alpha: float, beta: float,
                        evaluator: Callable[[chess.Board], float],
                        tt: Optional[TranspositionTable] = None) -> float:
    if depth == 0 or board.is_game_over():
        return evaluator(board)
    if tt is not None:
        val, _ = tt.lookup(board, depth, alpha, beta)
        if val is not None:
            return val
    min_eval = float('inf')
    best_move = None
    moves = list(board.legal_moves)
    if tt is not None:
        _, tt_move = tt.lookup(board, depth, alpha, beta)
        if tt_move is not None and tt_move in moves:
            moves.remove(tt_move)
            moves.insert(0, tt_move)
    for move in order_moves(board, moves):
        board.push(move)
        eval = evaluate_maximizing(board, depth - 1, alpha, beta, evaluator, tt)
        board.pop()
        if eval < min_eval:
            min_eval = eval
            best_move = move
        beta = min(beta, eval)
        if beta <= alpha:
            break
    if tt is not None:
        node_type = 'exact'
        if min_eval >= beta:
            node_type = 'upper'
        elif min_eval <= alpha:
            node_type = 'lower'
        tt.store(board, depth, min_eval, best_move, node_type, alpha, beta)
    return min_eval

def get_best_move(board: chess.Board, depth: int,
                  evaluator: Callable[[chess.Board], float],
                  use_tt: bool = True) -> Tuple[Optional[chess.Move], Optional[float]]:
    if board.is_game_over():
        return None, None
    tt = TranspositionTable() if use_tt else None
    best_move = None
    if board.turn == chess.WHITE:
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        for move in order_moves(board, list(board.legal_moves)):
            board.push(move)
            value = evaluate_minimizing(board, depth - 1, alpha, beta, evaluator, tt)
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
            value = evaluate_maximizing(board, depth - 1, alpha, beta, evaluator, tt)
            board.pop()
            if value < best_value:
                best_value = value
                best_move = move
            beta = min(beta, best_value)
    return best_move, best_value