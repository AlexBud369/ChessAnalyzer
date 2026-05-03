import chess
import numpy as np

def index_to_move(index: int) -> chess.Move:
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

def move_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def legal_policy_probs(policy_probs: np.ndarray, board: chess.Board) -> np.ndarray:

    legal_mask = np.zeros(4096, dtype=bool)
    for move in board.legal_moves:
        legal_mask[move_to_index(move)] = True
    masked = policy_probs * legal_mask
    if masked.sum() > 0:
        masked /= masked.sum()
    else:
        n_legal = sum(legal_mask)
        masked[legal_mask] = 1.0 / n_legal
    return masked