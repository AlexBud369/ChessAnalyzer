from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import chess
import torch

from chess_engine.encoding import board_to_planes
from chess_engine.moves import MOVE_ENCODER


@dataclass
class SearchResult:
    best_move: Optional[chess.Move]
    score_cp: float
    depth: int
    nodes: int
    principal_variation: List[chess.Move]


PIECE_CAPTURE_BONUS = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


class AlphaBetaSearch:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        max_depth: int = 4,
    ) -> None:
        self.model = model
        self.device = device
        self.max_depth = max_depth
        self.nodes = 0

    @torch.no_grad()
    def _nn_eval(
        self, board: chess.Board
    ) -> Tuple[float, List[Tuple[chess.Move, float]]]:
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        policy_logits = policy_logits.squeeze(0)
        v = float(value.item())
        top = MOVE_ENCODER.top_moves(policy_logits, board, n=40)
        return v * 100.0, top

    def _move_order(
        self, board: chess.Board, hints: List[Tuple[chess.Move, float]]
    ) -> List[chess.Move]:
        hint_map = {m: p for m, p in hints}
        moves = list(board.legal_moves)

        def key(m: chess.Move) -> float:
            score = hint_map.get(m, 0.0)
            if board.is_capture(m):
                victim = board.piece_at(m.to_square)
                if victim:
                    score += PIECE_CAPTURE_BONUS.get(victim.piece_type, 0) / 1000.0
            return score

        return sorted(moves, key=key, reverse=True)

    def search(self, board: chess.Board, depth: Optional[int] = None) -> SearchResult:
        depth = depth or self.max_depth
        self.nodes = 0
        score, best = self._negamax(board, depth, float("-inf"), float("inf"))
        pv: List[chess.Move] = []
        if best:
            pv.append(best)
        return SearchResult(
            best_move=best,
            score_cp=score,
            depth=depth,
            nodes=self.nodes,
            principal_variation=pv,
        )

    def _negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
    ) -> Tuple[float, Optional[chess.Move]]:
        self.nodes += 1
        if depth == 0 or board.is_game_over():
            score, _ = self._nn_eval(board)
            return score, None

        _, hints = self._nn_eval(board)
        moves = self._move_order(board, hints)
        best_move: Optional[chess.Move] = None
        best_score = float("-inf")

        for move in moves:
            board.push(move)
            child_score, _ = self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            score = -child_score
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score, best_move
