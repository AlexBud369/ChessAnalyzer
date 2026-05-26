from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import chess


PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

CENTER_SQUARES = {chess.D4, chess.E4, chess.D5, chess.E5, chess.C3, chess.F3,
                  chess.C6, chess.F6, chess.C4, chess.F4, chess.C5, chess.F5}


@dataclass
class FactorScore:
    name: str
    score_cp: float
    description: str


@dataclass
class ExpertAnalysis:
    total_cp: float
    factors: List[FactorScore] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_cp": self.total_cp,
            "total_pawns": self.total_cp / 100.0,
            "factors": [
                {"name": f.name, "score_cp": f.score_cp, "description": f.description}
                for f in self.factors
            ],
        }


class ExpertEvaluator:
    def evaluate(self, board: chess.Board) -> ExpertAnalysis:
        stm = board.turn
        factors: List[FactorScore] = []

        material = self._material(board, stm)
        factors.append(FactorScore("material", material, "Материальный баланс"))

        center = self._center_control(board, stm)
        factors.append(FactorScore("center", center, "Контроль центра"))

        king_safety = self._king_safety(board, stm)
        factors.append(
            FactorScore("king_safety", king_safety, "Безопасность короля")
        )

        pawns = self._pawn_structure(board, stm)
        factors.append(FactorScore("pawn_structure", pawns, "Пешечная структура"))

        activity = self._piece_activity(board, stm)
        factors.append(FactorScore("activity", activity, "Активность фигур"))

        bishops = self._bishop_pair(board, stm)
        factors.append(FactorScore("bishop_pair", bishops, "Пара слонов"))

        rooks = self._rook_files(board, stm)
        factors.append(FactorScore("rook_files", rooks, "Ладьи на открытых линиях"))

        total = sum(f.score_cp for f in factors)
        return ExpertAnalysis(total_cp=total, factors=factors)

    def _material(self, board: chess.Board, stm: chess.Color) -> float:
        white = sum(
            len(board.pieces(pt, chess.WHITE)) * PIECE_VALUES[pt]
            for pt in PIECE_VALUES
            if pt != chess.KING
        )
        black = sum(
            len(board.pieces(pt, chess.BLACK)) * PIECE_VALUES[pt]
            for pt in PIECE_VALUES
            if pt != chess.KING
        )
        diff = white - black
        return diff if stm == chess.WHITE else -diff

    def _center_control(self, board: chess.Board, stm: chess.Color) -> float:
        score = 0.0
        for sq in CENTER_SQUARES:
            w = bool(board.attackers(chess.WHITE, sq))
            b = bool(board.attackers(chess.BLACK, sq))
            if w and not b:
                score += 15
            elif b and not w:
                score -= 15
            elif w and b:
                score += 5 if stm == chess.WHITE else -5
        return score if stm == chess.WHITE else -score

    def _king_safety(self, board: chess.Board, stm: chess.Color) -> float:
        king_sq = board.king(stm)
        if king_sq is None:
            return 0.0
        attackers = len(board.attackers(not stm, king_sq))
        defenders = len(board.attackers(stm, king_sq))
        shield = 0
        rank = chess.square_rank(king_sq)
        file = chess.square_file(king_sq)
        for df in (-1, 0, 1):
            f = file + df
            if 0 <= f <= 7:
                sq = chess.square(f, rank + (1 if stm == chess.WHITE else -1))
                if 0 <= chess.square_rank(sq) <= 7:
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == stm:
                        shield += 1
        safety = (defenders - attackers) * 25 + shield * 20
        if board.is_check():
            safety -= 80 if board.turn == stm else 0
        return safety

    def _pawn_structure(self, board: chess.Board, stm: chess.Color) -> float:
        score = 0.0
        pawns = board.pieces(chess.PAWN, stm)
        files_occupied = set()
        for sq in pawns:
            f = chess.square_file(sq)
            if f in files_occupied:
                score -= 25
            files_occupied.add(f)
            r = chess.square_rank(sq)
            if stm == chess.WHITE:
                if r >= 4:
                    score += 10 * (r - 3)
            else:
                if r <= 3:
                    score += 10 * (4 - r)
        return score

    @staticmethod
    def _attack_square_count(mask: int) -> int:
        return mask.bit_count()

    def _piece_activity(self, board: chess.Board, stm: chess.Color) -> float:
        mobility = 0
        for sq in board.pieces(chess.KNIGHT, stm):
            mobility += self._attack_square_count(board.attacks_mask(sq)) - 2
        for sq in board.pieces(chess.BISHOP, stm):
            mobility += self._attack_square_count(board.attacks_mask(sq)) - 7
        for sq in board.pieces(chess.ROOK, stm):
            mobility += self._attack_square_count(board.attacks_mask(sq)) - 10
        for sq in board.pieces(chess.QUEEN, stm):
            mobility += self._attack_square_count(board.attacks_mask(sq)) - 20
        return mobility * 2

    def _bishop_pair(self, board: chess.Board, stm: chess.Color) -> float:
        bishops = len(board.pieces(chess.BISHOP, stm))
        opp = len(board.pieces(chess.BISHOP, not stm))
        if bishops >= 2 and opp < 2:
            return 40
        return 0

    def _rook_files(self, board: chess.Board, stm: chess.Color) -> float:
        score = 0.0
        for sq in board.pieces(chess.ROOK, stm):
            f = chess.square_file(sq)
            open_file = True
            semi_open = True
            for r in range(8):
                p = board.piece_at(chess.square(f, r))
                if p and p.piece_type == chess.PAWN:
                    open_file = False
                    if p.color == stm:
                        semi_open = False
            if open_file:
                score += 35
            elif semi_open:
                score += 18
        return score
