from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import math
import chess

from chess_engine.expert import ExpertAnalysis, ExpertEvaluator


@dataclass
class CombinedEvaluation:
    total_cp: float
    total_pawns: float
    win_probability: float
    nn_cp: float
    expert_cp: float
    nn_weight: float
    expert_weight: float
    phase: str
    expert_analysis: ExpertAnalysis

    def to_dict(self) -> Dict:
        return {
            "total_cp": self.total_cp,
            "total_pawns": self.total_pawns,
            "win_probability": self.win_probability,
            "nn_cp": self.nn_cp,
            "expert_cp": self.expert_cp,
            "nn_weight": self.nn_weight,
            "expert_weight": self.expert_weight,
            "phase": self.phase,
            "expert": self.expert_analysis.to_dict(),
        }


def value_to_cp(nn_value: float) -> float:
    if abs(nn_value) >= 0.999:
        return math.copysign(800, nn_value)
    return math.atanh(max(-0.999, min(0.999, nn_value))) / 0.3 * 100


def cp_to_win_probability(cp: float) -> float:
    return 1.0 / (1.0 + math.exp(-cp / 400.0))


class HybridCombiner:
    def __init__(
        self,
        opening_threshold: int = 12,
        endgame_threshold: int = 20,
        nn_weight_opening: float = 0.35,
        nn_weight_middlegame: float = 0.65,
        nn_weight_endgame: float = 0.75,
    ) -> None:
        self.opening_threshold = opening_threshold
        self.endgame_threshold = endgame_threshold
        self.nn_weight_opening = nn_weight_opening
        self.nn_weight_middlegame = nn_weight_middlegame
        self.nn_weight_endgame = nn_weight_endgame
        self.expert = ExpertEvaluator()

    def game_phase(self, board: chess.Board) -> str:
        fullmove = board.fullmove_number
        piece_count = len(board.piece_map())
        if fullmove <= self.opening_threshold:
            return "opening"
        if piece_count <= self.endgame_threshold:
            return "endgame"
        return "middlegame"

    def nn_weight_for_phase(self, phase: str) -> float:
        if phase == "opening":
            return self.nn_weight_opening
        if phase == "endgame":
            return self.nn_weight_endgame
        return self.nn_weight_middlegame

    def combine(
        self,
        board: chess.Board,
        nn_value: float,
        expert: Optional[ExpertAnalysis] = None,
    ) -> CombinedEvaluation:
        phase = self.game_phase(board)
        w_nn = self.nn_weight_for_phase(phase)
        w_exp = 1.0 - w_nn

        nn_cp = value_to_cp(nn_value)
        if expert is None:
            expert = self.expert.evaluate(board)
        exp_cp = expert.total_cp

        total_cp = w_nn * nn_cp + w_exp * exp_cp
        stm = board.turn
        if stm == chess.BLACK:
            pass
        win_prob = cp_to_win_probability(total_cp)

        return CombinedEvaluation(
            total_cp=total_cp,
            total_pawns=total_cp / 100.0,
            win_probability=win_prob,
            nn_cp=nn_cp,
            expert_cp=exp_cp,
            nn_weight=w_nn,
            expert_weight=w_exp,
            phase=phase,
            expert_analysis=expert,
        )
