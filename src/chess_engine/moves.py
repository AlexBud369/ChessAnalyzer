from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import chess
import numpy as np
import torch


MoveKey = Tuple[int, int, Optional[int]]


class MoveEncoder:
    def __init__(self) -> None:
        self.move_to_index: Dict[MoveKey, int] = {}
        self.index_to_move: List[chess.Move] = []
        self._build_vocabulary()

    def _build_vocabulary(self) -> None:
        idx = 0
        for from_sq in range(64):
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                key: MoveKey = (from_sq, to_sq, None)
                self.move_to_index[key] = idx
                self.index_to_move.append(chess.Move(from_sq, to_sq))
                idx += 1

        for from_sq in range(64):
            fr = chess.square_rank(from_sq)
            ff = chess.square_file(from_sq)
            for to_sq in range(64):
                if from_sq == to_sq:
                    continue
                tr = chess.square_rank(to_sq)
                tf = chess.square_file(to_sq)
                for promo in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
                    if not self._is_underpromotion(fr, ff, tr, tf, promo):
                        continue
                    key = (from_sq, to_sq, promo)
                    if key in self.move_to_index:
                        continue
                    self.move_to_index[key] = idx
                    self.index_to_move.append(
                        chess.Move(from_sq, to_sq, promotion=promo)
                    )
                    idx += 1

    @staticmethod
    def _is_underpromotion(
        fr: int, ff: int, tr: int, tf: int, promo: int
    ) -> bool:
        if promo not in (chess.ROOK, chess.BISHOP, chess.KNIGHT):
            return False
        if fr == 6 and tr == 7 and abs(tf - ff) <= 1:
            return True
        if fr == 1 and tr == 0 and abs(tf - ff) <= 1:
            return True
        return False

    @property
    def num_moves(self) -> int:
        return len(self.index_to_move)

    def move_key(self, move: chess.Move) -> MoveKey:
        promo = move.promotion
        if promo == chess.QUEEN:
            promo = None
        return (move.from_square, move.to_square, promo)

    def encode_move(self, move: chess.Move) -> int:
        key = self.move_key(move)
        if key not in self.move_to_index:
            if move.promotion == chess.QUEEN:
                key = (move.from_square, move.to_square, None)
            if key not in self.move_to_index:
                raise ValueError(f"Ход не в словаре: {move.uci()}")
        return self.move_to_index[key]

    def decode_index(self, index: int) -> chess.Move:
        return self.index_to_move[index]

    def legal_mask(self, board: chess.Board) -> np.ndarray:
        mask = np.zeros(self.num_moves, dtype=bool)
        for move in board.legal_moves:
            key = self.move_key(move)
            if key in self.move_to_index:
                mask[self.move_to_index[key]] = True
            elif move.promotion == chess.QUEEN:
                key = (move.from_square, move.to_square, None)
                if key in self.move_to_index:
                    mask[self.move_to_index[key]] = True
        return mask

    def apply_policy_mask(
        self, logits: torch.Tensor, board: chess.Board
    ) -> torch.Tensor:
        mask = self.legal_mask(board)
        out = logits.clone()
        illegal = ~torch.from_numpy(mask).to(logits.device)
        out[illegal] = float("-inf")
        return out

    def policy_to_probs(
        self, logits: torch.Tensor, board: chess.Board
    ) -> np.ndarray:
        masked = self.apply_policy_mask(logits, board)
        probs = torch.softmax(masked, dim=-1).cpu().numpy()
        return probs

    def top_moves(
        self,
        logits: torch.Tensor,
        board: chess.Board,
        n: int = 10,
    ) -> List[Tuple[chess.Move, float]]:
        probs = self.policy_to_probs(logits, board)
        indices = np.argsort(probs)[::-1][:n]
        result = []
        for i in indices:
            if probs[i] <= 0:
                break
            move = self.decode_index(int(i))
            if move in board.legal_moves:
                result.append((move, float(probs[i])))
        return result


MOVE_ENCODER = MoveEncoder()
