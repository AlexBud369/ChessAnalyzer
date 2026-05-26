from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chess
import torch
import yaml

from chess_engine.combiner import HybridCombiner
from chess_engine.encoding import board_to_planes
from chess_engine.expert import ExpertEvaluator
from chess_engine.model import ChessNet, build_model_from_config
from chess_engine.moves import MOVE_ENCODER
from chess_engine.search import AlphaBetaSearch


class ChessEngine:
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ) -> None:
        with open(config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model_from_config(self.config).to(self.device)
        self.model.eval()

        ckpt = checkpoint_path or self.config["paths"]["best_checkpoint"]
        if Path(ckpt).exists():
            state = torch.load(ckpt, map_location=self.device, weights_only=True)
            if "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            else:
                self.model.load_state_dict(state)
        else:
            print(f"Предупреждение: чекпоинт {ckpt} не найден, используются случайные веса.")

        c = self.config.get("combiner", {})
        self.combiner = HybridCombiner(
            opening_threshold=c.get("opening_threshold", 12),
            endgame_threshold=c.get("endgame_threshold", 20),
            nn_weight_opening=c.get("nn_weight_opening", 0.35),
            nn_weight_middlegame=c.get("nn_weight_middlegame", 0.65),
            nn_weight_endgame=c.get("nn_weight_endgame", 0.75),
        )
        self.expert = ExpertEvaluator()
        self.search_depth = self.config.get("search", {}).get("default_depth", 4)

    @torch.no_grad()
    def _forward(self, board: chess.Board) -> tuple[float, torch.Tensor]:
        planes = board_to_planes(board)
        x = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        policy_logits, value = self.model(x)
        nn_value = float(value.item())
        return nn_value, policy_logits.squeeze(0)

    def analyze(
        self,
        fen: str,
        depth: int = 0,
        top_moves: int = 10,
    ) -> Dict[str, Any]:
        board = chess.Board(fen)
        nn_value, policy_logits = self._forward(board)
        expert = self.expert.evaluate(board)
        combined = self.combiner.combine(board, nn_value, expert)
        candidates = MOVE_ENCODER.top_moves(policy_logits, board, n=top_moves)

        result: Dict[str, Any] = {
            "fen": fen,
            "turn": "white" if board.turn == chess.WHITE else "black",
            "evaluation": combined.to_dict(),
            "nn_value": nn_value,
            "top_moves": [
                {"uci": m.uci(), "san": board.san(m), "probability": p}
                for m, p in candidates
            ],
        }

        if depth > 0:
            searcher = AlphaBetaSearch(self.model, self.device, max_depth=depth)
            sr = searcher.search(board, depth)
            result["search"] = {
                "depth": sr.depth,
                "nodes": sr.nodes,
                "score_cp": sr.score_cp,
                "best_move": sr.best_move.uci() if sr.best_move else None,
                "pv": [m.uci() for m in sr.principal_variation],
            }
        return result

    def best_move(
        self, fen: str, depth: Optional[int] = None
    ) -> Optional[chess.Move]:
        board = chess.Board(fen)
        d = depth if depth is not None else self.search_depth
        if d <= 0:
            _, logits = self._forward(board)
            top = MOVE_ENCODER.top_moves(logits, board, n=1)
            return top[0][0] if top else None
        searcher = AlphaBetaSearch(self.model, self.device, max_depth=d)
        return searcher.search(board, d).best_move
