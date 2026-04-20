"""
Модуль для оценки шахматных позиций с помощью обученной нейросети.
класс NNEvaluator загружает модель и нормализацинные параметры и
метод evaluate_board для получения оценки позиции
"""

import json
import torch
import chess
from src.utils.position_to_tensor_converter import fen_to_tensor_18ch
from src.engine.chess_value_net import ChessValueNet

class NNEvaluator:
    _instance = None

    def __new__(cls, model_path="models/chess_nn.pth", norm_path="models/norm_params.json"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path="models/chess_nn.pth", norm_path="models/norm_params.json"):
        if self._initialized:
            return
        self._initialized = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"NNEvaluator использует устройство: {self.device}")

        with open(norm_path, "r") as f:
            norm_params = json.load(f)
        self.max_abs = norm_params["max_abs"]
        print(f"Загружены параметры нормализации: max_abs = {self.max_abs}")

        self.model = ChessValueNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Модель загружена из {model_path}")

    def evaluate_board(self, board: chess.Board) -> float:
        """
        Оценивает позицию с помощью нейросети.
        :param board: объект python-chess Board
        :return: оценка в пешках (положительная = преимущество белых)
        """
        if board.is_checkmate():
            return -self.max_abs if board.turn == chess.WHITE else self.max_abs
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        fen = board.fen()
        tensor_np = fen_to_tensor_18ch(fen)
        tensor = torch.tensor(tensor_np, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            pred_norm = self.model(tensor).item()

        score = pred_norm * self.max_abs
        return score