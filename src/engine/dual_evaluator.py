import torch
import chess
import numpy as np
from src.engine.chess_dual_net import ChessDualNet
from src.utils.position_to_tensor_converter import fen_to_tensor_18ch

class DualEvaluator:
    _instance = None
    _model = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def initialize(self, model_path: str, device=None):
        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device

        self._model = ChessDualNet(input_channels=18, dropout=0.1)
        state_dict = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()
        print(f"DualEvaluator инициализирован на {self._device}, модель: {model_path}")

    def evaluate(self, board: chess.Board):
        tensor = fen_to_tensor_18ch(board.fen())
        tensor = torch.tensor(tensor, dtype=torch.float32).permute(2,0,1).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            value, policy_logits = self._model(tensor)
            value = value.item()
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
        return value, policy_probs

    def get_value(self, board: chess.Board) -> float:
        value, _ = self.evaluate(board)
        return value

    def get_policy(self, board: chess.Board) -> np.ndarray:
        _, policy = self.evaluate(board)
        return policy