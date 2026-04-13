import chess
from chess_constants import PROMOTION_RANK_WHITE, PROMOTION_RANK_BLACK, DEFAULT_PROMOTION
from messages import (
    RESULT_CHECKMATE_WHITE, RESULT_CHECKMATE_BLACK,
    RESULT_STALEMATE, RESULT_INSUFFICIENT_MATERIAL,
    RESULT_FIFTY_MOVES, RESULT_REPETITION
)
from src.engine.minimax_search import get_best_move
from src.engine.evaluation import evaluate

class GameState:
    """
    Управляет состоянием шахматной партии: доска, история ходов, навигация по нотации
    """
    def __init__(self, initial_fen=chess.STARTING_FEN):
        self.initial_fen = initial_fen
        self.moves = []
        self.san_moves = []
        self.current_index = 0
        self.board = chess.Board(initial_fen)

    def reset_to_start(self):
        self.set_initial_fen(chess.STARTING_FEN)

    def set_initial_fen(self, fen):
        self.initial_fen = fen
        self.moves = []
        self.san_moves = []
        self.current_index = 0
        self.board = chess.Board(fen)

    def _is_promotion_move(self, from_square, to_square):
        piece = self.board.piece_at(from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        to_rank = chess.square_rank(to_square)
        return (piece.color == chess.WHITE and to_rank == PROMOTION_RANK_WHITE) or \
               (piece.color == chess.BLACK and to_rank == PROMOTION_RANK_BLACK)

    def is_promotion_move(self, from_square, to_square):
        return self._is_promotion_move(from_square, to_square)

    def _create_move(self, from_square, to_square, promotion):
        if self._is_promotion_move(from_square, to_square):
            promotion = promotion or DEFAULT_PROMOTION
            return chess.Move(from_square, to_square, promotion=promotion)
        return chess.Move(from_square, to_square)

    def try_move(self, from_square, to_square, promotion=None):
        move = self._create_move(from_square, to_square, promotion)

        if move not in self.board.legal_moves:
            return False

        if self.current_index < len(self.moves):
            self.moves = self.moves[:self.current_index]
            self.san_moves = self.san_moves[:self.current_index]

        san = self.board.san(move)

        self.board.push(move)
        self.moves.append(move)
        self.san_moves.append(san)
        self.current_index = len(self.moves)
        return True

    def go_to_index(self, index):
        if 0 <= index <= len(self.moves):
            self.current_index = index
            self.board = chess.Board(self.initial_fen)
            for move in self.moves[:index]:
                self.board.push(move)
            return True
        return False

    def get_san_moves(self):
        return self.san_moves.copy()

    def get_game_result(self):
        board = self.board
        if board.is_checkmate():
            winner = "Белые" if board.turn == chess.BLACK else "Чёрные"
            message = RESULT_CHECKMATE_WHITE if winner == "Белые" else RESULT_CHECKMATE_BLACK
            return {"is_terminal": True, "message": message, "code": "checkmate"}
        if board.is_stalemate():
            return {"is_terminal": True, "message": RESULT_STALEMATE, "code": "stalemate"}
        if board.is_insufficient_material():
            return {"is_terminal": True, "message": RESULT_INSUFFICIENT_MATERIAL, "code": "insufficient_material"}
        if board.is_fifty_moves():
            return {"is_terminal": True, "message": RESULT_FIFTY_MOVES, "code": "fifty_moves"}
        if board.is_repetition(3):
            return {"is_terminal": True, "message": RESULT_REPETITION, "code": "repetition"}
        return None

    def get_best_move(self, depth: int = 3, evaluator=None):
        """
        Возвращает лучший ход и его оценку,
        используя минимакс с указанной глубиной и функцией оценки.
        Если evaluator не передан, используется материальная оценка.
        """
        if evaluator is None:
            evaluator = evaluate
        return get_best_move(self.board, depth, evaluator)