import pygame
import chess
import textwrap
from src.utils.position_evalution import material_score

class InfoPanel:
    def __init__(self, rect, font):
        self.rect = rect
        self.font = font
        self.fen = ""
        self.material = 0.0
        self.turn = "Белые"
        self.analysis_eval = None
        self.analysis_best_move = None

    def update(self, game_state):
        self.fen = game_state.board.fen()
        self.material = material_score(game_state.board)
        self.turn = "Белые" if game_state.board.turn == chess.WHITE else "Чёрные"

    def update_analysis(self, evaluation, best_move_san):
        self.analysis_eval = evaluation
        self.analysis_best_move = best_move_san

    def draw(self, screen):
        pygame.draw.rect(screen, (220, 220, 220), self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        y = self.rect.y + 5
        y = self._draw_line(screen, f"Очередь: {self.turn}", y)
        y = self._draw_line(screen, f"Материал: {self.material:+.2f}", y)

        if self.analysis_eval is not None:
            y = self._draw_line(screen, f"Оценка: {self.analysis_eval:+.2f}", y)

        if self.analysis_best_move is not None:
            y = self._draw_line(screen, f"Лучший ход: {self.analysis_best_move}", y)

        fen_lines = textwrap.wrap(self.fen, width=45)
        for line in fen_lines:
            y = self._draw_line(screen, line, y)
            if y > self.rect.bottom - 5:
                break

    def _draw_line(self, screen, text, y):
        surf = self.font.render(text, True, (0, 0, 0))
        screen.blit(surf, (self.rect.x + 5, y))
        return y + 25