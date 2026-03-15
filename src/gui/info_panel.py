import pygame
import chess
import textwrap
from src.utils.chess_utils import material_score

class InfoPanel:
    def __init__(self, rect, font):
        self.rect = rect
        self.font = font
        self.fen = ""
        self.material = 0.0
        self.turn = "Белые"

    def update(self, game_state):
        self.fen = game_state.board.fen()
        self.material = material_score(game_state.board)
        self.turn = "Белые" if game_state.board.turn == chess.WHITE else "Чёрные"

    def draw(self, screen):
        pygame.draw.rect(screen, (220, 220, 220), self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        y = self.rect.y + 5
        turn_surf = self.font.render(f"Очередь: {self.turn}", True, (0, 0, 0))
        screen.blit(turn_surf, (self.rect.x + 5, y))
        y += 25

        material_surf = self.font.render(f"Материал: {self.material:+.2f}", True, (0, 0, 0))
        screen.blit(material_surf, (self.rect.x + 5, y))
        y += 25

        fen_lines = textwrap.wrap(self.fen, width=45)
        for line in fen_lines:
            fen_surf = self.font.render(line, True, (0, 0, 0))
            screen.blit(fen_surf, (self.rect.x + 5, y))
            y += 25
            if y > self.rect.bottom - 5:
                break