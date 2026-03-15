import pygame
import chess
from src.constants import LIGHT, DARK, SQUARE_SIZE
from src.chess_constants import BOARD_RANKS, BOARD_FILES, MAX_RANK_INDEX

class BoardDrawer:
    def __init__(self, screen, piece_images):
        self.screen = screen
        self.piece_images = piece_images
        self.square_size = SQUARE_SIZE
        self.board = None
        self.drag_from = None

    def set_board(self, board):
        self.board = board

    def set_drag_from(self, pos):
        self.drag_from = pos

    def draw_board(self):
        for row in range(BOARD_RANKS):
            for col in range(BOARD_FILES):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                rect = pygame.Rect(
                    col * self.square_size,
                    row * self.square_size,
                    self.square_size, self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)

    def draw_pieces(self):
        if self.board is None:
            return

        for row in range(BOARD_RANKS):
            for col in range(BOARD_FILES):
                if self.drag_from and (row, col) == self.drag_from:
                    continue

                square = (MAX_RANK_INDEX - row) * BOARD_FILES + col
                piece = self.board.piece_at(square)
                if piece:
                    color_code = 'w' if piece.color == chess.WHITE else 'b'
                    piece_type = piece.symbol().upper()
                    code = color_code + piece_type
                    image = self.piece_images.get(code)
                    if image:
                        scaled = pygame.transform.smoothscale(
                            image, (self.square_size, self.square_size)
                        )
                        self.screen.blit(scaled,
                                         (col * self.square_size, row * self.square_size))

    def highlight_square(self, row, col, color):
        rect = pygame.Rect(
            col * self.square_size, row * self.square_size,
            self.square_size, self.square_size
        )
        s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        s.fill(color)
        self.screen.blit(s, rect.topleft)