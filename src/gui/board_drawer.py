import pygame
from src.constants import LIGHT, DARK, SQUARE_SIZE

class BoardDrawer:
    def __init__(self, screen, piece_images):
        self.screen = screen
        self.piece_images = piece_images
        self.square_size = SQUARE_SIZE
        self.board = None

    def set_board(self, board):
        self.board = board

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = LIGHT if (row + col) % 2 == 0 else DARK
                rect = pygame.Rect(col * self.square_size,
                                   row * self.square_size,
                                   self.square_size, self.square_size)
                pygame.draw.rect(self.screen, color, rect)

    def draw_pieces(self):
        if self.board is None:
            return
        for row in range(8):
            for col in range(8):
                piece_code = self.board[row][col]
                if piece_code == '':
                    continue
                image = self.piece_images.get(piece_code)
                if image:
                    scaled = pygame.transform.smoothscale(image,
                                    (self.square_size, self.square_size))
                    self.screen.blit(scaled,
                                     (col * self.square_size, row * self.square_size))