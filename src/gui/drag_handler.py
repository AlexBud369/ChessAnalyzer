import pygame
from ..utils.helpers import get_square_under_mouse

class DragHandler:
    def __init__(self, board_drawer):
        self.board_drawer = board_drawer
        self.dragging = False
        self.drag_piece = None
        self.drag_start_pos = None
        self.drag_offset = (0, 0)
        self.mouse_pos = (0, 0)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            row, col = get_square_under_mouse(pos)
            if row is not None and 0 <= row < 8 and 0 <= col < 8:
                piece = self.board_drawer.board[row][col]
                if piece:
                    self.dragging = True
                    self.drag_piece = piece
                    self.drag_start_pos = (row, col)

                    square_x = col * self.board_drawer.square_size
                    square_y = row * self.board_drawer.square_size
                    self.drag_offset = (pos[0] - square_x, pos[1] - square_y)

                    self.board_drawer.board[row][col] = ''

        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.mouse_pos = event.pos

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                pos = event.pos
                row, col = get_square_under_mouse(pos)
                if row is not None and 0 <= row < 8 and 0 <= col < 8:
                    self.board_drawer.board[row][col] = self.drag_piece
                else:
                    sr, sc = self.drag_start_pos
                    self.board_drawer.board[sr][sc] = self.drag_piece

                self.dragging = False
                self.drag_piece = None

    def draw_dragged_piece(self, screen):
        if self.dragging and self.drag_piece:
            image = self.board_drawer.piece_images.get(self.drag_piece)
            if image:
                scaled = pygame.transform.smoothscale(image,
                            (self.board_drawer.square_size, self.board_drawer.square_size))
                x = self.mouse_pos[0] - self.drag_offset[0]
                y = self.mouse_pos[1] - self.drag_offset[1]
                screen.blit(scaled, (x, y))