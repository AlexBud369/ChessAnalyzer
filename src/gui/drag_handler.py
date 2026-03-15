import pygame
import chess
from src.utils.helpers import get_square_under_mouse
from src.chess_constants import (
    BOARD_FILES, MAX_RANK_INDEX,
    MIN_SQUARE_INDEX, MAX_SQUARE_INDEX,
    INITIAL_DRAG_OFFSET, INITIAL_MOUSE_POS
)

class DragHandler:
    def __init__(self, app, board_drawer):
        self.app = app
        self.board_drawer = board_drawer
        self.dragging = False
        self.drag_piece = None
        self.drag_start_pos = None
        self.drag_offset = INITIAL_DRAG_OFFSET
        self.mouse_pos = INITIAL_MOUSE_POS

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == pygame.BUTTON_LEFT:
            self._handle_mouse_down(event.pos)
        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_motion(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == pygame.BUTTON_LEFT:
            self._handle_mouse_up(event.pos)

    def _handle_mouse_down(self, pos):
        if self.app.game_state.board.is_game_over():
            return

        row, col = get_square_under_mouse(pos)
        if row is None:
            return

        square = (MAX_RANK_INDEX - row) * BOARD_FILES + col
        piece = self.app.game_state.board.piece_at(square)

        if piece and piece.color == self.app.game_state.board.turn:
            self.dragging = True
            self.drag_piece = piece
            self.drag_start_pos = (row, col)
            self.board_drawer.set_drag_from((row, col))

            square_x = col * self.board_drawer.square_size
            square_y = row * self.board_drawer.square_size
            self.drag_offset = (pos[0] - square_x, pos[1] - square_y)

    def _handle_mouse_motion(self, pos):
        if self.dragging:
            self.mouse_pos = pos

    def _handle_mouse_up(self, pos):
        if not self.dragging:
            return

        row, col = get_square_under_mouse(pos)
        if row is not None:
            from_square = (MAX_RANK_INDEX - self.drag_start_pos[0]) * BOARD_FILES + self.drag_start_pos[1]
            to_square = (MAX_RANK_INDEX - row) * BOARD_FILES + col

            if MIN_SQUARE_INDEX <= from_square <= MAX_SQUARE_INDEX and \
               MIN_SQUARE_INDEX <= to_square <= MAX_SQUARE_INDEX:
                self.app.attempt_move(from_square, to_square)

        self.dragging = False
        self.drag_piece = None
        self.board_drawer.set_drag_from(None)
        self.drag_offset = INITIAL_DRAG_OFFSET
        self.mouse_pos = INITIAL_MOUSE_POS

    def draw_dragged_piece(self, screen):
        if not self.dragging or not self.drag_piece:
            return

        color_code = 'w' if self.drag_piece.color == chess.WHITE else 'b'
        piece_type = self.drag_piece.symbol().upper()
        code = color_code + piece_type
        image = self.board_drawer.piece_images.get(code)

        if image:
            scaled = pygame.transform.smoothscale(
                image, (self.board_drawer.square_size, self.board_drawer.square_size)
            )
            x = self.mouse_pos[0] - self.drag_offset[0]
            y = self.mouse_pos[1] - self.drag_offset[1]
            screen.blit(scaled, (x, y))