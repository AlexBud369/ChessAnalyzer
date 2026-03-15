import pygame
from src.constants import (
    BLACK, PANEL_BG_COLOR, HIGHLIGHT_COLOR, ROW_BG_COLOR, HINT_TEXT_COLOR
)

class HistoryPanel:
    """Панель отображения истории ходов в две колонки (белые/чёрные).
    Позволяет нажимать на ход и переходить к одной из прошлых позиций
    """

    ITEM_HEIGHT = 30
    TEXT_PADDING = 5
    COLUMNS = 2

    def __init__(self, rect, font, game_state, on_move_selected):
        self.rect = rect
        self.font = font
        self.game_state = game_state
        self.on_move_selected = on_move_selected
        self.scroll_offset = 0
        self.visible_items = rect.height // self.ITEM_HEIGHT

    def handle_event(self, event):
        if event.type == pygame.MOUSEWHEEL:
            self.scroll_offset -= event.y
            total_moves = len(self.game_state.moves)
            total_entries = (total_moves + 1) // self.COLUMNS
            max_offset = max(0, total_entries - self.visible_items)
            self.scroll_offset = max(0, min(self.scroll_offset, max_offset))

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                y = event.pos[1] - self.rect.y
                row = y // self.ITEM_HEIGHT + self.scroll_offset
                half_width = self.rect.width // self.COLUMNS
                if event.pos[0] < self.rect.x + half_width:
                    move_idx = row * self.COLUMNS
                else:
                    move_idx = row * self.COLUMNS + 1

                if 0 <= move_idx < len(self.game_state.moves):
                    self.on_move_selected(move_idx + 1)

    def draw(self, screen):
        pygame.draw.rect(screen, PANEL_BG_COLOR, self.rect)
        pygame.draw.rect(screen, BLACK, self.rect, 2)

        moves = self.game_state.get_san_moves()
        if not moves:
            text = self.font.render("Нет ходов", True, HINT_TEXT_COLOR)
            text_rect = text.get_rect(center=self.rect.center)
            screen.blit(text, text_rect)
            return

        total_moves = len(moves)
        start_row = self.scroll_offset
        end_row = min(start_row + self.visible_items, (total_moves + 1) // self.COLUMNS)
        y_offset = self.rect.y

        for row in range(start_row, end_row):
            white_idx = row * self.COLUMNS
            black_idx = row * self.COLUMNS + 1
            move_number = row + 1

            white_move = moves[white_idx] if white_idx < total_moves else ""
            black_move = moves[black_idx] if black_idx < total_moves else ""

            line = f"{move_number}. {white_move:<10} {black_move}"

            current = self.game_state.current_index
            is_current_white = (white_idx + 1) == current
            is_current_black = (black_idx + 1) == current
            bg_color = HIGHLIGHT_COLOR if (is_current_white or is_current_black) else ROW_BG_COLOR

            pygame.draw.rect(screen, bg_color,
                             (self.rect.x, y_offset, self.rect.width, self.ITEM_HEIGHT))

            text_surf = self.font.render(line, True, BLACK)
            screen.blit(text_surf, (self.rect.x + self.TEXT_PADDING, y_offset + self.TEXT_PADDING))

            y_offset += self.ITEM_HEIGHT