import pygame
import sys
from src.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_SIZE,
    NEW_GAME_POS, ANALYZE_POS, BUTTON_WIDTH, BUTTON_HEIGHT,
    START_BOARD, FONT_SIZE, WHITE, BLACK
)
from src.gui.board_drawer import BoardDrawer
from src.gui.buttons import Button
from src.gui.drag_handler import DragHandler
from src.utils.image_loader import load_piece_images


class App:
    """Главный класс приложения, управляющий циклом, событиями и отрисовкой."""
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess Analyzer")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, FONT_SIZE)

        self.piece_images = load_piece_images()

        self.board_drawer = BoardDrawer(self.screen, self.piece_images)
        self.board_drawer.set_board([row[:] for row in START_BOARD])

        self.buttons = self._create_buttons()
        self.drag_handler = DragHandler(self.board_drawer)

        self.running = True

    def _create_buttons(self):
        """Создаёт и возвращает список кнопок."""
        new_game_btn = Button(
            NEW_GAME_POS[0], NEW_GAME_POS[1],
            BUTTON_WIDTH, BUTTON_HEIGHT, "Новая игра", self.font
        )
        analyze_btn = Button(
            ANALYZE_POS[0], ANALYZE_POS[1],
            BUTTON_WIDTH, BUTTON_HEIGHT, "Анализ", self.font
        )
        return [new_game_btn, analyze_btn]

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Обработка кнопок
            for btn in self.buttons:
                if btn.handle_event(event):
                    if btn.text == "Новая игра":
                        self.board_drawer.set_board([row[:] for row in START_BOARD])
                        self.drag_handler.dragging = False
                    elif btn.text == "Анализ":
                        print("Анализ (текущая доска):", self.board_drawer.board)

            self.drag_handler.handle_event(event)

    def draw(self):
        """Отрисовывает все элементы."""
        self.screen.fill(WHITE)

        self.board_drawer.draw_board()
        self.board_drawer.draw_pieces()
        self.drag_handler.draw_dragged_piece(self.screen)

        for btn in self.buttons:
            btn.draw(self.screen)

        turn_text = self.font.render("Ход белых", True, BLACK)
        self.screen.blit(turn_text, (50, BOARD_SIZE + 70))

        pygame.display.flip()

    def run(self):
        """Главный цикл приложения."""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()