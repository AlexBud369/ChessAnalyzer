import pygame
import sys
import chess
import tkinter as tk
from tkinter import filedialog

from src.constants import (
    WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_SIZE,
    BUTTON_WIDTH, BUTTON_HEIGHT,
    FONT_SIZE, WHITE,
    HISTORY_PANEL_WIDTH, INFO_PANEL_HEIGHT
)
from src.gui.board_drawer import BoardDrawer
from src.gui.buttons import Button
from src.gui.drag_handler import DragHandler
from src.gui.history_panel import HistoryPanel
from src.gui.info_panel import InfoPanel
from src.gui.promotion_dialog import PromotionDialog
from src.gui.game_over_dialog import GameOverDialog

from src.engine import evaluation
from src.game.game_state import GameState
from src.utils.image_loader import load_piece_images
from src.messages import (
    BUTTON_NEW_GAME, BUTTON_ANALYZE, BUTTON_SAVE_FEN, BUTTON_LOAD_FEN,
    APP_CAPTION
)


class App:
    """Главный класс приложения."""

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption(APP_CAPTION)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, FONT_SIZE)

        self.piece_images = load_piece_images()
        self.game_state = GameState()

        self.history_panel = self._create_history_panel()
        self.info_panel = self._create_info_panel()
        self.board_drawer = BoardDrawer(self.screen, self.piece_images)
        self.board_drawer.set_board(self.game_state.board)

        self.drag_handler = DragHandler(self, self.board_drawer)

        self.buttons = self._init_buttons()

        self.running = True

    def _create_history_panel(self):
        rect = pygame.Rect(
            BOARD_SIZE + 10, 10,
            HISTORY_PANEL_WIDTH, WINDOW_HEIGHT - INFO_PANEL_HEIGHT - 20
        )
        return HistoryPanel(rect, self.font, self.game_state, self.navigate_to_move)

    def _create_info_panel(self):
        rect = pygame.Rect(
            BOARD_SIZE + 10, WINDOW_HEIGHT - INFO_PANEL_HEIGHT - 10,
            HISTORY_PANEL_WIDTH, INFO_PANEL_HEIGHT
        )
        panel = InfoPanel(rect, self.font)
        panel.update(self.game_state)
        return panel

    def _init_buttons(self):
        btn_x = 50
        btn_y = BOARD_SIZE + 20
        return [
            Button(btn_x, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_NEW_GAME, self.font),
            Button(btn_x + BUTTON_WIDTH + 10, btn_y, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_ANALYZE, self.font),
            Button(btn_x + 2 * (BUTTON_WIDTH + 10), btn_y, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_SAVE_FEN, self.font),
            Button(btn_x, btn_y + BUTTON_HEIGHT + 10, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_LOAD_FEN, self.font),
        ]

    def attempt_move(self, from_square, to_square):
        """Пытается выполнить ход. Возвращает True при успехе."""
        promotion = None
        if self.game_state.is_promotion_move(from_square, to_square):
            temp_move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if temp_move not in self.game_state.board.legal_moves:
                return False
            dialog = PromotionDialog(self.screen, self.game_state.board.turn)
            promotion = dialog.run()

        if self.game_state.try_move(from_square, to_square, promotion):
            self.update_after_move()
            return True
        return False

    def update_after_move(self):
        """Обновляет интерфейс после изменения позиции"""
        self.info_panel.update(self.game_state)
        self.board_drawer.set_board(self.game_state.board)

        result = self.game_state.get_game_result()
        if result:
            dialog = GameOverDialog(self.screen, result['message'], result['code'])
            dialog.run()

    def navigate_to_move(self, index):
        """Переходит к позиции после указанного хода (1-based)"""
        if self.game_state.go_to_index(index):
            self.update_after_move()

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            for btn in self.buttons:
                if btn.handle_event(event):
                    if btn.text == BUTTON_NEW_GAME:
                        self._handle_new_game()
                    elif btn.text == BUTTON_ANALYZE:
                        self._handle_analyze()
                    elif btn.text == BUTTON_SAVE_FEN:
                        self._handle_save_fen()
                    elif btn.text == BUTTON_LOAD_FEN:
                        self._handle_load_fen()

            self.history_panel.handle_event(event)
            self.drag_handler.handle_event(event)

    def _handle_new_game(self):
        self.game_state.reset_to_start()
        self.update_after_move()

    def _handle_analyze(self):
        best_move, evaluation_value = self.game_state.get_best_move(depth=3, evaluator=evaluation.evaluate_nn)
        if best_move is not None:
            san = self.game_state.board.san(best_move)
            self.info_panel.update_analysis(evaluation_value, san)
            print(f"Лучший ход: {san}, оценка: {evaluation_value:.2f}")
        else:
            self.info_panel.update_analysis(None, None)
            print("Игра окончена, ходов нет.")
        self.info_panel.update(self.game_state)

    def _handle_save_fen(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".fen",
            filetypes=[("FEN files", "*.fen"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.game_state.board.fen())

    def _handle_load_fen(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("FEN files", "*.fen"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "r") as f:
                fen = f.read().strip()
            try:
                self.game_state.set_initial_fen(fen)
                self.update_after_move()
            except ValueError as e:
                print("Ошибка загрузки FEN:", e)

    def save_fen(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.asksaveasfilename(
            defaultextension=".fen",
            filetypes=[("FEN files", "*.fen"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "w") as f:
                f.write(self.game_state.board.fen())

    def load_fen(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            filetypes=[("FEN files", "*.fen"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, "r") as f:
                fen = f.read().strip()
            try:
                self.game_state.set_initial_fen(fen)
                self.update_after_move()
            except ValueError as e:
                print("Ошибка загрузки FEN:", e)

    def render(self):
        """Отрисовывает все элементы интерфейса."""
        self.screen.fill(WHITE)

        self.board_drawer.draw_board()
        self.board_drawer.draw_pieces()
        self.drag_handler.draw_dragged_piece(self.screen)

        self.history_panel.draw(self.screen)
        self.info_panel.draw(self.screen)

        for btn in self.buttons:
            btn.draw(self.screen)

        pygame.display.flip()

    def run(self):
        """Главный цикл приложения."""
        while self.running:
            self.process_events()
            self.render()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


def main():
    app = App()
    app.run()


if __name__ == "__main__":
    main()