from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import pygame

from chess_engine.engine import ChessEngine

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
SAVES_DIR = ROOT / "saves"

SQUARE_SIZE = 64
BOARD_PX = SQUARE_SIZE * 8
PANEL_W = 300
WIN_W = BOARD_PX + PANEL_W
WIN_H = max(BOARD_PX, 680)

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
SELECT = (186, 202, 68)
LAST = (246, 246, 130)
LEGAL = (106, 186, 106)
BTN = (70, 110, 160)
BTN_H = (90, 140, 200)
TEXT = (240, 240, 240)
BG = (45, 45, 48)


def piece_key(piece: chess.Piece) -> str:
    c = "w" if piece.color == chess.WHITE else "b"
    return f"{c}{piece.symbol().upper()}"


def _prepare_piece_image(img: pygame.Surface) -> pygame.Surface:
    img = img.convert_alpha()
    img.set_colorkey((0, 0, 0), pygame.RLEACCEL)
    return pygame.transform.smoothscale(img, (SQUARE_SIZE, SQUARE_SIZE))


class Button:
    def __init__(self, rect: pygame.Rect, label: str) -> None:
        self.rect = rect
        self.label = label

    def draw(self, surf: pygame.Surface, font: pygame.font.Font) -> None:
        mouse = pygame.mouse.get_pos()
        col = BTN_H if self.rect.collidepoint(mouse) else BTN
        pygame.draw.rect(surf, col, self.rect, border_radius=4)
        txt = font.render(self.label, True, TEXT)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, pos: Tuple[int, int]) -> bool:
        return self.rect.collidepoint(pos)


class ChessGUI:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("ChessEngine")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.font_sm = pygame.font.SysFont("consolas", 14)
        self.font_lg = pygame.font.SysFont("consolas", 18, bold=True)

        self.pieces = self._load_images()
        self.engine: Optional[ChessEngine] = None
        self.engine_lock = threading.Lock()
        self.search_depth = 4

        self.board = chess.Board()
        self.move_history: List[chess.Move] = []
        self.selected: Optional[int] = None
        self.last_move: Optional[Tuple[int, int]] = None
        self.human_color = chess.WHITE
        self.flip = False
        self.vs_model = False
        self.ai_thinking = False
        self.status = "Готово. Загрузка модели..."
        self.analysis_lines: List[str] = []
        self.notation: List[str] = []

        self.buttons = self._make_buttons()
        self._start_engine_load()

    def _load_images(self) -> Dict[str, pygame.Surface]:
        imgs: Dict[str, pygame.Surface] = {}
        for f in IMAGES_DIR.glob("*.png"):
            key = f.stem
            raw = pygame.image.load(str(f))
            imgs[key] = _prepare_piece_image(raw)
        return imgs

    def _make_buttons(self) -> List[Button]:
        x0 = BOARD_PX + 12
        w, h, gap = 276, 34, 8
        labels = [
            "Анализ (ходы + оценка)",
            "Лучший ход (поиск)",
            "Игра: человек vs модель",
            "Сменить цвет человека",
            "Новая партия",
            "Загрузить PGN",
            "Сохранить PGN",
        ]
        btns = []
        y = 12
        for lab in labels:
            btns.append(Button(pygame.Rect(x0, y, w, h), lab))
            y += h + gap
        return btns

    def _start_engine_load(self) -> None:
        def worker() -> None:
            try:
                ckpt = ROOT / "checkpoints" / "best.pt"
                eng = ChessEngine(
                    checkpoint_path=str(ckpt) if ckpt.exists() else None,
                    config_path=str(ROOT / "config.yaml"),
                )
                with self.engine_lock:
                    self.engine = eng
                    self.search_depth = eng.search_depth
                self.status = "Модель загружена."
            except Exception as e:
                self.status = f"Ошибка модели: {e}"

        threading.Thread(target=worker, daemon=True).start()

    def _sq_at(self, mx: int, my: int) -> Optional[int]:
        if mx >= BOARD_PX or my >= BOARD_PX:
            return None
        f = mx // SQUARE_SIZE
        r = 7 - (my // SQUARE_SIZE)
        if self.flip:
            f, r = 7 - f, 7 - r
        return chess.square(f, r)

    def _sq_rect(self, sq: int) -> pygame.Rect:
        f, r = chess.square_file(sq), chess.square_rank(sq)
        if self.flip:
            f, r = 7 - f, 7 - r
        x = f * SQUARE_SIZE
        y = (7 - r) * SQUARE_SIZE
        return pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)

    def _rebuild_notation(self) -> None:
        b = chess.Board()
        self.notation = []
        for i, mv in enumerate(self.move_history):
            san = b.san(mv)
            b.push(mv)
            if i % 2 == 0:
                self.notation.append(f"{i // 2 + 1}. {san}")
            else:
                self.notation[-1] += f" {san}"

    def _reset_game(self) -> None:
        self.board = chess.Board()
        self.move_history = []
        self.selected = None
        self.last_move = None
        self.notation = []
        self.analysis_lines = []
        self.status = "Новая партия."
        if self.vs_model and self.board.turn != self.human_color:
            self._schedule_ai_move()

    def _toggle_vs_model(self) -> None:
        self.vs_model = not self.vs_model
        self.status = (
            "Режим: человек vs модель ВКЛ." if self.vs_model else "Режим: анализ (свободные ходы)."
        )
        if self.vs_model and self.board.turn != self.human_color and not self.board.is_game_over():
            self._schedule_ai_move()

    def _toggle_human_color(self) -> None:
        self.human_color = chess.BLACK if self.human_color == chess.WHITE else chess.WHITE
        self.flip = self.human_color == chess.BLACK
        self.status = (
            "Вы играете белыми." if self.human_color == chess.WHITE else "Вы играете чёрными."
        )
        if self.vs_model and self.board.turn != self.human_color and not self.board.is_game_over():
            self._schedule_ai_move()

    def _apply_move(self, move: chess.Move) -> None:
        if move not in self.board.legal_moves:
            return
        self.board.push(move)
        self.move_history.append(move)
        self._rebuild_notation()
        self.last_move = (move.from_square, move.to_square)
        self.selected = None
        if self.board.is_game_over():
            self.status = f"Конец: {self.board.result()}"
            return
        if self.vs_model and self.board.turn != self.human_color:
            self._schedule_ai_move()

    def _schedule_ai_move(self) -> None:
        if self.ai_thinking or self.engine is None:
            return
        self.ai_thinking = True
        self.status = "Модель думает..."

        fen = self.board.fen()
        depth = self.search_depth

        def worker() -> None:
            try:
                with self.engine_lock:
                    eng = self.engine
                move = eng.best_move(fen, depth=depth) if eng else None
                if move and move in self.board.legal_moves:
                    san = self.board.san(move)
                    self._apply_move(move)
                    self.status = f"Модель: {san}"
            except Exception as e:
                self.status = f"Ошибка хода модели: {e}"
            finally:
                self.ai_thinking = False

        threading.Thread(target=worker, daemon=True).start()

    def _run_analysis(self, with_search: bool = False) -> None:
        if self.engine is None:
            self.status = "Модель ещё загружается..."
            return
        fen = self.board.fen()
        depth = self.search_depth if with_search else 0
        self.status = "Анализ..."

        def worker() -> None:
            try:
                with self.engine_lock:
                    eng = self.engine
                data = eng.analyze(fen, depth=depth, top_moves=8)
                ev = data["evaluation"]
                lines = [
                    f"Оценка: {ev['total_pawns']:+.2f} пеш.",
                    f"Win%: {ev['win_probability']*100:.1f}%",
                    f"NN: {ev['nn_cp']:+.0f} cp | Эксперт: {ev['expert_cp']:+.0f} cp",
                    f"Фаза: {ev['phase']}",
                    "— лучшие ходы (NN) —",
                ]
                for i, m in enumerate(data["top_moves"][:8], 1):
                    lines.append(
                        f"{i}. {m['san']} ({m['uci']})  {m['probability']*100:.1f}%"
                    )
                if "search" in data and data["search"]["best_move"]:
                    s = data["search"]
                    lines.append("— поиск —")
                    lines.append(
                        f"depth {s['depth']}: {s['best_move']}  "
                        f"score {s['score_cp']:+.0f} cp"
                    )
                self.analysis_lines = lines
                self.status = "Анализ готов."
            except Exception as e:
                self.status = f"Ошибка анализа: {e}"

        threading.Thread(target=worker, daemon=True).start()

    def _load_pgn(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Загрузить PGN",
                filetypes=[("PGN", "*.pgn"), ("All", "*.*")],
                initialdir=str(SAVES_DIR if SAVES_DIR.exists() else ROOT),
            )
            root.destroy()
            if not path:
                return
            with open(path, encoding="utf-8") as f:
                game = chess.pgn.read_game(f)
            if game is None:
                self.status = "PGN пуст."
                return
            self.board = game.end().board()
            self.move_history = []
            node = game
            while node.variations:
                node = node.variation(0)
                if node.move:
                    self.move_history.append(node.move)
            self._rebuild_notation()
            self.last_move = None
            if len(self.move_history) >= 1:
                m = self.move_history[-1]
                self.last_move = (m.from_square, m.to_square)
            self.status = f"Загружено: {Path(path).name}"
            if self.vs_model and self.board.turn != self.human_color:
                self._schedule_ai_move()
        except Exception as e:
            self.status = f"Ошибка загрузки: {e}"

    def _save_pgn(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            SAVES_DIR.mkdir(exist_ok=True)
            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Сохранить PGN",
                defaultextension=".pgn",
                filetypes=[("PGN", "*.pgn")],
                initialdir=str(SAVES_DIR),
                initialfile="game.pgn",
            )
            root.destroy()
            if not path:
                return
            game = chess.pgn.Game()
            game.headers["Event"] = "ChessEngine GUI"
            game.headers["Site"] = "local"
            node = game
            replay = chess.Board()
            for mv in self.move_history:
                node = node.add_variation(mv)
                replay.push(mv)
            with open(path, "w", encoding="utf-8") as f:
                print(game, file=f)
            self.status = f"Сохранено: {Path(path).name}"
        except Exception as e:
            self.status = f"Ошибка сохранения: {e}"

    def _on_click(self, pos: Tuple[int, int]) -> None:
        if self.ai_thinking:
            return
        for i, btn in enumerate(self.buttons):
            if btn.hit(pos):
                if i == 0:
                    self._run_analysis(with_search=False)
                elif i == 1:
                    self._run_analysis(with_search=True)
                elif i == 2:
                    self._toggle_vs_model()
                elif i == 3:
                    self._toggle_human_color()
                elif i == 4:
                    self._reset_game()
                elif i == 5:
                    self._load_pgn()
                elif i == 6:
                    self._save_pgn()
                return

        if self.vs_model and self.board.turn != self.human_color:
            return
        sq = self._sq_at(*pos)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        if self.selected is None:
            if piece and piece.color == self.board.turn:
                self.selected = sq
            return
        if sq == self.selected:
            self.selected = None
            return
        move = self._find_move(self.selected, sq)
        if move:
            self._apply_move(move)
        elif piece and piece.color == self.board.turn:
            self.selected = sq
        else:
            self.selected = None

    def _find_move(self, fr: int, to: int) -> Optional[chess.Move]:
        for m in self.board.legal_moves:
            if m.from_square == fr and m.to_square == to:
                return m
        p = self.board.piece_at(fr)
        if p and p.piece_type == chess.PAWN and chess.square_rank(to) in (0, 7):
            return chess.Move(fr, to, promotion=chess.QUEEN)
        return None

    def _draw_board(self) -> None:
        legal_targets = set()
        if self.selected is not None:
            for m in self.board.legal_moves:
                if m.from_square == self.selected:
                    legal_targets.add(m.to_square)

        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            is_light = (f + r) % 2 == 0
            col = LIGHT if is_light else DARK
            if self.last_move and sq in self.last_move:
                col = LAST
            if self.selected == sq:
                col = SELECT
            if sq in legal_targets:
                col = LEGAL
            pygame.draw.rect(self.screen, col, self._sq_rect(sq))

        for sq in chess.SQUARES:
            p = self.board.piece_at(sq)
            if not p:
                continue
            key = piece_key(p)
            img = self.pieces.get(key)
            if img:
                self.screen.blit(img, self._sq_rect(sq))

    def _draw_panel(self) -> None:
        px = BOARD_PX
        pygame.draw.rect(self.screen, BG, pygame.Rect(px, 0, PANEL_W, WIN_H))
        y = 12 + len(self.buttons) * (34 + 8) + 4
        for btn in self.buttons:
            btn.draw(self.screen, self.font_sm)

        lines = [self.status, "", "— нотация —"] + self.notation[-12:]
        for ln in lines:
            surf = self.font_sm.render(ln[:42], True, TEXT)
            self.screen.blit(surf, (px + 12, y))
            y += 18
            if y > WIN_H - 200:
                break

        y = max(y + 8, WIN_H - 190)
        self.screen.blit(self.font_lg.render("— анализ —", True, TEXT), (px + 12, y))
        y += 22
        for ln in self.analysis_lines[:8]:
            surf = self.font_sm.render(ln[:40], True, TEXT)
            self.screen.blit(surf, (px + 12, y))
            y += 17

    def run(self) -> None:
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    self._on_click(ev.pos)

            self.screen.fill(BG)
            self._draw_board()
            self._draw_panel()
            pygame.display.flip()
            self.clock.tick(60)


def main() -> None:
    ChessGUI().run()


if __name__ == "__main__":
    main()
