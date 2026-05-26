from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from chess_engine.engine import ChessEngine

ROOT = Path(__file__).resolve().parent
DEFAULT_SF = ROOT / "stockfish" / "stockfish-windows-x86-64-avx2.exe"
DEFAULT_OUT = ROOT / "saves" / "eval"


def find_stockfish(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        for p in sorted(path.rglob("stockfish*.exe")):
            return p
        for p in sorted(path.rglob("stockfish")):
            if p.is_file():
                return p
    raise FileNotFoundError(
        f"Stockfish не найден: {path}. Укажите --stockfish путь к .exe"
    )


def score_from_result(result: str, our_color: chess.Color) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if our_color == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if our_color == chess.BLACK else 0.0
    return 0.5


def play_one(
    our_engine: ChessEngine,
    sf: chess.engine.SimpleEngine,
    our_color: chess.Color,
    depth: int,
    sf_limit: chess.engine.Limit,
    *,
    game_index: int,
    total_games: int,
    sf_elo: int,
    out_dir: Path,
) -> tuple[str, int, Path]:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "ChessEngine vs Stockfish"
    game.headers["Site"] = "evaluate.py"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = (
        "ChessEngine" if our_color == chess.WHITE else f"Stockfish Elo {sf_elo}"
    )
    game.headers["Black"] = (
        "ChessEngine" if our_color == chess.BLACK else f"Stockfish Elo {sf_elo}"
    )
    game.headers["Depth"] = str(depth)

    node = game
    plies = 0
    our_color_name = "белые" if our_color == chess.WHITE else "чёрные"

    print(f"\n--- Игра {game_index}/{total_games} | модель {our_color_name} ---", flush=True)

    while not board.is_game_over(claim_draw=True):
        side = "модель" if board.turn == our_color else "Stockfish"
        t0 = time.time()

        if board.turn == our_color:
            move = our_engine.best_move(board.fen(), depth=depth)
        else:
            move = sf.play(board, sf_limit).move

        if move is None or move not in board.legal_moves:
            break

        san = board.san(move)
        board.push(move)
        node = node.add_variation(move)
        plies += 1
        dt = time.time() - t0

        fullmove = (plies + 1) // 2
        print(
            f"  [{game_index}/{total_games}] "
            f"полуход {plies} (ход {fullmove}): {side} {san}  [{dt:.1f}s]",
            flush=True,
        )

    result = board.result(claim_draw=True) or "*"
    game.headers["Result"] = result

    out_dir.mkdir(parents=True, exist_ok=True)
    slug = result.replace("/", "-")
    pgn_path = out_dir / f"game_{game_index:03d}_{slug}.pgn"
    with open(pgn_path, "w", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")

    return result, plies, pgn_path


def main() -> None:
    p = argparse.ArgumentParser(description="Матчи ChessEngine vs Stockfish")
    p.add_argument("--games", type=int, default=5)
    p.add_argument("--elo", type=int, default=1320, help="UCI_Elo Stockfish")
    p.add_argument("--depth", type=int, default=3, help="Глубина нашего alpha-beta")
    p.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "best.pt"))
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--stockfish", default=str(DEFAULT_SF))
    p.add_argument("--sf-time", type=float, default=0.1, help="Секунд на ход SF")
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT),
        help="Папка для сохранения PGN (по умолчанию saves/eval)",
    )
    args = p.parse_args()

    sf_path = find_stockfish(Path(args.stockfish))
    out_dir = Path(args.out_dir)

    print(f"Stockfish: {sf_path}")
    print(f"Модель: {args.checkpoint}, depth={args.depth}")
    print(f"Партий: {args.games}, UCI_Elo={args.elo}")
    print(f"PGN: {out_dir}\n", flush=True)

    our = ChessEngine(checkpoint_path=args.checkpoint, config_path=args.config)
    sf_limit = chess.engine.Limit(time=args.sf_time)

    total_score = 0.0
    t0 = time.time()

    with chess.engine.SimpleEngine.popen_uci(str(sf_path)) as sf:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})
        try:
            sf_opts = sf.options
            if "Skill Level" in sf_opts and args.elo <= 1350:
                sf.configure({"Skill Level": 3})
        except chess.engine.EngineError:
            pass

        for g in range(1, args.games + 1):
            our_color = chess.WHITE if g % 2 == 1 else chess.BLACK
            color_name = "белые" if our_color == chess.WHITE else "чёрные"
            result, plies, pgn_path = play_one(
                our,
                sf,
                our_color,
                args.depth,
                sf_limit,
                game_index=g,
                total_games=args.games,
                sf_elo=args.elo,
                out_dir=out_dir,
            )
            sc = score_from_result(result, our_color)
            total_score += sc
            print(
                f"  >> Итог игры {g}: {result} | {plies} полуходов | "
                f"очко={sc} | сохранено: {pgn_path.name}",
                flush=True,
            )

    elapsed = time.time() - t0
    avg = total_score / args.games
    print(
        f"\nИтого: {total_score}/{args.games} очков ({avg*100:.0f}%) "
        f"за {elapsed/60:.1f} мин",
        flush=True,
    )
    print(f"PGN-файлы: {out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
