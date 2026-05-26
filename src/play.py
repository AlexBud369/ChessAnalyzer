import argparse

import chess

from chess_engine.engine import ChessEngine


def main() -> None:
    p = argparse.ArgumentParser(description="Игра против гибридного движка")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--human", choices=("white", "black"), default="white")
    args = p.parse_args()

    engine = ChessEngine(checkpoint_path=args.checkpoint)
    board = chess.Board()

    print("Команды: ход в UCI (e2e4) или SAN, 'quit', 'fen <fen>', 'analyze'")
    while not board.is_game_over():
        print("\n", board)
        print(board.fen())

        if (board.turn == chess.WHITE and args.human == "white") or (
            board.turn == chess.BLACK and args.human == "black"
        ):
            raw = input("Ваш ход: ").strip()
            if raw == "quit":
                break
            if raw.startswith("fen "):
                board.set_fen(raw[4:].strip())
                continue
            if raw == "analyze":
                import json
                print(json.dumps(engine.analyze(board.fen(), depth=args.depth), indent=2))
                continue
            try:
                if len(raw) in (4, 5):
                    move = chess.Move.from_uci(raw)
                else:
                    move = board.parse_san(raw)
            except ValueError as e:
                print(f"Неверный ход: {e}")
                continue
        else:
            move = engine.best_move(board.fen(), depth=args.depth)
            if move is None:
                print("Движок не нашёл ход.")
                break
            print(f"Движок: {board.san(move)} ({move.uci()})")

        if move not in board.legal_moves:
            print("Нелегальный ход.")
            continue
        board.push(move)

    print("Итог:", board.result())


if __name__ == "__main__":
    main()
