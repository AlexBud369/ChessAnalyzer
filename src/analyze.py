import argparse
import json

from chess_engine.engine import ChessEngine


def main() -> None:
    p = argparse.ArgumentParser(description="Анализ шахматной позиции")
    p.add_argument("fen", nargs="?", default=chess_start_fen())
    p.add_argument("--depth", type=int, default=0, help="Глубина alpha-beta (0 = только NN)")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    engine = ChessEngine(checkpoint_path=args.checkpoint, config_path=args.config)
    result = engine.analyze(args.fen, depth=args.depth)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def chess_start_fen() -> str:
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


if __name__ == "__main__":
    main()
