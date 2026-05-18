import sys
import os
import argparse
import chess
import chess.engine
import chess.pgn
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.engine.dual_evaluator import DualEvaluator
from src.engine.mcts import MCTS

CONFIG = {
    "model_path": "src/model2/chess_dual_best.pth",
    "engine_path": "src/stockfish/stockfish-windows-x86-64-avx2.exe",
    "num_games": 5,
    "mcts_iterations": 400,
    "elo": 1320,
    "time_limit": 1.0,
    "output_pgn": "matches_vs_stockfish.pgn"
}


def play_game(evaluator, engine_path, time_limit=1.0, mcts_iterations=400, model_white=True, elo=1320):
    """
    Сыграть одну партию.
    Возвращает (результат, game_object, board) где game_object - партия в формате chess.pgn.Game.
    """
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"Test vs Stockfish {elo}"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "Model" if model_white else "Stockfish"
    game.headers["Black"] = "Stockfish" if model_white else "Model"
    node = game

    if elo < 1320:
        print(f"Предупреждение: Stockfish требует UCI_Elo >= 1320. Установлено {elo} -> 1320")
        elo = 1320

    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        stockfish_limit = chess.engine.Limit(time=time_limit)
        mcts = MCTS(evaluator, num_iterations=mcts_iterations)

        while not board.is_game_over():
            if board.turn == chess.WHITE:
                if model_white:
                    move = mcts.get_best_move(board)
                else:
                    result = engine.play(board, stockfish_limit)
                    move = result.move
            else:
                if not model_white:
                    move = mcts.get_best_move(board)
                else:
                    result = engine.play(board, stockfish_limit)
                    move = result.move
            node = node.add_variation(move)
            board.push(move)

        if board.is_checkmate():
            winner = "White" if board.turn == chess.BLACK else "Black"
            result_str = "1-0" if winner == "White" else "0-1"
        else:
            result_str = "1/2-1/2"
        game.headers["Result"] = result_str
        return result_str, game, board


def run_match(model_path, engine_path, num_games=5, mcts_iterations=400,
              pgn_output="matches_vs_stockfish.pgn", elo=1320, time_limit=1.0):
    """
    Запускает матч из num_games партий.
    Сохраняет каждую партию в PGN-файл (дописывает).
    Выводит счёт после каждой партии.
    """
    print(f"Загрузка модели: {model_path}")
    evaluator = DualEvaluator()
    evaluator.initialize(model_path)

    scores = {"model": 0, "stockfish": 0, "draw": 0}
    played_games = 0

    os.makedirs(os.path.dirname(os.path.abspath(pgn_output)) or '.', exist_ok=True)

    for i in range(num_games):
        model_white = (i % 2 == 0)
        try:
            result_str, game, board = play_game(
                evaluator, engine_path,
                time_limit=time_limit,
                mcts_iterations=mcts_iterations,
                model_white=model_white,
                elo=elo
            )
            played_games += 1
        except Exception as e:
            print(f"\nОшибка в партии {i+1}: {e}")
            continue

        if result_str == "1/2-1/2":
            scores["draw"] += 1
        elif (result_str == "1-0" and model_white) or (result_str == "0-1" and not model_white):
            scores["model"] += 1
        else:
            scores["stockfish"] += 1

        print(f"Партия {i+1}: {result_str} -> Счёт: модель {scores['model']} : {scores['stockfish']} (ничьи {scores['draw']})")

        with open(pgn_output, "a", encoding="utf-8") as f:
            exporter = chess.pgn.StringExporter(headers=True, variations=False)
            f.write(game.accept(exporter) + "\n\n")

    if played_games == 0:
        print("\nНи одной партии не было сыграно из-за ошибок.")
        return

    total_points = scores["model"] + scores["draw"] / 2
    percent = total_points / played_games * 100
    print(f"Сыграно партий: {played_games} (запланировано {num_games})")
    print(f"Побед модели: {scores['model']}")
    print(f"Побед Stockfish: {scores['stockfish']}")
    print(f"Ничьих: {scores['draw']}")
    print(f"Очков модели: {total_points} / {played_games} ({percent:.1f}%)")
    print(f"PGN-файл сохранён: {os.path.abspath(pgn_output)}")
    if percent >= 45:
        print("Модель достигла уровня ~1600-1800 Elo!")
    else:
        print("Модель пока слабее. Рекомендуется дообучение на self-play.")


def main():
    parser = argparse.ArgumentParser(description="Тестирование модели против Stockfish")
    parser.add_argument("--model-path", type=str, default=CONFIG["model_path"],
                        help="Путь к модели")
    parser.add_argument("--engine-path", type=str, default=CONFIG["engine_path"],
                        help="Путь к исполняемому файлу Stockfish")
    parser.add_argument("--games", type=int, default=CONFIG["num_games"],
                        help="Количество партий")
    parser.add_argument("--mcts-iterations", type=int, default=CONFIG["mcts_iterations"],
                        help="Число итераций MCTS на ход")
    parser.add_argument("--elo", type=int, default=CONFIG["elo"],
                        help="Рейтинг Stockfish (минимум 1320)")
    parser.add_argument("--time-limit", type=float, default=CONFIG["time_limit"],
                        help="Лимит времени на ход для Stockfish (секунды)")
    parser.add_argument("--output", type=str, default=CONFIG["output_pgn"],
                        help="Имя выходного PGN-файла")
    args = parser.parse_args()

    if not os.path.exists(args.engine_path):
        print(f"Ошибка: Stockfish не найден по пути {args.engine_path}")
        print("Укажите правильный путь через --engine-path или измените CONFIG['engine_path']")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"Ошибка: Модель не найдена по пути {args.model_path}")
        sys.exit(1)

    # Принудительно корректируем elo, если ниже 1320
    if args.elo < 1320:
        print(f"Внимание: Установлен Elo={args.elo}, но Stockfish требует минимум 1320. Будет использовано 1320.")
        args.elo = 1320

    print(f"Настройки: Elo={args.elo}, MCTS итераций={args.mcts_iterations}, партий={args.games}, time_limit={args.time_limit}с")
    run_match(
        model_path=args.model_path,
        engine_path=args.engine_path,
        num_games=args.games,
        mcts_iterations=args.mcts_iterations,
        pgn_output=args.output,
        elo=args.elo,
        time_limit=args.time_limit
    )


if __name__ == "__main__":
    main()