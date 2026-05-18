import os
import sys
import argparse
import numpy as np
import torch
import chess
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
import glob
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.engine.dual_evaluator import DualEvaluator
from src.engine.mcts import MCTS
from src.utils.position_to_tensor_converter import fen_to_tensor_18ch


def play_one_game(params):
    """
    Генерация одной партии.
    params: (game_id, model_path, mcts_iterations, temp_start, temp_end, max_moves, seed)
    Возвращает dict с данными партии и информацией об исходе.
    """
    game_id, model_path, mcts_iter, temp_start, temp_end, max_moves, seed = params
    np.random.seed(seed)
    torch.manual_seed(seed)

    evaluator = DualEvaluator()
    evaluator.initialize(model_path)
    mcts = MCTS(evaluator, num_iterations=mcts_iter)

    board = chess.Board()
    game_positions = []
    move_count = 0

    while move_count < max_moves:
        if board.is_game_over():
            break

        t = temp_start * (1 - move_count / max_moves) + temp_end * (move_count / max_moves)

        mcts.reset_tree()
        policy = mcts.get_policy_prob(board, temperature=t)

        state = fen_to_tensor_18ch(board.fen())
        policy_array = np.zeros(4096, dtype=np.float32)
        for move, prob in policy.items():
            idx = move.from_square * 64 + move.to_square
            policy_array[idx] = prob

        game_positions.append({
            'state': state,
            'policy': policy_array,
            'player_turn': board.turn
        })

        move = mcts.sample_move(board, temperature=t)
        board.push(move)
        move_count += 1

        if board.is_game_over():
            break

    if board.is_checkmate():
        winner_is_white = (board.turn == chess.BLACK)
        result_value = 1.0 if winner_is_white else -1.0
        termination = 'checkmate'
    else:
        result_value = 0.0
        if board.is_stalemate():
            termination = 'stalemate'
        elif board.is_insufficient_material():
            termination = 'insufficient_material'
        elif board.is_seventyfive_moves():
            termination = '75_moves'
        elif board.is_fivefold_repetition():
            termination = '5fold_repetition'
        elif board.is_repetition(3):
            termination = '3fold_repetition'
        elif board.is_fifty_moves():
            termination = '50_moves'
        elif move_count >= max_moves:
            termination = 'max_moves_limit'
        else:
            termination = 'other_draw'

    for pos in game_positions:
        if pos['player_turn'] == chess.WHITE:
            pos['value'] = result_value
        else:
            pos['value'] = -result_value

    # Собираем массивы
    states = np.array([p['state'] for p in game_positions], dtype=np.float32)
    policies = np.array([p['policy'] for p in game_positions], dtype=np.float32)
    values = np.array([p['value'] for p in game_positions], dtype=np.float32).reshape(-1, 1)

    return {
        'game_id': game_id,
        'states': states,
        'policies': policies,
        'values': values,
        'termination': termination,
        'num_moves': move_count
    }


def generate_dataset_parallel(num_games, output_dir, model_path, mcts_iterations,
                              temp_start, temp_end, max_moves, num_workers=None):
    """
    Генерирует num_games партий параллельно.
    Каждая партия сохраняется в отдельный .npz в output_dir/games/
    После завершения всех партий объединяет их в output_dir/merged_selfplay.npz
    """
    os.makedirs(output_dir, exist_ok=True)
    games_dir = os.path.join(output_dir, 'games')
    os.makedirs(games_dir, exist_ok=True)

    for f in glob.glob(os.path.join(games_dir, 'game_*.npz')):
        os.remove(f)

    if num_workers is None:
        num_workers = min(cpu_count(), 12)
    print(f"Using {num_workers} parallel processes")

    seeds = np.random.randint(0, 2**31, size=num_games)
    tasks = [(i, model_path, mcts_iterations, temp_start, temp_end, max_moves, seeds[i])
             for i in range(num_games)]

    start_time = time.time()
    results = []
    with Pool(processes=num_workers) as pool:
        for res in tqdm(pool.imap_unordered(play_one_game, tasks), total=num_games,
                        desc="Generating games"):
            results.append(res)
            tmp_file = os.path.join(games_dir, f"game_{res['game_id']:06d}.npz")
            np.savez_compressed(tmp_file,
                                states=res['states'],
                                policies=res['policies'],
                                values=res['values'],
                                termination=res['termination'],
                                num_moves=res['num_moves'])

    elapsed = time.time() - start_time
    print(f"\nGeneration finished in {elapsed:.1f} seconds")
    print(f"Average game time: {elapsed/num_games:.2f} sec")

    print("Merging all games into one dataset...")
    all_states = []
    all_policies = []
    all_values = []
    termination_stats = {}

    for tmp_file in sorted(glob.glob(os.path.join(games_dir, "game_*.npz"))):
        data = np.load(tmp_file)
        all_states.append(data['states'])
        all_policies.append(data['policies'])
        all_values.append(data['values'])
        term = str(data['termination'].item() if hasattr(data['termination'], 'item') else data['termination'])
        termination_stats[term] = termination_stats.get(term, 0) + 1

    final_states = np.concatenate(all_states, axis=0)
    final_policies = np.concatenate(all_policies, axis=0)
    final_values = np.concatenate(all_values, axis=0)

    merged_path = os.path.join(output_dir, 'merged_selfplay.npz')
    np.savez_compressed(merged_path,
                        states=final_states,
                        policies=final_policies,
                        values=final_values)

    print(f"\nMerged dataset saved to: {merged_path}")
    print(f"Total positions: {len(final_states)}")
    print("\nGame termination statistics:")
    for term, count in sorted(termination_stats.items()):
        print(f"  {term:25s}: {count:4d} ({100*count/num_games:.1f}%)")

    stats_path = os.path.join(output_dir, 'generation_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Generated {num_games} games on {datetime.now()}\n")
        f.write(f"Parameters: mcts_iter={mcts_iterations}, max_moves={max_moves}, "
                f"temp={temp_start}->{temp_end}\n")
        f.write(f"Total positions: {len(final_states)}\n")
        for term, count in termination_stats.items():
            f.write(f"{term}: {count}\n")
    print(f"\nStats saved to {stats_path}")

    return merged_path


def copy_existing_dataset(source_file, dest_dir):
    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    dest_file = os.path.join(dest_dir, os.path.basename(source_file))
    if os.path.exists(dest_file):
        print(f"File {dest_file} already exists. Overwrite? (y/n)")
        resp = input().strip().lower()
        if resp != 'y':
            print("Aborted.")
            return
    shutil.copy2(source_file, dest_file)
    print(f"Copied {source_file} -> {dest_file}")


def main():
    parser = argparse.ArgumentParser(description="Self-play data generation with correct game endings")
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Команда generate
    gen_parser = subparsers.add_parser('generate', help='Generate self-play games')
    gen_parser.add_argument('--num-games', type=int, default=100, help='Number of games to generate')
    gen_parser.add_argument('--model-path', type=str, default='src/model2/chess_dual_best.pth')
    gen_parser.add_argument('--mcts-iterations', type=int, default=80)
    gen_parser.add_argument('--temperature-start', type=float, default=1.0)
    gen_parser.add_argument('--temperature-end', type=float, default=0.2)
    gen_parser.add_argument('--max-moves', type=int, default=120, help='Max moves per game')
    gen_parser.add_argument('--workers', type=int, default=None, help='Number of parallel processes')
    gen_parser.add_argument('--output-dir', type=str, default='src/data/chess_m',
                            help='Directory to save results (merged + games/)')

    copy_parser = subparsers.add_parser('copy', help='Copy existing .npz to chess_m folder')
    copy_parser.add_argument('source', type=str, help='Path to existing .npz file (e.g., selfplay_data_v1.npz)')
    copy_parser.add_argument('--dest-dir', type=str, default='src/data/chess_m', help='Destination directory')

    args = parser.parse_args()

    if args.command == 'copy':
        copy_existing_dataset(args.source, args.dest_dir)
        return

    if args.command != 'generate':
        parser.print_help()
        return

    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        sys.exit(1)

    print("Self-play generation started with parameters:")
    print(f"  Model: {args.model_path}")
    print(f"  Games: {args.num_games}")
    print(f"  MCTS iterations: {args.mcts_iterations}")
    print(f"  Max moves: {args.max_moves}")
    print(f"  Temperature: {args.temperature_start} -> {args.temperature_end}")
    print(f"  Output directory: {args.output_dir}")
    print()

    print("Warming up model on CPU...")
    evaluator = DualEvaluator()
    evaluator.initialize(args.model_path)
    _ = evaluator.evaluate(chess.Board())
    print("Model ready.\n")

    generate_dataset_parallel(
        num_games=args.num_games,
        output_dir=args.output_dir,
        model_path=args.model_path,
        mcts_iterations=args.mcts_iterations,
        temp_start=args.temperature_start,
        temp_end=args.temperature_end,
        max_moves=args.max_moves,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()