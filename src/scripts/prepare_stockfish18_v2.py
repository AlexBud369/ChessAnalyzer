import os
import sys
import numpy as np
import pandas as pd
import chess
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.position_to_tensor_converter import fen_to_tensor_18ch
from scripts.evaluation_parser import parse_evaluation

CSV_PATH = "src/data/chess_data/stockfish_position_evaluations.csv"
OUTPUT_NPZ = "src/data/chess_data/dual_dataset_v1.npz"
MAX_POSITIONS = None
MAX_EVAL_PAWNS = 30.0
USE_TANH = True


def normalize_evaluation(pawn_score: float, max_pawns=MAX_EVAL_PAWNS, use_tanh=True):
    clipped = max(-max_pawns, min(max_pawns, pawn_score))
    if use_tanh:
        return np.tanh(clipped / max_pawns)
    else:
        return clipped / max_pawns

def uci_to_index(uci_str: str) -> int:
    move = chess.Move.from_uci(uci_str)
    return move.from_square * 64 + move.to_square

def main():
    print("Чтение CSV...")
    df = pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    if MAX_POSITIONS:
        df = df.head(MAX_POSITIONS)
    print(f"Всего позиций в CSV: {len(df)}")

    states = []
    values = []
    moves = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Обработка"):
        fen = row['fen']
        try:
            board = chess.Board(fen)
        except Exception:
            continue

        tensor = fen_to_tensor_18ch(fen)   # (8,8,18)
        states.append(tensor)

        eval_str = row['evaluation']
        pawn_score = parse_evaluation(eval_str)
        if pawn_score is None:
            states.pop()
            continue

        value = normalize_evaluation(pawn_score, MAX_EVAL_PAWNS, USE_TANH)
        values.append(value)

        move_uci = row['move_1']
        if not move_uci or move_uci == '':
            states.pop()
            values.pop()
            continue
        try:
            move_idx = uci_to_index(move_uci)
            moves.append(move_idx)
        except Exception:
            states.pop()
            values.pop()
            continue

    states_arr = np.array(states, dtype=np.float32)
    states_arr = np.transpose(states_arr, (0, 3, 1, 2))
    values_arr = np.array(values, dtype=np.float32)
    moves_arr = np.array(moves, dtype=np.int64)

    print(f"\nГотово: {len(states_arr)} позиций.")
    print(f"states shape: {states_arr.shape}")
    print(f"values shape: {values_arr.shape}")
    print(f"moves shape : {moves_arr.shape}")
    print(f"Статистика value: min={values_arr.min():.3f}, max={values_arr.max():.3f}, "
          f"mean={values_arr.mean():.3f}, std={values_arr.std():.3f}")

    np.savez_compressed(OUTPUT_NPZ,
                        states=states_arr,
                        values=values_arr,
                        moves=moves_arr)
    print(f"\nСохранено в: {OUTPUT_NPZ}")

if __name__ == "__main__":
    main()